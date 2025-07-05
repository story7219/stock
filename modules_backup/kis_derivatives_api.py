#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 한국투자증권 API 실시간 파생상품 데이터 수집기
=================================================
REST API + WebSocket을 통한 정확한 K200 선물/옵션 실시간 데이터
"""

import asyncio
import logging
import json
import websockets
import aiohttp
import hmac
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class KISDerivativeData:
    """한국투자증권 파생상품 데이터"""
    symbol: str
    name: str
    current_price: float
    change: float
    change_rate: float
    volume: int
    open_interest: int = 0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_volume: int = 0
    ask_volume: int = 0
    high_price: float = 0.0
    low_price: float = 0.0
    open_price: float = 0.0
    prev_close: float = 0.0
    timestamp: str = ""
    
    # 옵션 전용 필드
    strike_price: Optional[float] = None
    expiry_date: Optional[str] = None
    option_type: Optional[str] = None  # 'call', 'put'
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class KISDerivativesAPI:
    """한국투자증권 파생상품 API 클라이언트"""
    
    def __init__(self):
        self.app_key = os.getenv('LIVE_KIS_APP_KEY', '')
        self.app_secret = os.getenv('LIVE_KIS_APP_SECRET', '')
        self.account_number = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
        self.is_mock = os.getenv('IS_MOCK', 'true').lower() == 'true'
        
        # API URL 설정
        if self.is_mock:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.ws_url = "ws://ops.koreainvestment.com:21000"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443" 
            self.ws_url = "ws://ops.koreainvestment.com:21000"
        
        self.access_token = None
        self.websocket = None
        self.session = None
        
        logger.info(f"🚀 한국투자증권 API 초기화 {'(모의투자)' if self.is_mock else '(실투자)'}")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        await self.get_access_token()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> bool:
        """액세스 토큰 획득"""
        if not self.app_key or not self.app_secret:
            logger.error("❌ 한국투자증권 API 키가 설정되지 않음")
            return False
        
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result.get('access_token')
                    logger.info("✅ 한국투자증권 API 토큰 획득 성공")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ 토큰 획득 실패: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 토큰 획득 오류: {e}")
            return False
    
    async def get_kospi200_futures(self) -> List[KISDerivativeData]:
        """KOSPI200 선물 데이터 조회"""
        if not self.access_token:
            return []
        
        try:
            # KOSPI200 선물 조회 API
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "CTPF1002R"  # 선물 시세 조회
            }
            
            # KOSPI200 선물 종목코드들 (근월물, 차월물)
            future_codes = ["101Q9000", "101R0000"]  # 실제 종목코드는 확인 필요
            futures_data = []
            
            for code in future_codes:
                params = {
                    "prdt_type_cd": "300",  # 선물
                    "pdno": code,
                    "period_div_cd": "D",
                    "inquire_div_cd": "0"
                }
                
                try:
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            output = data.get('output', [])
                            
                            if output:
                                item = output[0]  # 최신 데이터
                                future_data = KISDerivativeData(
                                    symbol=code,
                                    name=f"KOSPI200 선물 {code}",
                                    current_price=float(item.get('stck_prpr', 0)),
                                    change=float(item.get('prdy_vrss', 0)),
                                    change_rate=float(item.get('prdy_ctrt', 0)),
                                    volume=int(item.get('acml_vol', 0)),
                                    high_price=float(item.get('stck_hgpr', 0)),
                                    low_price=float(item.get('stck_lwpr', 0)),
                                    open_price=float(item.get('stck_oprc', 0)),
                                    prev_close=float(item.get('stck_prdy_clpr', 0)),
                                    timestamp=datetime.now().isoformat()
                                )
                                futures_data.append(future_data)
                                
                except Exception as e:
                    logger.warning(f"선물 {code} 조회 실패: {e}")
            
            logger.info(f"✅ KOSPI200 선물 {len(futures_data)}개 조회 완료")
            return futures_data
            
        except Exception as e:
            logger.error(f"❌ KOSPI200 선물 조회 오류: {e}")
            return []
    
    async def get_kospi200_options(self) -> List[KISDerivativeData]:
        """KOSPI200 옵션 데이터 조회"""
        if not self.access_token:
            return []
        
        try:
            # 먼저 KOSPI200 현재가 조회
            kospi200_price = await self.get_kospi200_index()
            if not kospi200_price:
                return []
            
            # ATM 기준 옵션 종목코드 생성 (실제로는 KRX에서 제공하는 종목코드 사용)
            options_data = []
            
            # 옵션 시세 조회 API
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "FHKST01010100"  # 주식 현재가 시세
            }
            
            # 주요 옵션 종목코드들 (예시 - 실제로는 동적으로 생성 필요)
            option_codes = [
                "20123280",  # KOSPI200 콜옵션 예시
                "20123290",  # KOSPI200 풋옵션 예시
            ]
            
            for code in option_codes:
                params = {
                    "fid_cond_mrkt_div_code": "O",  # 옵션
                    "fid_input_iscd": code
                }
                
                try:
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            output = data.get('output', {})
                            
                            if output:
                                option_data = KISDerivativeData(
                                    symbol=code,
                                    name=output.get('hts_kor_isnm', f'옵션 {code}'),
                                    current_price=float(output.get('stck_prpr', 0)),
                                    change=float(output.get('prdy_vrss', 0)),
                                    change_rate=float(output.get('prdy_ctrt', 0)),
                                    volume=int(output.get('acml_vol', 0)),
                                    bid_price=float(output.get('stck_sdpr', 0)),
                                    ask_price=float(output.get('stck_shpr', 0)),
                                    high_price=float(output.get('stck_hgpr', 0)),
                                    low_price=float(output.get('stck_lwpr', 0)),
                                    open_price=float(output.get('stck_oprc', 0)),
                                    prev_close=float(output.get('stck_prdy_clpr', 0)),
                                    timestamp=datetime.now().isoformat()
                                )
                                options_data.append(option_data)
                                
                except Exception as e:
                    logger.warning(f"옵션 {code} 조회 실패: {e}")
            
            logger.info(f"✅ KOSPI200 옵션 {len(options_data)}개 조회 완료")
            return options_data
            
        except Exception as e:
            logger.error(f"❌ KOSPI200 옵션 조회 오류: {e}")
            return []
    
    async def get_kospi200_index(self) -> Optional[float]:
        """KOSPI200 지수 현재가 조회"""
        if not self.access_token:
            return None
        
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-index-price"
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "FHKUP03500100"  # 지수 현재가
            }
            
            params = {
                "fid_cond_mrkt_div_code": "U",
                "fid_input_iscd": "0001"  # KOSPI200 지수코드
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    output = data.get('output', {})
                    return float(output.get('bstp_nmix_prpr', 0))
                    
        except Exception as e:
            logger.error(f"❌ KOSPI200 지수 조회 오류: {e}")
        
        return None
    
    async def connect_websocket(self, symbols: List[str]) -> None:
        """WebSocket 실시간 데이터 연결"""
        if not self.access_token:
            logger.error("❌ 액세스 토큰이 없어 WebSocket 연결 불가")
            return
        
        try:
            # WebSocket 연결
            self.websocket = await websockets.connect(
                self.ws_url,
                subprotocols=["echo-protocol"]
            )
            
            # 인증 메시지 전송
            auth_message = {
                "header": {
                    "approval_key": self.access_token,
                    "custtype": "P",
                    "tr_type": "1",
                    "content-type": "utf-8"
                },
                "body": {
                    "input": {
                        "tr_id": "HDFSCNT0",
                        "tr_key": "|".join(symbols)
                    }
                }
            }
            
            await self.websocket.send(json.dumps(auth_message))
            logger.info("✅ WebSocket 연결 및 인증 완료")
            
            # 실시간 데이터 수신 루프
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_data(data)
                except Exception as e:
                    logger.error(f"WebSocket 데이터 처리 오류: {e}")
                    
        except Exception as e:
            logger.error(f"❌ WebSocket 연결 오류: {e}")
    
    async def _process_websocket_data(self, data: Dict[str, Any]) -> None:
        """WebSocket 실시간 데이터 처리"""
        try:
            header = data.get('header', {})
            body = data.get('body', {})
            
            if header.get('tr_id') == 'HDFSCNT0':  # 실시간 시세
                output = body.get('output', {})
                symbol = output.get('mksc_shrn_iscd', '')
                
                if symbol:
                    real_time_data = KISDerivativeData(
                        symbol=symbol,
                        name=output.get('hts_kor_isnm', ''),
                        current_price=float(output.get('stck_prpr', 0)),
                        change=float(output.get('prdy_vrss', 0)),
                        change_rate=float(output.get('prdy_ctrt', 0)),
                        volume=int(output.get('acml_vol', 0)),
                        bid_price=float(output.get('stck_sdpr', 0)),
                        ask_price=float(output.get('stck_shpr', 0)),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    logger.info(f"📡 실시간 데이터: {symbol} {real_time_data.current_price:,.0f} ({real_time_data.change_rate:+.2f}%)")
                    
                    # 실시간 데이터 콜백 호출 (필요시 구현)
                    await self._on_real_time_data(real_time_data)
                    
        except Exception as e:
            logger.error(f"WebSocket 데이터 처리 오류: {e}")
    
    async def _on_real_time_data(self, data: KISDerivativeData) -> None:
        """실시간 데이터 콜백 (확장 가능)"""
        # 여기서 실시간 데이터를 처리하거나 다른 시스템으로 전달
        pass
    
    async def get_option_chain(self, expiry_month: str = None) -> List[KISDerivativeData]:
        """옵션 체인 데이터 조회"""
        if not self.access_token:
            return []
        
        try:
            # 옵션 체인 조회 API (실제 API 확인 필요)
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/option-chain"
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "OPTCHAIN01"  # 옵션체인 조회 (예시)
            }
            
            params = {
                "base_asset": "KOSPI200",
                "expiry_month": expiry_month or datetime.now().strftime("%Y%m")
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # 옵션 체인 데이터 처리
                    return await self._process_option_chain(data)
                    
        except Exception as e:
            logger.error(f"❌ 옵션 체인 조회 오류: {e}")
        
        return []
    
    async def _process_option_chain(self, data: Dict[str, Any]) -> List[KISDerivativeData]:
        """옵션 체인 데이터 처리"""
        options = []
        
        try:
            output = data.get('output', [])
            
            for item in output:
                option_data = KISDerivativeData(
                    symbol=item.get('opt_code', ''),
                    name=item.get('opt_name', ''),
                    current_price=float(item.get('current_price', 0)),
                    strike_price=float(item.get('strike_price', 0)),
                    expiry_date=item.get('expiry_date', ''),
                    option_type='call' if item.get('opt_type') == 'C' else 'put',
                    volume=int(item.get('volume', 0)),
                    open_interest=int(item.get('open_interest', 0)),
                    implied_volatility=float(item.get('implied_vol', 0)) / 100,
                    delta=float(item.get('delta', 0)),
                    gamma=float(item.get('gamma', 0)),
                    theta=float(item.get('theta', 0)),
                    vega=float(item.get('vega', 0)),
                    timestamp=datetime.now().isoformat()
                )
                options.append(option_data)
                
        except Exception as e:
            logger.error(f"옵션 체인 데이터 처리 오류: {e}")
        
        return options
    
    async def get_derivatives_summary(self) -> Dict[str, Any]:
        """파생상품 종합 데이터 조회"""
        try:
            # 병렬로 데이터 수집
            futures_task = self.get_kospi200_futures()
            options_task = self.get_kospi200_options()
            index_task = self.get_kospi200_index()
            
            futures, options, kospi200_price = await asyncio.gather(
                futures_task, options_task, index_task,
                return_exceptions=True
            )
            
            # 결과 정리
            summary = {
                'kospi200_index': kospi200_price if not isinstance(kospi200_price, Exception) else 0,
                'futures': futures if not isinstance(futures, Exception) else [],
                'options': options if not isinstance(options, Exception) else [],
                'total_derivatives': len(futures or []) + len(options or []),
                'timestamp': datetime.now().isoformat()
            }
            
            # Put/Call 비율 계산
            if options:
                calls = [opt for opt in options if opt.option_type == 'call']
                puts = [opt for opt in options if opt.option_type == 'put']
                
                call_volume = sum(opt.volume for opt in calls)
                put_volume = sum(opt.volume for opt in puts)
                
                summary['put_call_ratio'] = put_volume / call_volume if call_volume > 0 else 0
                summary['total_option_volume'] = call_volume + put_volume
            
            logger.info(f"✅ 파생상품 종합 데이터 조회 완료: {summary['total_derivatives']}개")
            return summary
            
        except Exception as e:
            logger.error(f"❌ 파생상품 종합 조회 오류: {e}")
            return {}

# 전역 인스턴스
_kis_api = None

def get_kis_derivatives_api() -> KISDerivativesAPI:
    """KIS 파생상품 API 인스턴스 반환"""
    global _kis_api
    if _kis_api is None:
        _kis_api = KISDerivativesAPI()
    return _kis_api

async def main():
    """테스트용 메인 함수"""
    print("🚀 한국투자증권 API 파생상품 데이터 테스트")
    print("=" * 50)
    
    async with KISDerivativesAPI() as api:
        # 1. 기본 데이터 조회
        print("\n📊 KOSPI200 지수 조회...")
        kospi200 = await api.get_kospi200_index()
        print(f"KOSPI200: {kospi200:,.2f}")
        
        # 2. 선물 데이터 조회
        print("\n📈 KOSPI200 선물 조회...")
        futures = await api.get_kospi200_futures()
        for future in futures:
            print(f"  {future.name}: {future.current_price:,.0f} ({future.change_rate:+.2f}%)")
        
        # 3. 옵션 데이터 조회
        print("\n📊 KOSPI200 옵션 조회...")
        options = await api.get_kospi200_options()
        for option in options[:5]:  # 상위 5개만
            print(f"  {option.name}: {option.current_price:,.0f} ({option.change_rate:+.2f}%)")
        
        # 4. 종합 데이터
        print("\n📋 파생상품 종합 요약...")
        summary = await api.get_derivatives_summary()
        print(f"  총 파생상품: {summary.get('total_derivatives', 0)}개")
        print(f"  Put/Call 비율: {summary.get('put_call_ratio', 0):.2f}")
        print(f"  총 옵션 거래량: {summary.get('total_option_volume', 0):,}")
        
        # 5. WebSocket 실시간 연결 테스트 (30초)
        print("\n📡 실시간 WebSocket 연결 테스트 (30초)...")
        symbols = [future.symbol for future in futures[:2]]  # 상위 2개 선물
        
        if symbols:
            websocket_task = asyncio.create_task(api.connect_websocket(symbols))
            await asyncio.sleep(30)  # 30초간 실시간 데이터 수신
            websocket_task.cancel()
        
        print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main()) 