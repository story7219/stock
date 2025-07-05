#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇺🇸 한국투자증권 API - 미국 지수 선물/옵션 데이터 수집
================================================
KIS API를 통한 미국 파생상품 실시간 데이터 수집
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class KISUSDerivativeData:
    """한투 미국 파생상품 데이터 구조"""
    symbol: str
    name: str
    underlying: str
    derivative_type: str  # 'future', 'option'
    contract_type: str = ""  # 'call', 'put' for options
    strike_price: float = 0.0
    expiry_date: str = ""
    current_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    change: float = 0.0
    change_percent: float = 0.0
    currency: str = "USD"
    exchange: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'underlying': self.underlying,
            'derivative_type': self.derivative_type,
            'contract_type': self.contract_type,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date,
            'current_price': self.current_price,
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'change': self.change,
            'change_percent': self.change_percent,
            'currency': self.currency,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat()
        }

class KISUSDerivativesAPI:
    """한국투자증권 미국 파생상품 API 클라이언트"""
    
    def __init__(self):
        self.app_key = os.getenv('LIVE_KIS_APP_KEY', '')
        self.app_secret = os.getenv('LIVE_KIS_APP_SECRET', '')
        self.account_number = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
        self.is_mock = os.getenv('IS_MOCK', 'true').lower() == 'true'
        
        # API 엔드포인트
        if self.is_mock:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        
        logger.info(f"🇺🇸 KIS 미국 파생상품 API 초기화 ({'모의투자' if self.is_mock else '실투자'})")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # 액세스 토큰 획득
        await self.get_access_token()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> bool:
        """액세스 토큰 획득"""
        if not self.app_key or not self.app_secret:
            logger.error("KIS API 키가 설정되지 않음")
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
                    logger.info("✅ KIS 액세스 토큰 획득 성공")
                    return True
                else:
                    logger.error(f"KIS 토큰 획득 실패: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"KIS 토큰 획득 오류: {e}")
            return False
    
    async def get_us_futures_list(self) -> List[KISUSDerivativeData]:
        """미국 지수 선물 리스트 조회"""
        if not self.access_token:
            logger.warning("액세스 토큰이 없음")
            return []
        
        try:
            # 미국 선물 종목 조회 API
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/trading/inquire-product-baseinfo"
            
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS00000300",  # 해외선물 종목기본정보조회
                "custtype": "P"
            }
            
            # 주요 미국 지수 선물 심볼들
            us_futures_symbols = [
                "ES",    # S&P 500 E-mini Future
                "NQ",    # NASDAQ 100 E-mini Future
                "YM",    # Dow Jones E-mini Future
                "RTY",   # Russell 2000 E-mini Future
                "VX",    # VIX Future
            ]
            
            futures = []
            
            for symbol in us_futures_symbols:
                params = {
                    "EXCD": "CME",  # 시카고상품거래소
                    "PDNO": symbol
                }
                
                try:
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get('output', {})
                            
                            if result:
                                future = KISUSDerivativeData(
                                    symbol=result.get('pdno', symbol),
                                    name=result.get('prdt_name', f"{symbol} Future"),
                                    underlying=symbol.replace("ES", "SPX").replace("NQ", "NDX").replace("YM", "DJI"),
                                    derivative_type='future',
                                    current_price=float(result.get('last', 0)),
                                    change=float(result.get('diff', 0)),
                                    change_percent=float(result.get('rate', 0)),
                                    exchange=result.get('excd', 'CME'),
                                    expiry_date=result.get('expr_date', '')
                                )
                                futures.append(future)
                                logger.info(f"✅ {symbol} 선물 정보 수집")
                        else:
                            logger.warning(f"⚠️ {symbol} 선물 조회 실패: {response.status}")
                            
                except Exception as e:
                    logger.error(f"{symbol} 선물 조회 오류: {e}")
                    continue
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
            
            return futures
            
        except Exception as e:
            logger.error(f"미국 선물 리스트 조회 오류: {e}")
            return []
    
    async def get_us_options_list(self) -> List[KISUSDerivativeData]:
        """미국 지수 옵션 리스트 조회"""
        if not self.access_token:
            logger.warning("액세스 토큰이 없음")
            return []
        
        try:
            # 미국 옵션 종목 조회 API
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/trading/inquire-option-product"
            
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS76240000",  # 해외옵션 종목조회
                "custtype": "P"
            }
            
            # 주요 미국 지수 옵션
            us_option_underlyings = [
                "SPX",   # S&P 500 Index Options
                "NDX",   # NASDAQ 100 Index Options
                "DJX",   # Dow Jones Index Options
                "RUT",   # Russell 2000 Index Options
            ]
            
            options = []
            
            for underlying in us_option_underlyings:
                params = {
                    "EXCD": "CBOE",  # 시카고옵션거래소
                    "PDNO": underlying,
                    "GUBN": "0"  # 전체
                }
                
                try:
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get('output', [])
                            
                            if not isinstance(results, list):
                                results = [results] if results else []
                            
                            for result in results[:20]:  # 상위 20개만
                                option = KISUSDerivativeData(
                                    symbol=result.get('pdno', ''),
                                    name=result.get('prdt_name', ''),
                                    underlying=underlying,
                                    derivative_type='option',
                                    contract_type=result.get('optn_type', '').lower(),
                                    strike_price=float(result.get('exer_px', 0)),
                                    current_price=float(result.get('last', 0)),
                                    bid=float(result.get('bid', 0)),
                                    ask=float(result.get('ask', 0)),
                                    volume=int(result.get('tvol', 0)),
                                    change=float(result.get('diff', 0)),
                                    change_percent=float(result.get('rate', 0)),
                                    exchange=result.get('excd', 'CBOE'),
                                    expiry_date=result.get('expr_date', '')
                                )
                                options.append(option)
                            
                            logger.info(f"✅ {underlying} 옵션 {len(results)}개 수집")
                        else:
                            logger.warning(f"⚠️ {underlying} 옵션 조회 실패: {response.status}")
                            
                except Exception as e:
                    logger.error(f"{underlying} 옵션 조회 오류: {e}")
                    continue
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
            
            return options
            
        except Exception as e:
            logger.error(f"미국 옵션 리스트 조회 오류: {e}")
            return []
    
    async def get_us_derivative_quote(self, symbol: str, exchange: str = "CME") -> Optional[KISUSDerivativeData]:
        """미국 파생상품 실시간 시세 조회"""
        if not self.access_token:
            return None
        
        try:
            # 해외선물옵션 현재가 조회
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/quotations/inquire-present-balance"
            
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS00000100",  # 해외선물옵션 현재가
                "custtype": "P"
            }
            
            params = {
                "EXCD": exchange,
                "SYMB": symbol
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('output', {})
                    
                    if result:
                        return KISUSDerivativeData(
                            symbol=symbol,
                            name=result.get('prdt_name', symbol),
                            underlying=result.get('und_symb', ''),
                            derivative_type='future' if exchange == 'CME' else 'option',
                            current_price=float(result.get('last', 0)),
                            bid=float(result.get('bid', 0)),
                            ask=float(result.get('ask', 0)),
                            volume=int(result.get('tvol', 0)),
                            change=float(result.get('diff', 0)),
                            change_percent=float(result.get('rate', 0)),
                            exchange=exchange
                        )
                else:
                    logger.warning(f"시세 조회 실패 {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"시세 조회 오류 {symbol}: {e}")
            return None
    
    async def get_us_derivatives_summary(self) -> Dict[str, Any]:
        """미국 파생상품 종합 현황"""
        try:
            # 선물과 옵션 데이터 수집
            futures = await self.get_us_futures_list()
            options = await self.get_us_options_list()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'futures': {
                    'count': len(futures),
                    'data': [f.to_dict() for f in futures]
                },
                'options': {
                    'count': len(options),
                    'data': [o.to_dict() for o in options[:10]]  # 상위 10개만
                },
                'total_derivatives': len(futures) + len(options),
                'exchanges': list(set([f.exchange for f in futures + options])),
                'available_underlyings': list(set([f.underlying for f in futures + options]))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"종합 현황 조회 오류: {e}")
            return {}
    
    async def test_api_connectivity(self) -> Dict[str, Any]:
        """API 연결성 테스트"""
        test_results = {
            'token_status': False,
            'futures_api': False,
            'options_api': False,
            'quote_api': False,
            'error_messages': []
        }
        
        try:
            # 1. 토큰 테스트
            if self.access_token:
                test_results['token_status'] = True
            else:
                test_results['error_messages'].append("액세스 토큰 없음")
            
            # 2. 선물 API 테스트
            try:
                futures = await self.get_us_futures_list()
                if futures:
                    test_results['futures_api'] = True
                else:
                    test_results['error_messages'].append("선물 데이터 없음")
            except Exception as e:
                test_results['error_messages'].append(f"선물 API 오류: {e}")
            
            # 3. 옵션 API 테스트
            try:
                options = await self.get_us_options_list()
                if options:
                    test_results['options_api'] = True
                else:
                    test_results['error_messages'].append("옵션 데이터 없음")
            except Exception as e:
                test_results['error_messages'].append(f"옵션 API 오류: {e}")
            
            # 4. 시세 API 테스트
            try:
                quote = await self.get_us_derivative_quote("ES", "CME")
                if quote:
                    test_results['quote_api'] = True
                else:
                    test_results['error_messages'].append("시세 데이터 없음")
            except Exception as e:
                test_results['error_messages'].append(f"시세 API 오류: {e}")
            
        except Exception as e:
            test_results['error_messages'].append(f"전체 테스트 오류: {e}")
        
        return test_results

# 헬퍼 함수
async def get_kis_us_derivatives() -> KISUSDerivativesAPI:
    """KIS 미국 파생상품 API 팩토리"""
    return KISUSDerivativesAPI()

# 테스트용 메인 함수
async def main():
    """KIS 미국 파생상품 API 테스트"""
    print("🇺🇸 한국투자증권 미국 파생상품 API 테스트")
    print("=" * 60)
    
    async with KISUSDerivativesAPI() as api:
        # 1. API 연결성 테스트
        print("\n1️⃣ API 연결성 테스트...")
        test_results = await api.test_api_connectivity()
        
        print(f"✅ 토큰 상태: {'성공' if test_results['token_status'] else '실패'}")
        print(f"✅ 선물 API: {'성공' if test_results['futures_api'] else '실패'}")
        print(f"✅ 옵션 API: {'성공' if test_results['options_api'] else '실패'}")
        print(f"✅ 시세 API: {'성공' if test_results['quote_api'] else '실패'}")
        
        if test_results['error_messages']:
            print("\n❌ 오류 메시지:")
            for error in test_results['error_messages']:
                print(f"   - {error}")
        
        # 2. 미국 선물 조회
        print("\n2️⃣ 미국 지수 선물 조회...")
        futures = await api.get_us_futures_list()
        print(f"✅ 선물 종목: {len(futures)}개")
        
        for future in futures:
            print(f"   📈 {future.symbol}: {future.name} - ${future.current_price:.2f} "
                  f"({future.change_percent:+.2f}%)")
        
        # 3. 미국 옵션 조회
        print("\n3️⃣ 미국 지수 옵션 조회...")
        options = await api.get_us_options_list()
        print(f"✅ 옵션 종목: {len(options)}개")
        
        for option in options[:5]:  # 상위 5개만
            print(f"   📊 {option.symbol}: {option.name} - ${option.current_price:.2f} "
                  f"(Strike: ${option.strike_price:.0f})")
        
        # 4. 종합 현황
        print("\n4️⃣ 종합 현황...")
        summary = await api.get_us_derivatives_summary()
        print(f"✅ 총 파생상품: {summary.get('total_derivatives', 0)}개")
        print(f"✅ 거래소: {', '.join(summary.get('exchanges', []))}")
        print(f"✅ 기초자산: {', '.join(summary.get('available_underlyings', []))}")
    
    print("\n🎯 결론:")
    print("   - 한국투자증권 API를 통해 미국 지수 선물/옵션 데이터 수집 가능 여부 확인됨")
    print("   - 실시간 시세 및 기본 정보 조회 기능 테스트 완료")
    print("   - 추가 기능: 실시간 WebSocket 스트리밍, 주문 기능 등")

if __name__ == "__main__":
    asyncio.run(main()) 
# -*- coding: utf-8 -*-
"""
🇺🇸 한국투자증권 API - 미국 지수 선물/옵션 데이터 수집
================================================
KIS API를 통한 미국 파생상품 실시간 데이터 수집
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class KISUSDerivativeData:
    """한투 미국 파생상품 데이터 구조"""
    symbol: str
    name: str
    underlying: str
    derivative_type: str  # 'future', 'option'
    contract_type: str = ""  # 'call', 'put' for options
    strike_price: float = 0.0
    expiry_date: str = ""
    current_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    change: float = 0.0
    change_percent: float = 0.0
    currency: str = "USD"
    exchange: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'underlying': self.underlying,
            'derivative_type': self.derivative_type,
            'contract_type': self.contract_type,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date,
            'current_price': self.current_price,
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'change': self.change,
            'change_percent': self.change_percent,
            'currency': self.currency,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat()
        }

class KISUSDerivativesAPI:
    """한국투자증권 미국 파생상품 API 클라이언트"""
    
    def __init__(self):
        self.app_key = os.getenv('LIVE_KIS_APP_KEY', '')
        self.app_secret = os.getenv('LIVE_KIS_APP_SECRET', '')
        self.account_number = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
        self.is_mock = os.getenv('IS_MOCK', 'true').lower() == 'true'
        
        # API 엔드포인트
        if self.is_mock:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        
        logger.info(f"🇺🇸 KIS 미국 파생상품 API 초기화 ({'모의투자' if self.is_mock else '실투자'})")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # 액세스 토큰 획득
        await self.get_access_token()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> bool:
        """액세스 토큰 획득"""
        if not self.app_key or not self.app_secret:
            logger.error("KIS API 키가 설정되지 않음")
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
                    logger.info("✅ KIS 액세스 토큰 획득 성공")
                    return True
                else:
                    logger.error(f"KIS 토큰 획득 실패: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"KIS 토큰 획득 오류: {e}")
            return False
    
    async def get_us_futures_list(self) -> List[KISUSDerivativeData]:
        """미국 지수 선물 리스트 조회"""
        if not self.access_token:
            logger.warning("액세스 토큰이 없음")
            return []
        
        try:
            # 미국 선물 종목 조회 API
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/trading/inquire-product-baseinfo"
            
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS00000300",  # 해외선물 종목기본정보조회
                "custtype": "P"
            }
            
            # 주요 미국 지수 선물 심볼들
            us_futures_symbols = [
                "ES",    # S&P 500 E-mini Future
                "NQ",    # NASDAQ 100 E-mini Future
                "YM",    # Dow Jones E-mini Future
                "RTY",   # Russell 2000 E-mini Future
                "VX",    # VIX Future
            ]
            
            futures = []
            
            for symbol in us_futures_symbols:
                params = {
                    "EXCD": "CME",  # 시카고상품거래소
                    "PDNO": symbol
                }
                
                try:
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get('output', {})
                            
                            if result:
                                future = KISUSDerivativeData(
                                    symbol=result.get('pdno', symbol),
                                    name=result.get('prdt_name', f"{symbol} Future"),
                                    underlying=symbol.replace("ES", "SPX").replace("NQ", "NDX").replace("YM", "DJI"),
                                    derivative_type='future',
                                    current_price=float(result.get('last', 0)),
                                    change=float(result.get('diff', 0)),
                                    change_percent=float(result.get('rate', 0)),
                                    exchange=result.get('excd', 'CME'),
                                    expiry_date=result.get('expr_date', '')
                                )
                                futures.append(future)
                                logger.info(f"✅ {symbol} 선물 정보 수집")
                        else:
                            logger.warning(f"⚠️ {symbol} 선물 조회 실패: {response.status}")
                            
                except Exception as e:
                    logger.error(f"{symbol} 선물 조회 오류: {e}")
                    continue
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
            
            return futures
            
        except Exception as e:
            logger.error(f"미국 선물 리스트 조회 오류: {e}")
            return []
    
    async def get_us_options_list(self) -> List[KISUSDerivativeData]:
        """미국 지수 옵션 리스트 조회"""
        if not self.access_token:
            logger.warning("액세스 토큰이 없음")
            return []
        
        try:
            # 미국 옵션 종목 조회 API
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/trading/inquire-option-product"
            
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS76240000",  # 해외옵션 종목조회
                "custtype": "P"
            }
            
            # 주요 미국 지수 옵션
            us_option_underlyings = [
                "SPX",   # S&P 500 Index Options
                "NDX",   # NASDAQ 100 Index Options
                "DJX",   # Dow Jones Index Options
                "RUT",   # Russell 2000 Index Options
            ]
            
            options = []
            
            for underlying in us_option_underlyings:
                params = {
                    "EXCD": "CBOE",  # 시카고옵션거래소
                    "PDNO": underlying,
                    "GUBN": "0"  # 전체
                }
                
                try:
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get('output', [])
                            
                            if not isinstance(results, list):
                                results = [results] if results else []
                            
                            for result in results[:20]:  # 상위 20개만
                                option = KISUSDerivativeData(
                                    symbol=result.get('pdno', ''),
                                    name=result.get('prdt_name', ''),
                                    underlying=underlying,
                                    derivative_type='option',
                                    contract_type=result.get('optn_type', '').lower(),
                                    strike_price=float(result.get('exer_px', 0)),
                                    current_price=float(result.get('last', 0)),
                                    bid=float(result.get('bid', 0)),
                                    ask=float(result.get('ask', 0)),
                                    volume=int(result.get('tvol', 0)),
                                    change=float(result.get('diff', 0)),
                                    change_percent=float(result.get('rate', 0)),
                                    exchange=result.get('excd', 'CBOE'),
                                    expiry_date=result.get('expr_date', '')
                                )
                                options.append(option)
                            
                            logger.info(f"✅ {underlying} 옵션 {len(results)}개 수집")
                        else:
                            logger.warning(f"⚠️ {underlying} 옵션 조회 실패: {response.status}")
                            
                except Exception as e:
                    logger.error(f"{underlying} 옵션 조회 오류: {e}")
                    continue
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
            
            return options
            
        except Exception as e:
            logger.error(f"미국 옵션 리스트 조회 오류: {e}")
            return []
    
    async def get_us_derivative_quote(self, symbol: str, exchange: str = "CME") -> Optional[KISUSDerivativeData]:
        """미국 파생상품 실시간 시세 조회"""
        if not self.access_token:
            return None
        
        try:
            # 해외선물옵션 현재가 조회
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/quotations/inquire-present-balance"
            
            headers = {
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS00000100",  # 해외선물옵션 현재가
                "custtype": "P"
            }
            
            params = {
                "EXCD": exchange,
                "SYMB": symbol
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('output', {})
                    
                    if result:
                        return KISUSDerivativeData(
                            symbol=symbol,
                            name=result.get('prdt_name', symbol),
                            underlying=result.get('und_symb', ''),
                            derivative_type='future' if exchange == 'CME' else 'option',
                            current_price=float(result.get('last', 0)),
                            bid=float(result.get('bid', 0)),
                            ask=float(result.get('ask', 0)),
                            volume=int(result.get('tvol', 0)),
                            change=float(result.get('diff', 0)),
                            change_percent=float(result.get('rate', 0)),
                            exchange=exchange
                        )
                else:
                    logger.warning(f"시세 조회 실패 {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"시세 조회 오류 {symbol}: {e}")
            return None
    
    async def get_us_derivatives_summary(self) -> Dict[str, Any]:
        """미국 파생상품 종합 현황"""
        try:
            # 선물과 옵션 데이터 수집
            futures = await self.get_us_futures_list()
            options = await self.get_us_options_list()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'futures': {
                    'count': len(futures),
                    'data': [f.to_dict() for f in futures]
                },
                'options': {
                    'count': len(options),
                    'data': [o.to_dict() for o in options[:10]]  # 상위 10개만
                },
                'total_derivatives': len(futures) + len(options),
                'exchanges': list(set([f.exchange for f in futures + options])),
                'available_underlyings': list(set([f.underlying for f in futures + options]))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"종합 현황 조회 오류: {e}")
            return {}
    
    async def test_api_connectivity(self) -> Dict[str, Any]:
        """API 연결성 테스트"""
        test_results = {
            'token_status': False,
            'futures_api': False,
            'options_api': False,
            'quote_api': False,
            'error_messages': []
        }
        
        try:
            # 1. 토큰 테스트
            if self.access_token:
                test_results['token_status'] = True
            else:
                test_results['error_messages'].append("액세스 토큰 없음")
            
            # 2. 선물 API 테스트
            try:
                futures = await self.get_us_futures_list()
                if futures:
                    test_results['futures_api'] = True
                else:
                    test_results['error_messages'].append("선물 데이터 없음")
            except Exception as e:
                test_results['error_messages'].append(f"선물 API 오류: {e}")
            
            # 3. 옵션 API 테스트
            try:
                options = await self.get_us_options_list()
                if options:
                    test_results['options_api'] = True
                else:
                    test_results['error_messages'].append("옵션 데이터 없음")
            except Exception as e:
                test_results['error_messages'].append(f"옵션 API 오류: {e}")
            
            # 4. 시세 API 테스트
            try:
                quote = await self.get_us_derivative_quote("ES", "CME")
                if quote:
                    test_results['quote_api'] = True
                else:
                    test_results['error_messages'].append("시세 데이터 없음")
            except Exception as e:
                test_results['error_messages'].append(f"시세 API 오류: {e}")
            
        except Exception as e:
            test_results['error_messages'].append(f"전체 테스트 오류: {e}")
        
        return test_results

# 헬퍼 함수
async def get_kis_us_derivatives() -> KISUSDerivativesAPI:
    """KIS 미국 파생상품 API 팩토리"""
    return KISUSDerivativesAPI()

# 테스트용 메인 함수
async def main():
    """KIS 미국 파생상품 API 테스트"""
    print("🇺🇸 한국투자증권 미국 파생상품 API 테스트")
    print("=" * 60)
    
    async with KISUSDerivativesAPI() as api:
        # 1. API 연결성 테스트
        print("\n1️⃣ API 연결성 테스트...")
        test_results = await api.test_api_connectivity()
        
        print(f"✅ 토큰 상태: {'성공' if test_results['token_status'] else '실패'}")
        print(f"✅ 선물 API: {'성공' if test_results['futures_api'] else '실패'}")
        print(f"✅ 옵션 API: {'성공' if test_results['options_api'] else '실패'}")
        print(f"✅ 시세 API: {'성공' if test_results['quote_api'] else '실패'}")
        
        if test_results['error_messages']:
            print("\n❌ 오류 메시지:")
            for error in test_results['error_messages']:
                print(f"   - {error}")
        
        # 2. 미국 선물 조회
        print("\n2️⃣ 미국 지수 선물 조회...")
        futures = await api.get_us_futures_list()
        print(f"✅ 선물 종목: {len(futures)}개")
        
        for future in futures:
            print(f"   📈 {future.symbol}: {future.name} - ${future.current_price:.2f} "
                  f"({future.change_percent:+.2f}%)")
        
        # 3. 미국 옵션 조회
        print("\n3️⃣ 미국 지수 옵션 조회...")
        options = await api.get_us_options_list()
        print(f"✅ 옵션 종목: {len(options)}개")
        
        for option in options[:5]:  # 상위 5개만
            print(f"   📊 {option.symbol}: {option.name} - ${option.current_price:.2f} "
                  f"(Strike: ${option.strike_price:.0f})")
        
        # 4. 종합 현황
        print("\n4️⃣ 종합 현황...")
        summary = await api.get_us_derivatives_summary()
        print(f"✅ 총 파생상품: {summary.get('total_derivatives', 0)}개")
        print(f"✅ 거래소: {', '.join(summary.get('exchanges', []))}")
        print(f"✅ 기초자산: {', '.join(summary.get('available_underlyings', []))}")
    
    print("\n🎯 결론:")
    print("   - 한국투자증권 API를 통해 미국 지수 선물/옵션 데이터 수집 가능 여부 확인됨")
    print("   - 실시간 시세 및 기본 정보 조회 기능 테스트 완료")
    print("   - 추가 기능: 실시간 WebSocket 스트리밍, 주문 기능 등")

if __name__ == "__main__":
    asyncio.run(main()) 