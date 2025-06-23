"""
한국투자증권 API 연동 모듈
REST API, WebSocket API 자동 토큰 관리 시스템
🚀 오전 7시 자동 토큰 갱신 및 API 요청 한도 최적화
"""

import asyncio
import logging
import json
import os
import time
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import requests
import websocket
import threading
from urllib.parse import urlencode
import schedule

logger = logging.getLogger(__name__)

@dataclass
class KISToken:
    """KIS API 토큰 정보"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 86400  # 24시간
    issued_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_expired(self) -> bool:
        """토큰 만료 여부 확인"""
        return datetime.now() > (self.issued_at + timedelta(seconds=self.expires_in - 300))  # 5분 여유
    
    @property
    def authorization_header(self) -> str:
        """Authorization 헤더 값"""
        return f"{self.token_type} {self.access_token}"

@dataclass
class KISConfig:
    """KIS API 설정"""
    app_key: str
    app_secret: str
    account_number: str
    account_product_code: str = "01"
    base_url: str = "https://openapi.koreainvestment.com:9443"
    websocket_url: str = "ws://ops.koreainvestment.com:21000"
    is_mock: bool = False
    
    def __post_init__(self):
        if self.is_mock:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.websocket_url = "ws://ops.koreainvestment.com:31000"

class KISAPIManager:
    """한국투자증권 API 관리자"""
    
    def __init__(self, config: KISConfig):
        self.config = config
        self.token: Optional[KISToken] = None
        self.session = requests.Session()
        self.websocket_client: Optional[websocket.WebSocketApp] = None
        self.request_count = 0
        self.request_limit = 1000  # 일일 요청 제한
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # 100ms 딜레이
        
        # 자동 토큰 갱신 스케줄 설정
        self._setup_token_scheduler()
        
    def _setup_token_scheduler(self):
        """토큰 자동 갱신 스케줄러 설정"""
        # 매일 오전 7시에 토큰 갱신
        schedule.every().day.at("07:00").do(self._refresh_token_job)
        
        # 토큰 만료 30분 전에도 갱신
        schedule.every(30).minutes.do(self._check_and_refresh_token)
        
        # 스케줄러 백그라운드 실행
        threading.Thread(target=self._run_scheduler, daemon=True).start()
    
    def _run_scheduler(self):
        """스케줄러 실행"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크
    
    def _refresh_token_job(self):
        """토큰 갱신 작업"""
        try:
            logger.info("🔄 정기 토큰 갱신 시작 (오전 7시)")
            self.get_access_token(force_refresh=True)
            logger.info("✅ 정기 토큰 갱신 완료")
        except Exception as e:
            logger.error(f"❌ 정기 토큰 갱신 실패: {e}")
    
    def _check_and_refresh_token(self):
        """토큰 만료 체크 및 갱신"""
        if self.token and self.token.is_expired:
            try:
                logger.info("🔄 토큰 만료로 인한 자동 갱신")
                self.get_access_token(force_refresh=True)
                logger.info("✅ 토큰 자동 갱신 완료")
            except Exception as e:
                logger.error(f"❌ 토큰 자동 갱신 실패: {e}")
    
    def get_access_token(self, force_refresh: bool = False) -> KISToken:
        """액세스 토큰 획득"""
        if self.token and not self.token.is_expired and not force_refresh:
            return self.token
        
        try:
            url = f"{self.config.base_url}/oauth2/tokenP"
            headers = {
                "content-type": "application/json; charset=utf-8"
            }
            data = {
                "grant_type": "client_credentials",
                "appkey": self.config.app_key,
                "appsecret": self.config.app_secret
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            token_data = response.json()
            
            self.token = KISToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in", 86400)
            )
            
            logger.info(f"✅ KIS 토큰 획득 성공 (만료: {self.token.issued_at + timedelta(seconds=self.token.expires_in)})")
            return self.token
            
        except Exception as e:
            logger.error(f"❌ KIS 토큰 획득 실패: {e}")
            raise
    
    def _wait_for_rate_limit(self):
        """API 요청 제한 대기"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count >= self.request_limit:
            logger.warning(f"⚠️ 일일 API 요청 제한 도달: {self.request_count}/{self.request_limit}")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None, tr_id: Optional[str] = None) -> Dict:
        """API 요청 실행"""
        self._wait_for_rate_limit()
        
        # 토큰 확인 및 갱신
        if not self.token or self.token.is_expired:
            self.get_access_token(force_refresh=True)
        
        url = f"{self.config.base_url}{endpoint}"
        headers = {
            "Authorization": self.token.authorization_header,
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "content-type": "application/json; charset=utf-8"
        }
        
        if tr_id:
            headers["tr_id"] = tr_id
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=headers, json=data)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ KIS API 요청 실패 [{method} {endpoint}]: {e}")
            raise
    
    async def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """주식 현재가 조회"""
        try:
            endpoint = "/uapi/domestic-stock/v1/quotations/inquire-price"
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol
            }
            
            result = self._make_request("GET", endpoint, params=params, tr_id="FHKST01010100")
            
            if result.get("rt_cd") == "0":
                output = result.get("output", {})
                return {
                    "symbol": symbol,
                    "price": float(output.get("stck_prpr", 0)),
                    "change": float(output.get("prdy_vrss", 0)),
                    "change_rate": float(output.get("prdy_ctrt", 0)),
                    "volume": int(output.get("acml_vol", 0)),
                    "timestamp": datetime.now()
                }
            else:
                logger.error(f"❌ 주식 현재가 조회 실패: {result.get('msg1', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ 주식 현재가 조회 실패 {symbol}: {e}")
            return {}
    
    async def get_kospi200_list(self) -> List[str]:
        """코스피200 종목 리스트 조회"""
        try:
            endpoint = "/uapi/domestic-stock/v1/quotations/inquire-index-timeseries"
            params = {
                "fid_cond_mrkt_div_code": "U",
                "fid_input_iscd": "0001",  # 코스피200 지수
                "fid_input_date_1": (datetime.now() - timedelta(days=1)).strftime("%Y%m%d"),
                "fid_input_date_2": datetime.now().strftime("%Y%m%d"),
                "fid_period_div_code": "D"
            }
            
            result = self._make_request("GET", endpoint, params=params, tr_id="FHKUP03500100")
            
            # 실제로는 별도의 API로 코스피200 구성 종목을 가져와야 함
            # 여기서는 주요 종목들을 하드코딩
            kospi200_symbols = [
                "005930",  # 삼성전자
                "000660",  # SK하이닉스
                "373220",  # LG에너지솔루션
                "207940",  # 삼성바이오로직스
                "005380",  # 현대차
                "051910",  # LG화학
                "035420",  # NAVER
                "012330",  # 현대모비스
                "028260",  # 삼성물산
                "006400",  # 삼성SDI
            ]
            
            logger.info(f"✅ 코스피200 종목 리스트 조회 완료: {len(kospi200_symbols)}개")
            return kospi200_symbols
            
        except Exception as e:
            logger.error(f"❌ 코스피200 종목 리스트 조회 실패: {e}")
            return []
    
    def start_websocket(self, symbols: List[str], callback_func):
        """WebSocket 실시간 데이터 수신 시작"""
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    callback_func(data)
                except Exception as e:
                    logger.error(f"❌ WebSocket 메시지 처리 실패: {e}")
            
            def on_error(ws, error):
                logger.error(f"❌ WebSocket 오류: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("🔌 WebSocket 연결 종료")
            
            def on_open(ws):
                logger.info("🔌 WebSocket 연결 성공")
                
                # 종목 구독 요청
                for symbol in symbols:
                    subscribe_data = {
                        "header": {
                            "approval_key": self.token.access_token,
                            "custtype": "P",
                            "tr_type": "1",
                            "content-type": "utf-8"
                        },
                        "body": {
                            "input": {
                                "tr_id": "H0STCNT0",
                                "tr_key": symbol
                            }
                        }
                    }
                    ws.send(json.dumps(subscribe_data))
            
            self.websocket_client = websocket.WebSocketApp(
                self.config.websocket_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # 별도 스레드에서 WebSocket 실행
            websocket_thread = threading.Thread(
                target=self.websocket_client.run_forever,
                daemon=True
            )
            websocket_thread.start()
            
            logger.info("🚀 WebSocket 실시간 데이터 수신 시작")
            
        except Exception as e:
            logger.error(f"❌ WebSocket 시작 실패: {e}")
    
    def stop_websocket(self):
        """WebSocket 중지"""
        if self.websocket_client:
            self.websocket_client.close()
            logger.info("🔌 WebSocket 연결 중지")

class KISDataCollector:
    """KIS API 기반 데이터 수집기"""
    
    def __init__(self, config: KISConfig):
        self.api_manager = KISAPIManager(config)
        self.realtime_data = {}
        
    async def collect_kospi200_data(self) -> List[Dict[str, Any]]:
        """코스피200 종목 데이터 수집"""
        try:
            logger.info("🚀 KIS API를 통한 코스피200 데이터 수집 시작")
            
            # 토큰 획득
            self.api_manager.get_access_token()
            
            # 코스피200 종목 리스트 조회
            symbols = await self.api_manager.get_kospi200_list()
            
            # 각 종목의 현재가 정보 수집
            stock_data = []
            for symbol in symbols[:20]:  # API 제한으로 20개만 테스트
                try:
                    price_data = await self.api_manager.get_stock_price(symbol)
                    if price_data:
                        stock_data.append(price_data)
                    
                    # API 제한 준수를 위한 딜레이
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ 종목 {symbol} 데이터 수집 실패: {e}")
                    continue
            
            logger.info(f"✅ KIS API 코스피200 데이터 수집 완료: {len(stock_data)}개 종목")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ KIS API 데이터 수집 실패: {e}")
            return []
    
    def start_realtime_monitoring(self, symbols: List[str]):
        """실시간 데이터 모니터링 시작"""
        def realtime_callback(data):
            try:
                # 실시간 데이터 처리
                symbol = data.get("body", {}).get("output", {}).get("mksc_shrn_iscd")
                if symbol:
                    self.realtime_data[symbol] = {
                        "price": data.get("body", {}).get("output", {}).get("stck_prpr"),
                        "volume": data.get("body", {}).get("output", {}).get("cntg_vol"),
                        "timestamp": datetime.now()
                    }
                    logger.debug(f"📊 실시간 데이터 업데이트: {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ 실시간 데이터 처리 실패: {e}")
        
        # WebSocket 시작
        self.api_manager.start_websocket(symbols, realtime_callback)
    
    def get_realtime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """실시간 데이터 조회"""
        return self.realtime_data.get(symbol)

# 사용 예시
async def main():
    """KIS API 테스트"""
    # 환경변수에서 설정 로드
    config = KISConfig(
        app_key=os.getenv("KIS_APP_KEY", ""),
        app_secret=os.getenv("KIS_APP_SECRET", ""),
        account_number=os.getenv("KIS_ACCOUNT_NUMBER", ""),
        is_mock=os.getenv("IS_MOCK", "true").lower() == "true"
    )
    
    collector = KISDataCollector(config)
    
    # 코스피200 데이터 수집
    data = await collector.collect_kospi200_data()
    print(f"수집된 데이터: {len(data)}개")
    
    # 실시간 모니터링 시작 (선택사항)
    # collector.start_realtime_monitoring(["005930", "000660"])

if __name__ == "__main__":
    asyncio.run(main()) 