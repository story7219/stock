"""
💎 핵심 트레이딩 엔진 (v6.0, 리팩토링)
- 타입 힌트 추가로 코드 안정성 향상
- 함수 분리 및 모듈화로 가독성 향상
- 성능 최적화 및 최신 코딩 표준 적용
- 전략 로직 완전 보존
"""
import asyncio
import json
import logging
import time
import threading
import base64
from datetime import datetime, timedelta, time as time_obj
from collections import deque
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum

import requests
import gspread
import websocket
from google.oauth2.service_account import Credentials

from utils.telegram_bot import TelegramNotifier
import config

logger = logging.getLogger(__name__)

# === 📊 데이터 모델 정의 ===

class OrderSide(Enum):
    """주문 방향"""
    BUY = "01"
    SELL = "02"

class OrderType(Enum):
    """주문 유형"""
    MARKET = "01"  # 시장가
    LIMIT = "00"   # 지정가

@dataclass
class PriceData:
    """가격 정보 데이터 클래스"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    name: Optional[str] = None

@dataclass
class BalanceInfo:
    """계좌 잔고 정보"""
    cash: float
    total_value: float
    positions: Dict[str, Dict[str, Union[int, float]]]
    profit_loss: float

@dataclass
class OrderRequest:
    """주문 요청 정보"""
    symbol: str
    side: OrderSide
    quantity: int
    price: float = 0
    order_type: OrderType = OrderType.MARKET

# === ⚡ 성능 최적화된 API 레이트 리미터 ===

class HighPerformanceRateLimiter:
    """성능 최적화된 레이트 리미터"""
    
    __slots__ = ('calls', 'period', 'call_times', '_lock')
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = deque()
        self._lock = threading.Lock()
    
    def __enter__(self) -> 'HighPerformanceRateLimiter':
        with self._lock:
            now = time.monotonic()
            
            # 만료된 호출 시간 제거 (최적화)
            while self.call_times and self.call_times[0] <= now - self.period:
                self.call_times.popleft()
            
            if len(self.call_times) >= self.calls:
                sleep_time = self.call_times[0] - (now - self.period)
                if sleep_time > 0:
                    logger.debug(f"⏱️ API 제한으로 {sleep_time:.2f}초 대기")
                    time.sleep(sleep_time)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with self._lock:
            self.call_times.append(time.monotonic())

# === 📈 일일 API 카운터 (최적화) ===

class OptimizedDailyApiCounter:
    """최적화된 일일 API 호출 카운터"""
    
    __slots__ = ('daily_limit', 'counter_file', 'today_count', 'last_reset_date', '_lock')
    
    def __init__(self, daily_limit: Optional[int]):
        self.daily_limit = daily_limit
        self.counter_file = "daily_api_count.json"
        self.today_count = 0
        self.last_reset_date = None
        self._lock = threading.Lock()
        self._load_counter()
    
    def _load_counter(self) -> None:
        """카운터 로드 (예외 처리 강화)"""
        try:
            import os
            if not os.path.exists(self.counter_file):
                self._reset_counter()
                return
            
            with open(self.counter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.today_count = data.get('count', 0)
                self.last_reset_date = data.get('date')
                
                # 날짜 변경 확인
                today = datetime.now().strftime('%Y-%m-%d')
                if self.last_reset_date != today:
                    self._reset_counter()
                    
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.warning(f"⚠️ API 카운터 로드 실패 (초기화): {e}")
            self._reset_counter()
    
    def _reset_counter(self) -> None:
        """카운터 초기화"""
        self.today_count = 0
        self.last_reset_date = datetime.now().strftime('%Y-%m-%d')
        self._save_counter()
    
    def _save_counter(self) -> None:
        """카운터 저장 (원자적 쓰기)"""
        try:
            import tempfile
            import os
            
            data = {
                'count': self.today_count,
                'date': self.last_reset_date
            }
            
            # 원자적 쓰기를 위한 임시 파일 사용
            with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           dir=os.path.dirname(self.counter_file),
                                           encoding='utf-8') as tmp_file:
                json.dump(data, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                temp_path = tmp_file.name
            
            # 원자적 교체
            if os.name == 'nt':  # Windows
                if os.path.exists(self.counter_file):
                    os.remove(self.counter_file)
                os.rename(temp_path, self.counter_file)
            else:  # Unix-like
                os.rename(temp_path, self.counter_file)
                
        except Exception as e:
            logger.error(f"❌ API 카운터 저장 실패: {e}")
    
    def can_make_request(self) -> bool:
        """API 호출 가능 여부 (스레드 안전)"""
        if self.daily_limit is None:
            return True
        
        with self._lock:
            return self.today_count < self.daily_limit
    
    def increment(self) -> None:
        """API 호출 카운터 증가 (스레드 안전)"""
        with self._lock:
            self.today_count += 1
            
            # 비동기 저장 (성능 최적화)
            if self.today_count % 10 == 0:  # 10회마다 저장
                self._save_counter()
            
            # 경고 로직
            if self.daily_limit:
                ratio = self.today_count / self.daily_limit
                if ratio >= 0.9:
                    logger.warning(f"🚨 일일 API 한도 90% 도달: {self.today_count}/{self.daily_limit}")
                elif ratio >= 0.8:
                    logger.warning(f"⚠️ 일일 API 한도 80% 도달: {self.today_count}/{self.daily_limit}")
    
    def get_remaining_calls(self) -> Union[int, float]:
        """남은 호출 횟수 반환"""
        if self.daily_limit is None:
            return float('inf')
        
        with self._lock:
            return max(0, self.daily_limit - self.today_count)

# === 🔐 고성능 토큰 관리자 ===

class OptimizedTokenManager:
    """최적화된 토큰 관리 시스템"""
    
    __slots__ = ('base_url', 'app_key', 'app_secret', 'limiter', 'token_file', 
                 'access_token', '_token_cache', '_lock', 'renewal_time')
    
    def __init__(self, base_url: str, app_key: str, app_secret: str, limiter: HighPerformanceRateLimiter):
        self.base_url = base_url
        self.app_key = app_key
        self.app_secret = app_secret
        self.limiter = limiter
        self.token_file = "kis_token.json"
        self.access_token = None
        self._token_cache = None
        self._lock = threading.Lock()
        self.renewal_time = time_obj(hour=7, minute=0)

    def _save_token(self, token_data: Dict[str, Any]) -> None:
        """토큰 저장 (원자적 쓰기)"""
        try:
            import tempfile
            import os
            
            token_data['expires_at'] = (
                datetime.now() + timedelta(seconds=token_data['expires_in'] - 600)
            ).isoformat()
            
            # 원자적 쓰기
            with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           dir=os.path.dirname(os.path.abspath(self.token_file)),
                                           encoding='utf-8') as tmp_file:
                json.dump(token_data, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                temp_path = tmp_file.name
            
            # 원자적 교체
            if os.name == 'nt':
                if os.path.exists(self.token_file):
                    os.remove(self.token_file)
                os.rename(temp_path, self.token_file)
            else:
                os.rename(temp_path, self.token_file)
                
            # 캐시 무효화
            self._token_cache = None
            logger.info("✅ 새 API 토큰 저장 완료")
            
        except Exception as e:
            logger.error(f"❌ 토큰 저장 실패: {e}")
    
    def _load_token(self) -> Optional[Dict[str, Any]]:
        """토큰 로드 (캐싱 최적화)"""
        if self._token_cache is not None:
            return self._token_cache
        
        try:
            import os
            if not os.path.exists(self.token_file):
                return None
            
            with open(self.token_file, 'r', encoding='utf-8') as f:
                self._token_cache = json.load(f)
                return self._token_cache
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"⚠️ 토큰 로드 실패: {e}")
            return None
    
    def _issue_new_token(self) -> Optional[Dict[str, Any]]:
        """새 토큰 발급 (에러 처리 강화)"""
        with self.limiter:
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {
                "content-type": "application/json",
                "User-Agent": "TradingBot/1.0"
            }
            body = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    data=json.dumps(body),
                    timeout=10  # 타임아웃 추가
                )
                response.raise_for_status()
                
                token_data = response.json()
                if 'access_token' not in token_data:
                    logger.error(f"❌ 토큰 응답에 access_token 없음: {token_data}")
                    return None
                
                self._save_token(token_data)
                logger.info("✅ 새 토큰 발급 성공")
                return token_data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ 토큰 발급 네트워크 오류: {e}")
                return None
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"❌ 토큰 발급 응답 파싱 오류: {e}")
                return None
    
    def get_valid_token(self) -> Optional[str]:
        """유효한 토큰 반환 (스레드 안전)"""
        with self._lock:
            token_data = self._load_token()
            
            # 토큰 유효성 검사
            if token_data and 'expires_at' in token_data:
                try:
                    expires_at = datetime.fromisoformat(token_data['expires_at'])
                    if expires_at > datetime.now():
                        self.access_token = token_data['access_token']
                        return self.access_token
                except ValueError:
                    logger.warning("⚠️ 토큰 만료 시간 파싱 실패")
            
            # 새 토큰 발급
            logger.info("🔄 토큰 갱신 중...")
            new_token_data = self._issue_new_token()
            if new_token_data:
                self.access_token = new_token_data['access_token']
            return self.access_token

            return None

# === 🏦 리팩토링된 핵심 거래 클래스 ===

class CoreTrader:
    """리팩토링된 핵심 거래 엔진"""
    
    def __init__(self):
        """초기화 - 성능 최적화 및 타입 안전성 강화"""
        self._load_configuration()
        self._initialize_services()
        self._initialize_rate_limiters()
        self._initialize_websocket_components()
        self._log_initialization_status()
    
    def _load_configuration(self) -> None:
        """설정 로드 및 검증"""
        # 기본 API 설정
        self.app_key = config.KIS_APP_KEY
        self.app_secret = config.KIS_APP_SECRET
        self.account_no = config.KIS_ACCOUNT_NO
        self.base_url = config.KIS_BASE_URL
        self.is_mock = config.IS_MOCK
        
        # 설정 검증
        missing_configs, _ = config.validate_config()
        if missing_configs:
            error_msg = f"❌ 필수 환경변수 누락: {missing_configs}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_services(self) -> None:
        """외부 서비스 초기화"""
        # 텔레그램 알림
        self.notifier = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID
        )
        
        # Google Sheets
        self.worksheet = self._initialize_gspread()

    def _initialize_rate_limiters(self) -> None:
        """레이트 리미터 초기화"""
        self.order_limiter = HighPerformanceRateLimiter(
            calls=config.ORDER_API_CALLS_PER_SEC, 
            period=1
        )
        self.market_data_limiter = HighPerformanceRateLimiter(
            calls=config.MARKET_DATA_API_CALLS_PER_SEC, 
            period=1
        )
        self.account_limiter = HighPerformanceRateLimiter(
            calls=config.ACCOUNT_API_CALLS_PER_SEC, 
            period=1
        )
        self.global_limiter = HighPerformanceRateLimiter(
            calls=config.TOTAL_API_CALLS_PER_SEC, 
            period=1
        )
        
        # 일일 카운터 및 토큰 관리자
        self.daily_counter = OptimizedDailyApiCounter(config.DAILY_API_LIMIT)
        self.token_manager = OptimizedTokenManager(
            self.base_url, self.app_key, self.app_secret, self.order_limiter
        )
    
    def _initialize_websocket_components(self) -> None:
        """WebSocket 컴포넌트 초기화"""
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_ws_connected = False
        self.realtime_prices: Dict[str, PriceData] = {}
        self.price_callbacks: List[Callable[[str, PriceData], None]] = []
        self._ws_lock = threading.Lock()
    
    def _log_initialization_status(self) -> None:
        """초기화 상태 로깅"""
        mode = "모의투자" if self.is_mock else "실전투자"
        remaining_calls = self.daily_counter.get_remaining_calls()
        
        logger.info(f"🔧 CoreTrader v6.0 초기화 완료 ({mode})")
        logger.info(f"📊 API 제한: 주문({config.ORDER_API_CALLS_PER_SEC}/s), "
                   f"시세({config.MARKET_DATA_API_CALLS_PER_SEC}/s), "
                   f"전체({config.TOTAL_API_CALLS_PER_SEC}/s)")
        logger.info(f"📈 일일 한도: {self.daily_counter.today_count}회 사용, "
                   f"{remaining_calls}회 남음")
    
    def initialize(self) -> bool:
        """시스템 초기화 (기존 호환성 유지)"""
        try:
            # 토큰 유효성 확인
            token = self.token_manager.get_valid_token()
            if not token:
                logger.error("❌ 유효한 API 토큰을 획득할 수 없습니다")
                return False
            
            logger.info("✅ CoreTrader 초기화 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ CoreTrader 초기화 실패: {e}")
            return False
    
    def _initialize_gspread(self) -> Optional[Any]:
        """Google Sheets 초기화 (에러 처리 강화)"""
        try:
            if not config.GOOGLE_SERVICE_ACCOUNT_FILE or not config.GOOGLE_SPREADSHEET_ID:
                logger.info("⚠️ Google Sheets 미설정 - 로깅 비활성화")
                return None
            
            service_account_info = json.loads(config.GOOGLE_SERVICE_ACCOUNT_FILE)
            creds = Credentials.from_service_account_info(service_account_info)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(config.GOOGLE_SPREADSHEET_ID)
            
            worksheet_name = config.GOOGLE_WORKSHEET_NAME or "거래기록"
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="1000", cols="20")
                # 헤더 추가
                headers = ["시간", "종목", "구분", "수량", "가격", "금액", "수수료", "메모"]
                worksheet.append_row(headers)
            
            logger.info(f"✅ Google Sheets 연동 성공: {worksheet_name}")
            return worksheet
            
        except Exception as e:
            logger.warning(f"⚠️ Google Sheets 초기화 실패: {e}")
            return None
    
    def _send_request(self, 
                     method: str, 
                     path: str, 
                     headers: Optional[Dict[str, str]] = None, 
                     params: Optional[Dict[str, Any]] = None, 
                     json_data: Optional[Dict[str, Any]] = None, 
                     max_retries: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """최적화된 API 요청 처리"""
        
        # 일일 한도 확인
        if not self.daily_counter.can_make_request():
            logger.error("❌ 일일 API 호출 한도 초과")
            return None
        
        # 토큰 획득
        token = self.token_manager.get_valid_token()
        if not token:
            logger.error("❌ 유효한 토큰 없음")
            return None
        
        # 헤더 구성
        request_headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "User-Agent": "TradingBot/6.0"
        }
        
        if headers:
            request_headers.update(headers)
        
        # 재시도 로직
        max_retries = max_retries or config.MAX_RETRY_ATTEMPTS
        url = f"{self.base_url}{path}"
        
        for attempt in range(max_retries):
            try:
                # API 호출 카운터 증가
                self.daily_counter.increment()
                
                if method.upper() == "GET":
                    response = requests.get(
                        url, 
                        headers=request_headers, 
                        params=params,
                        timeout=10
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url, 
                        headers=request_headers, 
                        json=json_data,
                        timeout=10
                    )
                else:
                    logger.error(f"❌ 지원하지 않는 HTTP 메서드: {method}")
                    return None
                
                response.raise_for_status()
                
                try:
                    result = response.json()
                    
                    # API 응답 코드 확인
                    if isinstance(result, dict):
                        rt_cd = result.get('rt_cd', '1')
                        if rt_cd != '0':
                            msg1 = result.get('msg1', 'Unknown error')
                            logger.warning(f"⚠️ API 응답 오류: {rt_cd} - {msg1}")
                            return None
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON 파싱 실패: {e}")
                    return None
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ API 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(config.RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    logger.error(f"❌ API 요청 최종 실패: {path}")
                    return None
        
        return None

    def get_current_price(self, symbol: str) -> Optional[Dict[str, Union[int, str]]]:
        """현재 주가 조회 (타입 안전성 강화)"""
        with self.global_limiter, self.market_data_limiter:
            response = self._send_request(
                "GET", 
                "/uapi/domestic-stock/v1/quotations/inquire-price",
                headers={"tr_id": "FHKST01010100"},
                params={
                    "fid_cond_mrkt_div_code": "J",
                    "fid_input_iscd": symbol
                }
            )
            
            if not response or not response.get('output'):
                logger.warning(f"⚠️ {symbol} 현재가 조회 실패")
                return None
            
            output = response['output']
            try:
                price = int(output.get('stck_prpr', 0))
                name = output.get('hts_kor_isnm', symbol)
                
                return {
                    'price': price,
                    'name': name,
                    'change_rate': float(output.get('prdy_ctrt', 0.0)),
                    'volume': int(output.get('acml_vol', 0))
                }
            except (ValueError, TypeError) as e:
                logger.error(f"❌ {symbol} 가격 데이터 파싱 실패: {e}")
                return None

    def get_balance(self, part: str = 'all') -> Optional[BalanceInfo]:
        """계좌 잔고 조회 (타입 안전성 및 성능 최적화)"""
        with self.global_limiter, self.account_limiter:
            tr_id = "VTTC8434R" if self.is_mock else "TTTC8434R"
            
            headers = {
                "tr_id": tr_id,
                "tr_cont": "",
                "custtype": "P",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": ""
            }
            
            account_parts = self.account_no.split('-')
            if len(account_parts) != 2:
                logger.error(f"❌ 잘못된 계좌번호 형식: {self.account_no}")
                return None
            
            params = {
                "CANO": account_parts[0],
                "ACNT_PRDT_CD": account_parts[1],
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "01",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "Y",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": ""
            }
            
            response = self._send_request(
                "GET", 
                "/uapi/domestic-stock/v1/trading/inquire-balance",
                headers=headers,
                params=params
            )
            
            if not response:
                logger.error("❌ 잔고 조회 API 호출 실패")
                return None
            
            return self._parse_balance_response(response)
    
    def _parse_balance_response(self, response: Dict[str, Any]) -> Optional[BalanceInfo]:
        """잔고 응답 파싱 (분리된 메서드)"""
        try:
            output1 = response.get('output1', [])
            output2 = response.get('output2', [{}])
            
            if not output2:
                logger.warning("⚠️ 잔고 응답에 output2 없음")
                return None
            
            summary = output2[0]
            
            # 현금 정보
            cash = float(summary.get('dnca_tot_amt', 0))  # 예수금총액
            total_value = float(summary.get('tot_evlu_amt', 0))  # 총평가금액
            profit_loss = float(summary.get('evlu_pfls_rt', 0))  # 평가손익률
            
            # 보유 종목 정보
            positions = {}
            for item in output1:
                if not item:
                    continue
                    
                symbol = item.get('pdno', '').strip()
                if not symbol:
                    continue
                
                try:
                    positions[symbol] = {
                        'name': item.get('prdt_name', '').strip(),
                        'quantity': int(item.get('hldg_qty', 0)),
                        'avg_price': float(item.get('pchs_avg_pric', 0)),
                        'current_price': float(item.get('prpr', 0)),
                        'profit_loss': float(item.get('evlu_pfls_amt', 0)),
                        'profit_loss_rate': float(item.get('evlu_pfls_rt', 0))
                    }
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ 종목 {symbol} 데이터 파싱 실패: {e}")
                    continue
            
            balance_info = BalanceInfo(
                cash=cash,
                total_value=total_value,
                positions=positions,
                profit_loss=profit_loss
            )
            
            logger.info(f"💰 잔고 조회 성공: 현금 {cash:,.0f}원, "
                       f"총평가 {total_value:,.0f}원, "
                       f"보유종목 {len(positions)}개")
            
            return balance_info
            
        except Exception as e:
            logger.error(f"❌ 잔고 응답 파싱 실패: {e}")
            return None

    def get_top_ranking_stocks(self, top_n: int = 10) -> List[Dict[str, Union[str, int, float]]]:
        """상위 랭킹 주식 조회 (타입 안전성 강화)"""
        with self.global_limiter, self.market_data_limiter:
            response = self._send_request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/volume-rank",
                headers={"tr_id": "FHPST01710000"},
                params={
                    "fid_cond_mrkt_div_code": "J",
                    "fid_cond_scr_div_code": "20171",
                    "fid_input_iscd": "0000",
                    "fid_div_cls_code": "0",
                    "fid_blng_cls_code": "0",
                    "fid_trgt_cls_code": "111111111",
                    "fid_trgt_exls_cls_code": "0000000000",
                    "fid_input_price_1": "",
                    "fid_input_price_2": "",
                    "fid_vol_cnt": str(top_n)
                }
            )
            
            if not response or not response.get('output'):
                logger.warning("⚠️ 랭킹 주식 조회 실패")
                return []
            
            return self._parse_ranking_stocks(response['output'], top_n)
    
    def _parse_ranking_stocks(self, stocks_data: List[Dict], top_n: int) -> List[Dict[str, Union[str, int, float]]]:
        """랭킹 주식 데이터 파싱"""
        ranking_stocks = []
        
        for i, stock in enumerate(stocks_data[:top_n]):
            try:
                ranking_stocks.append({
                    'rank': i + 1,
                    'symbol': stock.get('mksc_shrn_iscd', '').strip(),
                    'name': stock.get('hts_kor_isnm', '').strip(),
                    'price': int(stock.get('stck_prpr', 0)),
                    'change_rate': float(stock.get('prdy_ctrt', 0)),
                    'volume': int(stock.get('acml_vol', 0)),
                    'volume_rate': float(stock.get('vol_inrt', 0))
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ 랭킹 주식 {i+1} 데이터 파싱 실패: {e}")
                continue
        
        logger.info(f"📊 상위 {len(ranking_stocks)}개 종목 조회 완료")
        return ranking_stocks

    def execute_order(self, 
                     symbol: str, 
                     side: str, 
                     quantity: int, 
                     price: float = 0, 
                     log_payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """통합 주문 실행 (에러 처리 및 로깅 강화)"""
        
        # 입력 유효성 검사
        if not self._validate_order_inputs(symbol, side, quantity, price):
            return None
        
        with self.global_limiter, self.order_limiter:
            order_request = self._build_order_request(symbol, side, quantity, price)
            if not order_request:
                return None
            
            # 주문 실행
            response = self._send_order_request(order_request)
            if not response:
                return None
            
            # 결과 처리
            result = self._process_order_response(response, symbol, side, quantity, price)
            
            # 로깅 및 알림
            if result and result.get('success'):
                self._log_successful_order(result, log_payload)
                self._send_order_notification(result)
            else:
                self._log_failed_order(symbol, side, quantity, result)
            
            return result
    
    def _validate_order_inputs(self, symbol: str, side: str, quantity: int, price: float) -> bool:
        """주문 입력값 유효성 검사"""
        if not symbol or not symbol.strip():
            logger.error("❌ 종목코드가 없습니다")
            return False
        
        if side not in ['buy', 'sell', 'BUY', 'SELL']:
            logger.error(f"❌ 잘못된 주문 방향: {side}")
            return False
        
        if quantity <= 0:
            logger.error(f"❌ 잘못된 수량: {quantity}")
            return False
        
        if price < 0:
            logger.error(f"❌ 잘못된 가격: {price}")
            return False
        
        return True
    
    def _build_order_request(self, symbol: str, side: str, quantity: int, price: float) -> Optional[Dict[str, Any]]:
        """주문 요청 데이터 구성"""
        try:
            account_parts = self.account_no.split('-')
            if len(account_parts) != 2:
                logger.error(f"❌ 잘못된 계좌번호: {self.account_no}")
                return None
            
            # TR ID 설정
            side_upper = side.upper()
            if self.is_mock:
                tr_id = "VTTC0802U" if side_upper in ['BUY', '매수'] else "VTTC0801U"
            else:
                tr_id = "TTTC0802U" if side_upper in ['BUY', '매수'] else "TTTC0801U"
            
            # 주문 구분 코드
            ord_dvsn = "01" if price == 0 else "00"  # 시장가 : 지정가
            
            return {
                'headers': {"tr_id": tr_id},
                'json_data': {
                    "CANO": account_parts[0],
                    "ACNT_PRDT_CD": account_parts[1],
                    "PDNO": symbol,
                    "ORD_DVSN": ord_dvsn,
                    "ORD_QTY": str(quantity),
                    "ORD_UNPR": str(int(price)) if price > 0 else "0"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 주문 요청 구성 실패: {e}")
            return None
    
    def _send_order_request(self, order_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """주문 API 호출"""
        return self._send_request(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            headers=order_request['headers'],
            json_data=order_request['json_data']
        )
    
    def _process_order_response(self, response: Dict[str, Any], symbol: str, side: str, quantity: int, price: float) -> Dict[str, Any]:
        """주문 응답 처리"""
        if not response:
            return {'success': False, 'message': 'API 응답 없음'}
        
        rt_cd = response.get('rt_cd', '1')
        msg1 = response.get('msg1', 'Unknown error')
        
        result = {
            'success': rt_cd == '0',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'message': msg1,
            'timestamp': datetime.now().isoformat()
        }
        
        if rt_cd == '0':
            output = response.get('output', {})
            result.update({
                'order_id': output.get('KRX_FWDG_ORD_ORGNO', ''),
                'order_number': output.get('ODNO', ''),
                'order_time': output.get('ORD_TMD', '')
            })
            logger.info(f"✅ 주문 성공: {symbol} {side} {quantity}주 {price}원")
        else:
            logger.error(f"❌ 주문 실패: {symbol} {side} - {msg1}")
        
        return result
    
    def _log_successful_order(self, result: Dict[str, Any], log_payload: Optional[Dict[str, Any]]) -> None:
        """성공한 주문 로깅"""
        try:
            if self.worksheet:
                log_data = [
                    result['timestamp'],
                    result['symbol'],
                    result['side'],
                    result['quantity'],
                    result['price'],
                    result['quantity'] * result['price'],
                    log_payload.get('commission', 0) if log_payload else 0,
                    log_payload.get('memo', '') if log_payload else ''
                ]
                self.worksheet.append_row(log_data)
                logger.debug("📝 Google Sheets 로깅 완료")
        except Exception as e:
            logger.warning(f"⚠️ Google Sheets 로깅 실패: {e}")
    
    def _send_order_notification(self, result: Dict[str, Any]) -> None:
        """주문 알림 발송"""
        try:
            if self.notifier:
                message = (f"📈 주문 완료\n"
                          f"종목: {result['symbol']}\n"
                          f"방향: {result['side']}\n"
                          f"수량: {result['quantity']:,}주\n"
                          f"가격: {result['price']:,}원\n"
                          f"시간: {result['timestamp']}")
                
                self.notifier.send_message(message)
                logger.debug("📱 텔레그램 알림 발송 완료")
        except Exception as e:
            logger.warning(f"⚠️ 텔레그램 알림 실패: {e}")
    
    def _log_failed_order(self, symbol: str, side: str, quantity: int, result: Optional[Dict[str, Any]]) -> None:
        """실패한 주문 로깅"""
        error_msg = result.get('message', 'Unknown error') if result else 'No response'
        logger.error(f"❌ 주문 실패 로그: {symbol} {side} {quantity}주 - {error_msg}")

    def get_market_summary(self) -> Optional[Dict[str, Any]]:
        """시장 요약 정보 조회"""
        with self.global_limiter, self.market_data_limiter:
            response = self._send_request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-index-price",
                headers={"tr_id": "FHPST01020000"},
                params={
                    "fid_cond_mrkt_div_code": "U",
                    "fid_input_iscd": "0001"  # KOSPI
                }
            )
            
            if not response or not response.get('output'):
                logger.warning("⚠️ 시장 요약 조회 실패")
                return None
            
            output = response['output']
            try:
                return {
                    'index_name': output.get('hts_kor_isnm', 'KOSPI'),
                    'current_value': float(output.get('bstp_nmix_prpr', 0)),
                    'change_value': float(output.get('bstp_nmix_prdy_vrss', 0)),
                    'change_rate': float(output.get('prdy_ctrt', 0)),
                    'volume': int(output.get('acml_vol', 0)),
                    'timestamp': datetime.now().isoformat()
                }
            except (ValueError, TypeError) as e:
                logger.error(f"❌ 시장 요약 데이터 파싱 실패: {e}")
                return None

    def get_todays_trades_from_sheet(self) -> List[Dict[str, Any]]:
        """오늘의 거래 기록 조회 (Google Sheets)"""
        if not self.worksheet:
            logger.warning("⚠️ Google Sheets 미설정")
            return []
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            all_records = self.worksheet.get_all_records()
            
            todays_trades = []
            for record in all_records:
                if not record.get('시간'):
                    continue
                
                try:
                    trade_date = record['시간'][:10]  # YYYY-MM-DD 부분만
                    if trade_date == today:
                        todays_trades.append(record)
                except (IndexError, TypeError):
                    continue
            
            logger.info(f"📋 오늘의 거래 기록: {len(todays_trades)}건")
            return todays_trades
            
        except Exception as e:
            logger.error(f"❌ 거래 기록 조회 실패: {e}")
            return []

    # === 🔴 리팩토링된 WebSocket 실시간 시세 시스템 ===
    
    def _get_ws_approval_key(self) -> Optional[str]:
        """WebSocket 접속 승인키 발급 (에러 처리 강화)"""
        try:
            body = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "secretkey": self.app_secret
            }
            
            response = requests.post(
                f"{self.base_url}/oauth2/Approval",
                headers={"content-type": "application/json"},
                data=json.dumps(body),
                timeout=10
            )
            
            response.raise_for_status()
            result = response.json()
            
            approval_key = result.get('approval_key')
            if not approval_key:
                logger.error(f"❌ WebSocket 승인키 응답에 approval_key 없음: {result}")
                return None
            
            logger.info("✅ WebSocket 승인키 발급 성공")
            return approval_key
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ WebSocket 승인키 요청 실패: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ WebSocket 승인키 응답 파싱 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ WebSocket 승인키 발급 예상치 못한 오류: {e}")
            return None

    def _on_ws_open(self, ws: websocket.WebSocketApp) -> None:
        """WebSocket 연결 성공 핸들러"""
        with self._ws_lock:
            self.is_ws_connected = True
        logger.info("🔗 WebSocket 연결 성공")

    def _on_ws_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """WebSocket 메시지 수신 핸들러 (성능 최적화)"""
        try:
            # 메시지 타입 확인 (바이너리 vs 텍스트)
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            # 빈 메시지 필터링
            if not message or message.strip() == '':
                return
            
            # 시세 데이터 파싱
            price_data = self._parse_realtime_price_message(message)
            if not price_data:
                return
            
            # 실시간 가격 캐시 업데이트
            self._update_realtime_price_cache(price_data)
            
            # 콜백 함수들 실행 (비동기 처리)
            self._execute_price_callbacks(price_data)
            
        except Exception as e:
            logger.error(f"❌ WebSocket 메시지 처리 실패: {e}")
    
    def _parse_realtime_price_message(self, message: str) -> Optional[PriceData]:
        """실시간 시세 메시지 파싱 (최적화)"""
        try:
            # 메시지 포맷 확인 및 파싱
            if '|' not in message:
                return None
            
            parts = message.split('|')
            if len(parts) < 4:
                return None
            
            # 주식 체결가 데이터 확인
            if parts[0] != '0':  # 주식 체결가
                return None
            
            # 데이터 필드 파싱
            data_fields = parts[3].split('^') if len(parts) > 3 else []
            if len(data_fields) < 15:
                return None
            
            symbol = data_fields[0].strip()
            if not symbol:
                return None
            
            try:
                price = float(data_fields[2])
                volume = int(data_fields[12])
                
                return PriceData(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    timestamp=datetime.now(),
                    name=None  # 이름은 별도 조회 필요
                )
                
            except (ValueError, IndexError) as e:
                logger.debug(f"⚠️ 가격 데이터 파싱 실패 {symbol}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 실시간 시세 메시지 파싱 실패: {e}")
            return None

    def _update_realtime_price_cache(self, price_data: PriceData) -> None:
        """실시간 가격 캐시 업데이트 (스레드 안전)"""
        try:
            with self._ws_lock:
                self.realtime_prices[price_data.symbol] = price_data
                
                # 캐시 크기 제한 (메모리 최적화)
                if len(self.realtime_prices) > 1000:
                    # 가장 오래된 항목들 제거
                    oldest_symbols = sorted(
                        self.realtime_prices.keys(),
                        key=lambda s: self.realtime_prices[s].timestamp
                    )[:100]
                    
                    for symbol in oldest_symbols:
                        del self.realtime_prices[symbol]
                        
        except Exception as e:
            logger.error(f"❌ 실시간 가격 캐시 업데이트 실패: {e}")
    
    def _execute_price_callbacks(self, price_data: PriceData) -> None:
        """가격 변동 콜백 실행 (비동기 처리)"""
        if not self.price_callbacks:
            return
        
        def run_callbacks():
            for callback in self.price_callbacks[:]:  # 복사본으로 안전하게 순회
                try:
                    callback(price_data.symbol, price_data)
                except Exception as e:
                    logger.warning(f"⚠️ 가격 콜백 실행 실패: {e}")
        
        # 별도 스레드에서 실행하여 WebSocket 블로킹 방지
        threading.Thread(target=run_callbacks, daemon=True).start()

    def _on_ws_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """WebSocket 오류 핸들러"""
        logger.error(f"❌ WebSocket 오류: {error}")
        with self._ws_lock:
            self.is_ws_connected = False

    def _on_ws_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        """WebSocket 연결 종료 핸들러"""
        with self._ws_lock:
            self.is_ws_connected = False
        logger.info(f"🔌 WebSocket 연결 종료: {close_status_code} - {close_msg}")

    def start_realtime_price_feed(self, symbols: List[str] = None) -> bool:
        """실시간 시세 수신 시작 (타입 안전성 및 에러 처리 강화)"""
        if symbols is None:
            symbols = []
        
        if self.is_ws_connected:
            logger.warning("⚠️ WebSocket이 이미 연결되어 있습니다")
            return True
        
        try:
            # WebSocket 승인키 발급
            approval_key = self._get_ws_approval_key()
            if not approval_key:
                logger.error("❌ WebSocket 승인키 발급 실패")
                return False
            
            # WebSocket URL 구성
            ws_url = self._build_websocket_url()
            if not ws_url:
                logger.error("❌ WebSocket URL 구성 실패")
                return False
            
            # WebSocket 연결 설정
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # WebSocket 연결 시작 (백그라운드 스레드)
            self.ws_thread = threading.Thread(
                target=self._run_websocket,
                args=(approval_key, symbols),
                daemon=True
            )
            self.ws_thread.start()
            
            # 연결 확인 대기
            if self._wait_for_ws_connection():
                logger.info(f"✅ 실시간 시세 시작: {len(symbols)}개 종목")
                return True
            else:
                logger.error("❌ WebSocket 연결 타임아웃")
                return False
                
        except Exception as e:
            logger.error(f"❌ 실시간 시세 시작 실패: {e}")
            return False
    
    def _build_websocket_url(self) -> Optional[str]:
        """WebSocket URL 구성"""
        try:
            if self.is_mock:
                return "ws://ops.koreainvestment.com:31000"
            else:
                return "ws://ops.koreainvestment.com:21000"
        except Exception as e:
            logger.error(f"❌ WebSocket URL 구성 실패: {e}")
            return None
    
    def _run_websocket(self, approval_key: str, symbols: List[str]) -> None:
        """WebSocket 실행 (백그라운드 스레드)"""
        try:
            # WebSocket 연결 실행
            self.ws.run_forever()
            
            # 연결 성공 후 종목 구독
            if self.is_ws_connected and symbols:
                time.sleep(1)  # 연결 안정화 대기
                for symbol in symbols:
                    self._subscribe_symbol(symbol, approval_key)
                    time.sleep(0.1)  # 구독 간격
                    
        except Exception as e:
            logger.error(f"❌ WebSocket 실행 실패: {e}")
            with self._ws_lock:
                self.is_ws_connected = False
    
    def _wait_for_ws_connection(self, timeout: int = 5) -> bool:
        """WebSocket 연결 대기"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ws_connected:
                return True
            time.sleep(0.1)
        return False

    def _subscribe_symbol(self, symbol: str, approval_key: str) -> bool:
        """종목 구독 (에러 처리 강화)"""
        try:
            if not self.ws or not self.is_ws_connected:
                logger.warning(f"⚠️ WebSocket 연결되지 않음 - {symbol} 구독 실패")
                return False

            # 구독 메시지 구성
            subscribe_message = {
                "header": {
                    "approval_key": approval_key,
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
            
            # 구독 메시지 전송
            message_json = json.dumps(subscribe_message, ensure_ascii=False)
            self.ws.send(message_json)
            
            logger.debug(f"📡 종목 구독: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 종목 구독 실패 {symbol}: {e}")
            return False

    def stop_realtime_price_feed(self) -> None:
        """실시간 시세 수신 중지 (안전한 종료)"""
        try:
            logger.info("🛑 실시간 시세 중지 중...")
            
            # WebSocket 연결 상태 변경
            with self._ws_lock:
                self.is_ws_connected = False
            
            # WebSocket 연결 종료
            if self.ws:
                self.ws.close()
                self.ws = None
            
            # 스레드 정리
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=3)  # 3초 대기
                if self.ws_thread.is_alive():
                    logger.warning("⚠️ WebSocket 스레드가 3초 내에 종료되지 않음")
            
            # 캐시 정리
            with self._ws_lock:
                self.realtime_prices.clear()
            
            logger.info("✅ 실시간 시세 중지 완료")
            
        except Exception as e:
            logger.error(f"❌ 실시간 시세 중지 실패: {e}")

    def get_realtime_price(self, symbol: str) -> Optional[PriceData]:
        """실시간 가격 조회 (스레드 안전)"""
        with self._ws_lock:
            return self.realtime_prices.get(symbol)

    def add_price_callback(self, callback_func: Callable[[str, PriceData], None]) -> None:
        """가격 변동 콜백 함수 등록"""
        if callback_func not in self.price_callbacks:
            self.price_callbacks.append(callback_func)
            logger.debug("📎 가격 변동 콜백 함수 등록 완료")

    def remove_price_callback(self, callback_func: Callable[[str, PriceData], None]) -> None:
        """가격 변동 콜백 함수 제거"""
        try:
            self.price_callbacks.remove(callback_func)
            logger.debug("🗑️ 가격 변동 콜백 함수 제거 완료")
        except ValueError:
            logger.warning("⚠️ 제거하려는 콜백 함수가 등록되지 않음")

    def test_websocket_connection(self) -> bool:
        """WebSocket 연결 테스트"""
        logger.info("🧪 WebSocket 연결 테스트 시작...")
        
        test_symbols = ["005930"]  # 삼성전자
        
        def test_callback(symbol: str, price_data: PriceData) -> None:
            logger.info(f"✅ 테스트 콜백 수신: {symbol} - {price_data.price}원 "
                       f"(시간: {price_data.timestamp})")
        
        try:
            # 콜백 등록
            self.add_price_callback(test_callback)
            
            # WebSocket 시작
            if not self.start_realtime_price_feed(test_symbols):
                logger.error("❌ WebSocket 테스트 연결 실패")
                return False
            
            # 5초 대기
            logger.info("⏳ 5초간 데이터 수신 테스트...")
            time.sleep(5)
            
            # 연결 상태 확인
            is_connected = self.is_ws_connected
            
            # 정리
            self.remove_price_callback(test_callback)
            self.stop_realtime_price_feed()
            
            if is_connected:
                logger.info("✅ WebSocket 연결 테스트 성공")
                return True
            else:
                logger.error("❌ WebSocket 연결 테스트 실패 - 연결 끊어짐")
                return False
                
        except Exception as e:
            logger.error(f"❌ WebSocket 테스트 중 오류: {e}")
            return False

    # === 🔧 유틸리티 메서드 ===
    
    def get_connection_status(self) -> Dict[str, Union[bool, int, str]]:
        """연결 상태 정보 조회"""
        with self._ws_lock:
            return {
                'websocket_connected': self.is_ws_connected,
                'realtime_symbols_count': len(self.realtime_prices),
                'callbacks_count': len(self.price_callbacks),
                'daily_api_calls': self.daily_counter.today_count,
                'remaining_api_calls': self.daily_counter.get_remaining_calls(),
                'last_token_check': datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 건강도 체크"""
        try:
            # 토큰 유효성 체크
            token_valid = self.token_manager.get_valid_token() is not None
            
            # API 호출 가능 여부
            api_available = self.daily_counter.can_make_request()
            
            # WebSocket 상태
            ws_status = self.is_ws_connected
            
            # 전체 상태 평가
            overall_healthy = all([token_valid, api_available])
            
            return {
                'status': 'healthy' if overall_healthy else 'warning',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'token': 'ok' if token_valid else 'error',
                    'api_limit': 'ok' if api_available else 'limit_exceeded',
                    'websocket': 'connected' if ws_status else 'disconnected'
                },
                'details': self.get_connection_status()
            }
            
        except Exception as e:
            logger.error(f"❌ 건강도 체크 실패: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            } 