import os
import sys
import logging
import time
import asyncio
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import numpy as np

from kis_api import KIS_API
from market_analyzer import MarketAnalyzer
from oneil_scanner import ONeilScanner
from minervini_screener import MinerviniScreener
import google.generativeai as genai
import yfinance as yf

# 웹소켓 관련 import (선택적 로드)
try:
    import websocket
    import threading
    from queue import Queue
    WEBSOCKET_AVAILABLE = True
    logging.info("✅ 웹소켓 라이브러리 로드 성공")
except ImportError as e:
    WEBSOCKET_AVAILABLE = False
    logging.warning(f"⚠️ 웹소켓 라이브러리 없음: {e}")

# .env 파일 로드
load_dotenv()

# Windows 콘솔 인코딩 문제 해결
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 상수 정의
class TradingConstants:
    """거래 관련 상수"""
    MAX_POSITIONS = 10
    POSITION_SIZE_RATIO = 0.1
    STOP_LOSS_RATIO = -0.05
    TAKE_PROFIT_RATIO = 0.15
    AI_CONFIDENCE_THRESHOLD = 0.7
    REBALANCE_INTERVAL_MINUTES = 5
    MARKET_OPEN_TIME = "09:00"
    MARKET_CLOSE_TIME = "15:20"
    WEBSOCKET_RECONNECT_DELAY = 10
    PRICE_CACHE_TIMEOUT = 10

class DataSource(Enum):
    """데이터 소스 타입"""
    REST_API = "rest"
    WEBSOCKET = "websocket"
    AUTO = "auto"

@dataclass
class OrderInfo:
    """주문 정보 데이터 클래스"""
    action: str
    code: str
    name: str
    quantity: int
    reason: str = ""
    expected_price: int = 0

@dataclass
class ScoutStrategy:
    """척후병 전략 설정"""
    enabled: bool = True
    candidate_count: int = 5
    scout_count: int = 4
    final_count: int = 2
    scout_shares: int = 1
    evaluation_period: int = 3
    scout_positions: Dict = field(default_factory=dict)
    evaluation_start: Optional[datetime] = None
    candidates: List[str] = field(default_factory=list)

class StockNameCache:
    """종목명 캐시 관리 클래스"""
    
    def __init__(self):
        self._cache = {}
        self._default_names = {
            '005930': '삼성전자', '000660': 'SK하이닉스', '035420': 'NAVER',
            '005490': 'POSCO홀딩스', '051910': 'LG화학', '035720': '카카오',
            '006400': '삼성SDI', '028260': '삼성물산', '068270': '셀트리온',
            '207940': '삼성바이오로직스', '005380': '현대차', '000270': '기아',
            '012330': '현대모비스', '003550': 'LG', '066570': 'LG전자',
            '017670': 'SK텔레콤', '030200': 'KT', '036570': '엔씨소프트',
            '251270': '넷마블', '018260': '삼성에스디에스'
        }
    
    def get_name(self, stock_code: str, api_client=None) -> str:
        """종목명 조회 (캐시 우선, API 폴백)"""
        if stock_code in self._cache:
            return self._cache[stock_code]
        
        if stock_code in self._default_names:
            name = self._default_names[stock_code]
            self._cache[stock_code] = name
            return name
        
        if api_client:
            try:
                price_info = api_client.get_current_price(stock_code)
                if price_info and price_info.get('rt_cd') == '0':
                    api_name = price_info['output'].get('hts_kor_isnm', '').strip()
                    if api_name and api_name != stock_code:
                        self._cache[stock_code] = api_name
                        return api_name
            except Exception as e:
                logging.warning(f"API 종목명 조회 실패 ({stock_code}): {e}")
        
        return stock_code

class TelegramNotifier:
    """텔레그램 알림 관리"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def send_message(self, message: str, parse_mode: str = 'HTML'):
        """비동기 메시지 전송"""
        if not self.bot_token or not self.chat_id:
            return
        self.executor.submit(self._send_message_sync, message, parse_mode)
    
    def _send_message_sync(self, message: str, parse_mode: str):
        """동기 메시지 전송"""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {'chat_id': self.chat_id, 'text': message, 'parse_mode': parse_mode}
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logging.debug("📱 텔레그램 알림 전송 완료")
            else:
                logging.warning(f"📱 텔레그램 알림 전송 실패: {response.status_code}")
        except Exception as e:
            logging.error(f"📱 텔레그램 알림 전송 중 오류: {e}")
            
    def shutdown(self):
        self.executor.shutdown(wait=True)

class PortfolioManager:
    """포트폴리오 관리"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.portfolio: Dict[str, Dict] = {}
        self.cash_balance = 0
        self.total_assets = 0
    
    def sync_portfolio(self) -> bool:
        """포트폴리오 동기화"""
        try:
            logging.info("💰 포트폴리오 동기화를 시작합니다...")
            
            token_status = self.api_client.get_token_status()
            logging.info(f"🔑 토큰 상태: {token_status['status']} - {token_status['message']}")
            
            balance_info = self.api_client.get_balance()
            if not balance_info or balance_info.get('rt_cd') != '0':
                logging.error("❌ 계좌 잔고 조회 실패")
                return False
            
            success = self._parse_balance_info(balance_info)
            if success:
                logging.info(f"💰 현금잔고: {self.cash_balance:,}원")
                logging.info(f"💎 총자산: {self.total_assets:,}원")
                logging.info(f"📊 보유종목: {len(self.portfolio)}개")
            
            return success
            
        except Exception as e:
            logging.error(f"❌ 포트폴리오 동기화 중 오류: {e}", exc_info=True)
            return False
    
    def _parse_balance_info(self, balance_info: Dict) -> bool:
        """잔고 정보 파싱"""
        try:
            output1 = balance_info.get('output1', [])
            output2 = balance_info.get('output2', [])
            
            # 잔고 파싱
            if isinstance(output2, list) and len(output2) > 0:
                account_info = output2[0]
                cash_fields = ['dnca_tot_amt', 'nxdy_excc_amt', 'cma_evlu_amt']
                
                for field in cash_fields:
                    try:
                        value = account_info.get(field)
                        if value and str(value).replace('-', '').isdigit():
                            self.cash_balance = int(value)
                            break
                    except (ValueError, TypeError): continue
                
                asset_fields = ['tot_evlu_amt', 'evlu_amt_smtl_amt']
                for field in asset_fields:
                    try:
                        value = account_info.get(field)
                        if value and str(value).replace('-', '').isdigit():
                            self.total_assets = int(value)
                            break
                    except (ValueError, TypeError): continue
            
            # 모의투자 환경에서 잔고가 0인 경우 기본값 설정
            if self.cash_balance == 0 and self.total_assets == 0 and self.api_client.is_mock_env:
                self.cash_balance = 500000000
                self.total_assets = 500000000
            
            # 보유 종목 파싱
            self.portfolio = {}
            if isinstance(output1, list):
                for holding in output1:
                    if isinstance(holding, dict):
                        symbol = holding.get('pdno')
                        if symbol and symbol.strip():
                            # API가 주는 값은 정수/실수 형태의 문자열일 수 있으므로 float으로 먼저 변환
                            quantity = int(float(holding.get('hldg_qty', 0)))
                            if quantity > 0:
                                self.portfolio[symbol] = {
                                    'quantity': quantity,
                                    'avg_price': int(float(holding.get('pchs_avg_pric', 0))),
                                    'current_price': int(float(holding.get('prpr', 0))),
                                    'profit_loss': int(float(holding.get('evlu_pfls_amt', 0))),
                                    'profit_loss_rate': float(holding.get('evlu_pfls_rt', 0))
                                }
            
            return True
            
        except (ValueError, TypeError) as e:
            logging.error(f"❌ 잔고 정보 파싱 중 데이터 타입 오류: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ 잔고 정보 파싱 오류: {e}")
            return False
    
    def can_buy(self, price: int, quantity: int = 1) -> bool:
        """매수 가능 여부 확인"""
        return (price * quantity) <= self.cash_balance
    
    def get_position_size(self) -> int:
        """종목당 투자 금액 계산"""
        return int(self.total_assets * TradingConstants.POSITION_SIZE_RATIO)

class WebSocketManager:
    """웹소켓 연결 및 실시간 데이터 관리"""
    def __init__(self, api_client: KIS_API, is_mock: bool):
        if not WEBSOCKET_AVAILABLE:
            self.ws = None
            return
            
        self.api = api_client
        self.url = "ws://ops.koreainvestment.com:31000" if is_mock else "ws://ops.koreainvestment.com:21000"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.connected = False
        self.realtime_data: Dict[str, Dict] = {}
        self.subscribed_stocks = set()

    def start(self):
        if not WEBSOCKET_AVAILABLE: return
        logging.info("🔌 웹소켓 연결을 시작합니다...")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            self.ws = websocket.WebSocketApp(self.url,
                                             on_open=self._on_open,
                                             on_message=self._on_message,
                                             on_error=self._on_error,
                                             on_close=self._on_close)
            self.ws.run_forever()
            logging.info(f"🔄 {TradingConstants.WEBSOCKET_RECONNECT_DELAY}초 후 웹소켓 재연결을 시도합니다.")
            time.sleep(TradingConstants.WEBSOCKET_RECONNECT_DELAY)

    def _on_open(self, ws):
        logging.info("🔗 웹소켓 연결 성공!")
        self.connected = True
        # 기존 구독 종목들 재구독
        if self.subscribed_stocks:
            self.subscribe(list(self.subscribed_stocks))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'header' in data and data['header']['tr_id'] == 'H0STCNT0':
                body = data.get('body', {})
                code = body.get('mksc_shrn_iscd')
                if code:
                    self.realtime_data[code] = {
                        'price': int(body.get('stck_prpr', 0)),
                        'rate': float(body.get('prdy_ctrt', 0)),
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logging.error(f"❌ 웹소켓 메시지 처리 오류: {e}")

    def _on_error(self, ws, error):
        logging.error(f"❌ 웹소켓 오류: {error}")
        self.connected = False

    def _on_close(self, ws, status, msg):
        logging.warning(f"⚠️ 웹소켓 연결 종료: {status} {msg}")
        self.connected = False

    def subscribe(self, codes: List[str]):
        if not self.connected or not self.ws: return
        
        new_codes = [c for c in codes if c not in self.subscribed_stocks]
        if not new_codes: return

        approval_key = self.api.get_approval_key() # 웹소켓용 실시간 승인키
        for code in new_codes:
            msg = {
                "header": {"approval_key": approval_key, "custtype": "P", "tr_type": "1", "content-type": "utf-8"},
                "body": {"input": {"tr_id": "H0STCNT0", "tr_key": code}}
            }
            self.ws.send(json.dumps(msg))
            self.subscribed_stocks.add(code)
        logging.info(f"📡 실시간 구독 추가: {', '.join(new_codes)}")

    def get_price(self, code: str) -> Optional[Dict]:
        data = self.realtime_data.get(code)
        if data and (datetime.now() - data['timestamp']).seconds < TradingConstants.PRICE_CACHE_TIMEOUT:
            return data
        return None
        
    def close(self):
        if self.ws:
            self.ws.close()

class PriceDataManager:
    """가격 데이터 관리 (웹소켓/REST 하이브리드)"""
    def __init__(self, api_client, websocket_manager: Optional[WebSocketManager]):
        self.api = api_client
        self.ws_manager = websocket_manager

    def get_current_price(self, code: str) -> Optional[Dict]:
        # 1. 웹소켓에서 조회
        if self.ws_manager:
            ws_price = self.ws_manager.get_price(code)
            if ws_price:
                return {'rt_cd': '0', 'output': {'stck_prpr': ws_price['price'], 'prdy_ctrt': ws_price['rate']}}
        
        # 2. REST API로 폴백
        try:
            return self.api.get_current_price(code)
        except Exception as e:
            logging.error(f"❌ REST API 가격 조회 실패({code}): {e}")
            return None

class OrderExecutor:
    """주문 실행 관리"""
    
    def __init__(self, api_client, telegram_notifier: TelegramNotifier):
        self.api_client = api_client
        self.telegram_notifier = telegram_notifier
    
    def _safe_int(self, value: Union[str, int, float]) -> int:
        """안전한 정수 변환"""
        try:
            if isinstance(value, str):
                return int(float(value))
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def _execute_single_order(self, order: OrderInfo) -> Dict:
        """단일 주문 실행 (시장가 주문)"""
        try:
            logging.info(f"🔄 시장가 주문 실행 중: {order.action} {order.name}({order.code}) {order.quantity}주")
            
            if order.action.upper() == 'BUY':
                # 시장가 매수
                result = self.api_client.buy_order(
                    stock_code=order.code,
                    quantity=order.quantity,
                    price=0,  # 시장가
                    order_type="01"  # 시장가
                )
            elif order.action.upper() == 'SELL':
                # 시장가 매도
                result = self.api_client.sell_order(
                    stock_code=order.code,
                    quantity=order.quantity,
                    price=0,  # 시장가
                    order_type="01"  # 시장가
                )
            else:
                return {'status': 'error', 'message': f"알 수 없는 주문 타입: {order.action}", 'order': order}
            
            if result and result.get('rt_cd') == '0':
                order_no = result.get('output', {}).get('ODNO', 'N/A')
                return {
                    'status': 'success', 
                    'message': f"✅ 시장가 {order.action} {order.name}({order.code}) {order.quantity}주 - 성공 (주문번호: {order_no})", 
                    'order': order,
                    'order_no': order_no
                }
            else:
                error_code = result.get('rt_cd', 'N/A') if result else 'N/A'
                error_msg = result.get('msg1', '알 수 없는 오류') if result else '응답 없음'
                return {
                    'status': 'failed', 
                    'message': f"❌ 시장가 {order.action} {order.name}({order.code}) {order.quantity}주 - 실패 (코드: {error_code}): {error_msg}", 
                    'order': order
                }
        except Exception as e:
            return {
                'status': 'error', 
                'message': f"❌ 시장가 {order.action} {order.name}({order.code}) - 예외 오류: {str(e)}", 
                'order': order
            }
    
    def _send_order_results(self, results: List[Dict]):
        """주문 결과를 텔레그램으로 전송 (시장가 표시 추가)"""
        if not results:
            return
        
        success_orders = [r for r in results if r.get('status') == 'success']
        failed_orders = [r for r in results if r.get('status') == 'failed']
        
        message = f"""
📊 <b>주문 실행 결과</b>

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ <b>성공한 주문: {len(success_orders)}개</b>
"""
        
        for order in success_orders:
            action_emoji = "🛒" if order.get('action') == 'BUY' else "💰"
            order_type = order.get('order_type', '시장가')
            message += f"{action_emoji} {order.get('name', 'N/A')}({order.get('code', 'N/A')}) "
            message += f"{order.get('quantity', 0)}주 [{order_type}]\n"
            
            if order.get('price', 0) > 0:
                message += f"   💰 참고가: {order.get('price', 0):,}원\n"
            if order.get('order_no'):
                message += f"   📋 주문번호: {order.get('order_no')}\n"
        
        if failed_orders:
            message += f"\n❌ <b>실패한 주문: {len(failed_orders)}개</b>\n"
            for order in failed_orders:
                message += f"• {order.get('name', 'N/A')}({order.get('code', 'N/A')}): {order.get('message', 'N/A')}\n"
        
        message += f"\n🔍 시장가 주문은 즉시 체결됩니다. HTS에서 체결 결과를 확인해주세요."
        
        self.telegram_notifier.send_message(message)

class FibonacciStrategy:
    """피보나치 분할매수 전략 설정"""
    enabled: bool = True
    
    # 매수 전략 우선순위 (낮은 숫자가 높은 우선순위)
    strategy_priority: Dict[str, int] = field(default_factory=lambda: {
        'TREND_CHANGE': 1,  # 최우선: 추세전환 매수
        'PULLBACK': 2,      # 2순위: 눌림목 매수  
        'BREAKOUT': 3       # 3순위: 전고점 돌파 매수
    })
    
    pullback_ratios: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.618])
    breakout_multipliers: List[float] = field(default_factory=lambda: [1, 2, 3])
    trend_change_signals: List[str] = field(default_factory=lambda: ['MA_CROSS', 'VOLUME_SPIKE', 'MOMENTUM'])
    
    # 피보나치 수열 기반 매수 수량 (1, 1, 2, 3, 5, 8...)
    fibonacci_sequence: List[int] = field(default_factory=lambda: [1, 1, 2, 3, 5, 8, 13])
    
    # 각 전략별 현재 단계
    pullback_stage: Dict[str, int] = field(default_factory=dict)
    breakout_stage: Dict[str, int] = field(default_factory=dict)
    trend_change_stage: Dict[str, int] = field(default_factory=dict)

class TechnicalAnalyzer:
    """기술적 분석 도구"""
    
    @staticmethod
    def analyze_market_situation(price_data: Dict) -> str:
        """시장 상황 분석하여 최적 전략 결정"""
        current_price = price_data['current_price']
        recent_high = price_data['recent_high']
        recent_low = price_data['recent_low']
        volume_ratio = price_data['volume_ratio']
        price_history = price_data['price_history']
        
        # 현재가 위치 분석
        price_position = (current_price - recent_low) / (recent_high - recent_low)
        
        # 추세 강도 분석
        if len(price_history) >= 20:
            ma5 = np.mean(price_history[-5:])
            ma20 = np.mean(price_history[-20:])
            trend_strength = (ma5 - ma20) / ma20 if ma20 > 0 else 0
        else:
            trend_strength = 0
        
        # 상황별 우선 전략 결정
        if abs(trend_strength) < 0.02:  # 횡보 구간
            if price_position < 0.4:  # 저점 근처
                return "TREND_CHANGE_PRIORITY"  # 추세전환 우선 대기
            else:
                return "PULLBACK_PRIORITY"  # 눌림목 우선
        elif trend_strength > 0.05:  # 강한 상승 추세
            if volume_ratio > 2.0:  # 거래량 급증
                return "BREAKOUT_PRIORITY"  # 돌파 우선
            else:
                return "PULLBACK_PRIORITY"  # 눌림목 우선
        else:  # 약한 추세 또는 불확실
            return "TREND_CHANGE_PRIORITY"  # 추세전환 우선
    
    # ... existing methods ...

class AdvancedTrader:
    """제미나이 AI와 고급 시장 분석 기능이 통합된 차세대 자동매매 시스템"""
    
    def __init__(self, app_key: str, app_secret: str, account_no: str, gemini_api_key: str, is_mock: bool = True):
        """AI 통합 고급 트레이더 초기화"""
        try:
            logging.info("🚀 AI 통합 시스템을 초기화합니다...")
            
            self.is_mock = is_mock
            
            # 핵심 컴포넌트 초기화
            self.api = KIS_API(app_key, app_secret, account_no, is_mock_env=is_mock)
            self.market_analyzer = MarketAnalyzer(self.api, gemini_api_key)
            
            # 관리 컴포넌트 초기화
            self.stock_name_cache = StockNameCache()
            self.portfolio_manager = PortfolioManager(self.api)
            self.telegram_notifier = TelegramNotifier(self.api.telegram_bot_token, self.api.telegram_chat_id)
            self.order_executor = OrderExecutor(self.api, self.telegram_notifier)
            
            # 전략 스크리너 초기화
            self._init_strategy_screeners()
            
            # 전략 설정
            self.scout_strategy = ScoutStrategy()
            self.watch_list = ['005930', '000660', '035420', '005380', '051910']
            
            # AI 분석 결과 저장
            self.last_market_analysis = None
            self.last_ai_signal = None
            
            # 시작 알림
            self._send_startup_notification()
            
            # 웹소켓 관리 컴포넌트 초기화
            self.ws_manager = WebSocketManager(self.api, self.is_mock)
            self.ws_manager.start()
            
            # 기술적 분석 도구 초기화
            self.technical_analyzer = TechnicalAnalyzer()
            
            # 피보나치 전략 초기화
            self.fibonacci_strategy = FibonacciStrategy()
            
            logging.info("✅ 시스템 초기화가 완료되었습니다.")
        except Exception as e:
            logging.error(f"❌ 시스템 초기화 중 오류 발생: {e}")
            self.shutdown()
            raise
    
    def _init_strategy_screeners(self):
        """전략 스크리너 초기화"""
        try:
            self.oneil_scanner = ONeilScanner(self.api)
            logging.info("✅ ONeilScanner 초기화 완료")
        except Exception as e:
            logging.warning(f"⚠️ ONeilScanner 초기화 실패: {e}")
            self.oneil_scanner = None
        
        try:
            self.minervini_screener = MinerviniScreener(self.api)
            logging.info("✅ MinerviniScreener 초기화 완료")
        except Exception as e:
            logging.warning(f"⚠️ MinerviniScreener 초기화 실패: {e}")
            self.minervini_screener = None
    
    def _send_startup_notification(self):
        """시작 알림 전송"""
        env_name = 'MOCK(모의투자)' if self.is_mock else 'LIVE(실전투자)'
        startup_msg = f"""
🎯 <b>척후병 매수 전략 활성화</b>

📊 환경: {env_name}
⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📋 <b>전략 설정:</b>
🔍 후보: {self.scout_strategy.candidate_count}개 | 🎖️ 척후병: {self.scout_strategy.scout_count}개 | 🏆 최종: {self.scout_strategy.final_count}개
📅 오디션 기간: {self.scout_strategy.evaluation_period}일

🚀 AI 기반 종목 선별이 시작됩니다!
"""
        self.telegram_notifier.send_message(startup_msg)
    
    def get_stock_name(self, stock_code: str) -> str:
        """종목명 조회"""
        return self.stock_name_cache.get_name(stock_code, self.api)
    
    async def make_trading_decision(self, symbol: str) -> Dict:
        """AI 매매 결정"""
        try:
            logging.info(f"🤖 {symbol}에 대한 AI 매매 결정 시작...")
            
            market_regime = self.market_analyzer.market_regime
            
            oneil_analysis, minervini_analysis = {}, {}
            if self.oneil_scanner:
                oneil_analysis = await self.oneil_scanner.analyze_stock(symbol) or {}
            if self.minervini_screener:
                minervini_analysis = await self.minervini_screener.analyze_stock(symbol) or {}
            
            if not oneil_analysis and not minervini_analysis:
                return await self._basic_trading_decision(symbol)
            
            prompt = self._build_integrated_decision_prompt(symbol, market_regime, oneil_analysis, minervini_analysis)
            response = await self.market_analyzer.model.generate_content_async(prompt)
            decision = self.market_analyzer._parse_ai_json_response(response.text)
            
            if not decision: return {'symbol': symbol, 'action': 'HOLD'}
            
            if market_regime != "AGGRESSIVE_GROWTH" and decision.get('action') == 'BUY':
                decision['confidence'] = decision.get('confidence', 0.7) * 0.5
            
            logging.info(f"✅ AI 결정: {decision.get('action', 'HOLD')} (신뢰도: {decision.get('confidence', 0):.2f})")
            return decision
        except Exception as e:
            logging.error(f"❌ {symbol} 매매 결정 중 오류: {e}")
            return {'symbol': symbol, 'action': 'HOLD', 'reasoning': str(e)}
    
    async def _basic_trading_decision(self, symbol: str) -> Dict:
        """스크리너 없을 때의 기본 매매 결정"""
        try:
            price_info = self.api.get_current_price(symbol)
            if not price_info or price_info.get('rt_cd') != '0':
                return {'symbol': symbol, 'action': 'HOLD'}
            
            change_rate = float(price_info['output'].get('prdy_ctrt', 0))
            
            if change_rate > 3.0: action, confidence = 'BUY', 0.6
            elif change_rate < -5.0: action, confidence = 'SELL', 0.7
            else: action, confidence = 'HOLD', 0.5
            
            return {'symbol': symbol, 'action': action, 'confidence': confidence}
        except Exception as e:
            logging.error(f"❌ 기본 매매 결정 중 오류: {e}")
            return {'symbol': symbol, 'action': 'HOLD'}
    
    def _build_integrated_decision_prompt(self, symbol: str, market_regime: str, oneil_analysis: Dict, minervini_analysis: Dict) -> str:
        """통합 결정 프롬프트 생성"""
        return f"""
당신은 제시 리버모어와 같은 AI 트레이딩 마스터입니다. 두 추세 추종 분석가(오닐, 미너비니)의 보고서와 시장 체제를 종합하여 매매를 결정하세요.

## 분석 정보
1. 시장 체제: {market_regime}
2. 오닐 분석: {json.dumps(oneil_analysis, ensure_ascii=False, indent=2)}
3. 미너비니 분석: {json.dumps(minervini_analysis, ensure_ascii=False, indent=2)}

## 출력 형식 (JSON)
```json
{{
    "symbol": "{symbol}", "action": "BUY/SELL/HOLD", "confidence": 0.0-1.0, 
    "reasoning": "분석 근거", "stop_loss_price": 0
}}
```"""
    
    async def execute_scout_strategy(self):
        """척후병 전략 실행"""
        try:
            logging.info("🔍 척후병 전략을 시작합니다...")
            
            if not self.portfolio_manager.sync_portfolio():
                self.telegram_notifier.send_message("❌ 포트폴리오 동기화 실패. 척후병 전략 중단.")
                return
            
            if self.portfolio_manager.cash_balance == 0:
                self.telegram_notifier.send_message("❌ 현금잔고 0원. 척후병 전략 실행 불가.")
                return
            
            candidates = await self._select_candidate_stocks()
            if len(candidates) < self.scout_strategy.scout_count:
                self.telegram_notifier.send_message("⚠️ 후보 종목 부족으로 척후병 매수 중단.")
                return
            
            scout_orders = await self._create_scout_orders(candidates[:self.scout_strategy.scout_count])
            
            if scout_orders:
                self.order_executor.execute_orders(scout_orders)
                self.scout_strategy.evaluation_start = datetime.now()
                self.scout_strategy.scout_positions = {order.code: order for order in scout_orders}
                
                bought_list = [f"{order.name}({order.code})" for order in scout_orders]
                buy_msg = f"""✅ <b>척후병 매수 주문 완료</b>
🛒 주문 종목 ({len(scout_orders)}개):
{chr(10).join([f"• {stock} - 1주" for stock in bought_list])}
📅 3일간 오디션 시작"""
                self.telegram_notifier.send_message(buy_msg)
            else:
                self.telegram_notifier.send_message("⚠️ 매수 가능한 척후병 종목이 없습니다.")
        except Exception as e:
            logging.error(f"❌ 척후병 전략 실행 중 오류: {e}", exc_info=True)
            self.telegram_notifier.send_message(f"❌ 척후병 전략 오류: {str(e)[:100]}")
    
    async def _select_candidate_stocks(self) -> List[str]:
        """후보 종목 선정"""
        logging.info("🔍 후보 종목 선정을 시작합니다...")
        
        candidates = []
        ai_count = 0
        if (self.last_ai_signal and self.last_ai_signal.get('recommended_stocks')):
            ai_recs = self.last_ai_signal['recommended_stocks'][:3]
            candidates.extend(ai_recs)
            ai_count = len(ai_recs)
        
        quality_stocks = ['005930', '000660', '035420', '005490', '051910', '035720', '006400']
        
        available = [s for s in quality_stocks if s not in self.portfolio_manager.portfolio and s not in candidates]
        
        needed = self.scout_strategy.candidate_count - len(candidates)
        if needed > 0: candidates.extend(available[:needed])
        
        final_candidates = candidates[:self.scout_strategy.candidate_count]
        await self._send_candidate_selection_notification(final_candidates, ai_count)
        return final_candidates
    
    async def _send_candidate_selection_notification(self, candidates: List[str], ai_count: int):
        """후보 선정 완료 알림"""
        details = []
        for code in candidates:
            name = self.get_stock_name(code)
            price_info = self.api.get_current_price(code)
            if price_info and price_info.get('rt_cd') == '0':
                price = int(price_info['output']['stck_prpr'])
                rate = float(price_info['output'].get('prdy_ctrt', 0))
                details.append(f"{name}({code}): {price:,}원 ({rate:+.2f}%)")
            else:
                details.append(f"{name}({code}): 가격 조회 실패")
        
        list_str = "\n".join([f"{i+1}. {d}" for i, d in enumerate(details)])
        msg = f"""
✅ <b>후보 종목 선정 완료</b>
🤖 AI 추천: {ai_count}개 | 📈 우량주 선별: {len(candidates) - ai_count}개

📋 <b>선정된 후보 종목:</b>
{list_str}

🎖️ 다음 단계: 척후병 매수 ({self.scout_strategy.scout_count}개)
"""
        self.telegram_notifier.send_message(msg)
    
    async def _create_scout_orders(self, candidates: List[str]) -> List[OrderInfo]:
        """척후병 주문 생성"""
        orders = []
        for code in candidates:
            try:
                name = self.get_stock_name(code)
                price_info = self.api.get_current_price(code)
                if price_info and price_info.get('rt_cd') == '0':
                    price = int(price_info['output']['stck_prpr'])
                    if self.portfolio_manager.can_buy(price, 1):
                        orders.append(OrderInfo(action='BUY', code=code, name=name, quantity=1))
                    else:
                        logging.warning(f"💸 {name} 매수 불가 - 잔고 부족")
                else:
                    logging.error(f"❌ {name} 가격 조회 실패")
            except Exception as e:
                logging.error(f"❌ {code} 주문 생성 중 오류: {e}")
        return orders
    
    async def _manage_existing_positions(self) -> List[OrderInfo]:
        """기존 포지션 관리 (매도 결정)"""
        sell_orders = []
        if not self.portfolio_manager.portfolio: return sell_orders
        
        for code, stock_info in self.portfolio_manager.portfolio.items():
            try:
                decision = await self.make_trading_decision(code)
                if (decision.get('action') == 'SELL' and 
                    decision.get('confidence', 0) > TradingConstants.AI_CONFIDENCE_THRESHOLD):
                    sell_orders.append(OrderInfo(
                        action='SELL', code=code, name=self.get_stock_name(code),
                        quantity=stock_info['quantity']
                    ))
            except Exception as e:
                logging.error(f"❌ 포지션 분석 중 오류 ({code}): {e}")
        return sell_orders
    
    async def _find_new_positions(self) -> List[OrderInfo]:
        """신규 포지션 발굴 (매수 결정)"""
        buy_orders = []
        pm = self.portfolio_manager
        
        if len(pm.portfolio) >= TradingConstants.MAX_POSITIONS: return buy_orders
        if pm.cash_balance < pm.get_position_size(): return buy_orders
        
        candidates = [c for c in self.watch_list if c not in pm.portfolio]
        
        for code in candidates:
            try:
                decision = await self.make_trading_decision(code)
                if (decision.get('action') == 'BUY' and 
                    decision.get('confidence', 0) > TradingConstants.AI_CONFIDENCE_THRESHOLD):
                    
                    price_info = self.api.get_current_price(code)
                    if not price_info or price_info.get('rt_cd') != '0': continue
                    
                    price = int(price_info['output']['stck_prpr'])
                    investment = pm.get_position_size()
                    quantity = max(1, int(investment // price))
                    
                    if not pm.can_buy(price, quantity):
                        quantity = max(1, int(pm.cash_balance // price))
                        if not pm.can_buy(price, quantity): continue
                    
                    buy_orders.append(OrderInfo(
                        action='BUY', code=code, name=self.get_stock_name(code), quantity=quantity
                    ))
                    break # 한 번에 한 종목만
            except Exception as e:
                logging.error(f"❌ {code} 신규 포지션 분석 오류: {e}")
        return buy_orders
    
    async def rebalance_portfolio(self):
        """AI 기반 포트폴리오 리밸런싱"""
        logging.info("🔄 AI 기반 포트폴리오 리밸런싱 시작")
        
        try:
            await self.market_analyzer.get_market_regime_analysis()
            logging.info(f"🧭 현재 시장 체제: {self.market_analyzer.market_regime}")

            if not self.portfolio_manager.sync_portfolio(): return

            scout = self.scout_strategy
            if scout.enabled and not scout.evaluation_start:
                await self.execute_scout_strategy()
                return 
            
            elif scout.enabled and scout.evaluation_start:
                if datetime.now() >= scout.evaluation_start + timedelta(days=scout.evaluation_period):
                    logging.info("🎯 척후병 평가 및 포지션 정리 실행 (구현 필요)")
                else:
                    remaining_days = (scout.evaluation_start + timedelta(days=scout.evaluation_period) - datetime.now()).days
                    logging.info(f"⏳ 척후병 오디션 진행 중... (남은 기간: {remaining_days}일)")
                return
            
            # 일반 리밸런싱
            logging.info("⚙️ 일반 포트폴리오 리밸런싱을 실행합니다.")
            sell_orders = await self._manage_existing_positions()
            buy_orders = await self._find_new_positions()
            
            all_orders = sell_orders + buy_orders
            if all_orders: self.order_executor.execute_orders(all_orders)
            else: logging.info("📊 실행할 주문이 없습니다.")
                    
        except Exception as e:
            logging.error(f"❌ 리밸런싱 중 오류: {e}", exc_info=True)
            self.telegram_notifier.send_message(f"❌ 리밸런싱 오류: {str(e)[:100]}")
    
    async def run_trading_cycle(self):
        """메인 거래 사이클 (주문 상태 확인 추가)"""
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                cycle_start_time = datetime.now()
                
                logging.info(f"🔄 거래 사이클 #{cycle_count} 시작 - {cycle_start_time.strftime('%H:%M:%S')}")
                
                # 1. 포트폴리오 동기화
                if not self.sync_portfolio():
                    logging.error("❌ 포트폴리오 동기화 실패. 다음 사이클에서 재시도합니다.")
                    await asyncio.sleep(60)
                    continue
                
                # 2. 주문 상태 확인 (매 3번째 사이클마다)
                if cycle_count % 3 == 0:
                    self.check_order_status()
                
                # 3. 시장 체제 분석
                market_analysis = await self.get_market_regime_analysis()
                
                # 4. 거래 결정 및 실행
                await self._execute_trading_strategy(market_analysis)
                
                # 5. 사이클 완료 로깅
                cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                logging.info(f"✅ 거래 사이클 #{cycle_count} 완료 (소요시간: {cycle_duration:.1f}초)")
                
                # 6. 다음 사이클까지 대기
                await asyncio.sleep(self.rebalance_interval)
                
            except KeyboardInterrupt:
                logging.info("🛑 사용자에 의한 프로그램 종료 요청")
                break
            except Exception as e:
                logging.error(f"❌ 거래 사이클 중 오류: {e}")
                await asyncio.sleep(30)  # 오류 시 30초 대기 후 재시도

    def shutdown(self):
        """리소스 정리"""
        logging.info("시스템 종료 중...")
        if hasattr(self, 'telegram_notifier'):
            self.telegram_notifier.shutdown()
        if hasattr(self, 'api'):
            self.api.stop_token_scheduler()
        if hasattr(self, 'ws_manager'):
            self.ws_manager.close()
        logging.info("시스템이 종료되었습니다.")

    def check_order_status(self, order_no: str = None) -> Dict[str, Any]:
        """주문 체결 상태 확인"""
        try:
            logging.info("📋 주문 체결 상태를 확인합니다...")
            
            # 미체결 주문 조회
            pending_orders = self.api.get_pending_orders()
            
            if pending_orders and pending_orders.get('rt_cd') == '0':
                orders = pending_orders.get('output', [])
                
                if orders:
                    status_msg = f"""
📋 <b>미체결 주문 현황</b>

⏰ 조회 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 미체결 주문: {len(orders)}건

"""
                    for order in orders[:10]:  # 최대 10건만 표시
                        stock_name = order.get('prdt_name', 'N/A')
                        stock_code = order.get('pdno', 'N/A')
                        order_qty = self._safe_int(order.get('ord_qty', 0))
                        order_price = self._safe_int(order.get('ord_unpr', 0))
                        order_type = "매수" if order.get('sll_buy_dvsn_cd') == '02' else "매도"
                        
                        status_msg += f"• {order_type} {stock_name}({stock_code}) {order_qty}주 @ {order_price:,}원\n"
                    
                    self._send_telegram_message(status_msg)
                    return {'success': True, 'orders': orders}
                else:
                    no_orders_msg = """
📋 <b>미체결 주문 없음</b>

✅ 현재 미체결 주문이 없습니다.
🔍 모든 주문이 체결되었거나 주문이 없는 상태입니다.
"""
                    self._send_telegram_message(no_orders_msg)
                    return {'success': True, 'orders': []}
            else:
                error_msg = f"주문 조회 실패: {pending_orders.get('msg1', '알 수 없는 오류')}"
                logging.error(error_msg)
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"주문 상태 확인 중 오류: {e}"
            logging.error(error_msg)
            return {'success': False, 'error': error_msg}

    async def _execute_trading_strategy(self, market_analysis: Dict):
        """시장 상황에 따른 거래 전략 실행"""
        try:
            logging.info("🔄 거래 전략을 실행합니다...")
            
            # 시장 상황에 따른 전략 선택
            strategy = self.technical_analyzer.analyze_market_situation(market_analysis)
            
            if strategy == "TREND_CHANGE_PRIORITY":
                await self._execute_trend_change_strategy()
            elif strategy == "PULLBACK_PRIORITY":
                await self._execute_pullback_strategy()
            elif strategy == "BREAKOUT_PRIORITY":
                await self._execute_breakout_strategy()
            else:
                logging.info("📊 현재 매매 신호 없음. 휴식 중...")
        except Exception as e:
            logging.error(f"❌ 거래 전략 실행 중 오류: {e}")
            self.telegram_notifier.send_message(f"❌ 거래 전략 실행 오류: {str(e)[:100]}")

    async def _execute_trend_change_strategy(self):
        """추세전환 전략 실행"""
        try:
            logging.info("🔄 추세전환 전략을 실행합니다...")
            
            # 추세전환 전략 실행
            await self._execute_stock_fibonacci_strategy('005930')  # 예시로 삼성전자에 적용
        except Exception as e:
            logging.error(f"❌ 추세전환 전략 실행 중 오류: {e}")
            self.telegram_notifier.send_message(f"❌ 추세전환 전략 실행 오류: {str(e)[:100]}")

    async def _execute_pullback_strategy(self):
        """눌림목 전략 실행"""
        try:
            logging.info("🔄 눌림목 전략을 실행합니다...")
            
            # 눌림목 전략 실행
            await self._execute_stock_fibonacci_strategy('000660')  # 예시로 SK하이닉스에 적용
        except Exception as e:
            logging.error(f"❌ 눌림목 전략 실행 중 오류: {e}")
            self.telegram_notifier.send_message(f"❌ 눌림목 전략 실행 오류: {str(e)[:100]}")

    async def _execute_breakout_strategy(self):
        """돌파 전략 실행"""
        try:
            logging.info("🔄 돌파 전략을 실행합니다...")
            
            # 돌파 전략 실행
            await self._execute_stock_fibonacci_strategy('035420')  # 예시로 NAVER에 적용
        except Exception as e:
            logging.error(f"❌ 돌파 전략 실행 중 오류: {e}")
            self.telegram_notifier.send_message(f"❌ 돌파 전략 실행 오류: {str(e)[:100]}")

    async def _execute_stock_fibonacci_strategy(self, stock_code: str):
        """개별 종목에 대한 피보나치 전략 실행 (우선순위 적용)"""
        stock_name = self.get_stock_name(stock_code)
        logging.info(f"📊 {stock_name}({stock_code}) 피보나치 분석 시작...")
        
        # 현재 가격 및 기술적 데이터 수집
        price_data = await self._collect_price_data(stock_code)
        if not price_data:
            logging.error(f"❌ {stock_name} 가격 데이터 수집 실패")
            return
        
        current_price = price_data['current_price']
        recent_high = price_data['recent_high']
        recent_low = price_data['recent_low']
        volume_ratio = price_data['volume_ratio']
        price_history = price_data['price_history']
        volume_history = price_data['volume_history']
        
        # 시장 상황 분석
        market_situation = self.technical_analyzer.analyze_market_situation(price_data)
        logging.info(f"📈 {stock_name} 시장 상황: {market_situation}")
        
        # 모든 매수 신호 분석
        available_strategies = []
        
        # 1. 추세전환 매수 분석
        is_trend_change, signal_type = self.technical_analyzer.detect_trend_change(
            price_history, volume_history
        )
        if is_trend_change:
            available_strategies.append({
                'type': 'TREND_CHANGE',
                'priority': self.fibonacci_strategy.strategy_priority['TREND_CHANGE'],
                'reason': f'추세전환 신호: {signal_type}',
                'quantity': self._get_fibonacci_quantity(stock_code, 'trend_change'),
                'confidence': 0.9  # 추세전환은 높은 신뢰도
            })
        
        # 2. 눌림목 매수 분석
        is_pullback, fib_level = self.technical_analyzer.detect_pullback_opportunity(
            current_price, recent_high, recent_low
        )
        if is_pullback:
            available_strategies.append({
                'type': 'PULLBACK',
                'priority': self.fibonacci_strategy.strategy_priority['PULLBACK'],
                'reason': f'피보나치 {fib_level} 레벨 눌림목',
                'quantity': self._get_fibonacci_quantity(stock_code, 'pullback'),
                'confidence': 0.8
            })
        
        # 3. 전고점 돌파 매수 분석
        is_breakout = self.technical_analyzer.detect_breakout_opportunity(
            current_price, recent_high, volume_ratio
        )
        if is_breakout:
            available_strategies.append({
                'type': 'BREAKOUT',
                'priority': self.fibonacci_strategy.strategy_priority['BREAKOUT'],
                'reason': f'전고점 {recent_high:,}원 돌파 (거래량 {volume_ratio:.1f}배)',
                'quantity': self._get_fibonacci_quantity(stock_code, 'breakout'),
                'confidence': 0.7
            })
        
        if not available_strategies:
            logging.info(f"📊 {stock_name}: 현재 매수 신호 없음")
            return
        
        # 상황별 우선순위 적용하여 전략 선택
        selected_strategy = self._select_optimal_strategy(available_strategies, market_situation)
        
        if selected_strategy:
            await self._execute_fibonacci_orders(
                stock_code, stock_name, current_price, [selected_strategy]
            )
            
            # 실행된 전략 로깅
            logging.info(f"✅ {stock_name} 실행 전략: {selected_strategy['type']} (우선순위: {selected_strategy['priority']})")
    
    def _select_optimal_strategy(self, available_strategies: List[Dict], market_situation: str) -> Optional[Dict]:
        """시장 상황에 따른 최적 전략 선택"""
        if not available_strategies:
            return None
        
        # 시장 상황별 우선순위 조정
        situation_weights = {
            "TREND_CHANGE_PRIORITY": {'TREND_CHANGE': 0.5, 'PULLBACK': 0.3, 'BREAKOUT': 0.2},
            "PULLBACK_PRIORITY": {'PULLBACK': 0.5, 'TREND_CHANGE': 0.3, 'BREAKOUT': 0.2},
            "BREAKOUT_PRIORITY": {'BREAKOUT': 0.5, 'PULLBACK': 0.3, 'TREND_CHANGE': 0.2}
        }
        
        weights = situation_weights.get(market_situation, {
            'TREND_CHANGE': 0.4, 'PULLBACK': 0.35, 'BREAKOUT': 0.25
        })
        
        # 각 전략의 점수 계산 (우선순위 + 신뢰도 + 상황별 가중치)
        for strategy in available_strategies:
            strategy_type = strategy['type']
            
            # 점수 계산 (낮을수록 좋음)
            priority_score = strategy['priority']  # 1, 2, 3
            confidence_score = (1 - strategy['confidence']) * 5  # 신뢰도가 높을수록 낮은 점수
            situation_score = (1 - weights.get(strategy_type, 0.1)) * 3  # 상황 적합도
            
            strategy['total_score'] = priority_score + confidence_score + situation_score
        
        # 가장 낮은 점수(최적) 전략 선택
        selected = min(available_strategies, key=lambda x: x['total_score'])
        
        logging.info(f"🎯 전략 선택 결과:")
        for strategy in available_strategies:
            status = "✅ 선택됨" if strategy == selected else "⏸️ 대기"
            logging.info(f"   {strategy['type']}: 점수 {strategy['total_score']:.2f} {status}")
        
        return selected
    
    async def _execute_fibonacci_orders(self, stock_code: str, stock_name: str, current_price: int, strategies: List[Dict]):
        """피보나치 전략 주문 실행 (우선순위 순서대로)"""
        # 우선순위 순으로 정렬
        strategies.sort(key=lambda x: x['priority'])
        
        orders = []
        total_quantity = 0
        
        for strategy in strategies:
            quantity = strategy['quantity']
            total_quantity += quantity
            
            orders.append(OrderInfo(
                action='BUY',
                code=stock_code,
                name=stock_name,
                quantity=quantity,
                reason=f"피보나치 {strategy['type']}: {strategy['reason']}"
            ))
        
        if orders:
            # 잔고 확인
            total_cost = current_price * total_quantity
            if not self.portfolio_manager.can_buy(current_price, total_quantity):
                # 잔고 부족 시 가장 우선순위 높은 전략만 실행
                available_quantity = max(1, self.portfolio_manager.cash_balance // current_price)
                logging.warning(f"⚠️ 잔고 부족으로 최우선 전략만 실행: {orders[0].reason}")
                
                orders = [orders[0]]  # 첫 번째(최우선) 주문만
                orders[0].quantity = min(available_quantity, orders[0].quantity)
            
            # 주문 실행
            self.order_executor.execute_orders(orders)
            
            # 상세 알림 전송
            await self._send_fibonacci_notification(stock_name, stock_code, current_price, strategies, orders)
    
    async def _send_fibonacci_notification(self, stock_name: str, stock_code: str, current_price: int, 
                                         strategies: List[Dict], orders: List[OrderInfo]):
        """피보나치 전략 실행 알림 (우선순위 표시)"""
        strategy_names = {
            'TREND_CHANGE': '🔄 추세전환 매수',
            'PULLBACK': '📉 눌림목 매수',
            'BREAKOUT': '🚀 돌파 매수'
        }
        
        priority_emojis = {1: '🥇', 2: '🥈', 3: '🥉'}
        
        executed_strategies = []
        available_strategies = []
        
        for strategy in strategies:
            priority_emoji = priority_emojis.get(strategy['priority'], '📊')
            strategy_name = strategy_names.get(strategy['type'], strategy['type'])
            strategy_info = f"{priority_emoji} {strategy_name}: {strategy['reason']}"
            
            # 실제 실행된 전략인지 확인
            if any(order.reason.startswith(f"피보나치 {strategy['type']}") for order in orders):
                executed_strategies.append(strategy_info + " ✅")
            else:
                available_strategies.append(strategy_info + " ⏸️")
        
        total_quantity = sum(order.quantity for order in orders)
        total_amount = current_price * total_quantity
        
        message = f"""
🔢 <b>피보나치 분할매수 실행</b>

📊 종목: {stock_name}({stock_code})
💰 현재가: {current_price:,}원
📈 총 매수량: {total_quantity}주
💵 총 투자금액: {total_amount:,}원

🎯 <b>실행된 전략:</b>
{chr(10).join(executed_strategies)}

⏸️ <b>대기 중인 전략:</b>
{chr(10).join(available_strategies) if available_strategies else '없음'}

📋 <b>주문 내역:</b>
{chr(10).join([f"• {order.quantity}주 - {order.reason}" for order in orders])}

💡 <b>매수 우선순위:</b>
🥇 추세전환 → 🥈 눌림목 → 🥉 돌파

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.telegram_notifier.send_message(message)

async def main():
    """메인 실행 함수"""
    trader = None
    try:
        is_mock = os.getenv('IS_MOCK', 'true').lower() == 'true'
        env_prefix = "MOCK" if is_mock else "LIVE"
        
        app_key = os.getenv(f'{env_prefix}_KIS_APP_KEY')
        app_secret = os.getenv(f'{env_prefix}_KIS_APP_SECRET')
        account_no = os.getenv(f'{env_prefix}_KIS_ACCOUNT_NUMBER')
        gemini_api_key = os.getenv('GEMINI_API_KEY', "YOUR_GEMINI_API_KEY") # Replace with your key
        
        if not all([app_key, app_secret, account_no, gemini_api_key]):
            logging.error(f"{env_prefix} 환경의 필수 환경 변수가 설정되지 않았습니다.")
            return
        
        trader = AdvancedTrader(
            app_key=app_key, app_secret=app_secret, account_no=account_no,
            gemini_api_key=gemini_api_key, is_mock=is_mock
        )
        
        await trader.run_trading_cycle()
        
    except KeyboardInterrupt:
        logging.info("사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        logging.error(f"프로그램 실행 중 치명적 오류 발생: {e}", exc_info=True)
    finally:
        if trader:
            trader.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("프로그램을 종료합니다.")