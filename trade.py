"""
🚀 올인원 매매 시스템 v2.0
- 윌리엄 오닐 전략
- 제시 리버모어 기법  
- 척후병 매수 전략
- 피보나치 분할매수 (1,1,2,3,5,8,13...)
- 3가지 매수 전략 (추세전환, 눌림목, 전고점 돌파)
"""

import asyncio
import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import aiohttp
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class OrderType(Enum):
    """주문 타입"""
    MARKET = "01"  # 시장가
    LIMIT = "00"   # 지정가

class StrategyType(Enum):
    """전략 타입"""
    SCOUT = "척후병"
    FIBONACCI = "피보나치"
    TREND_REVERSAL = "추세전환"
    PULLBACK = "눌림목"
    BREAKOUT = "전고점돌파"

@dataclass
class Stock:
    """주식 정보"""
    symbol: str
    name: str = ""
    price: int = 0
    quantity: int = 0

@dataclass
class TradingConfig:
    """매매 설정"""
    url: str = "https://openapivts.koreainvestment.com:29443"
    fibonacci_sequence: List[int] = None
    max_candidates: int = 5
    scout_selection: int = 4
    final_selection: int = 2
    
    def __post_init__(self):
        if self.fibonacci_sequence is None:
            self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34]

class APIClient:
    """API 클라이언트 (속도 최적화)"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.key = os.getenv('MOCK_KIS_APP_KEY')
        self.secret = os.getenv('MOCK_KIS_APP_SECRET')
        self.account = os.getenv('MOCK_KIS_ACCOUNT_NUMBER')
        self.token: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_expires = 0
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def _ensure_token(self) -> bool:
        """토큰 자동 갱신 (캐싱)"""
        current_time = time.time()
        if self.token and current_time < self._token_expires:
            return True
        
        data = {
            "grant_type": "client_credentials",
            "appkey": self.key,
            "appsecret": self.secret
        }
        
        try:
            async with self.session.post(f"{self.config.url}/oauth2/tokenP", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.token = result.get('access_token')
                    self._token_expires = current_time + 3600  # 1시간 후 만료
                    logger.info("✅ 토큰 발급/갱신 완료")
                    return True
        except Exception as e:
            logger.error(f"❌ 토큰 발급 실패: {e}")
        
        return False
    
    async def _make_request(self, method: str, endpoint: str, tr_id: str, 
                          data: Dict = None, params: Dict = None) -> Optional[Dict]:
        """통합 API 요청 (에러 처리 강화)"""
        if not await self._ensure_token():
            return None
        
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.key,
            "appsecret": self.secret,
            "tr_id": tr_id
        }
        
        url = f"{self.config.url}{endpoint}"
        
        try:
            if method.upper() == "POST":
                async with self.session.post(url, headers=headers, json=data) as response:
                    return await response.json()
            else:
                async with self.session.get(url, headers=headers, params=params) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"❌ API 요청 실패: {e}")
            return None

class OrderExecutor:
    """주문 실행기 (책임 분리)"""
    
    def __init__(self, api_client: APIClient):
        self.api = api_client
    
    async def buy(self, symbol: str, quantity: int, order_type: OrderType = OrderType.MARKET) -> bool:
        """매수 주문"""
        data = {
            "CANO": self.api.account[:8],
            "ACNT_PRDT_CD": self.api.account[8:],
            "PDNO": symbol,
            "ORD_DVSN": order_type.value,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0"
        }
        
        logger.info(f"🛒 {symbol} {quantity}주 매수 주문")
        result = await self.api._make_request(
            "POST", 
            "/uapi/domestic-stock/v1/trading/order-cash",
            "VTTC0802U",
            data
        )
        
        if result and result.get('rt_cd') == '0':
            logger.info(f"✅ {symbol} {quantity}주 매수 성공")
            return True
        else:
            logger.error(f"❌ {symbol} 매수 실패: {result}")
            return False
    
    async def sell(self, symbol: str, quantity: int, order_type: OrderType = OrderType.MARKET) -> bool:
        """매도 주문"""
        data = {
            "CANO": self.api.account[:8],
            "ACNT_PRDT_CD": self.api.account[8:],
            "PDNO": symbol,
            "ORD_DVSN": order_type.value,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0"
        }
        
        logger.info(f"💰 {symbol} {quantity}주 매도 주문")
        result = await self.api._make_request(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            "VTTC0801U",
            data
        )
        
        if result and result.get('rt_cd') == '0':
            logger.info(f"✅ {symbol} {quantity}주 매도 성공")
            return True
        else:
            logger.error(f"❌ {symbol} 매도 실패: {result}")
            return False

class PortfolioManager:
    """포트폴리오 관리자"""
    
    def __init__(self, api_client: APIClient):
        self.api = api_client
    
    async def get_holdings(self) -> List[Stock]:
        """보유 종목 조회"""
        params = {
            "CANO": self.api.account[:8],
            "ACNT_PRDT_CD": self.api.account[8:],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        result = await self.api._make_request(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            "VTTC8434R",
            params=params
        )
        
        holdings = []
        if result and result.get('rt_cd') == '0':
            for item in result.get('output1', []):
                quantity = int(item.get('HLDG_QTY', 0))
                if quantity > 0:
                    stock = Stock(
                        symbol=item.get('PDNO'),
                        name=item.get('PRDT_NAME'),
                        quantity=quantity
                    )
                    holdings.append(stock)
                    logger.info(f"📊 보유: {stock.symbol}({stock.name}) {stock.quantity}주")
        
        return holdings

class TradingStrategy:
    """매매 전략 (전략 패턴)"""
    
    def __init__(self, executor: OrderExecutor, portfolio: PortfolioManager, config: TradingConfig):
        self.executor = executor
        self.portfolio = portfolio
        self.config = config
    
    async def scout_strategy(self, candidates: List[str]) -> List[str]:
        """척후병 매수 전략 - 5개 후보 → 4개 각 1주 → 3일 오디션 → 상위 2개"""
        logger.info("🔍 척후병 매수 전략 시작")
        
        # 1단계: 상위 4개 종목 선정
        selected_candidates = candidates[:self.config.scout_selection]
        logger.info(f"📋 선정된 후보: {selected_candidates}")
        
        # 2단계: 각 1주씩 매수 (병렬 처리로 속도 최적화)
        buy_tasks = [self.executor.buy(symbol, 1) for symbol in selected_candidates]
        results = await asyncio.gather(*buy_tasks, return_exceptions=True)
        
        successful_buys = []
        for symbol, result in zip(selected_candidates, results):
            if result is True:
                successful_buys.append(symbol)
        
        logger.info(f"✅ 매수 성공: {successful_buys}")
        
        # 3단계: 3일 오디션 (실제로는 성과 분석 로직)
        logger.info("⏰ 3일 오디션 기간 시뮬레이션...")
        await asyncio.sleep(1)  # 시뮬레이션
        
        # 4단계: 상위 2개 최종 선정
        final_selected = successful_buys[:self.config.final_selection]
        logger.info(f"🏆 최종 선정: {final_selected}")
        
        return final_selected
    
    async def fibonacci_strategy(self, symbol: str, total_amount: int = 100000) -> bool:
        """피보나치 분할매수 전략 (1,1,2,3,5,8,13...)"""
        logger.info(f"📈 {symbol} 피보나치 분할매수 시작 (총 {total_amount:,}원)")
        
        # 현재가 조회 (실제로는 시세 API 호출)
        current_price = 70000  # 임시 가격
        
        # 피보나치 비율로 분할매수
        fib_sum = sum(self.config.fibonacci_sequence[:5])  # 처음 5단계
        
        buy_tasks = []
        for i, ratio in enumerate(self.config.fibonacci_sequence[:5]):
            amount = int(total_amount * ratio / fib_sum)
            quantity = max(1, amount // current_price)
            
            logger.info(f"  📊 {i+1}단계: {quantity}주 (비율 {ratio}/{fib_sum})")
            buy_tasks.append(self.executor.buy(symbol, quantity))
        
        # 병렬 실행으로 속도 최적화
        results = await asyncio.gather(*buy_tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)
        
        logger.info(f"✅ 피보나치 매수 완료: {success_count}/{len(results)}건 성공")
        return success_count > 0
    
    async def trend_strategy(self, symbol: str, strategy_type: StrategyType) -> bool:
        """추세 기반 매수 전략"""
        logger.info(f"📊 {symbol} {strategy_type.value} 전략 실행")
        
        # 전략별 매수 수량 결정
        quantity_map = {
            StrategyType.TREND_REVERSAL: 2,  # 추세전환
            StrategyType.PULLBACK: 3,        # 눌림목
            StrategyType.BREAKOUT: 5         # 전고점 돌파
        }
        
        quantity = quantity_map.get(strategy_type, 1)
        
        # 전략별 신호 감지 (실제로는 기술적 분석 로직)
        signal_detected = await self._detect_signal(symbol, strategy_type)
        
        if signal_detected:
            logger.info(f"  🎯 {strategy_type.value} 신호 감지!")
            return await self.executor.buy(symbol, quantity)
        else:
            logger.info(f"  ⏸️ {strategy_type.value} 신호 없음")
            return False
    
    async def _detect_signal(self, symbol: str, strategy_type: StrategyType) -> bool:
        """신호 감지 로직 (실제로는 기술적 분석)"""
        # 시뮬레이션: 70% 확률로 신호 감지
        await asyncio.sleep(0.1)  # API 호출 시뮬레이션
        return True  # 임시로 항상 True

class TradingSystem:
    """통합 매매 시스템 (파사드 패턴)"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.api_client = None
        self.executor = None
        self.portfolio = None
        self.strategy = None
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저"""
        self.api_client = APIClient(self.config)
        await self.api_client.__aenter__()
        
        self.executor = OrderExecutor(self.api_client)
        self.portfolio = PortfolioManager(self.api_client)
        self.strategy = TradingStrategy(self.executor, self.portfolio, self.config)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """리소스 정리"""
        if self.api_client:
            await self.api_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def run_full_strategy(self, candidates: List[str] = None) -> Dict:
        """전체 전략 실행"""
        if candidates is None:
            candidates = ["005930", "000660", "035420", "005380", "051910"]  # 기본 후보
        
        logger.info("🚀 통합 매매 전략 시작")
        results = {}
        
        try:
            # 1. 척후병 전략
            selected_stocks = await self.strategy.scout_strategy(candidates)
            results['scout'] = selected_stocks
            
            # 2. 선정된 종목에 대해 추가 전략 실행 (병렬 처리)
            strategy_tasks = []
            for symbol in selected_stocks:
                # 피보나치 전략
                strategy_tasks.append(self.strategy.fibonacci_strategy(symbol))
                # 추세 전략들
                strategy_tasks.append(self.strategy.trend_strategy(symbol, StrategyType.TREND_REVERSAL))
                strategy_tasks.append(self.strategy.trend_strategy(symbol, StrategyType.PULLBACK))
                strategy_tasks.append(self.strategy.trend_strategy(symbol, StrategyType.BREAKOUT))
            
            # 모든 전략 병렬 실행
            strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
            results['strategies'] = strategy_results
            
            # 3. 최종 포트폴리오 확인
            final_holdings = await self.portfolio.get_holdings()
            results['final_holdings'] = final_holdings
            
            logger.info("✅ 통합 매매 전략 완료")
            
        except Exception as e:
            logger.error(f"❌ 전략 실행 중 오류: {e}")
            results['error'] = str(e)
        
        return results
    
    # 간단한 실행 함수들
    async def simple_buy(self, symbol: str = "005930", quantity: int = 1) -> bool:
        """간단 매수"""
        return await self.executor.buy(symbol, quantity)
    
    async def simple_sell(self, symbol: str = "005930", quantity: int = 1) -> bool:
        """간단 매도"""
        return await self.executor.sell(symbol, quantity)
    
    async def check_portfolio(self) -> List[Stock]:
        """포트폴리오 확인"""
        return await self.portfolio.get_holdings()

# 실행 함수들 (사용자 편의)
async def buy_test(symbol: str = "005930", qty: int = 1):
    """매수 테스트"""
    async with TradingSystem() as system:
        return await system.simple_buy(symbol, qty)

async def sell_test(symbol: str = "005930", qty: int = 1):
    """매도 테스트"""
    async with TradingSystem() as system:
        return await system.simple_sell(symbol, qty)

async def portfolio_check():
    """포트폴리오 확인"""
    async with TradingSystem() as system:
        return await system.check_portfolio()

async def full_auto_trading():
    """전체 자동매매"""
    async with TradingSystem() as system:
        return await system.run_full_strategy()

async def main():
    """메인 실행"""
    print("🤖 자동매매 시스템 v2.0")
    print("1. 매수 테스트: await buy_test()")
    print("2. 매도 테스트: await sell_test()")
    print("3. 포트폴리오: await portfolio_check()")
    print("4. 전체 자동매매: await full_auto_trading()")
    
    # 전체 자동매매 실행
    result = await full_auto_trading()
    print(f"📊 실행 결과: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 