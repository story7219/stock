# portfolio.py
# 포트폴리오 상태 (현금, 보유 종목)를 관리하고, 거래 내역을 기록하는 클래스
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from typing import Dict, List, Optional
import config
from trading.kis_api import KIS_API
from utils.logger import log_event
from utils.telegram_bot import TelegramBot

class StockStatus(Enum):
    """종목의 포트폴리오 내 상태"""
    SCOUTING = auto()  # 정찰 (오디션 진행 중)
    HOLDING = auto()   # 정식 보유

@dataclass
class Stock:
    """개별 종목의 상태 정보를 담는 데이터 클래스"""
    ticker: str
    quantity: int
    avg_price: float
    status: StockStatus
    purchase_time: datetime
    
    @property
    def total_investment(self) -> float:
        """총 투자 원금"""
        return self.avg_price * self.quantity

class PortfolioManager:
    """
    계좌의 현금, 보유/정찰 종목 등 전체 포트폴리오를 관리합니다.
    모든 자산 변경은 이 클래스를 통해 이루어져야 합니다.
    """
    def __init__(self, capital: float, kis_api: KIS_API, telegram_bot: TelegramBot):
        self.initial_capital = capital
        self.cash = capital
        self.holdings: Dict[str, Stock] = {}  # {ticker: Stock}
        self.kis_api = kis_api
        self.telegram_bot = telegram_bot
        self.trade_log: List[Dict] = []
        self.long_term_stock = config.LONG_TERM_STOCK
        self.long_term_amount = self.initial_capital * config.LONG_TERM_BUY_AMOUNT
        # 장기투자 종목 자동 매수
        self.add_long_term_stock()
        log_event("INFO", f"💰 포트폴리오 매니저 초기화. 초기 자본금: {capital:,.0f}원")

    def can_invest(self, invest_amount: float) -> bool:
        """현금 25% 이상을 반드시 남겨두는지 체크"""
        min_cash = self.initial_capital * config.MIN_CASH_RATIO
        return (self.cash - invest_amount) >= min_cash

    def add_scout(self, ticker: str, price: float, quantity: int):
        """정찰병(scout)을 포트폴리오에 추가합니다."""
        if ticker in self.holdings:
            log_event("WARNING", f"이미 포트폴리오에 포함된 종목({ticker})에 정찰병을 추가할 수 없습니다.")
            return

        investment = price * quantity
        if not self.can_invest(investment):
            log_event("WARNING", f"현금 25% 유지 조건으로 정찰병 추가 불가: {ticker}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"⚠️ 현금 25% 유지 조건으로 정찰병 추가 불가: {ticker}")
            return

        self.cash -= investment
        stock = Stock(
            ticker=ticker,
            quantity=quantity,
            avg_price=price,
            status=StockStatus.SCOUTING,
            purchase_time=datetime.now()
        )
        self.holdings[ticker] = stock
        log_event("INFO", f"🕵️ [정찰 시작] {ticker} | {quantity}주 | 평단가: {price:,.0f}원 | 잔여 현금: {self.cash:,.0f}원")

    def promote_stock(self, ticker: str, price: float, additional_quantity: int):
        """정찰병을 정식 보유 종목으로 승격시키고, 추가 매수합니다."""
        if ticker not in self.holdings or self.holdings[ticker].status != StockStatus.SCOUTING:
            log_event("WARNING", f"정찰 중이 아닌 종목({ticker})을 승격할 수 없습니다.")
            return
        
        additional_investment = price * additional_quantity
        if not self.can_invest(additional_investment):
            log_event("WARNING", f"현금 25% 유지 조건으로 본대 투입 불가: {ticker}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"⚠️ 현금 25% 유지 조건으로 본대 투입 불가: {ticker}")
            return
            
        self.cash -= additional_investment
        stock = self.holdings[ticker]
        
        # 평균 단가 재계산
        total_investment = stock.total_investment + additional_investment
        total_quantity = stock.quantity + additional_quantity
        stock.avg_price = total_investment / total_quantity
        
        stock.quantity = total_quantity
        stock.status = StockStatus.HOLDING
        stock.purchase_time = datetime.now() # 승격 시점을 기준으로 시간 초기화
        
        log_event("INFO", f"📈 [정식 편입] {ticker} | {additional_quantity}주 추가 매수 | 총 {stock.quantity}주 | 새 평단가: {stock.avg_price:,.0f}원")

    def remove_stock(self, ticker: str, sell_price: float, quantity: int, reason: str) -> str:
        """보유/정찰 종목을 포트폴리오에서 제거 (매도) 합니다."""
        if ticker not in self.holdings:
            log_event("WARNING", f"보유하지 않은 종목({ticker})을 매도할 수 없습니다.")
            return "오류: 보유하지 않은 종목"

        stock = self.holdings[ticker]
        if quantity > stock.quantity:
            log_event("WARNING", f"매도 수량({quantity})이 보유 수량({stock.quantity})보다 많습니다.")
            quantity = stock.quantity # 보유 수량만큼만 매도

        proceeds = sell_price * quantity
        self.cash += proceeds
        
        profit = (sell_price - stock.avg_price) * quantity
        profit_pct = ((sell_price / stock.avg_price) - 1) * 100
        
        # 거래 기록
        self._log_trade(stock, sell_price, quantity, profit, reason)
        
        stock.quantity -= quantity
        if stock.quantity == 0:
            del self.holdings[ticker]
            log_event("INFO", f"[전량 매도] {ticker} | 잔여 현금: {self.cash:,.0f}원")
        else:
            log_event("INFO", f"[부분 매도] {ticker} | 잔여 수량: {stock.quantity}주")
            
        return f"실현 손익: {profit:,.0f}원 ({profit_pct:.2f}%)"
    
    def _log_trade(self, stock: Stock, sell_price: float, quantity: int, profit: float, reason: str):
        """거래 내역을 기록합니다."""
        log = {
            "timestamp": datetime.now().isoformat(),
            "ticker": stock.ticker,
            "side": "SELL",
            "buy_price": stock.avg_price,
            "sell_price": sell_price,
            "quantity": quantity,
            "profit": profit,
            "reason": reason
        }
        self.trade_log.append(log)
        # 여기에 나중에 파일 저장 또는 DB 저장 로직 추가 가능
        log_event("INFO", f"[거래 기록] {log}")

    def get_stocks_by_status(self, status: StockStatus) -> List[Stock]:
        """특정 상태의 종목 리스트를 반환합니다."""
        return [stock for stock in self.holdings.values() if stock.status == status]

    def get_total_stock_count(self) -> int:
        """현재 포트폴리오에 있는 총 종목 수를 반환합니다 (정찰+보유)."""
        return len(self.holdings)

    def get_portfolio_summary(self) -> str:
        """현재 포트폴리오의 요약 정보를 문자열로 반환합니다."""
        total_asset_value = self.cash
        holding_details = []
        for stock in self.holdings.values():
            total_asset_value += stock.total_investment # 평가액 대신 투자원금 기준
            detail = (f"  - [{stock.status.name}] {stock.ticker}: {stock.quantity}주 @ {stock.avg_price:,.0f}원")
            holding_details.append(detail)
            
        summary = (
            f"총 자산: {total_asset_value:,.0f}원 (현금: {self.cash:,.0f}원)\n"
            f"보유 종목 ({len(self.holdings)}개):\n" +
            ("\n".join(holding_details) if holding_details else "  - 없음")
        )
        return summary 

    def add_long_term_stock(self):
        # 이미 보유 중이면 패스
        if self.long_term_stock in self.holdings:
            return
        price = self.kis_api.get_current_price(self.long_term_stock)
        quantity = int(self.long_term_amount // price)
        if quantity > 0:
            self.cash -= price * quantity
            stock = Stock(
                ticker=self.long_term_stock,
                quantity=quantity,
                avg_price=price,
                status=StockStatus.HOLDING,
                purchase_time=datetime.now()
            )
            self.holdings[self.long_term_stock] = stock
            log_event("INFO", f"🟢 [장기투자] {self.long_term_stock} {quantity}주 매수") 