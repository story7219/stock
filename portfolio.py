# portfolio.py
# 포트폴리오 상태를 관리하는 클래스 (리팩토링)

import pandas as pd
import config
from utils.logger import log_event
from trading.kis_api import KIS_API
from utils.telegram_bot import TelegramBot
from utils.gspread_client import gspread_client
import os
import csv
from enum import Enum, auto
from datetime import datetime

# --- 데이터 구조 정의 ---

class StockStatus(Enum):
    """보유 종목의 상태를 정의하는 열거형"""
    SCOUTING = auto()  # 척후병 파견 상태 (오디션 진행 중)
    HOLDING = auto()   # 정식 보유 상태 (본대 투입 완료)

class Stock:
    """개별 보유 주식의 정보를 관리하는 데이터 클래스"""
    def __init__(self, ticker: str, purchase_price: float, quantity: int, status: StockStatus):
        self.ticker = ticker
        self.purchase_price = purchase_price # 척후병 매수 시점의 단가
        self.avg_price = purchase_price      # 평균 매수 단가
        self.quantity = quantity
        self.status = status
        self.purchase_time = datetime.now()  # 객체 생성 시점(척후병 매수 시점)을 기록
        self.total_investment = purchase_price * quantity
        
    def __repr__(self):
        return (f"Stock(ticker={self.ticker}, status={self.status.name}, "
                f"avg_price={self.avg_price:,.0f}, quantity={self.quantity})")

    def add_investment(self, price: float, quantity: int):
        """본대 투입 시 정보를 업데이트 (평균단가, 총 투자금 등)"""
        new_investment = price * quantity
        self.total_investment += new_investment
        self.quantity += quantity
        self.avg_price = self.total_investment / self.quantity
        self.status = StockStatus.HOLDING # 상태를 정식 보유로 변경

# 로컬 로그파일 경로 설정
LOG_DIR = 'logs'
CSV_FILE_PATH = os.path.join(LOG_DIR, 'trade_log.csv')

class PortfolioManager:
    """포트폴리오 상태를 관리하고 기록하는 클래스 (리팩토링)"""
    def __init__(self, capital: float, kis_api: KIS_API, telegram_bot: TelegramBot):
        self.initial_capital = capital
        self.cash = capital
        self.holdings: dict[str, Stock] = {} # {티커: Stock 객체} 형태로 저장
        
        self.kis_api = kis_api
        self.telegram_bot = telegram_bot

    def _log_trade_to_csv(self, log_data: dict):
        """거래 내역을 로컬 CSV 파일에 기록합니다."""
        try:
            # 로그 디렉토리가 없으면 생성
            os.makedirs(LOG_DIR, exist_ok=True)
            
            # 파일이 존재하지 않으면 헤더를 먼저 씁니다.
            file_exists = os.path.isfile(CSV_FILE_PATH)
            
            # utf-8-sig 인코딩으로 Excel에서 한글 깨짐 방지
            with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
                
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerow(log_data)
            log_event("INFO", f"로컬 CSV 로그 저장 성공: {CSV_FILE_PATH}")
        except Exception as e:
            log_event("ERROR", f"로컬 CSV 로깅 실패: {e}")

    def _log_trade(self, ticker, action, quantity, price, strategy, reason=""):
        """거래 내역을 구글 스프레드시트와 로컬 CSV 파일에 이중으로 기록합니다."""
        log_data = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': strategy,
            'symbol': ticker,
            'action': action,
            'price': f"{price:,.0f}",  # 보기 좋게 쉼표 포맷 추가
            'quantity': quantity,
            'total_amount': f"{price * quantity:,.0f}", # 보기 좋게 쉼표 포맷 추가
            'reason': reason
        }
        
        # 1. 구글 스프레드시트에 기록
        try:
            # 구글 시트에는 숫자형 데이터로 저장하기 위해 포맷팅 안된 값으로 전달
            gspread_log_data = log_data.copy()
            gspread_log_data['price'] = price
            gspread_log_data['total_amount'] = price * quantity
            gspread_client.append_log(gspread_log_data, worksheet_name="trade_log")
        except Exception as e:
            log_event("ERROR", f"스프레드시트 로깅 실패: {e}")

        # 2. 로컬 CSV 파일에 기록
        self._log_trade_to_csv(log_data)

    def get_stocks_by_status(self, status: StockStatus) -> list[Stock]:
        """특정 상태에 있는 모든 주식 객체를 리스트로 반환합니다."""
        return [stock for stock in self.holdings.values() if stock.status == status]

    def add_scout(self, ticker: str, price: float, quantity: int):
        """척후병 매수 후 포트폴리오에 추가합니다."""
        if ticker in self.holdings:
            log_event("WARNING", f"[{ticker}] 이미 포트폴리오에 존재하는 종목이므로 척후병을 추가할 수 없습니다.")
            return

        scout_stock = Stock(ticker, price, quantity, StockStatus.SCOUTING)
        self.holdings[ticker] = scout_stock
        self.cash -= scout_stock.total_investment
        log_event("INFO", f"[척후병 파견] {scout_stock}")
        # self._log_trade(ticker, 'SCOUT_BUY', quantity, price, reason="Audition Start")

    def promote_stock(self, ticker: str, price: float, quantity: int):
        """척후병을 정식 보유 종목으로 승격시킵니다 (본대 투입)."""
        if ticker not in self.holdings or self.holdings[ticker].status != StockStatus.SCOUTING:
            log_event("WARNING", f"[{ticker}] 척후병 상태가 아니므로 본대를 투입할 수 없습니다.")
            return
            
        investment_cost = price * quantity
        self.cash -= investment_cost
        
        stock_to_promote = self.holdings[ticker]
        stock_to_promote.add_investment(price, quantity)
        
        log_event("INFO", f"[본대 투입] {stock_to_promote}")
        # self._log_trade(ticker, 'MAIN_BUY', quantity, price, reason="Audition Winner")

    def remove_stock(self, ticker: str, price: float, quantity: int, reason: str) -> str:
        """
        종목을 매도하고 포트폴리오에서 제거합니다.
        매도 후 수익 정보를 담은 문자열을 반환합니다.
        """
        if ticker not in self.holdings:
            log_event("WARNING", f"[{ticker}] 포트폴리오에 없는 종목을 매도 시도했습니다.")
            return "오류: 포트폴리오에 없는 종목"

        revenue = price * quantity
        self.cash += revenue
        
        # 실제 매도 수량과 보유 수량이 다를 수 있으므로, 전량 매도로 간주하고 제거
        removed_stock = self.holdings.pop(ticker)
        
        # 수익률 계산
        profit = revenue - removed_stock.total_investment
        profit_pct = (profit / removed_stock.total_investment) * 100 if removed_stock.total_investment > 0 else 0
        
        log_message = f"[매도 완료] {removed_stock.ticker} | 이유: {reason} | 수익: {profit:,.0f}원 ({profit_pct:.2f}%)"
        log_event("INFO", log_message)
        # self._log_trade(ticker, 'SELL', removed_stock.quantity, price, reason=reason)

        return f"수익: {profit:,.0f}원 ({profit_pct:.2f}%)"

    def get_total_stock_count(self) -> int:
        """포트폴리오에 있는 총 종목 수를 반환합니다 (척후병 포함)."""
        return len(self.holdings)

    def get_portfolio_summary(self) -> str:
        """현재 포트폴리오 요약 정보를 반환합니다."""
        scouting_count = len(self.get_stocks_by_status(StockStatus.SCOUTING))
        holding_count = len(self.get_stocks_by_status(StockStatus.HOLDING))
        
        summary = (
            f"현금: {self.cash:,.0f}원 | "
            f"정식보유: {holding_count}종목 | "
            f"척후병: {scouting_count}종목"
        )
        return summary 