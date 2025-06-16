# strategy/long_term_strategy.py
# 장기 투자 전략을 정의하고 실행합니다.

import config
from utils.logger import log_event

class LongTermStrategy:
    """
    장기 투자 전략을 관리하고 실행하는 클래스.
    - 보유 종목 관리 (분할 매수, 분할 매도)
    - 신규 투자 대상 탐색 및 매수 결정
    """
    def __init__(self, portfolio_manager, trader, candidates):
        self.pm = portfolio_manager
        self.trader = trader
        self.candidates = candidates

    def execute(self):
        """장기 투자 전략의 모든 로직을 실행합니다."""
        self._manage_holdings()
        self._find_new_investments()

    def _manage_holdings(self):
        """보유 중인 장기 투자 종목을 관리합니다."""
        long_term_holdings = self.pm.get_holdings_by_strategy('long_term')
        for ticker, data in long_term_holdings.items():
            current_price = self.trader.kis_api.get_current_price(ticker)
            if not current_price:
                log_event("WARNING", f"[장기 전략] {ticker}의 현재가 조회 실패. 건너뜁니다.")
                continue

            # 분할 매수 또는 분할 매도 로직 적용
            self._apply_buy_sell_rules(ticker, data, current_price)
    
    def _apply_buy_sell_rules(self, ticker, data, current_price):
        """장기 투자 종목에 대한 매수/매도 규칙을 적용합니다."""
        # 1. 분할 매수 규칙
        # 아직 목표 분할 매수 기간을 채우지 못했고, 추가 매수 여력이 있는 경우
        if data['buy_days'] < config.LONG_TERM_BUY_SPLIT_DAYS:
            # 하루 투자금액 계산
            daily_investment = self.pm.investment_per_stock / config.LONG_TERM_BUY_SPLIT_DAYS
            quantity_to_buy = int(daily_investment // current_price)
            if quantity_to_buy > 0:
                reason = f"장기 분할매수 ({data['buy_days'] + 1}일차)"
                self.trader._execute_buy(ticker, quantity_to_buy, current_price, 'long_term', reason)
                return # 분할매수를 실행했으면 해당 종목은 이번 사이클에서 추가 행동 없음

        # 2. 분할 매도 규칙 (수익 실현)
        profit_margin = (current_price / data['avg_price']) - 1
        if profit_margin >= config.LONG_TERM_SELL_PROFIT_TARGET:
            reason = f"장기 분할매도 (수익률: {profit_margin:.2%})"
            self.trader._execute_sell(ticker, config.LONG_TERM_SELL_SHARES_PER_DAY, current_price, reason)
            return

    def _find_new_investments(self):
        """신규 장기 투자 대상을 찾고 첫 매수를 실행합니다."""
        if self.pm.get_holdings_count('long_term') >= self.pm.long_term_slots:
            return

        log_event("INFO", "[장기 전략] 사전 준비된 후보군으로 신규 종목 탐색...")
        
        for ticker in self.candidates:
            if self.pm.get_holdings_count('long_term') >= self.pm.long_term_slots:
                break
            if ticker in self.pm.holdings:
                continue
            
            current_price = self.trader.kis_api.get_current_price(ticker)
            if current_price:
                daily_investment = self.pm.investment_per_stock / config.LONG_TERM_BUY_SPLIT_DAYS
                quantity = int(daily_investment // current_price)
                if quantity > 0:
                    self.trader._execute_buy(ticker, quantity, current_price, 'long_term', "신규 장기 투자 전략 편입")
                    # 신규 편입 시 하나만 처리하고 다음 사이클로 넘어감
                    break 