# strategy/short_term_strategy.py
# 단기 투자 전략을 정의하고 실행합니다.

import config
from data.fetcher import fetch_realtime_ohlcv
from strategy.gemini_signal import get_gemini_trading_signal
from utils.logger import log_event

class ShortTermStrategy:
    """
    단기 투자 전략을 관리하고 실행하는 클래스.
    - 보유 종목 관리 (손절, 트레일링 스탑)
    - 신규 투자 대상 탐색 및 매수 결정
    """
    def __init__(self, portfolio_manager, trader, candidates):
        self.pm = portfolio_manager
        self.trader = trader
        self.candidates = candidates

    def execute(self):
        """단기 투자 전략의 모든 로직을 실행합니다."""
        self._manage_holdings()
        self._find_new_investments()

    def _manage_holdings(self):
        """보유 중인 단기 투자 종목을 관리합니다."""
        short_term_holdings = self.pm.get_holdings_by_strategy('short_term')
        for ticker, data in short_term_holdings.items():
            current_price = self.trader.kis_api.get_current_price(ticker)
            if not current_price:
                log_event("WARNING", f"[단기 전략] {ticker}의 현재가 조회 실패. 건너뜁니다.")
                continue

            self._apply_selling_rules(ticker, data, current_price)
    
    def _apply_selling_rules(self, ticker, data, current_price):
        """단기 투자 종목에 대한 매도 규칙(손절, 트레일링 스탑)을 적용합니다."""
        data['highest_price_since_buy'] = max(data.get('highest_price_since_buy', 0), current_price)
        avg_price = data['avg_price']

        # 1. 손절 규칙
        stop_loss_price = avg_price * (1 + config.SHORT_TERM_STOP_LOSS_RATIO)
        if current_price <= stop_loss_price:
            reason = f"단기 손절 (매수가: {avg_price:,.0f}, 현재가: {current_price:,.0f})"
            self.trader._execute_sell(ticker, data['quantity'], current_price, reason)
            return True # 매도 실행 시 추가 규칙 확인 중단

        # 2. 트레일링 스탑 규칙
        profit_target_price = avg_price * (1 + config.SHORT_TERM_PROFIT_TARGET_FOR_TRAILING_STOP)
        if data['highest_price_since_buy'] >= profit_target_price:
            trailing_stop_price = data['highest_price_since_buy'] * (1 - config.SHORT_TERM_TRAILING_STOP_RATIO)
            if current_price <= trailing_stop_price:
                reason = f"단기 트레일링 스탑 (최고가: {data['highest_price_since_buy']:,.0f}, 현재가: {current_price:,.0f})"
                self.trader._execute_sell(ticker, data['quantity'], current_price, reason)
                return True
        return False

    def _find_new_investments(self):
        """신규 단기 투자 대상을 찾고 AI 분석 후 매수합니다."""
        if self.pm.get_holdings_count('short_term') >= self.pm.short_term_slots:
            return

        log_event("INFO", "[단기 전략] 사전 준비된 후보군으로 신규 종목 스캔...")
        
        for ticker in self.candidates:
            if self.pm.get_holdings_count('short_term') >= self.pm.short_term_slots:
                break
            if ticker in self.pm.holdings or ticker in self.pm.analysis_failed_tickers:
                continue

            log_event("INFO", f"[단기 전략] '{ticker}' 종목 AI 분석 요청...")
            try:
                if self._evaluate_and_buy(ticker):
                    # 매수 성공 시 다음 사이클까지 대기
                    break 
            except Exception as e:
                log_event("ERROR", f"[{ticker}] 단기 투자 AI 분석 중 오류 발생: {e}")
                self.pm.analysis_failed_tickers.add(ticker)

    def _evaluate_and_buy(self, ticker):
        """종목을 평가하고 조건에 맞으면 매수합니다."""
        yf_ticker = f"{ticker}.KS"
        # 5분봉 데이터로 최근 5일치 분석
        df = fetch_realtime_ohlcv(yf_ticker, period="5d", interval="5m") 
        
        if df is None or df.empty:
            log_event("WARNING", f"[{ticker}] AI 분석을 위한 데이터 수집 실패. 건너뜁니다.")
            self.pm.analysis_failed_tickers.add(ticker)
            return False

        signal_data = get_gemini_trading_signal(df, ticker)
        
        if signal_data and signal_data.get('signal') == '매수':
            price = df['Close'].iloc[-1]
            quantity = int(self.pm.investment_per_stock // price)
            if quantity > 0:
                return self.trader._execute_buy(ticker, quantity, price, 'short_term', signal_data.get('reason'))
        else:
            reason = signal_data.get('reason', '신호 없음')
            log_event("INFO", f"[단기 전략] AI 분석 결과 '보류'. 종목: {ticker}, 이유: {reason}")
        
        return False 