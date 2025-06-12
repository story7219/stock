# trading/trader.py
# '실시간 오디션'과 '듀얼 스탑' 전략에 기반한 매매 실행 및 관리를 총괄하는 클래스 (AI 분석 기능 추가)

import asyncio
from datetime import datetime, timedelta, time
import config
from data.fetcher import MarketPrism
from portfolio import PortfolioManager, StockStatus, Stock
from trading.kis_api import KIS_API
from utils.logger import log_event
from utils.telegram_bot import TelegramBot
from utils.chart_generator import generate_stock_chart
from analysis.gemini_analyzer import analyze_chart_with_gemini
from typing import List, Dict

class Trader:
    """
    (리팩토링) '실시간 오디션' 및 '듀얼 스탑' 전략에 따라 거래를 실행하고 관리합니다.
    장 초반에는 Gemini AI 분석을 통해 후보를 선별하는 지능형 로직을 포함합니다.
    """
    def __init__(self, portfolio_manager: PortfolioManager, kis_api: KIS_API, telegram_bot: TelegramBot):
        self.pm = portfolio_manager
        self.kis_api = kis_api
        self.telegram_bot = telegram_bot
        self.market_prism = MarketPrism(kis_api)
        # 설정에서 시간 문자열을 time 객체로 변환
        self.gemini_start_time = datetime.strptime(config.GEMINI_ANALYSIS_START_TIME, '%H:%M').time()
        self.gemini_end_time = datetime.strptime(config.GEMINI_ANALYSIS_END_TIME, '%H:%M').time()

    async def run_trading_cycle(self):
        """(비동기) 한 번의 거래 사이클을 실행합니다."""
        log_event("INFO", "=== 비동기 거래 사이클 시작 ===")
        
        # 1. 포트폴리오 내역 정리 (청산 및 오디션 심사)
        await self._manage_portfolio()
        
        # 2. 포트폴리오에 여유가 생겼으면 신규 오디션 시작
        await self._start_new_audition()
        
        log_event("INFO", "=== 비동기 거래 사이클 종료 ===")

    async def _manage_portfolio(self):
        """보유/정찰 중인 종목에 대한 청산 및 오디션 심사를 수행합니다."""
        scouting_stocks = self.pm.get_stocks_by_status(StockStatus.SCOUTING)
        holding_stocks = self.pm.get_stocks_by_status(StockStatus.HOLDING)
        
        all_stocks_to_check = scouting_stocks + holding_stocks
        if not all_stocks_to_check:
            log_event("INFO", "관리할 보유/정찰 종목이 없습니다.")
            return

        tickers_to_fetch = [stock.ticker for stock in all_stocks_to_check]
        price_data = await self.kis_api.fetch_prices_in_parallel(tickers_to_fetch)
        
        # 청산 및 오디션 심사 동시 진행
        exit_tasks = self._apply_exit_strategies(holding_stocks, price_data)
        audition_tasks = self._process_audition_results(scouting_stocks, price_data)
        await asyncio.gather(exit_tasks, audition_tasks)

    async def _apply_exit_strategies(self, holding_stocks: List, price_data: dict):
        """(비동기) '듀얼 스탑' 청산 전략을 적용합니다."""
        if not holding_stocks: return
        log_event("INFO", f"[청산 점검] 정식 보유 {len(holding_stocks)}개 종목 점검.")
        
        tasks = []
        for stock in holding_stocks:
            current_price = price_data.get(stock.ticker)
            if not current_price: continue

            time_limit = stock.purchase_time + timedelta(minutes=config.EXIT_STRATEGY_TIME_LIMIT_MINUTES)
            if datetime.now() >= time_limit:
                reason = f"시간 초과 ({config.EXIT_STRATEGY_TIME_LIMIT_MINUTES}분)"
                tasks.append(self._execute_sell(stock, current_price, reason))
                continue

            stop_loss_price = stock.avg_price * (1 + config.EXIT_STRATEGY_STOP_LOSS_PCT / 100)
            if current_price <= stop_loss_price:
                reason = f"손절매 ({config.EXIT_STRATEGY_STOP_LOSS_PCT}%) 도달"
                tasks.append(self._execute_sell(stock, current_price, reason))
        
        if tasks: await asyncio.gather(*tasks)

    async def _process_audition_results(self, scouting_stocks: List, price_data: dict):
        """(비동기) 척후병들의 성과를 평가하고 승자와 패자를 결정합니다."""
        if not scouting_stocks: return
        log_event("INFO", f"[오디션 심사] 척후병 {len(scouting_stocks)}개 종목 심사.")
        
        winners, losers = [], []
        for stock in scouting_stocks:
            current_price = price_data.get(stock.ticker)
            if not current_price: continue
            
            profit_pct = ((current_price / stock.purchase_price) - 1) * 100
            if profit_pct >= config.AUDITION_WINNER_MIN_PROFIT_PCT:
                winners.append({'stock': stock, 'price': current_price, 'profit_pct': profit_pct})
            else:
                losers.append({'stock': stock, 'price': current_price})

        sell_tasks = [self._execute_sell(item['stock'], item['price'], "오디션 탈락") for item in losers]
        
        buy_tasks = []
        if winners:
            winners.sort(key=lambda x: x['profit_pct'], reverse=True)
            
            available_slots = config.MAX_STOCKS_TO_HOLD - len(self.pm.get_stocks_by_status(StockStatus.HOLDING))
            final_winners = winners[:available_slots]
            
            if final_winners:
                investment_per_stock = self.pm.initial_capital / config.MAX_STOCKS_TO_HOLD
                for item in final_winners:
                    stock, price = item['stock'], item['price']
                    remaining_investment = investment_per_stock - stock.total_investment
                    quantity = int(remaining_investment // price)
                    if quantity > 0:
                        buy_tasks.append(self._execute_buy(stock.ticker, price, quantity, promote=True))

        await asyncio.gather(*sell_tasks, *buy_tasks)

    def _is_gemini_analysis_time(self) -> bool:
        """현재 시간이 Gemini AI 분석을 수행할 시간대인지 확인합니다."""
        if not config.USE_GEMINI_ANALYSIS:
            return False
        now_time = datetime.now().time()
        return self.gemini_start_time <= now_time <= self.gemini_end_time

    async def _start_new_audition(self):
        """(비동기) 포트폴리오에 여유가 있을 경우, 새로운 오디션을 시작합니다."""
        if self.pm.get_total_stock_count() >= config.MAX_STOCKS_TO_HOLD:
            return # 포트폴리오 꽉 찼으면 신규 진입 안함

        log_event("INFO", "[신규 오디션] 포트폴리오에 여유 공간 확인. 오디션을 시작합니다.")
        
        try:
            candidates = self.market_prism.find_market_prism_candidates(top_n=config.AUDITION_CANDIDATE_COUNT * 2) # 후보군 넉넉하게
            if not candidates:
                log_event("WARNING", "[신규 오디션] 마켓 프리즘 후보 종목이 없습니다.")
                return

            # AI 분석 시간대인 경우, 후보를 AI로 필터링
            if self._is_gemini_analysis_time():
                final_candidates = await self._filter_candidates_with_gemini(candidates)
            else:
                final_candidates = candidates[:config.AUDITION_CANTIDATE_COUNT] # AI 시간 아니면 그냥 상위 종목 선택

            if not final_candidates:
                log_event("INFO", "[신규 오디션] 최종 후보 종목이 없어 척후병을 파견하지 않습니다.")
                return

            # 최종 후보들에게 척후병 파견
            prices = await self.kis_api.fetch_prices_in_parallel(final_candidates)
            buy_tasks = []
            for ticker in final_candidates:
                price = prices.get(ticker)
                # 이미 포폴에 있거나, 정찰병 보낼 자리가 없으면 중단
                if ticker in self.pm.holdings or self.pm.get_total_stock_count() >= config.MAX_STOCKS_TO_HOLD:
                    continue
                if price:
                    buy_tasks.append(self._execute_buy(ticker, price, config.AUDITION_SCOUT_BUY_AMOUNT, promote=False))
            
            if buy_tasks:
                await asyncio.gather(*buy_tasks)

        except Exception as e:
            log_event("ERROR", f"[신규 오디션] 후보 탐색 또는 척후병 파견 중 오류: {e}")

    async def _filter_candidates_with_gemini(self, candidates: List[str]) -> List[str]:
        """Gemini AI로 후보군을 필터링합니다."""
        log_event("INFO", f"🤖 [Gemini 분석 시간] 후보 {len(candidates)}종목에 대한 AI 분석 시작.")
        self.telegram_bot.send_message(f"🤖 AI 분석 시작 (후보: {len(candidates)}개)")
        
        approved_candidates = []
        for ticker in candidates:
            chart_path = generate_stock_chart(ticker)
            if not chart_path: continue

            analysis_result = analyze_chart_with_gemini(chart_path)
            if not analysis_result: continue
            
            decision = analysis_result.get("decision")
            reason = analysis_result.get("reason")
            
            msg = f"[{ticker}] AI 분석 결과: *{decision}*\n> {reason}"
            self.telegram_bot.send_message(msg)
            
            if decision == "BUY":
                approved_candidates.append(ticker)

        log_event("SUCCESS", f"AI 분석 결과, 최종 후보: {approved_candidates}")
        self.telegram_bot.send_message(f"✅ AI 분석 완료. 최종 후보: {approved_candidates}")
        return approved_candidates

    async def _execute_buy(self, ticker: str, price: float, quantity: int, promote: bool = False):
        """(비동기) 매수 주문 및 포트폴리오 업데이트"""
        if self.pm.cash < quantity * price:
            log_event("WARNING", f"자본 부족으로 매수 불가: {ticker}")
            return

        order_result = self.kis_api.create_order('buy', ticker, quantity)
        if order_result and order_result.get('order_id'):
            if promote:
                self.pm.promote_stock(ticker, price, quantity)
                msg = f"📈 [본대 투입] {ticker} | {quantity}주 | {price:,.0f}원"
            else:
                self.pm.add_scout(ticker, price, quantity)
                msg = f"🕵️ [척후병 파견] {ticker} | {quantity}주 | {price:,.0f}원"
            self.telegram_bot.send_message(msg)
            log_event("SUCCESS", msg)
        else:
            log_event("ERROR", f"[{ticker}] 매수 주문 실패. API 응답: {order_result}")

    async def _execute_sell(self, stock: Stock, price: float, reason: str):
        """(비동기) 매도 주문 및 포트폴리오 업데이트"""
        order_result = self.kis_api.create_order('sell', stock.ticker, stock.quantity)
        if order_result and order_result.get('order_id'):
            profit_info = self.pm.remove_stock(stock.ticker, price, stock.quantity, reason)
            msg = (f"💰 [매도] {stock.ticker}\n"
                   f"- 이유: {reason}\n"
                   f"- 매도 단가: {price:,.0f}원\n"
                   f"- {profit_info}")
            self.telegram_bot.send_message(msg)
            log_event("SUCCESS", f"[매도 성공] {stock.ticker} | 이유: {reason}")
        else:
            log_event("ERROR", f"[{stock.ticker}] 매도 주문 실패. API 응답: {order_result}") 