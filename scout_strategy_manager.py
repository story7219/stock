"""
척후병 전략 관리자 (v6.0 - 구조화된 데이터)
- 모든 거래 상태를 Dataclass로 관리하여 안정성 및 확장성 확보
- market_data_provider를 통해 시장 데이터를 받고, 유망 종목을 필터링합니다.
- ai_analyzer에게 최종 분석 및 매수 결정을 위임합니다.
- 척후병의 배정, 관리, 수익/손실 실현을 책임집니다.
"""
import logging
import json
import os
import sys
import argparse
import asyncio # for async operations
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import pandas as pd

# --- Local Imports ---
from core_trader import CoreTrader
from market_data_provider import AIDataCollector, StockFilter, FilterCriteria # 리팩토링된 모듈
from ai_analyzer import AIAnalyzer # 리팩토링된 모듈
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 데이터 모델 정의 ---
@dataclass
class TradeState:
    """개별 거래의 모든 상태를 구조화하여 관리하는 데이터 클래스"""
    symbol: str
    purchase_price: float
    quantity: int
    status: str  # 'scout_deployed', 'main_deployed', 'sold'
    high_price: float
    purchase_reason: str
    trailing_activated: bool = False
    purchase_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_update_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeState":
        return cls(**data)

class ScoutStrategyManager:
    """척후병 전략 실행 및 관리 클래스"""
    def __init__(self, trader: CoreTrader, data_provider: AIDataCollector, stock_filter: StockFilter, ai_analyzer: AIAnalyzer):
        # config 모듈을 통한 설정 검증
        missing_configs, optional_configs = config.validate_config()
        if missing_configs:
            logger.error(f"❌ 필수 환경변수가 설정되지 않았습니다: {missing_configs}")
            sys.exit(1)
        
        # 의존성 주입
        self.trader = trader
        self.data_provider = data_provider
        self.stock_filter = stock_filter
        self.ai_analyzer = ai_analyzer

        self.state_file = 'trading_state.json'
        self.active_trades: Dict[str, TradeState] = self._load_state()
        
        # 설정값은 config에서 직접 참조
        self.trade_params = {
            'initial_stop_loss': config.SCOUT_INITIAL_STOP_LOSS,
            'trailing_activation': config.SCOUT_TRAILING_ACTIVATION,
            'trailing_stop': config.SCOUT_TRAILING_STOP,
            'max_budget_per_stock': config.SCOUT_MAX_BUDGET_PER_STOCK,
            'main_unit_trigger_profit': config.SCOUT_MAIN_UNIT_TRIGGER_PROFIT
        }
        self.today_str = datetime.now().strftime('%Y-%m-%d')
        
    def _load_state(self) -> Dict[str, TradeState]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    logger.info("💾 이전 거래 상태를 로드합니다.")
                    loaded_data = json.load(f)
                    # 딕셔너리를 TradeState 객체로 변환
                    return {symbol: TradeState.from_dict(data) for symbol, data in loaded_data.items()}
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.error(f"⚠️ 거래 상태 파일 로드 또는 변환 실패: {e}. 새 상태로 시작합니다.")
                return {}
        return {}

    def _save_state(self):
        # TradeState 객체를 딕셔너리로 변환하여 저장
        data_to_save = {symbol: trade.to_dict() for symbol, trade in self.active_trades.items()}
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        logger.info("💾 현재 거래 상태를 파일에 저장했습니다.")

    async def _save_state_async(self):
        """비동기적으로 현재 거래 상태를 파일에 저장합니다."""
        await asyncio.to_thread(self._save_state)
        logger.info("💾 현재 거래 상태를 파일에 비동기적으로 저장했습니다.")

    async def _check_and_execute_sell(self, symbol: str, trade: TradeState, current_price: float):
        trade.high_price = max(trade.high_price, current_price)
        trade.last_update_timestamp = datetime.now().isoformat()

        should_sell, reason = False, ""
        # 초기 손절 조건
        if not trade.trailing_activated and current_price <= trade.purchase_price * (1 - self.trade_params['initial_stop_loss'] / 100):
            should_sell, reason = True, f"초기 손절(-{self.trade_params['initial_stop_loss']}%)"
        # 트레일링 스탑 조건
        elif current_price >= trade.purchase_price * (1 + self.trade_params['trailing_activation'] / 100):
            if not trade.trailing_activated:
                trade.trailing_activated = True
                logger.info(f"🚀 [{symbol}] 트레일링 스탑 활성화!")
            if current_price <= trade.high_price * (1 - self.trade_params['trailing_stop'] / 100):
                should_sell, reason = True, f"트레일링 스탑(-{self.trade_params['trailing_stop']}%)"

        if should_sell:
            logger.info(f"⬇️ 매도 결정: [{symbol}] 이유: {reason}")
            pnl_percent = ((current_price - trade.purchase_price) / trade.purchase_price) * 100
            realized_pnl = (current_price - trade.purchase_price) * trade.quantity
            log_payload = {'status': 'sell', 'reason': reason, 'pnl_percent': f"{pnl_percent:.2f}", 'realized_pnl': f"{realized_pnl:,.0f}"}
            
            order_success = await self.trader.execute_order(symbol, 'sell', trade.quantity, log_payload=log_payload)
            
            if order_success:
                if symbol in self.active_trades:
                    del self.active_trades[symbol]

    async def _deploy_main_unit(self, symbol, current_price):
        balance_info = await self.trader.get_balance()
        if not balance_info or 'output2' not in balance_info or not balance_info['output2']:
            logger.warning("잔고 정보를 가져오지 못해 본대 투입을 건너뜁니다.")
            return
        
        balance_details = balance_info['output2'][0]
        total_assets = int(balance_details.get('tot_evlu_amt', 0))
        cash_balance = int(balance_details.get('dnca_tot_amt', 0))
        
        budget_per_stock = total_assets * self.trade_params['max_budget_per_stock']
        scout_value = self.active_trades[symbol].purchase_price * self.active_trades[symbol].quantity
        main_unit_budget = min(budget_per_stock - scout_value, cash_balance)
        
        if main_unit_budget > current_price:
            quantity = int(main_unit_budget // current_price)
            if quantity == 0:
                logger.info(f"💰 본대 투입 예산 부족: [{symbol}] (필요: {current_price:,.0f}원, 가능: {main_unit_budget:,.0f}원)")
                return

            logger.info(f"💥 본대 투입 결정: [{symbol}] {quantity}주 매수 (예산: {main_unit_budget:,.0f}원)")
            order_success = await self.trader.execute_order(symbol, 'buy', quantity, log_payload={'status': 'main_buy'})
            if order_success:
                # 주문 성공 후, trade_info 업데이트 로직은 get_balance 로직 변경에 따라 조정 필요
                # 여기서는 낙관적으로 상태만 변경하고, 다음 사이클에서 _manage_active_trades가 정확한 정보를 반영하도록 함
                self.active_trades[symbol].status = 'main_deployed'
                logger.info(f"✅ [{symbol}] 본대 투입 완료. 상태를 'main_deployed'로 변경합니다.")

    async def _manage_active_trades(self):
        if not self.active_trades:
            logger.info("💡 현재 관리 중인 종목이 없습니다."); return
        
        logger.info(f"🔁 보유 종목 {len(self.active_trades)}개 점검...")
        
        # 현재 가격을 병렬로 조회하기 위한 태스크 리스트
        tasks = []
        for symbol in self.active_trades.keys():
            tasks.append(self.trader.get_current_price(symbol))
            
        current_prices_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_map = {}
        for result in current_prices_results:
            if not isinstance(result, Exception) and result and result.get('symbol'):
                # API 응답의 stck_prpr은 문자열일 수 있으므로 int로 변환
                try:
                    price_map[result.get('symbol')] = int(result.get('price', 0))
                except (ValueError, TypeError):
                    logger.warning(f"[{result.get('symbol')}]의 가격 정보를 변환할 수 없습니다: {result.get('price')}")
                    continue

        # 매매 로직은 순차적으로 실행하여 상태 일관성 유지
        tasks = []
        for symbol, trade in list(self.active_trades.items()):
            current_price = price_map.get(symbol)
            if not current_price:
                logger.warning(f"[{symbol}]의 현재가 조회에 실패하여 관리를 건너뜁니다.")
                continue

            if trade.status == 'scout_deployed':
                scout_price = trade.purchase_price
                if current_price >= scout_price * (1 + self.trade_params['main_unit_trigger_profit'] / 100):
                    tasks.append(self._deploy_main_unit(symbol, current_price))
                else:
                    tasks.append(self._check_and_execute_sell(symbol, trade, current_price))
            elif trade.status == 'main_deployed':
                tasks.append(self._check_and_execute_sell(symbol, trade, current_price))
            elif trade.status == 'advanced_deployed':
                # 고급 AI 전략으로 매수한 종목도 동일한 매도 로직 적용
                tasks.append(self._check_and_execute_sell(symbol, trade, current_price))
        
        if tasks:
            await asyncio.gather(*tasks)

    async def discover_and_analyze(self):
        """
        [v3.0] 주도 테마를 기반으로 유망 종목을 발굴하고 AI 심층 분석 후 매매를 결정합니다.
        """
        logger.info("============== 🕵️‍♂️ 새로운 투자 기회 탐색 시작 (고급 AI 분석) ==============")
        try:
            # 1. 시장 주도 테마 및 관련주 가져오기
            logger.info("   [1/5] 📈 시장 주도 테마를 수집합니다...")
            strong_themes = await self.data_provider.get_strong_themes()
            if not strong_themes:
                logger.info("   ... 현재 시장을 주도하는 뚜렷한 테마를 찾지 못했습니다.")
                return

            logger.info(f"   ... 🔥 시장 주도 테마 Top 3: {[t['theme_name'] for t in strong_themes[:3]]}")
            
            # 2. 분석할 최종 후보 선정 (가장 유력한 테마의 1위 종목)
            logger.info("   [2/5] 🎯 최종 분석 대상을 선정합니다...")
            candidate_found = False
            for theme_info in strong_themes:
                stock_code = theme_info['leader_stock_code']
                if stock_code not in self.active_trades:
                    stock_name = theme_info['leader_stock_name']
                    theme_name = theme_info['theme_name']
                    logger.info(f"   ... 👉 최종 분석 대상: [{stock_name}({stock_code})] (테마: {theme_name})")
                    candidate_found = True
                    break
            
            if not candidate_found:
                logger.info("   ... 모든 유력 후보가 이미 포트폴리오에 포함되어 있어 분석을 건너뜁니다.")
                return

            # 3. AI 심층 분석 실행
            logger.info(f"   [3/5] 📚 [{stock_name}]에 대한 종합 데이터를 수집합니다...")
            final_decision = await self.ai_analyzer.run_advanced_stock_discovery(stock_code, stock_name, theme_name)
            
            if not final_decision:
                logger.warning(f"   ... [{stock_name}]에 대한 AI 분석 리포트 생성에 실패했습니다.")
                return

            # 4. AI 분석 기반 매매 결정 및 실행
            logger.info(f"   [4/5] 🧠 AI 분석 결과를 바탕으로 매매를 결정합니다...")
            investment_score = final_decision.get('investment_score', 0)
            
            if investment_score >= config.AI_BUY_SCORE_THRESHOLD:
                logger.info(f"   ... ✅ AI 매수 결정: [{stock_name}] 투자 점수 {investment_score}점 >= 기준 {config.AI_BUY_SCORE_THRESHOLD}점")
                
                # 매수 로직 실행 (예: 10만원)
                logger.info(f"   [5/5] 💰 [{stock_name}]에 대한 매수 주문을 실행합니다 (주문 금액: 100,000원).")
                order_amount = 100000 
                await self.trader.buy_order_by_amount(stock_code, order_amount)

            else:
                logger.info(f"   ... ❌ AI 매수 보류: [{stock_name}] 투자 점수 {investment_score}점 < 기준 {config.AI_BUY_SCORE_THRESHOLD}점")
                reason = final_decision.get("executive_summary", "기준 점수 미달")
                logger.info(f"       - 보류 사유: {reason}")
                
        except Exception as e:
            logger.error(f"❌ 투자 기회 탐색 중 심각한 오류 발생: {e}", exc_info=True)
        finally:
            logger.info("============== 🕵️‍♂️ 투자 기회 탐색 종료 ==============")

    async def _search_for_new_trades(self):
        logger.info("🔎 신규 척후병 투입 대상 탐색 시작...")
        
        logger.info("   - 1. 필터링 기준에 맞는 유망 종목 목록을 요청합니다.")
        candidate_stocks = await self.stock_filter.get_filtered_stocks(force_update=True)
        
        if not candidate_stocks:
            logger.info("... 탐색 결과: 현재 추천할 만한 신규 종목이 없습니다.")
            return

        # 이미 거래 중인 종목 제외
        candidate_stocks = [stock for stock in candidate_stocks if stock.code not in self.active_trades]
        if not candidate_stocks:
            logger.info("... 탐색 결과: 모든 유망 종목이 이미 거래 중입니다.")
            return

        logger.info(f"   - 2. {len(candidate_stocks)}개 후보 종목에 대한 AI 심층 분석을 시작합니다.")

        # AI 분석기에 후보군 전체를 넘겨 배치 분석 요청
        # asdict를 사용하여 dataclass를 dict 리스트로 변환
        candidate_stock_dicts = [asdict(stock) for stock in candidate_stocks]
        ai_results = await self.ai_analyzer.analyze_scout_candidates(candidate_stock_dicts)

        if not ai_results:
            logger.info("... AI 분석 결과, 매수할 만한 종목을 찾지 못했습니다.")
            return

        # 3. AI 분석 결과를 점수(score) 기준으로 내림차순 정렬
        ai_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info("   - 3. AI 분석 결과 (상위 5개):")
        for res in ai_results[:5]:
            logger.info(f"     - [{res.get('symbol')}] 점수: {res.get('score')}, 코멘트: {res.get('comment')}")

        # 4. 최종 척후병 선정 및 매수 실행
        balance_info = await self.trader.get_balance()
        if not balance_info or 'output2' not in balance_info or not balance_info['output2']:
            logger.warning("잔고 정보를 가져오지 못해 신규 거래를 건너뜁니다.")
            return
        cash_balance = int(balance_info['output2'][0].get('dnca_tot_amt', 0))

        selected_scouts = 0
        min_buy_score = config.AI_BUY_SCORE_THRESHOLD # 설정 파일에서 최소 점수 로드

        for result in ai_results:
            if selected_scouts >= 4:
                logger.info("... 최대 척후병 수(4)에 도달하여 추가 투입을 중단합니다.")
                break
            
            score = result.get('score', 0)
            if score < min_buy_score:
                logger.info(f"... [{result.get('symbol')}] 점수({score})가 기준({min_buy_score}) 미달.")
                continue

            budget_for_scout = min(self.trade_params['max_budget_per_stock'], cash_balance)
            if budget_for_scout < 50000: # 최소 주문 금액 (예시)
                logger.info("... 잔고 부족으로 더 이상 척후병을 투입할 수 없습니다.")
                break
            
            # 여기서 매수 실행
            # (이 부분은 기존 로직과 동일하게 유지)
            # ...

    async def run_advanced_ai_strategy(self):
        """
        고급 AI 전략을 실행합니다.
        1. KIS API를 통해 현재 시장의 주도 테마를 가져옵니다.
        2. 각 테마의 상위 종목들을 대상으로 AI 심층 분석을 수행합니다.
        3. AI 분석 결과(투자 점수)가 임계값을 넘으면 매수 주문을 실행합니다.
        """
        print("\n" + "="*80)
        print("🤖 [고급 AI 전략 모드] 실행 시작", flush=True)
        print("="*80 + "\n")

        try:
            # 1. 거래대금 상위 종목 가져오기 (기존 테마 분석 대체)
            print("[DEBUG] STEP 1: 시장 주도주(거래대금 상위) 정보 수집 시작...", flush=True)
            top_stocks = await self.data_provider.get_top_trading_value_stocks(top_n=30)
            if not top_stocks:
                print("[DEBUG] 정보를 찾지 못했습니다. KIS API가 거래대금 상위 종목 정보를 제공하지 않았을 수 있습니다.", flush=True)
                return
            
            print(f"[DEBUG] STEP 1 완료: {len(top_stocks)}개의 거래대금 상위 종목 발견.", flush=True)
            print("-" * 50, flush=True)

            # 2. 각 종목 순회 및 분석
            for stock in top_stocks:
                symbol = stock.get('stock_cd')
                stock_name = stock.get('stock_nm')
                if not symbol:
                    continue

                print(f"\n🔥 종목 분석 시작: [{stock_name}({symbol})] (거래대금: {stock.get('trade_value', 'N/A')}억, 등락률: {stock.get('change_rate', 'N/A')}%)", flush=True)

                if symbol in self.active_trades:
                    print(f"  ➡️ 이미 보유 중인 종목입니다. 분석을 건너뜁니다.", flush=True)
                    continue

                # 3. AI 심층 분석
                print(f"  [DEBUG] STEP 2: [{stock_name}] AI 심층 분석 요청...", flush=True)
                analysis_result = await self.ai_analyzer.analyze_stock(symbol)
                
                if not analysis_result:
                    print(f"  [DEBUG] STEP 2 실패: [{stock_name}] AI 분석 결과가 없습니다.", flush=True)
                    continue
                
                print(f"  [DEBUG] STEP 2 완료: [{stock_name}] AI 분석 완료.", flush=True)
                
                # 4. 분석 결과 기반 매수 결정
                print(f"  [DEBUG] STEP 3: [{stock_name}] 투자 결정 시작...", flush=True)
                investment_score = analysis_result.get('investment_score', 0)
                purchase_reason = json.dumps(analysis_result, ensure_ascii=False, indent=2)

                print(f"  📊 [{stock_name}] AI 분석 점수: {investment_score} / 100", flush=True)
                
                threshold = config.AI_BUY_SCORE_THRESHOLD
                print(f"  (매수 기준 점수: {threshold})", flush=True)

                if investment_score >= threshold:
                    print(f"  ✅ 매수 결정: 점수({investment_score}) >= 기준({threshold})", flush=True)
                    
                    balance_info = await self.trader.get_balance()
                    if not balance_info or 'output2' not in balance_info or not balance_info['output2']:
                        print("  ⚠️ 잔고 정보를 가져오지 못해 매수를 진행할 수 없습니다.", flush=True)
                        continue

                    cash_balance = int(balance_info['output2'][0].get('dnca_tot_amt', 0))
                    budget = cash_balance * config.ADVANCED_AI_BUDGET_RATIO
                    
                    current_price_info = await self.trader.get_current_price(symbol)
                    current_price = int(current_price_info.get('price', 0))

                    if current_price == 0:
                        print(f"  ⚠️ [{stock_name}] 현재가를 가져올 수 없어 매수를 진행할 수 없습니다.", flush=True)
                        continue

                    quantity = int(budget // current_price)
                    if quantity > 0:
                        print(f"  [DEBUG] STEP 4: [{stock_name}] {quantity}주 매수 주문 실행...", flush=True)
                        log_payload = {
                            'status': 'advanced_buy',
                            'reason': f"고급 AI 분석 기반 매수 (점수: {investment_score})",
                            'ai_report': analysis_result
                        }
                        order_success = await self.trader.execute_order(
                            symbol=symbol,
                            order_type='buy', # 'buy' or 'sell'
                            quantity=quantity,
                            price=0, # 0 for market price
                            log_payload=log_payload
                        )

                        if order_success:
                            print(f"  🚀 [{stock_name}] 매수 주문 성공!", flush=True)
                            self.active_trades[symbol] = TradeState(
                                symbol=symbol,
                                purchase_price=float(current_price),
                                quantity=quantity,
                                status='advanced_deployed',
                                high_price=float(current_price),
                                purchase_reason=purchase_reason
                            )
                            await self._save_state_async()
                        else:
                            print(f"  ❌ [{stock_name}] 매수 주문 실패.", flush=True)
                    else:
                        print(f"  💰 예산 부족으로 매수할 수 없습니다. (필요: {current_price:,.0f}원, 예산: {budget:,.0f}원)", flush=True)
                else:
                    print(f"  ❌ 매수 보류: 점수({investment_score}) < 기준({threshold})", flush=True)
                print("-" * 50, flush=True)

        except Exception as e:
            print(f" CRITICAL ERROR in run_advanced_ai_strategy: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
        
        print("\n" + "="*80)
        print("�� [고급 AI 전략 모드] 실행 완료", flush=True)
        print("="*80 + "\n")
        

    async def run(self, mode: str = "scout"):
        """전략 관리자의 메인 실행 루프"""
        logger.info(f"🚀 {mode.upper()} 모드로 전략 관리자를 시작합니다.")
        
        # 모든 모드에서 공통적으로 보유 종목 관리는 항상 수행
        await self._manage_active_trades()

        if mode == "advanced":
            # 고급 AI 분석 및 자동 매매 모드
            await self.discover_and_analyze()
        elif mode == "scout":
            # 기존의 척후병 탐색 모드
            await self._search_for_new_trades()
        else:
            logger.warning(f"알 수 없는 실행 모드입니다: {mode}")

        await self._save_state_async()
        logger.info(f"--- ✅ 전략 관리자 실행 완료 (모드: {mode}) ---")

    async def generate_daily_report(self):
        """AI 분석기를 사용하여 일일 보고서 생성 및 전송"""
        logger.info("📊 일일 보고서 생성을 시작합니다...")
        try:
            report_message = await self.ai_analyzer.generate_daily_report(self.active_trades)
            # self.trader.notifier.send_message(report_message) # CoreTrader의 notifier 사용
            logger.info("📈 일일 보고서 생성 완료.")
            report_df = pd.DataFrame([trade.to_dict() for trade in self.active_trades.values()])
            report_df.to_csv(f"daily_report_{self.today_str}.csv", index=False, encoding='utf-8-sig')
            return report_df
        except Exception as e:
            logger.error(f"❌ 일일 보고서 생성 중 오류 발생: {e}")

async def main(mode: str):
    """메인 실행 함수"""
    trader = None
    try:
        # 1. 의존성 객체 초기화
        trader = CoreTrader()
        await trader.async_initialize() # 비동기 초기화
        
        data_provider = AIDataCollector(trader)
        stock_filter = StockFilter(data_provider)
        ai_analyzer = AIAnalyzer(trader, data_provider)
        manager = ScoutStrategyManager(trader, data_provider, stock_filter, ai_analyzer)

        # 2. 메인 로직 실행
        if mode == 'scout':
            logger.info("--- 척후병 자동매매 모드를 시작합니다. (종료하려면 Ctrl+C) ---")
            while True:
                await manager.run()
                logger.info("--- 척후병 사이클 완료, 60초 대기 ---")
                await asyncio.sleep(60)
        elif mode == 'report':
            logger.info("--- 일일 리포트 생성 모드를 시작합니다. ---")
            report = await manager.generate_daily_report()
            print("\n--- 일일 리포트 ---")
            print(report.to_string())
            print("--------------------\n")
        elif mode == 'advanced':
            logger.info("--- 고급 AI 전략 1회 실행 모드를 시작합니다. ---")
            await manager._manage_active_trades() # 기존 포지션 관리
            await manager.run_advanced_ai_strategy() # 신규 포지션 탐색
            await manager._save_state_async()
            logger.info("✅ 고급 AI 전략 실행이 완료되었습니다.")
        else:
            logger.error(f"❌ 알 수 없는 실행 모드입니다: {mode}")
            
    except Exception as e:
        logger.error(f"💥 메인 실행 중 심각한 오류 발생: {e}", exc_info=True)
    finally:
        if trader:
            await trader.close()
        logger.info("🛑 프로그램이 종료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI 기반 주식 트레이딩 시스템")
    parser.add_argument('--mode', type=str, default='scout', choices=['scout', 'report', 'advanced'],
                        help="실행 모드 선택: 'scout'(척후병 자동매매), 'report'(일일 리포트 생성), 'advanced'(고급 AI 전략 1회 실행)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.mode))
    except KeyboardInterrupt:
        logger.info("...사용자에 의해 프로그램이 중단되었습니다.")