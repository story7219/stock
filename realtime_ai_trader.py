"""
실시간 AI 트레이딩 시스템 (v4.0 - GitHub Actions 최적화)
- 환경 변수에서 설정을 로드하고, --action 플래그로 실행 모드 제어.
- 거래, 분석, 리포팅 로직을 모두 포함하는 통합 컨트롤 타워.
"""
import logging
import json
import os
import sys
import argparse
import google.generativeai as genai
from datetime import datetime
from core_trader import CoreTrader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimeAITrader:
    def __init__(self):
        self.trader = CoreTrader()
        self.state_file = 'trading_state.json'
        self.active_trades = self._load_state()
        self.trade_params = {
            'initial_stop_loss': 4.0, 'trailing_activation': 6.0, 'trailing_stop': 3.0,
            'max_budget_per_stock': 0.05, 'main_unit_trigger_profit': 2.5
        }
        self.ai_model = self._initialize_ai_model()
        self.today_str = datetime.now().strftime('%Y-%m-%d')
        
    def _initialize_ai_model(self):
        try:
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logger.warning("⚠️ GEMINI_API_KEY 환경변수가 없어 AI 기능이 비활성화됩니다.")
                return None
            genai.configure(api_key=api_key)
            logger.info("✅ Gemini AI 모델이 성공적으로 설정되었습니다.")
            return genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            logger.error(f"❌ Gemini AI 설정 실패: {e}"); return None

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                logger.info("💾 이전 거래 상태를 로드합니다.")
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.active_trades, f, indent=4)
        logger.info("💾 현재 거래 상태를 파일에 저장했습니다.")

    def _check_and_execute_sell(self, symbol, trade_info, current_price):
        purchase_price = trade_info['purchase_price']
        trade_info['high_price'] = max(trade_info.get('high_price', purchase_price), current_price)
        
        should_sell, reason = False, ""
        # 1. 초기 손절
        if not trade_info.get('trailing_activated', False) and current_price <= purchase_price * (1 - self.trade_params['initial_stop_loss'] / 100):
            should_sell, reason = True, f"초기 손절(-{self.trade_params['initial_stop_loss']}%)"
        # 2. 트레일링 스탑
        elif current_price >= purchase_price * (1 + self.trade_params['trailing_activation'] / 100):
            if not trade_info.get('trailing_activated', False):
                trade_info['trailing_activated'] = True
                logger.info(f"🚀 [{symbol}] 트레일링 스탑 활성화!")
            if current_price <= trade_info['high_price'] * (1 - self.trade_params['trailing_stop'] / 100):
                should_sell, reason = True, f"트레일링 스탑(-{self.trade_params['trailing_stop']}%)"

        if should_sell:
            logger.info(f"⬇️ 매도 결정: [{symbol}] 이유: {reason}")
            pnl = ((current_price - purchase_price) / purchase_price) * 100
            if self.trader.execute_order(symbol, 'sell', trade_info['quantity'], log_payload={'status': reason, 'pnl_percent': f"{pnl:.2f}%"}):
                del self.active_trades[symbol]
        
    def _deploy_main_unit(self, symbol, current_price):
        balance = self.trader.get_balance('all')
        if not balance: return
        total_assets = int(balance['output2'][0]['tot_evlu_amt'])
        cash_balance = int(balance['output2'][0]['dnca_tot_amt'])
        
        budget_per_stock = total_assets * self.trade_params['max_budget_per_stock']
        scout_value = self.active_trades[symbol]['scout_info']['price'] * self.active_trades[symbol]['scout_info']['quantity']
        main_unit_budget = min(budget_per_stock - scout_value, cash_balance)
        
        if main_unit_budget > current_price:
            quantity = int(main_unit_budget // current_price)
            logger.info(f"💥 본대 투입 결정: [{symbol}] {quantity}주 매수 (예산: {main_unit_budget:,.0f}원)")
            if self.trader.execute_order(symbol, 'buy', quantity, log_payload={'status': 'main_buy'}):
                updated_stock_info = self.trader.get_balance(symbol)
                self.active_trades[symbol].update({
                    'purchase_price': float(updated_stock_info['pchs_avg_prc']),
                    'quantity': int(updated_stock_info['hldg_qty']),
                    'status': 'main_deployed'
                })

    def _manage_active_trades(self):
        if not self.active_trades:
            logger.info("💡 현재 관리 중인 종목이 없습니다."); return
        
        logger.info(f"🔁 보유 종목 {len(self.active_trades)}개 점검...")
        for symbol, trade_info in list(self.active_trades.items()):
            current_info = self.trader.get_current_price(symbol)
            if not current_info: continue
            
            if trade_info['status'] == 'scout_deployed':
                scout_price = trade_info['scout_info']['price']
                if current_info['price'] >= scout_price * (1 + self.trade_params['main_unit_trigger_profit'] / 100):
                    self._deploy_main_unit(symbol, current_info['price'])
                else:
                    self._check_and_execute_sell(symbol, trade_info, current_info['price'])
            elif trade_info['status'] == 'main_deployed':
                self._check_and_execute_sell(symbol, trade_info, current_info['price'])

    def _get_ai_decision(self, stock_info):
        if not self.ai_model: return {'action': 'hold', 'reason': 'AI 모델 비활성화'}
        prompt = f"""당신은 '상한가 따라잡기' 전문 AI 트레이더입니다. 지금 소액의 척후병을 보낼 가치가 있는지 판단하고 JSON으로만 답해주세요.
- 분석 대상: {stock_info['name']}({stock_info['symbol']}), 현재가: {stock_info['price']}
- 분석 후 "action": "buy" 또는 "hold" 와 "reason": "핵심 근거"를 JSON 형식으로 반환하세요."""
        try:
            response = self.ai_model.generate_content(prompt)
            return json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        except Exception as e:
            logger.error(f"AI 결정 과정 오류: {e}"); return None

    def _search_for_new_trades(self):
        logger.info("🔎 신규 매수 대상 '오디션' 시작...")
        top_stocks = self.trader.get_top_ranking_stocks(top_n=10)
        cash_balance = self.trader.get_balance('cash')
        
        selected_scouts = 0
        for stock in top_stocks:
            if selected_scouts >= 4 or stock['symbol'] in self.active_trades: continue
            if cash_balance < stock['price']:
                logger.warning("현금 부족으로 더 이상 척후병을 보낼 수 없습니다."); break

            decision = self._get_ai_decision(stock)
            if decision and decision.get('action') == 'buy':
                logger.info(f"⬆️ AI 매수 추천: [{stock['symbol']}] 이유: {decision.get('reason')}")
                if self.trader.execute_order(stock['symbol'], 'buy', 1, log_payload={'status': 'scout_buy'}):
                    self.active_trades[stock['symbol']] = {
                        'purchase_price': stock['price'], 'quantity': 1, 'status': 'scout_deployed',
                        'scout_info': {'price': stock['price'], 'quantity': 1}
                    }
                    selected_scouts += 1
                    cash_balance -= stock['price']
    
    def run(self):
        logger.info("="*50 + "\n🚀 실시간 AI 트레이더 시스템 가동\n" + "="*50)
        self._manage_active_trades()
        self._search_for_new_trades()
        self._save_state()
        logger.info("✅ 사이클 완료. 현재 보유 종목 수: %d", len(self.active_trades))

    def generate_daily_report(self):
        logger.info("🚀 AI 코치 일일 리포트 생성을 시작합니다...")
        trades = self.trader.get_todays_trades_from_sheet()
        market = self.trader.get_market_summary()
        prompt = f"""당신은 유능한 AI 트레이딩 코치입니다. 아래 오늘의 거래 내역과 시장 상황을 종합하여 '일일 반성 리포트'를 Markdown 형식으로 작성해 주세요.
- **시장 요약:**\n{market}
- **거래 기록:**\n{trades}
- **리포트 가이드:** 시장 리뷰, 종합 평가, 잘된/아쉬운 매매 분석, 내일을 위한 제언을 포함하세요."""
        
        if self.ai_model:
            response = self.ai_model.generate_content(prompt)
            report_text = response.text
        else:
            report_text = "AI 모델이 비활성화되어 리포트를 생성할 수 없습니다."
        
        self.trader.notifier.send_message(f"### 🤖 AI 코치 데일리 리포트 ({self.today_str})\n\n{report_text}")
        logger.info("✅ 리포트 생성 및 전송 완료!")

def main():
    parser = argparse.ArgumentParser(description="실시간 AI 트레이더 및 리포트 생성기")
    parser.add_argument('action', choices=['run', 'report'], help="'run'(자동매매) 또는 'report'(리포트 생성)")
    args = parser.parse_args()
    trader = RealtimeAITrader()
    if args.action == 'run': trader.run()
    elif args.action == 'report': trader.generate_daily_report()

if __name__ == "__main__":
    main() 