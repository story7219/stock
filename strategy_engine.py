"""
사용자 맞춤 투자 전략 엔진
- 300일 최저가 기준 횡보 완료 종목 탐지
- 추세전환, 피보나치, 전고점 돌파 매수 신호 감지
"""
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import config
from core_trader import CoreTrader

logger = logging.getLogger(__name__)

class StrategyEngine:
    """사용자 맞춤 투자 전략 실행 엔진"""
    
    def __init__(self):
        self.trader = CoreTrader()
        
        self.target_stocks = [
            "005930",  # 삼성전자
            "000660",  # SK하이닉스
            "035420",  # NAVER
            "051910",  # LG화학
            "006400",  # 삼성SDI
            "035720",  # 카카오
            "068270",  # 셀트리온
            "005380",  # 현대차
            "096770",  # SK이노베이션
            "017670",  # SK텔레콤
        ]
        
    def get_daily_data(self, stock_code: str, days: int = 300):
        """일봉 데이터 조회 (300일)"""
        try:
            path = "/uapi/domestic-stock/v1/quotations/inquire-daily-price"
            headers = self.trader._get_common_headers()
            headers["tr_id"] = "FHKST01010400"
            
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
                "fid_org_adj_prc": "1",
                "fid_period_div_code": "D"
            }
            
            response = requests.get(f"{self.trader.base_url}{path}", headers=headers, params=params)
            data = response.json()
            
            if data.get('rt_cd') == '0' and 'output' in data:
                df = pd.DataFrame(data['output'])
                df['stck_bsop_date'] = pd.to_datetime(df['stck_bsop_date'])
                df = df.sort_values('stck_bsop_date').tail(days)
                
                numeric_cols = ['stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_prpr', 'acml_vol']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                return df
            return None
            
        except Exception as e:
            logger.error(f"{stock_code} 일봉 데이터 조회 실패: {e}")
            return None
    
    def check_300day_low_and_consolidation(self, df: pd.DataFrame):
        """300일 최저가 + 40~100일 횡보 확인"""
        if len(df) < 300:
            return False, {}
            
        low_300 = df['stck_lwpr'].min()
        low_date_idx = df[df['stck_lwpr'] == low_300].index[-1]
        
        after_low = df.loc[low_date_idx:]
        days_since_low = len(after_low)
        
        if not (40 <= days_since_low <= 100):
            return False, {}
            
        consolidation_data = after_low
        max_price = consolidation_data['stck_hgpr'].max()
        price_range = (max_price - low_300) / low_300
        
        if price_range > 0.20:
            return False, {}
            
        return True, {
            'low_300': low_300,
            'days_since_low': days_since_low,
            'consolidation_range': price_range,
            'current_price': df.iloc[-1]['stck_prpr']
        }
    
    def check_trend_reversal(self, df: pd.DataFrame):
        """추세전환 신호 확인 (30/60 이평선 OR 일목균형표 2역전호)"""
        if len(df) < 60:
            return False
            
        # 30일, 60일 이동평균
        df['ma30'] = df['stck_prpr'].rolling(30).mean()
        df['ma60'] = df['stck_prpr'].rolling(60).mean()
        ma_signal = df.iloc[-1]['ma30'] > df.iloc[-1]['ma60']
        
        # 일목균형표 2역전호
        ichimoku_signal = False
        if len(df) >= 52:
            high_9 = df['stck_hgpr'].rolling(9).max()
            low_9 = df['stck_lwpr'].rolling(9).min()
            tenkan_sen = (high_9 + low_9) / 2
            
            high_26 = df['stck_hgpr'].rolling(26).max()
            low_26 = df['stck_lwpr'].rolling(26).min()
            kijun_sen = (high_26 + low_26) / 2
            
            current_tenkan = tenkan_sen.iloc[-1]
            current_kijun = kijun_sen.iloc[-1]
            current_price = df.iloc[-1]['stck_prpr']
            
            if len(df) >= 26:
                price_26_ago = df.iloc[-26]['stck_prpr']
                tenkan_above_kijun = current_tenkan > current_kijun
                chikou_above_price = current_price > price_26_ago
                
                if tenkan_above_kijun and chikou_above_price:
                    ichimoku_signal = True
        
        if ma_signal or ichimoku_signal:
            return True
                
        return False
    
    def check_fibonacci_pullback(self, df: pd.DataFrame, low_300: float):
        """피보나치 눌림목 매수 신호 (31.8%, 50%, 61.8%)"""
        if len(df) < 20:
            return False
            
        recent_high = df.tail(50)['stck_hgpr'].max()
        current_price = df.iloc[-1]['stck_prpr']
        
        fib_range = recent_high - low_300
        fib_318 = recent_high - (fib_range * 0.318)
        fib_500 = recent_high - (fib_range * 0.500)
        fib_618 = recent_high - (fib_range * 0.618)
        
        tolerance = 0.02
        
        if abs(current_price - fib_318) / fib_318 <= tolerance:
            return True
        if abs(current_price - fib_500) / fib_500 <= tolerance:
            return True
        if abs(current_price - fib_618) / fib_618 <= tolerance:
            return True
        
        return False
    
    def check_breakout_signal(self, df: pd.DataFrame):
        """전고점 돌파 매수 신호 (10~20일)"""
        if len(df) < 25:
            return False
            
        current_price = df.iloc[-1]['stck_prpr']
        
        for days in range(10, 21):
            if len(df) >= days:
                high_period = df.tail(days)['stck_hgpr'].max()
                
                if current_price > high_period:
                    avg_volume = df.tail(20)['acml_vol'].mean()
                    current_volume = df.iloc[-1]['acml_vol']
                    
                    if current_volume > avg_volume * 1.5:
                        return True
                        
        return False
    
    def scan_and_execute_strategy(self):
        """전체 전략 스캔 및 실행"""
        buy_signals = []
        
        for stock_code in self.target_stocks:
            try:
                logger.info(f"📊 {stock_code} 분석 중...")
                
                df = self.get_daily_data(stock_code)
                if df is None or len(df) < 300:
                    continue
                
                is_consolidation, consol_info = self.check_300day_low_and_consolidation(df)
                if not is_consolidation:
                    continue
                
                logger.info(f"✅ {stock_code} 횡보 완료 ({consol_info['days_since_low']}일)")
                
                signals = []
                
                if self.check_trend_reversal(df):
                    signals.append("추세전환")
                
                if self.check_fibonacci_pullback(df, consol_info['low_300']):
                    signals.append("피보나치_눌림목")
                
                if self.check_breakout_signal(df):
                    signals.append("전고점_돌파")
                
                if signals:
                    stock_info = self.trader.get_current_price(stock_code)
                    buy_signals.append({
                        'stock_code': stock_code,
                        'stock_name': stock_info.get('name', stock_code) if stock_info else stock_code,
                        'current_price': consol_info['current_price'],
                        'signals': signals,
                        'consolidation_info': consol_info
                    })
                    
                    logger.info(f"🎯 {stock_code} 매수 신호: {', '.join(signals)}")
                
            except Exception as e:
                logger.error(f"{stock_code} 분석 중 오류: {e}")
                continue
        
        return buy_signals
    
    def execute_buy_orders(self, buy_signals: list):
        """매수 신호에 따른 실제 주문 실행"""
        if not buy_signals:
            logger.info("📭 매수 신호가 없습니다.")
            return
        
        balance = self.trader.get_balance()
        if not balance or balance.get('rt_cd') != '0':
            logger.error("잔고 조회 실패")
            return
        
        available_cash = int(balance['output2'][0].get('dnca_tot_amt', 0))
        investment_per_signal = available_cash * 0.10
        
        for signal in buy_signals:
            try:
                stock_code = signal['stock_code']
                current_price = signal['current_price']
                
                quantity = int(investment_per_signal / current_price)
                if quantity < 1:
                    continue
                
                result = self.trader.place_order(
                    stock_code=stock_code,
                    order_type="BUY",
                    quantity=quantity,
                    price=0
                )
                
                if result and result.get('rt_cd') == '0':
                    logger.info(f"✅ {signal['stock_name']} 매수 주문 성공: {quantity}주")
                    
                    if self.trader.telegram_bot:
                        message = f"""
🎯 전략 매수 실행!
종목: {signal['stock_name']}({stock_code})
수량: {quantity}주
가격: {current_price:,}원
신호: {', '.join(signal['signals'])}
                        """
                        self.trader.telegram_bot.send_message(message)
                
            except Exception as e:
                logger.error(f"{signal['stock_code']} 매수 주문 실패: {e}")

def run_strategy():
    """전략 실행 메인 함수"""
    logger.info("🚀 투자 전략 실행 시작")
    
    strategy = StrategyEngine()
    buy_signals = strategy.scan_and_execute_strategy()
    strategy.execute_buy_orders(buy_signals)
    
    logger.info("✅ 투자 전략 실행 완료")

if __name__ == "__main__":
    run_strategy() 