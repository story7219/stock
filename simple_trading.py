"""
초간단 개인투자자 트레이딩 시스템
- 파일 1개로 모든 기능 완성
- 설정 5줄, 실행 1줄
- 복잡한 기능 모두 제거
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 간단 설정
CAPITAL = 10_000_000  # 투자금 1천만원
MAX_STOCKS = 5        # 최대 5종목
STOP_LOSS = -0.15     # 15% 손절
TAKE_PROFIT = 0.25    # 25% 익절

# 로깅
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class SimpleTrading:
    """초간단 트레이딩 시스템"""
    
    def __init__(self):
        self.cash = CAPITAL
        self.stocks = {}  # {종목코드: {'수량': int, '매수가': float}}
        
        # 코스피 상위 20종목 (실제로는 API에서 가져옴)
        self.stock_list = [
            '005930', '000660', '035420', '005380', '006400',
            '051910', '035720', '028260', '068270', '207940',
            '066570', '323410', '003670', '096770', '000270',
            '105560', '055550', '017670', '034730', '018260'
        ]
    
    def get_price(self, stock_code):
        """주가 가져오기 (시뮬레이션)"""
        # 실제로는 한국투자증권 API 사용
        np.random.seed(int(stock_code) + int(datetime.now().strftime('%Y%m%d')))
        base_price = int(stock_code) // 100 * 1000 + 50000
        change = np.random.uniform(-0.05, 0.05)  # ±5% 변동
        return int(base_price * (1 + change))
    
    def calculate_score(self, stock_code):
        """종목 점수 계산 (0~100점)"""
        np.random.seed(int(stock_code))
        
        # 간단한 점수 계산 (실제로는 기술적/기본적 분석)
        momentum = np.random.uniform(0, 40)      # 모멘텀 점수
        value = np.random.uniform(0, 30)         # 가치 점수  
        volume = np.random.uniform(0, 20)        # 거래량 점수
        trend = np.random.uniform(0, 10)         # 추세 점수
        
        return momentum + value + volume + trend
    
    def should_buy(self, stock_code):
        """매수 판단"""
        if len(self.stocks) >= MAX_STOCKS:
            return False, "최대 보유 종목 수 초과"
        
        if stock_code in self.stocks:
            return False, "이미 보유 중"
        
        score = self.calculate_score(stock_code)
        if score >= 70:
            return True, f"매수 신호 (점수: {score:.1f})"
        
        return False, f"점수 부족 ({score:.1f})"
    
    def should_sell(self, stock_code):
        """매도 판단"""
        if stock_code not in self.stocks:
            return False, "보유하지 않음"
        
        current_price = self.get_price(stock_code)
        buy_price = self.stocks[stock_code]['매수가']
        return_rate = (current_price - buy_price) / buy_price
        
        if return_rate <= STOP_LOSS:
            return True, f"손절 ({return_rate:.1%})"
        
        if return_rate >= TAKE_PROFIT:
            return True, f"익절 ({return_rate:.1%})"
        
        return False, f"보유 ({return_rate:.1%})"
    
    def buy_stock(self, stock_code):
        """주식 매수"""
        price = self.get_price(stock_code)
        investment = self.cash // MAX_STOCKS  # 균등 분할 투자
        quantity = investment // price
        
        if quantity == 0:
            return False, "투자금 부족"
        
        cost = price * quantity
        self.cash -= cost
        self.stocks[stock_code] = {'수량': quantity, '매수가': price}
        
        logger.info(f"✅ 매수: {stock_code} {quantity:,}주 @ {price:,}원")
        return True, f"매수 완료"
    
    def sell_stock(self, stock_code):
        """주식 매도"""
        if stock_code not in self.stocks:
            return False, "보유하지 않음"
        
        price = self.get_price(stock_code)
        quantity = self.stocks[stock_code]['수량']
        revenue = price * quantity
        
        buy_price = self.stocks[stock_code]['매수가']
        profit = revenue - (buy_price * quantity)
        return_rate = profit / (buy_price * quantity)
        
        self.cash += revenue
        del self.stocks[stock_code]
        
        logger.info(f"✅ 매도: {stock_code} {quantity:,}주 @ {price:,}원 ({return_rate:+.1%})")
        return True, f"매도 완료 ({return_rate:+.1%})"
    
    def daily_trading(self):
        """일일 트레이딩 실행"""
        logger.info(f"\n📊 {datetime.now().strftime('%Y-%m-%d')} 트레이딩 시작")
        logger.info(f"💰 현금: {self.cash:,}원, 보유종목: {len(self.stocks)}개")
        
        # 1. 매도 검토
        sell_list = list(self.stocks.keys())
        for stock_code in sell_list:
            should_sell, reason = self.should_sell(stock_code)
            if should_sell:
                self.sell_stock(stock_code)
        
        # 2. 매수 검토
        for stock_code in self.stock_list:
            should_buy, reason = self.should_buy(stock_code)
            if should_buy:
                success, msg = self.buy_stock(stock_code)
                if not success:
                    break
        
        # 3. 현황 출력
        total_value = self.cash
        for stock_code, info in self.stocks.items():
            current_price = self.get_price(stock_code)
            value = current_price * info['수량']
            total_value += value
            return_rate = (current_price - info['매수가']) / info['매수가']
            logger.info(f"📈 {stock_code}: {info['수량']:,}주 ({return_rate:+.1%})")
        
        total_return = (total_value - CAPITAL) / CAPITAL
        logger.info(f"💎 총 자산: {total_value:,}원 ({total_return:+.1%})")
        
        return total_value, total_return

# 실행
if __name__ == "__main__":
    trader = SimpleTrading()
    trader.daily_trading() 