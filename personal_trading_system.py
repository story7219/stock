"""
개인투자자 현실형 트레이딩 시스템
- 소액 투자 최적화 (100만원~1억원)
- 간단하고 실용적인 전략
- 한국 주식시장 특화
- 무료/저비용 도구 활용
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 개인투자자용 라이브러리 (무료/저비용)
import yfinance as yf  # 해외 주식용
import requests
from bs4 import BeautifulSoup
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PersonalTradingConfig:
    """개인투자자 현실형 설정"""
    # 투자 규모
    total_capital: int = 10_000_000  # 1천만원 기준
    max_stocks: int = 10  # 최대 10종목
    min_position_size: float = 0.05  # 최소 5% (50만원)
    max_position_size: float = 0.20  # 최대 20% (200만원)
    
    # 리스크 관리 (현실적)
    stop_loss: float = -0.15  # 15% 손절
    take_profit: float = 0.30  # 30% 익절
    max_daily_loss: float = -0.05  # 일일 최대 손실 5%
    
    # 거래 설정
    trading_fee: float = 0.00015  # 0.015% (증권사 수수료)
    slippage: float = 0.001  # 0.1% 슬리피지
    min_trade_amount: int = 100_000  # 최소 거래금액 10만원
    
    # 전략 설정
    rebalance_days: int = 7  # 주 1회 리밸런싱
    lookback_period: int = 60  # 60일 데이터 분석
    momentum_period: int = 20  # 20일 모멘텀

class KoreanStockData:
    """한국 주식 데이터 수집 (무료 소스 활용)"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def get_kospi_top_stocks(self, count: int = 50) -> List[str]:
        """코스피 상위 종목 리스트"""
        # 실제로는 한국투자증권 API나 네이버 금융에서 가져옴
        # 여기서는 샘플 데이터
        top_stocks = [
            '005930',  # 삼성전자
            '000660',  # SK하이닉스
            '035420',  # NAVER
            '005380',  # 현대차
            '006400',  # 삼성SDI
            '051910',  # LG화학
            '035720',  # 카카오
            '028260',  # 삼성물산
            '068270',  # 셀트리온
            '207940',  # 삼성바이오로직스
            '066570',  # LG전자
            '323410',  # 카카오뱅크
            '003670',  # 포스코홀딩스
            '096770',  # SK이노베이션
            '000270',  # 기아
            '105560',  # KB금융
            '055550',  # 신한지주
            '017670',  # SK텔레콤
            '034730',  # SK
            '018260',  # 삼성에스디에스
        ]
        return top_stocks[:count]
    
    def get_stock_price_data(self, stock_code: str, days: int = 100) -> pd.DataFrame:
        """개별 종목 가격 데이터 (시뮬레이션)"""
        # 실제로는 한국투자증권 API나 크롤링으로 구현
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 실제 주식 가격 패턴 시뮬레이션
        np.random.seed(int(stock_code))
        
        # 기본 가격 (종목코드에 따라 다르게)
        base_price = int(stock_code) // 1000 * 1000 + 50000
        
        # 가격 변동 시뮬레이션 (실제 주식 패턴과 유사하게)
        returns = np.random.normal(0.001, 0.025, days)  # 일평균 0.1%, 변동성 2.5%
        
        # 트렌드 추가 (일부 종목은 상승, 일부는 하락)
        trend = 0.0002 if int(stock_code) % 3 == 0 else -0.0001
        returns += trend
        
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))  # 최소 1000원
        
        # 거래량 시뮬레이션
        volumes = np.random.lognormal(12, 0.5, days).astype(int)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.03)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.03)) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        return df.set_index('date')
    
    def get_fundamental_data(self, stock_code: str) -> Dict:
        """기본적 분석 데이터 (간단한 지표만)"""
        # 실제로는 네이버 금융, 다음 금융 등에서 크롤링
        np.random.seed(int(stock_code))
        
        return {
            'per': np.random.uniform(5, 25),  # PER
            'pbr': np.random.uniform(0.5, 3.0),  # PBR
            'roe': np.random.uniform(5, 20),  # ROE
            'debt_ratio': np.random.uniform(20, 80),  # 부채비율
            'dividend_yield': np.random.uniform(0, 5),  # 배당수익률
            'market_cap': np.random.uniform(1000, 100000)  # 시가총액 (억원)
        }

class SimpleIndicators:
    """간단하고 효과적인 기술적 지표"""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """단순이동평균"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """지수이동평균"""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 지표"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

class PersonalTradingStrategy:
    """개인투자자용 실용적 전략"""
    
    def __init__(self, config: PersonalTradingConfig):
        self.config = config
        self.indicators = SimpleIndicators()
        
    def calculate_momentum_score(self, price_data: pd.DataFrame) -> float:
        """모멘텀 점수 계산 (0~100)"""
        if len(price_data) < self.config.momentum_period:
            return 50  # 중립
        
        close_prices = price_data['close']
        
        # 1. 가격 모멘텀 (40점)
        price_momentum = (close_prices.iloc[-1] / close_prices.iloc[-self.config.momentum_period] - 1) * 100
        price_score = min(max(price_momentum * 2 + 50, 0), 40)
        
        # 2. 이동평균 위치 (30점)
        sma_20 = self.indicators.sma(close_prices, 20)
        sma_60 = self.indicators.sma(close_prices, 60)
        
        ma_score = 0
        if close_prices.iloc[-1] > sma_20.iloc[-1]:
            ma_score += 15
        if sma_20.iloc[-1] > sma_60.iloc[-1]:
            ma_score += 15
        
        # 3. RSI (20점)
        rsi = self.indicators.rsi(close_prices)
        current_rsi = rsi.iloc[-1]
        
        if 30 <= current_rsi <= 70:  # 적정 구간
            rsi_score = 20
        elif current_rsi < 30:  # 과매도
            rsi_score = 15
        elif current_rsi > 70:  # 과매수
            rsi_score = 5
        else:
            rsi_score = 10
        
        # 4. 거래량 (10점)
        volume_avg = price_data['volume'].rolling(20).mean()
        volume_score = 10 if price_data['volume'].iloc[-1] > volume_avg.iloc[-1] else 5
        
        total_score = price_score + ma_score + rsi_score + volume_score
        return min(max(total_score, 0), 100)
    
    def calculate_value_score(self, fundamental_data: Dict) -> float:
        """가치 점수 계산 (0~100)"""
        score = 50  # 기본 점수
        
        # PER 점수 (30점)
        per = fundamental_data.get('per', 15)
        if per < 10:
            score += 30
        elif per < 15:
            score += 20
        elif per < 20:
            score += 10
        elif per > 30:
            score -= 10
        
        # PBR 점수 (25점)
        pbr = fundamental_data.get('pbr', 1.5)
        if pbr < 1:
            score += 25
        elif pbr < 1.5:
            score += 15
        elif pbr < 2:
            score += 10
        elif pbr > 3:
            score -= 10
        
        # ROE 점수 (25점)
        roe = fundamental_data.get('roe', 10)
        if roe > 15:
            score += 25
        elif roe > 10:
            score += 15
        elif roe > 5:
            score += 10
        else:
            score -= 10
        
        # 부채비율 점수 (20점)
        debt_ratio = fundamental_data.get('debt_ratio', 50)
        if debt_ratio < 30:
            score += 20
        elif debt_ratio < 50:
            score += 10
        elif debt_ratio > 80:
            score -= 15
        
        return min(max(score, 0), 100)
    
    def calculate_final_score(self, momentum_score: float, value_score: float) -> float:
        """최종 점수 계산 (모멘텀 60% + 가치 40%)"""
        return momentum_score * 0.6 + value_score * 0.4
    
    def should_buy(self, price_data: pd.DataFrame, fundamental_data: Dict) -> Tuple[bool, float, str]:
        """매수 신호 판단"""
        momentum_score = self.calculate_momentum_score(price_data)
        value_score = self.calculate_value_score(fundamental_data)
        final_score = self.calculate_final_score(momentum_score, value_score)
        
        # 매수 조건
        buy_signal = False
        reason = ""
        
        if final_score >= 75:
            buy_signal = True
            reason = f"강력 매수 (점수: {final_score:.1f})"
        elif final_score >= 65:
            buy_signal = True
            reason = f"매수 (점수: {final_score:.1f})"
        else:
            reason = f"관망 (점수: {final_score:.1f})"
        
        # 추가 안전장치
        close_prices = price_data['close']
        rsi = self.indicators.rsi(close_prices)
        
        if rsi.iloc[-1] > 80:  # 과매수
            buy_signal = False
            reason += " - 과매수 구간"
        
        return buy_signal, final_score, reason
    
    def should_sell(self, price_data: pd.DataFrame, buy_price: float) -> Tuple[bool, str]:
        """매도 신호 판단"""
        current_price = price_data['close'].iloc[-1]
        return_rate = (current_price - buy_price) / buy_price
        
        # 손절/익절 조건
        if return_rate <= self.config.stop_loss:
            return True, f"손절 ({return_rate:.1%})"
        
        if return_rate >= self.config.take_profit:
            return True, f"익절 ({return_rate:.1%})"
        
        # 기술적 매도 신호
        rsi = self.indicators.rsi(price_data['close'])
        if rsi.iloc[-1] > 80:
            momentum_score = self.calculate_momentum_score(price_data)
            if momentum_score < 40:
                return True, f"기술적 매도 (RSI: {rsi.iloc[-1]:.1f}, 모멘텀: {momentum_score:.1f})"
        
        return False, f"보유 ({return_rate:.1%})"

class PersonalPortfolioManager:
    """개인투자자용 포트폴리오 관리"""
    
    def __init__(self, config: PersonalTradingConfig):
        self.config = config
        self.positions = {}  # {stock_code: {'quantity': int, 'buy_price': float, 'buy_date': datetime}}
        self.cash = config.total_capital
        self.transaction_history = []
        
    def calculate_position_size(self, stock_price: float, score: float) -> int:
        """포지션 크기 계산"""
        # 점수에 따른 가중치 (60점 이상만 투자)
        if score < 60:
            return 0
        
        # 기본 투자 비중 계산
        weight = min(max((score - 60) / 40 * 0.15 + 0.05, self.config.min_position_size), self.config.max_position_size)
        
        # 투자 금액
        investment_amount = self.cash * weight
        
        # 최소 투자 금액 체크
        if investment_amount < self.config.min_trade_amount:
            return 0
        
        # 주식 수량 계산 (단주 불가)
        quantity = int(investment_amount / stock_price)
        
        return quantity
    
    def buy_stock(self, stock_code: str, price: float, quantity: int, reason: str) -> bool:
        """주식 매수"""
        total_cost = price * quantity * (1 + self.config.trading_fee + self.config.slippage)
        
        if total_cost > self.cash:
            logger.warning(f"자금 부족: {stock_code} 매수 불가")
            return False
        
        if len(self.positions) >= self.config.max_stocks:
            logger.warning(f"최대 보유 종목 수 초과: {stock_code} 매수 불가")
            return False
        
        # 매수 실행
        self.positions[stock_code] = {
            'quantity': quantity,
            'buy_price': price,
            'buy_date': datetime.now()
        }
        
        self.cash -= total_cost
        
        # 거래 기록
        self.transaction_history.append({
            'date': datetime.now(),
            'action': 'BUY',
            'stock_code': stock_code,
            'price': price,
            'quantity': quantity,
            'amount': total_cost,
            'reason': reason
        })
        
        logger.info(f"✅ 매수: {stock_code} {quantity}주 @ {price:,}원 ({reason})")
        return True
    
    def sell_stock(self, stock_code: str, price: float, reason: str) -> bool:
        """주식 매도"""
        if stock_code not in self.positions:
            return False
        
        position = self.positions[stock_code]
        quantity = position['quantity']
        
        # 매도 금액 (수수료 차감)
        sell_amount = price * quantity * (1 - self.config.trading_fee - self.config.slippage)
        
        # 수익률 계산
        buy_amount = position['buy_price'] * quantity
        profit_loss = sell_amount - buy_amount
        return_rate = profit_loss / buy_amount
        
        # 매도 실행
        self.cash += sell_amount
        del self.positions[stock_code]
        
        # 거래 기록
        self.transaction_history.append({
            'date': datetime.now(),
            'action': 'SELL',
            'stock_code': stock_code,
            'price': price,
            'quantity': quantity,
            'amount': sell_amount,
            'profit_loss': profit_loss,
            'return_rate': return_rate,
            'reason': reason
        })
        
        logger.info(f"✅ 매도: {stock_code} {quantity}주 @ {price:,}원 ({return_rate:.1%}, {reason})")
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """포트폴리오 현재 가치"""
        total_stock_value = 0
        position_values = {}
        
        for stock_code, position in self.positions.items():
            current_price = current_prices.get(stock_code, position['buy_price'])
            position_value = current_price * position['quantity']
            total_stock_value += position_value
            
            position_values[stock_code] = {
                'quantity': position['quantity'],
                'buy_price': position['buy_price'],
                'current_price': current_price,
                'position_value': position_value,
                'profit_loss': position_value - (position['buy_price'] * position['quantity']),
                'return_rate': (current_price - position['buy_price']) / position['buy_price']
            }
        
        total_value = self.cash + total_stock_value
        
        return {
            'cash': self.cash,
            'stock_value': total_stock_value,
            'total_value': total_value,
            'positions': position_values,
            'return_rate': (total_value - self.config.total_capital) / self.config.total_capital
        }

class PersonalTradingSystem:
    """개인투자자 현실형 통합 시스템"""
    
    def __init__(self, config: PersonalTradingConfig = None):
        self.config = config or PersonalTradingConfig()
        self.data_source = KoreanStockData()
        self.strategy = PersonalTradingStrategy(self.config)
        self.portfolio = PersonalPortfolioManager(self.config)
        
        logger.info(f"🏠 개인투자자 트레이딩 시스템 초기화 (자본금: {self.config.total_capital:,}원)")
    
    async def daily_analysis(self) -> Dict:
        """일일 분석 및 매매 신호"""
        logger.info("📊 일일 시장 분석 시작")
        
        # 관심 종목 리스트
        stock_list = self.data_source.get_kospi_top_stocks(30)
        
        analysis_results = {
            'date': datetime.now(),
            'buy_candidates': [],
            'sell_candidates': [],
            'portfolio_status': {}
        }
        
        # 1. 매수 후보 분석
        logger.info("🔍 매수 후보 분석 중...")
        for stock_code in stock_list:
            if stock_code in self.portfolio.positions:
                continue  # 이미 보유 중인 종목은 스킵
            
            try:
                # 데이터 수집
                price_data = self.data_source.get_stock_price_data(stock_code, self.config.lookback_period)
                fundamental_data = self.data_source.get_fundamental_data(stock_code)
                
                # 매수 신호 분석
                should_buy, score, reason = self.strategy.should_buy(price_data, fundamental_data)
                
                if should_buy:
                    current_price = price_data['close'].iloc[-1]
                    quantity = self.portfolio.calculate_position_size(current_price, score)
                    
                    if quantity > 0:
                        analysis_results['buy_candidates'].append({
                            'stock_code': stock_code,
                            'price': current_price,
                            'quantity': quantity,
                            'score': score,
                            'reason': reason,
                            'investment_amount': current_price * quantity
                        })
                
            except Exception as e:
                logger.error(f"❌ {stock_code} 분석 실패: {e}")
        
        # 2. 보유 종목 매도 분석
        logger.info("📈 보유 종목 분석 중...")
        current_prices = {}
        
        for stock_code in self.portfolio.positions.keys():
            try:
                price_data = self.data_source.get_stock_price_data(stock_code, 30)
                current_price = price_data['close'].iloc[-1]
                current_prices[stock_code] = current_price
                
                buy_price = self.portfolio.positions[stock_code]['buy_price']
                should_sell, reason = self.strategy.should_sell(price_data, buy_price)
                
                if should_sell:
                    analysis_results['sell_candidates'].append({
                        'stock_code': stock_code,
                        'current_price': current_price,
                        'buy_price': buy_price,
                        'reason': reason
                    })
                
            except Exception as e:
                logger.error(f"❌ {stock_code} 매도 분석 실패: {e}")
        
        # 3. 포트폴리오 현황
        analysis_results['portfolio_status'] = self.portfolio.get_portfolio_value(current_prices)
        
        return analysis_results
    
    def execute_trades(self, analysis_results: Dict) -> Dict:
        """매매 실행"""
        execution_results = {
            'executed_buys': [],
            'executed_sells': [],
            'skipped_trades': []
        }
        
        # 1. 매도 먼저 실행 (현금 확보)
        for sell_candidate in analysis_results['sell_candidates']:
            success = self.portfolio.sell_stock(
                sell_candidate['stock_code'],
                sell_candidate['current_price'],
                sell_candidate['reason']
            )
            
            if success:
                execution_results['executed_sells'].append(sell_candidate)
            else:
                execution_results['skipped_trades'].append(sell_candidate)
        
        # 2. 매수 실행 (점수 높은 순으로)
        buy_candidates = sorted(analysis_results['buy_candidates'], key=lambda x: x['score'], reverse=True)
        
        for buy_candidate in buy_candidates:
            success = self.portfolio.buy_stock(
                buy_candidate['stock_code'],
                buy_candidate['price'],
                buy_candidate['quantity'],
                buy_candidate['reason']
            )
            
            if success:
                execution_results['executed_buys'].append(buy_candidate)
            else:
                execution_results['skipped_trades'].append(buy_candidate)
        
        return execution_results
    
    def generate_daily_report(self, analysis_results: Dict, execution_results: Dict) -> str:
        """일일 리포트 생성"""
        portfolio_status = analysis_results['portfolio_status']
        
        report = f"""
📊 개인투자자 트레이딩 일일 리포트
날짜: {datetime.now().strftime('%Y-%m-%d %H:%M')}

💰 포트폴리오 현황:
- 총 자산: {portfolio_status['total_value']:,.0f}원
- 현금: {portfolio_status['cash']:,.0f}원
- 주식: {portfolio_status['stock_value']:,.0f}원
- 수익률: {portfolio_status['return_rate']:.2%}

📈 보유 종목 ({len(portfolio_status['positions'])}개):
"""
        
        for stock_code, position in portfolio_status['positions'].items():
            report += f"- {stock_code}: {position['quantity']:,}주 ({position['return_rate']:+.1%})\n"
        
        report += f"""
🔄 오늘의 거래:
- 매수: {len(execution_results['executed_buys'])}건
- 매도: {len(execution_results['executed_sells'])}건
"""
        
        for buy in execution_results['executed_buys']:
            report += f"  ✅ 매수: {buy['stock_code']} {buy['quantity']:,}주 @ {buy['price']:,.0f}원\n"
        
        for sell in execution_results['executed_sells']:
            report += f"  ✅ 매도: {sell['stock_code']} @ {sell['current_price']:,.0f}원\n"
        
        if analysis_results['buy_candidates']:
            report += f"\n🎯 매수 후보 ({len(analysis_results['buy_candidates'])}개):\n"
            for candidate in analysis_results['buy_candidates'][:5]:  # 상위 5개만
                report += f"- {candidate['stock_code']}: {candidate['score']:.1f}점 ({candidate['reason']})\n"
        
        return report

# 사용 예시
async def main():
    """메인 실행 함수"""
    
    # 개인투자자 설정 (1천만원 기준)
    config = PersonalTradingConfig(
        total_capital=10_000_000,  # 1천만원
        max_stocks=8,              # 최대 8종목
        stop_loss=-0.15,           # 15% 손절
        take_profit=0.25           # 25% 익절
    )
    
    # 시스템 초기화
    trading_system = PersonalTradingSystem(config)
    
    # 일일 분석 실행
    analysis_results = await trading_system.daily_analysis()
    
    # 매매 실행
    execution_results = trading_system.execute_trades(analysis_results)
    
    # 리포트 생성
    daily_report = trading_system.generate_daily_report(analysis_results, execution_results)
    
    print(daily_report)
    
    # 텔레그램으로 리포트 전송 (선택사항)
    # await send_telegram_message(daily_report)

if __name__ == "__main__":
    asyncio.run(main()) 