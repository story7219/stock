"""
자동 백테스트 실행기
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.scout_strategy import ScoutStrategyManager
from strategies.fibonacci_strategy import FibonacciStrategyManager
from strategies.technical_analyzer import TechnicalAnalyzer

class StrategyBacktester:
    """전략 백테스트 실행기"""
    
    def __init__(self):
        self.initial_capital = 10000000  # 1천만원
        self.start_date = datetime.now() - timedelta(days=365)  # 1년간
        self.end_date = datetime.now()
        
        # 테스트용 종목 리스트 (코스피 대형주)
        self.test_symbols = [
            '005930.KS',  # 삼성전자
            '000660.KS',  # SK하이닉스
            '035420.KS',  # NAVER
            '051910.KS',  # LG화학
            '006400.KS',  # 삼성SDI
        ]
    
    def download_test_data(self) -> Dict[str, pd.DataFrame]:
        """테스트 데이터 다운로드"""
        data = {}
        
        for symbol in self.test_symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d'
                )
                
                if not df.empty:
                    data[symbol] = df
                    print(f"✅ {symbol} 데이터 다운로드 완료")
                else:
                    print(f"❌ {symbol} 데이터 없음")
                    
            except Exception as e:
                print(f"❌ {symbol} 다운로드 실패: {e}")
        
        return data
    
    def run_scout_strategy_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """척후병 전략 백테스트"""
        results = {
            'strategy_name': 'scout_strategy',
            'trades': [],
            'daily_returns': [],
            'portfolio_values': []
        }
        
        try:
            # 간단한 척후병 전략 시뮬레이션
            portfolio_value = self.initial_capital
            cash = self.initial_capital
            positions = {}
            
            # 각 날짜별로 시뮬레이션
            dates = sorted(set().union(*[df.index for df in data.values()]))
            
            for i, date in enumerate(dates[20:]):  # 20일 후부터 시작 (기술적 지표 계산용)
                daily_data = {}
                for symbol, df in data.items():
                    if date in df.index:
                        daily_data[symbol] = df.loc[date]
                
                if len(daily_data) < 3:  # 최소 3개 종목 필요
                    continue
                
                # 척후병 선정 (간단한 모멘텀 기반)
                candidates = []
                for symbol, price_data in daily_data.items():
                    if symbol in data:
                        df = data[symbol]
                        recent_data = df[df.index <= date].tail(20)
                        
                        if len(recent_data) >= 20:
                            momentum = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-20] - 1)
                            volume_ratio = recent_data['Volume'].iloc[-5:].mean() / recent_data['Volume'].iloc[-20:-5].mean()
                            
                            candidates.append({
                                'symbol': symbol,
                                'momentum': momentum,
                                'volume_ratio': volume_ratio,
                                'price': price_data['Close']
                            })
                
                # 상위 4개 선정
                candidates.sort(key=lambda x: x['momentum'] * x['volume_ratio'], reverse=True)
                selected = candidates[:4]
                
                # 매수 시뮬레이션 (각 1주씩)
                for candidate in selected:
                    symbol = candidate['symbol']
                    price = candidate['price']
                    
                    if cash >= price and symbol not in positions:
                        positions[symbol] = {
                            'shares': 1,
                            'entry_price': price,
                            'entry_date': date
                        }
                        cash -= price
                        
                        results['trades'].append({
                            'symbol': symbol,
                            'action': 'buy',
                            'price': price,
                            'shares': 1,
                            'date': date.strftime('%Y-%m-%d')
                        })
                
                # 3일 후 성과 평가 및 매도 (간단화)
                if i % 3 == 0:  # 3일마다 리밸런싱
                    for symbol in list(positions.keys()):
                        if symbol in daily_data:
                            current_price = daily_data[symbol]['Close']
                            position = positions[symbol]
                            
                            # 수익률 계산
                            return_rate = (current_price / position['entry_price'] - 1)
                            
                            # 매도 (간단한 규칙: -5% 손절, +10% 익절)
                            if return_rate <= -0.05 or return_rate >= 0.10:
                                cash += current_price * position['shares']
                                
                                results['trades'].append({
                                    'symbol': symbol,
                                    'action': 'sell',
                                    'price': current_price,
                                    'shares': position['shares'],
                                    'date': date.strftime('%Y-%m-%d'),
                                    'return': return_rate
                                })
                                
                                del positions[symbol]
                
                # 포트폴리오 가치 계산
                position_value = sum(
                    daily_data.get(symbol, {}).get('Close', pos['entry_price']) * pos['shares']
                    for symbol, pos in positions.items()
                )
                portfolio_value = cash + position_value
                
                results['daily_returns'].append({
                    'date': date.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'positions_value': position_value
                })
            
            # 최종 정리
            results['final_portfolio_value'] = portfolio_value
            results['total_return'] = (portfolio_value / self.initial_capital) - 1
            results['total_trades'] = len(results['trades'])
            
        except Exception as e:
            print(f"❌ 척후병 전략 백테스트 실패: {e}")
            results['error'] = str(e)
        
        return results
    
    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """성과 지표 계산"""
        metrics = {}
        
        try:
            if 'daily_returns' not in results or not results['daily_returns']:
                return {'error': '일일 수익률 데이터 없음'}
            
            # 일일 수익률 계산
            daily_values = [r['portfolio_value'] for r in results['daily_returns']]
            daily_returns = []
            
            for i in range(1, len(daily_values)):
                daily_return = (daily_values[i] / daily_values[i-1]) - 1
                daily_returns.append(daily_return)
            
            if not daily_returns:
                return {'error': '수익률 계산 불가'}
            
            returns_array = np.array(daily_returns)
            
            # 주요 지표 계산
            metrics.update({
                'total_return': results.get('total_return', 0),
                'avg_return': np.mean(returns_array),
                'volatility': np.std(returns_array) * np.sqrt(252),  # 연환산
                'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
                'max_drawdown': self.calculate_max_drawdown(daily_values),
                'win_rate': self.calculate_win_rate(results.get('trades', [])),
                'total_trades': results.get('total_trades', 0),
                'max_consecutive_losses': self.calculate_max_consecutive_losses(results.get('trades', []))
            })
            
        except Exception as e:
            print(f"❌ 성과 지표 계산 실패: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """최대 손실 계산"""
        if not portfolio_values:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """승률 계산"""
        if not trades:
            return 0
        
        sell_trades = [t for t in trades if t.get('action') == 'sell' and 'return' in t]
        if not sell_trades:
            return 0
        
        winning_trades = len([t for t in sell_trades if t['return'] > 0])
        return winning_trades / len(sell_trades)
    
    def calculate_max_consecutive_losses(self, trades: List[Dict]) -> int:
        """최대 연속 손실 계산"""
        if not trades:
            return 0
        
        sell_trades = [t for t in trades if t.get('action') == 'sell' and 'return' in t]
        if not sell_trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sell_trades:
            if trade['return'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

def main():
    """메인 실행 함수"""
    print("🚀 자동 백테스트 시작...")
    
    backtester = StrategyBacktester()
    
    # 테스트 데이터 다운로드
    print("📥 테스트 데이터 다운로드 중...")
    data = backtester.download_test_data()
    
    if not data:
        print("❌ 테스트 데이터가 없습니다.")
        return
    
    # 척후병 전략 백테스트
    print("📈 척후병 전략 백테스트 실행 중...")
    scout_results = backtester.run_scout_strategy_backtest(data)
    
    # 성과 지표 계산
    print("📊 성과 지표 계산 중...")
    performance_metrics = backtester.calculate_performance_metrics(scout_results)
    
    # 최종 결과 생성
    final_results = {
        'start_date': backtester.start_date.strftime('%Y-%m-%d'),
        'end_date': backtester.end_date.strftime('%Y-%m-%d'),
        'initial_capital': backtester.initial_capital,
        'test_symbols': backtester.test_symbols,
        'strategy_performance': {
            'scout_strategy': scout_results
        },
        **performance_metrics
    }
    
    # JSON 출력 (GitHub Actions에서 읽음)
    print(json.dumps(final_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 