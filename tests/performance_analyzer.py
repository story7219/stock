"""
성능 분석 및 리포트 생성
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class PerformanceAnalyzer:
    """성능 분석기"""
    
    def __init__(self):
        self.results_file = 'backtest_results.json'
        self.output_dir = 'backtest_charts'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self) -> dict:
        """백테스트 결과 로드"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 결과 파일 로드 실패: {e}")
            return {}
    
    def generate_performance_charts(self, results: dict):
        """성능 차트 생성"""
        try:
            # 포트폴리오 가치 차트
            self.plot_portfolio_value(results)
            
            # 일일 수익률 분포
            self.plot_returns_distribution(results)
            
            # 드로우다운 차트
            self.plot_drawdown(results)
            
            # 거래 분석
            self.plot_trade_analysis(results)
            
            print("✅ 성능 차트 생성 완료")
            
        except Exception as e:
            print(f"❌ 차트 생성 실패: {e}")
    
    def plot_portfolio_value(self, results: dict):
        """포트폴리오 가치 변화 차트"""
        strategy_data = results.get('strategy_performance', {}).get('scout_strategy', {})
        daily_returns = strategy_data.get('daily_returns', [])
        
        if not daily_returns:
            return
        
        dates = [datetime.strptime(r['date'], '%Y-%m-%d') for r in daily_returns]
        values = [r['portfolio_value'] for r in daily_returns]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, linewidth=2, color='blue', label='포트폴리오 가치')
        plt.axhline(y=results.get('initial_capital', 10000000), color='red', linestyle='--', label='초기 자본')
        
        plt.title('포트폴리오 가치 변화', fontsize=16, fontweight='bold')
        plt.xlabel('날짜')
        plt.ylabel('포트폴리오 가치 (원)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/portfolio_value.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_returns_distribution(self, results: dict):
        """수익률 분포 차트"""
        strategy_data = results.get('strategy_performance', {}).get('scout_strategy', {})
        daily_returns = strategy_data.get('daily_returns', [])
        
        if len(daily_returns) < 2:
            return
        
        # 일일 수익률 계산
        values = [r['portfolio_value'] for r in daily_returns]
        returns = [(values[i] / values[i-1] - 1) * 100 for i in range(1, len(values))]
        
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'평균: {np.mean(returns):.2f}%')
        
        plt.title('일일 수익률 분포', fontsize=16, fontweight='bold')
        plt.xlabel('일일 수익률 (%)')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown(self, results: dict):
        """드로우다운 차트"""
        strategy_data = results.get('strategy_performance', {}).get('scout_strategy', {})
        daily_returns = strategy_data.get('daily_returns', [])
        
        if not daily_returns:
            return
        
        dates = [datetime.strptime(r['date'], '%Y-%m-%d') for r in daily_returns]
        values = [r['portfolio_value'] for r in daily_returns]
        
        # 드로우다운 계산
        peak = values[0]
        drawdowns = []
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(dates, drawdowns, 0, alpha=0.3, color='red', label='드로우다운')
        plt.plot(dates, drawdowns, color='red', linewidth=1)
        
        plt.title('드로우다운 분석', fontsize=16, fontweight='bold')
        plt.xlabel('날짜')
        plt.ylabel('드로우다운 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trade_analysis(self, results: dict):
        """거래 분석 차트"""
        strategy_data = results.get('strategy_performance', {}).get('scout_strategy', {})
        trades = strategy_data.get('trades', [])
        
        sell_trades = [t for t in trades if t.get('action') == 'sell' and 'return' in t]
        
        if not sell_trades:
            return
        
        returns = [t['return'] * 100 for t in sell_trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 거래별 수익률
        ax1.bar(range(len(returns)), returns, 
                color=['green' if r > 0 else 'red' for r in returns], alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('거래별 수익률')
        ax1.set_xlabel('거래 번호')
        ax1.set_ylabel('수익률 (%)')
        ax1.grid(True, alpha=0.3)
        
        # 승률 파이 차트
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        if win_count + loss_count > 0:
            ax2.pie([win_count, loss_count], 
                   labels=[f'승리 ({win_count})', f'손실 ({loss_count})'],
                   colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
            ax2.set_title('승률 분석')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/trade_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """메인 실행 함수"""
    print("📊 성능 분석 시작...")
    
    analyzer = PerformanceAnalyzer()
    results = analyzer.load_results()
    
    if results:
        analyzer.generate_performance_charts(results)
        print("✅ 성능 분석 완료!")
    else:
        print("❌ 분석할 데이터가 없습니다.")

if __name__ == "__main__":
    main() 