"""
모멘텀 기반 추세필터 전략 (Momentum Trend Filter Strategy)
3M/6M 수익률 상승 추세와 기술적 지표 활용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..common import BaseStrategy, StrategyResult

class MomentumTrendStrategy(BaseStrategy):
    """
    모멘텀 기반 추세필터 전략
    - 3개월/6개월 수익률 상승 추세 확인
    - 52주 신고가 대비 현재가 위치 분석
    - 변동성 대비 수익률(샤프비율) 고려
    - 지속적인 상승 모멘텀 종목 선별
    """
    
    def __init__(self, 
                 min_3m_return=0.05,
                 min_6m_return=0.10,
                 max_volatility=0.4,
                 min_near_high=0.8,
                 top_n=20):
        """
        전략 초기화
        
        Args:
            min_3m_return: 최소 3개월 수익률 기준
            min_6m_return: 최소 6개월 수익률 기준
            max_volatility: 최대 변동성 기준
            min_near_high: 52주 신고가 대비 최소 비율
            top_n: 선택할 상위 종목 수
        """
        self.min_3m_return = min_3m_return
        self.min_6m_return = min_6m_return
        self.max_volatility = max_volatility
        self.min_near_high = min_near_high
        self.top_n = top_n
        self.name = "모멘텀 기반 추세필터 전략"
        self.description = "지속적인 상승 추세와 기술적 강세를 보이는 종목 선별"
    
    def analyze(self, data: pd.DataFrame) -> StrategyResult:
        """
        모멘텀 추세필터 분석 실행
        
        Args:
            data: 주식 데이터 DataFrame
            
        Returns:
            StrategyResult: 분석 결과
        """
        # 1단계: 기본 필터링 및 데이터 준비
        required_cols = ['3M_Return', '6M_Return', 'Volatility', '52W_High', 'Close']
        filtered_data = data.dropna(subset=required_cols).copy()
        
        if len(filtered_data) < 10:
            return StrategyResult(
                selected_stocks=[],
                strategy_name=self.name,
                total_score=0,
                analysis_summary="분석 가능한 데이터가 부족합니다."
            )
        
        # 2단계: 추세 필터링
        # 기본 모멘텀 조건
        momentum_filter = (
            (filtered_data['3M_Return'] >= self.min_3m_return) &
            (filtered_data['6M_Return'] >= self.min_6m_return) &
            (filtered_data['Volatility'] <= self.max_volatility)
        )
        
        # 52주 신고가 근접도 계산
        filtered_data['near_high_ratio'] = filtered_data['Close'] / filtered_data['52W_High']
        high_filter = filtered_data['near_high_ratio'] >= self.min_near_high
        
        # 추세 가속도 계산 (6개월 > 3개월 수익률이면 가속)
        filtered_data['momentum_acceleration'] = (
            filtered_data['6M_Return'] > filtered_data['3M_Return'] * 2
        ).astype(int)
        
        # 3단계: 모멘텀 점수 계산
        scaler = MinMaxScaler()
        
        # 3개월 수익률 점수 (높을수록 좋음)
        filtered_data['return_3m_score'] = scaler.fit_transform(
            filtered_data[['3M_Return']]
        ).flatten()
        
        # 6개월 수익률 점수 (높을수록 좋음)
        filtered_data['return_6m_score'] = scaler.fit_transform(
            filtered_data[['6M_Return']]
        ).flatten()
        
        # 변동성 점수 (낮을수록 좋음)
        filtered_data['volatility_score'] = 1 - scaler.fit_transform(
            filtered_data[['Volatility']]
        ).flatten()
        
        # 신고가 근접 점수
        filtered_data['high_proximity_score'] = scaler.fit_transform(
            filtered_data[['near_high_ratio']]
        ).flatten()
        
        # 샤프비율 계산 (6개월 수익률 / 변동성)
        filtered_data['sharpe_ratio'] = filtered_data['6M_Return'] / (filtered_data['Volatility'] + 0.01)
        filtered_data['sharpe_score'] = scaler.fit_transform(
            filtered_data[['sharpe_ratio']]
        ).flatten()
        
        # 4단계: 종합 모멘텀 점수 계산
        filtered_data['momentum_score'] = (
            filtered_data['return_3m_score'] * 0.2 +
            filtered_data['return_6m_score'] * 0.3 +
            filtered_data['volatility_score'] * 0.2 +
            filtered_data['high_proximity_score'] * 0.15 +
            filtered_data['sharpe_score'] * 0.1 +
            filtered_data['momentum_acceleration'] * 0.05
        )
        
        # 5단계: 품질 필터 적용
        quality_stocks = filtered_data[momentum_filter & high_filter].copy()
        
        # 6단계: 추가 기술적 필터링
        if len(quality_stocks) > 0:
            # ROE가 있는 경우 추가 고려
            if 'ROE' in quality_stocks.columns:
                roe_filter = quality_stocks['ROE'] > 10
                quality_stocks = quality_stocks[roe_filter]
            
            # PER이 있는 경우 극단적 고평가 제외
            if 'PER' in quality_stocks.columns:
                per_filter = (quality_stocks['PER'] > 0) & (quality_stocks['PER'] < 100)
                quality_stocks = quality_stocks[per_filter]
        
        # 7단계: 최종 종목 선택
        if len(quality_stocks) >= self.top_n:
            final_selection = quality_stocks.nlargest(self.top_n, 'momentum_score')
        else:
            # 조건을 만족하는 종목이 부족하면 전체에서 선택
            final_selection = filtered_data.nlargest(self.top_n, 'momentum_score')
        
        # 8단계: 결과 생성
        selected_stocks = []
        total_score = 0
        
        for idx, row in final_selection.iterrows():
            # 모멘텀 점수를 100점 만점으로 변환
            final_score = row['momentum_score'] * 100
            
            stock_info = {
                'ticker': row['Ticker'],
                'market': row['Market'],
                'current_price': row['Close'],
                'score': round(final_score, 2),
                'return_3m': round(row['3M_Return'] * 100, 1),
                'return_6m': round(row['6M_Return'] * 100, 1),
                'volatility': round(row['Volatility'] * 100, 1),
                'near_high_ratio': round(row['near_high_ratio'] * 100, 1),
                'sharpe_ratio': round(row['sharpe_ratio'], 2),
                'momentum_acceleration': bool(row['momentum_acceleration']),
                'per': row.get('PER', 'N/A'),
                'roe': row.get('ROE', 'N/A'),
                'reason': f"모멘텀점수 {final_score:.1f} (3M:{row['3M_Return']*100:.1f}% 6M:{row['6M_Return']*100:.1f}% 신고가:{row['near_high_ratio']*100:.0f}%)"
            }
            
            selected_stocks.append(stock_info)
            total_score += final_score
        
        # 분석 요약 생성
        avg_3m = final_selection['3M_Return'].mean() * 100
        avg_6m = final_selection['6M_Return'].mean() * 100
        avg_vol = final_selection['Volatility'].mean() * 100
        avg_high_ratio = final_selection['near_high_ratio'].mean() * 100
        acceleration_count = final_selection['momentum_acceleration'].sum()
        
        analysis_summary = f"""
        📊 모멘텀 추세필터 분석 결과
        
        ✅ 선택된 종목: {len(selected_stocks)}개
        🎯 평균 모멘텀점수: {total_score/len(selected_stocks):.1f}/100점
        
        📈 수익률 현황:
        🚀 평균 3개월 수익률: {avg_3m:.1f}%
        🌟 평균 6개월 수익률: {avg_6m:.1f}%
        📊 평균 변동성: {avg_vol:.1f}%
        🎯 평균 신고가 근접도: {avg_high_ratio:.1f}%
        ⚡ 모멘텀 가속 종목: {acceleration_count}개
        
        🎯 필터 조건:
        - 최소 3개월 수익률: {self.min_3m_return*100:.0f}% 이상
        - 최소 6개월 수익률: {self.min_6m_return*100:.0f}% 이상
        - 최대 변동성: {self.max_volatility*100:.0f}% 이하
        - 52주 신고가 근접도: {self.min_near_high*100:.0f}% 이상
        
        💡 투자 포인트:
        지속적인 상승 추세와 기술적 강세를 보이는 모멘텀 종목 선별
        """
        
        return StrategyResult(
            selected_stocks=selected_stocks,
            strategy_name=self.name,
            total_score=round(total_score / len(selected_stocks) if selected_stocks else 0, 2),
            analysis_summary=analysis_summary.strip()
        )
    
    def get_strategy_info(self) -> dict:
        """전략 정보 반환"""
        return {
            'name': self.name,
            'description': self.description,
            'type': '모멘텀 추세추종',
            'risk_level': '중상',
            'holding_period': '1-3개월',
            'rebalancing': '월별',
            'criteria': {
                'min_3m_return': f"{self.min_3m_return*100:.0f}% 이상",
                'min_6m_return': f"{self.min_6m_return*100:.0f}% 이상",
                'max_volatility': f"{self.max_volatility*100:.0f}% 이하",
                'min_near_high': f"{self.min_near_high*100:.0f}% 이상"
            }
        } 