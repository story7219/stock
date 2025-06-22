"""
팩터 점수조합 전략 (Multi-Factor Scoring Strategy)
Value + Quality + Momentum + Volatility 통합 점수 시스템
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..common import BaseStrategy, StrategyResult

class FactorScoringStrategy(BaseStrategy):
    """
    다중 팩터 점수조합 전략
    - Value: PER, PBR, EV/EBITDA, PSR 낮을수록 좋음
    - Quality: ROE, ROIC, Operating_Margin 높을수록 좋음  
    - Momentum: 3M/6M 수익률 높을수록 좋음
    - Low Volatility: 변동성 낮을수록 좋음
    """
    
    def __init__(self, 
                 value_weight=0.3, 
                 quality_weight=0.3, 
                 momentum_weight=0.3, 
                 volatility_weight=0.1,
                 top_n=20):
        """
        전략 초기화
        
        Args:
            value_weight: 가치 팩터 가중치
            quality_weight: 품질 팩터 가중치  
            momentum_weight: 모멘텀 팩터 가중치
            volatility_weight: 변동성 팩터 가중치
            top_n: 선택할 상위 종목 수
        """
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.momentum_weight = momentum_weight
        self.volatility_weight = volatility_weight
        self.top_n = top_n
        self.name = "팩터 점수조합 전략"
        self.description = "Value + Quality + Momentum + Low Volatility 통합 스코어링"
        
        # 가중치 합이 1이 되도록 정규화
        total_weight = value_weight + quality_weight + momentum_weight + volatility_weight
        self.value_weight /= total_weight
        self.quality_weight /= total_weight
        self.momentum_weight /= total_weight
        self.volatility_weight /= total_weight
    
    def analyze(self, data: pd.DataFrame) -> StrategyResult:
        """
        팩터 점수조합 분석 실행
        
        Args:
            data: 주식 데이터 DataFrame
            
        Returns:
            StrategyResult: 분석 결과
        """
        # 1단계: 기본 필터링 및 데이터 준비
        required_cols = ['PER', 'PBR', 'ROE', 'ROIC', '6M_Return', 'Volatility']
        filtered_data = data.dropna(subset=required_cols).copy()
        
        if len(filtered_data) < 10:
            return StrategyResult(
                selected_stocks=[],
                strategy_name=self.name,
                total_score=0,
                analysis_summary="분석 가능한 데이터가 부족합니다."
            )
        
        # 2단계: 각 팩터별 점수 계산 (0-1 정규화)
        scaler = MinMaxScaler()
        
        # Value 팩터 (낮을수록 좋음 - 역정규화)
        value_factors = ['PER', 'PBR']
        available_value_factors = [col for col in value_factors if col in filtered_data.columns]
        
        if available_value_factors:
            # PER, PBR이 낮을수록 좋으므로 역정규화 (1에서 빼기)
            value_scores = []
            for factor in available_value_factors:
                # 이상치 제거 (99% 분위수 이상 제거)
                factor_data = filtered_data[factor].copy()
                threshold = factor_data.quantile(0.99)
                factor_data = np.where(factor_data > threshold, threshold, factor_data)
                
                # 정규화 후 역변환 (낮을수록 좋음)
                normalized = scaler.fit_transform(factor_data.reshape(-1, 1)).flatten()
                value_scores.append(1 - normalized)
            
            filtered_data['value_score'] = np.mean(value_scores, axis=0)
        else:
            filtered_data['value_score'] = 0.5
        
        # Quality 팩터 (높을수록 좋음)
        quality_factors = ['ROE', 'ROIC', 'Operating_Margin']
        available_quality_factors = [col for col in quality_factors if col in filtered_data.columns]
        
        if available_quality_factors:
            quality_scores = []
            for factor in available_quality_factors:
                factor_data = filtered_data[factor].copy()
                # 음수 값을 0으로 처리
                factor_data = np.where(factor_data < 0, 0, factor_data)
                
                normalized = scaler.fit_transform(factor_data.reshape(-1, 1)).flatten()
                quality_scores.append(normalized)
            
            filtered_data['quality_score'] = np.mean(quality_scores, axis=0)
        else:
            filtered_data['quality_score'] = 0.5
        
        # Momentum 팩터 (높을수록 좋음)
        momentum_factors = ['3M_Return', '6M_Return']
        available_momentum_factors = [col for col in momentum_factors if col in filtered_data.columns]
        
        if available_momentum_factors:
            momentum_scores = []
            for factor in available_momentum_factors:
                factor_data = filtered_data[factor].copy()
                normalized = scaler.fit_transform(factor_data.reshape(-1, 1)).flatten()
                momentum_scores.append(normalized)
            
            filtered_data['momentum_score'] = np.mean(momentum_scores, axis=0)
        else:
            filtered_data['momentum_score'] = 0.5
        
        # Low Volatility 팩터 (낮을수록 좋음)
        if 'Volatility' in filtered_data.columns:
            volatility_data = filtered_data['Volatility'].copy()
            normalized_vol = scaler.fit_transform(volatility_data.reshape(-1, 1)).flatten()
            filtered_data['volatility_score'] = 1 - normalized_vol  # 낮을수록 좋으므로 역변환
        else:
            filtered_data['volatility_score'] = 0.5
        
        # 3단계: 종합 점수 계산
        filtered_data['composite_score'] = (
            filtered_data['value_score'] * self.value_weight +
            filtered_data['quality_score'] * self.quality_weight +
            filtered_data['momentum_score'] * self.momentum_weight +
            filtered_data['volatility_score'] * self.volatility_weight
        )
        
        # 4단계: 추가 필터링 (기본 품질 기준)
        quality_filter = (
            (filtered_data['PER'] > 0) & (filtered_data['PER'] < 50) &
            (filtered_data['ROE'] > 5) &
            (filtered_data['Debt_Ratio'] < 1.0) if 'Debt_Ratio' in filtered_data.columns else True
        )
        
        quality_stocks = filtered_data[quality_filter].copy()
        
        # 5단계: 상위 종목 선택
        if len(quality_stocks) >= self.top_n:
            final_selection = quality_stocks.nlargest(self.top_n, 'composite_score')
        else:
            final_selection = filtered_data.nlargest(self.top_n, 'composite_score')
        
        # 6단계: 결과 생성
        selected_stocks = []
        total_score = 0
        
        for idx, row in final_selection.iterrows():
            # 종합 점수를 100점 만점으로 변환
            final_score = row['composite_score'] * 100
            
            stock_info = {
                'ticker': row['Ticker'],
                'market': row['Market'],
                'current_price': row['Close'],
                'score': round(final_score, 2),
                'value_score': round(row['value_score'] * 100, 1),
                'quality_score': round(row['quality_score'] * 100, 1),
                'momentum_score': round(row['momentum_score'] * 100, 1),
                'volatility_score': round(row['volatility_score'] * 100, 1),
                'per': row['PER'],
                'roe': row['ROE'],
                'return_6m': row.get('6M_Return', 0),
                'volatility': row.get('Volatility', 0),
                'reason': f"종합점수 {final_score:.1f} (V:{row['value_score']*100:.0f} Q:{row['quality_score']*100:.0f} M:{row['momentum_score']*100:.0f} Vol:{row['volatility_score']*100:.0f})"
            }
            
            selected_stocks.append(stock_info)
            total_score += final_score
        
        # 분석 요약 생성
        avg_scores = final_selection[['value_score', 'quality_score', 'momentum_score', 'volatility_score']].mean()
        
        analysis_summary = f"""
        📊 팩터 점수조합 분석 결과
        
        ✅ 선택된 종목: {len(selected_stocks)}개
        🎯 평균 종합점수: {total_score/len(selected_stocks):.1f}/100점
        
        📈 팩터별 평균 점수:
        💰 Value 점수: {avg_scores['value_score']*100:.1f}/100 (가중치: {self.value_weight:.1%})
        🏆 Quality 점수: {avg_scores['quality_score']*100:.1f}/100 (가중치: {self.quality_weight:.1%})
        🚀 Momentum 점수: {avg_scores['momentum_score']*100:.1f}/100 (가중치: {self.momentum_weight:.1%})
        📉 Low Vol 점수: {avg_scores['volatility_score']*100:.1f}/100 (가중치: {self.volatility_weight:.1%})
        
        🎯 전략 특징:
        - 다중 팩터를 통합한 정량적 종목 선별
        - 가치, 품질, 모멘텀, 안정성을 균형있게 고려
        - 각 팩터를 0-1로 정규화하여 객관적 비교
        
        💡 투자 포인트:
        저평가된 우량 기업 중 상승 모멘텀과 안정성을 갖춘 종목 선별
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
            'type': '다중팩터 퀀트전략',
            'risk_level': '중간',
            'holding_period': '3-6개월',
            'rebalancing': '월별',
            'factor_weights': {
                'value': f"{self.value_weight:.1%}",
                'quality': f"{self.quality_weight:.1%}",
                'momentum': f"{self.momentum_weight:.1%}",
                'volatility': f"{self.volatility_weight:.1%}"
            }
        } 