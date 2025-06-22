"""
마법공식 전략 (Magic Formula Strategy)
조엘 그린블라트의 마법공식: ROIC 높음 + PER 낮음
"""

import pandas as pd
import numpy as np
from ..common import BaseStrategy, StrategyResult

class MagicFormulaStrategy(BaseStrategy):
    """
    조엘 그린블라트의 마법공식 전략
    - ROIC(자본수익률)가 높은 기업을 선별
    - PER(주가수익비율)이 낮은 기업을 선별
    - 두 조건을 만족하는 기업에 투자
    """
    
    def __init__(self, min_roic=10.0, max_per=20.0, top_n=20):
        """
        전략 초기화
        
        Args:
            min_roic: 최소 ROIC 기준 (%)
            max_per: 최대 PER 기준
            top_n: 선택할 상위 종목 수
        """
        self.min_roic = min_roic
        self.max_per = max_per
        self.top_n = top_n
        self.name = "마법공식 전략"
        self.description = "ROIC 높음 + PER 낮음 조합으로 우량 저평가 기업 발굴"
    
    def analyze(self, data: pd.DataFrame) -> StrategyResult:
        """
        마법공식 분석 실행
        
        Args:
            data: 주식 데이터 DataFrame
            
        Returns:
            StrategyResult: 분석 결과
        """
        # 1단계: 기본 필터링 (유효한 데이터만)
        filtered_data = data.dropna(subset=['PER', 'ROIC']).copy()
        filtered_data = filtered_data[
            (filtered_data['PER'] > 0) & 
            (filtered_data['ROIC'] > 0)
        ]
        
        if len(filtered_data) == 0:
            return StrategyResult(
                selected_stocks=[],
                strategy_name=self.name,
                total_score=0,
                analysis_summary="분석 가능한 데이터가 없습니다."
            )
        
        # 2단계: 마법공식 점수 계산
        # ROIC 순위 (높을수록 좋음 - 순위가 낮을수록 점수 높음)
        filtered_data['roic_rank'] = filtered_data['ROIC'].rank(ascending=False)
        
        # PER 순위 (낮을수록 좋음 - 순위가 낮을수록 점수 높음)  
        filtered_data['per_rank'] = filtered_data['PER'].rank(ascending=True)
        
        # 마법공식 점수 = ROIC 순위 + PER 순위 (낮을수록 좋음)
        filtered_data['magic_score'] = filtered_data['roic_rank'] + filtered_data['per_rank']
        
        # 3단계: 추가 품질 필터링
        quality_filter = (
            (filtered_data['ROIC'] >= self.min_roic) &
            (filtered_data['PER'] <= self.max_per) &
            (filtered_data['ROE'] > 10) &  # 최소 ROE 기준
            (filtered_data['Debt_Ratio'] < 0.7)  # 부채비율 70% 이하
        )
        
        quality_stocks = filtered_data[quality_filter].copy()
        
        # 4단계: 마법공식 점수 기준 정렬 및 상위 종목 선택
        if len(quality_stocks) > 0:
            final_selection = quality_stocks.nsmallest(self.top_n, 'magic_score')
        else:
            # 품질 필터를 통과한 종목이 없으면 전체에서 선택
            final_selection = filtered_data.nsmallest(self.top_n, 'magic_score')
        
        # 5단계: 결과 생성
        selected_stocks = []
        total_score = 0
        
        for idx, row in final_selection.iterrows():
            # 개별 종목 점수 계산 (0-100 점수로 정규화)
            roic_score = min(100, (row['ROIC'] / 30) * 100)  # ROIC 30% 기준
            per_score = max(0, 100 - (row['PER'] / 30) * 100)  # PER 30 기준
            magic_individual_score = (roic_score + per_score) / 2
            
            stock_info = {
                'ticker': row['Ticker'],
                'market': row['Market'],
                'current_price': row['Close'],
                'score': round(magic_individual_score, 2),
                'per': row['PER'],
                'roic': row['ROIC'],
                'roe': row['ROE'],
                'debt_ratio': row['Debt_Ratio'],
                'magic_rank': int(row['magic_score']),
                'reason': f"ROIC {row['ROIC']:.1f}% (순위: {int(row['roic_rank'])}), PER {row['PER']:.1f} (순위: {int(row['per_rank'])})"
            }
            
            selected_stocks.append(stock_info)
            total_score += magic_individual_score
        
        # 분석 요약 생성
        avg_roic = final_selection['ROIC'].mean()
        avg_per = final_selection['PER'].mean()
        avg_roe = final_selection['ROE'].mean()
        
        analysis_summary = f"""
        📊 마법공식 분석 결과
        
        ✅ 선택된 종목: {len(selected_stocks)}개
        📈 평균 ROIC: {avg_roic:.1f}%
        💰 평균 PER: {avg_per:.1f}
        🏆 평균 ROE: {avg_roe:.1f}%
        
        🎯 마법공식 기준:
        - ROIC 순위 + PER 순위가 낮은 종목 우선 선택
        - 최소 ROIC: {self.min_roic}% 이상
        - 최대 PER: {self.max_per} 이하
        - 추가 품질 필터: ROE 10% 이상, 부채비율 70% 이하
        
        💡 투자 포인트:
        좋은 기업(높은 ROIC)을 저렴한 가격(낮은 PER)에 매수하는 전략
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
            'type': '가치투자 + 품질투자',
            'risk_level': '중간',
            'holding_period': '1-3년',
            'rebalancing': '분기별',
            'criteria': {
                'min_roic': f"{self.min_roic}% 이상",
                'max_per': f"{self.max_per} 이하",
                'top_n': f"상위 {self.top_n}개 종목"
            }
        } 