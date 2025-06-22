"""
종합 점수 계산기

여러 투자 전략의 점수를 종합하여 최종 점수를 계산합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CompositeScorer:
    """종합 점수 계산기"""
    
    def __init__(self):
        """초기화"""
        self.score_thresholds = {
            'excellent': 80,
            'good': 70,
            'fair': 60,
            'poor': 50
        }
    
    def calculate_weighted_score(self, strategy_scores: Dict[str, float], 
                               weights: Dict[str, float]) -> float:
        """가중 평균 점수 계산"""
        try:
            total_weighted_score = 0
            total_weight = 0
            
            for strategy_name, weight in weights.items():
                score_key = f'{strategy_name}_score'
                if score_key in strategy_scores:
                    score = strategy_scores[score_key]
                    if score > 0:  # 유효한 점수만 사용
                        total_weighted_score += score * weight
                        total_weight += weight
            
            if total_weight > 0:
                return total_weighted_score / total_weight
            else:
                return 0
                
        except Exception as e:
            logger.error(f"가중 점수 계산 오류: {e}")
            return 0
    
    def calculate_confidence_score(self, strategy_scores: Dict[str, float]) -> float:
        """신뢰도 점수 계산 (유효한 전략 수와 점수 분산 기반)"""
        try:
            valid_scores = [score for key, score in strategy_scores.items() 
                          if key.endswith('_score') and score > 0]
            
            if not valid_scores:
                return 0
            
            # 유효한 전략 수 기반 신뢰도 (최대 20개 전략)
            strategy_count_factor = min(len(valid_scores) / 20, 1.0) * 60
            
            # 점수 일관성 기반 신뢰도 (분산이 낮을수록 높은 신뢰도)
            if len(valid_scores) > 1:
                score_std = np.std(valid_scores)
                consistency_factor = max(0, 40 - score_std)
            else:
                consistency_factor = 20
            
            return strategy_count_factor + consistency_factor
            
        except Exception as e:
            logger.error(f"신뢰도 계산 오류: {e}")
            return 0
    
    def get_score_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= self.score_thresholds['excellent']:
            return 'A+'
        elif score >= self.score_thresholds['good']:
            return 'A'
        elif score >= self.score_thresholds['fair']:
            return 'B'
        elif score >= self.score_thresholds['poor']:
            return 'C'
        else:
            return 'D'
    
    def get_recommendation_strength(self, composite_score: float, 
                                  confidence: float) -> str:
        """추천 강도 결정"""
        try:
            if composite_score >= 80 and confidence >= 80:
                return '강력 추천'
            elif composite_score >= 70 and confidence >= 70:
                return '추천'
            elif composite_score >= 60 and confidence >= 60:
                return '보통'
            elif composite_score >= 50:
                return '약한 추천'
            else:
                return '비추천'
                
        except Exception as e:
            logger.error(f"추천 강도 계산 오류: {e}")
            return '분석 불가'
    
    def identify_dominant_strategy(self, strategy_scores: Dict[str, float]) -> str:
        """지배적인 전략 식별"""
        try:
            valid_strategies = {}
            
            for key, score in strategy_scores.items():
                if key.endswith('_score') and score > 0:
                    strategy_name = key.replace('_score', '')
                    valid_strategies[strategy_name] = score
            
            if not valid_strategies:
                return '분석 불가'
            
            # 최고 점수 전략 반환
            dominant_strategy = max(valid_strategies.items(), key=lambda x: x[1])
            return dominant_strategy[0]
            
        except Exception as e:
            logger.error(f"지배 전략 식별 오류: {e}")
            return '분석 불가'
    
    def classify_investment_style(self, strategy_scores: Dict[str, float]) -> str:
        """투자 스타일 분류"""
        try:
            # 투자 스타일별 전략 그룹
            style_groups = {
                '가치투자': ['buffett', 'graham', 'munger'],
                '성장투자': ['lynch', 'oneill', 'fisher', 'wood'],
                '매크로': ['soros', 'dalio', 'druckenmiller'],
                '기술적분석': ['williams', 'raschke', 'livermore', 'tudor_jones'],
                '시스템매매': ['dennis', 'seykota', 'henry'],
                '퀀트': ['greenblatt', 'k_fisher'],
                '패시브': ['bogle'],
                '혁신성장': ['minervini']
            }
            
            style_scores = {}
            
            # 각 스타일별 평균 점수 계산
            for style, strategies in style_groups.items():
                valid_scores = []
                for strategy in strategies:
                    score_key = f'{strategy}_score'
                    if score_key in strategy_scores and strategy_scores[score_key] > 0:
                        valid_scores.append(strategy_scores[score_key])
                
                if valid_scores:
                    style_scores[style] = np.mean(valid_scores)
                else:
                    style_scores[style] = 0
            
            # 최고 점수 스타일 반환
            if style_scores:
                best_style = max(style_scores.items(), key=lambda x: x[1])
                return best_style[0]
            else:
                return '혼합'
                
        except Exception as e:
            logger.error(f"투자 스타일 분류 오류: {e}")
            return '분석 불가'
    
    def calculate_risk_score(self, strategy_scores: Dict[str, float]) -> float:
        """위험 점수 계산 (높을수록 위험)"""
        try:
            # 고위험 전략들
            high_risk_strategies = ['soros', 'livermore', 'williams', 'raschke', 'tudor_jones']
            
            # 저위험 전략들  
            low_risk_strategies = ['buffett', 'graham', 'bogle', 'dalio']
            
            high_risk_score = 0
            low_risk_score = 0
            
            for strategy in high_risk_strategies:
                score_key = f'{strategy}_score'
                if score_key in strategy_scores:
                    high_risk_score += strategy_scores[score_key]
            
            for strategy in low_risk_strategies:
                score_key = f'{strategy}_score'
                if score_key in strategy_scores:
                    low_risk_score += strategy_scores[score_key]
            
            # 위험 점수 = 고위험 점수 비율
            total_score = high_risk_score + low_risk_score
            if total_score > 0:
                return (high_risk_score / total_score) * 100
            else:
                return 50  # 중간 위험
                
        except Exception as e:
            logger.error(f"위험 점수 계산 오류: {e}")
            return 50

    def calculate_financial_health_score(self, stock_data: Dict) -> float:
        """재무건전성 점수 계산"""
        scores = []
        
        # 부채비율 점수 (낮을수록 좋음)
        debt_ratio = stock_data.get('debt_ratio', 100)
        if debt_ratio <= 30:
            debt_score = 100
        elif debt_ratio <= 50:
            debt_score = 80 - (debt_ratio - 30) * 2
        elif debt_ratio <= 80:
            debt_score = 40 - (debt_ratio - 50) * 1
        else:
            debt_score = max(0, 10 - (debt_ratio - 80) * 0.5)
        scores.append(debt_score)
        
        # 유동비율 점수 (높을수록 좋음)
        current_ratio = stock_data.get('current_ratio', 0)
        if current_ratio >= 2.0:
            current_score = 100
        elif current_ratio >= 1.5:
            current_score = 70 + (current_ratio - 1.5) * 60
        elif current_ratio >= 1.0:
            current_score = 40 + (current_ratio - 1.0) * 60
        else:
            current_score = current_ratio * 40
        scores.append(min(100, current_score))
        
        # 이자보상비율 점수
        interest_coverage = stock_data.get('interest_coverage_ratio', 0)
        if interest_coverage >= 10:
            interest_score = 100
        elif interest_coverage >= 5:
            interest_score = 70 + (interest_coverage - 5) * 6
        elif interest_coverage >= 2:
            interest_score = 40 + (interest_coverage - 2) * 10
        else:
            interest_score = interest_coverage * 20
        scores.append(min(100, interest_score))
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_profitability_score(self, stock_data: Dict) -> float:
        """수익성 점수 계산"""
        scores = []
        
        # ROE 점수
        roe = stock_data.get('roe', 0)
        if roe >= 20:
            roe_score = 100
        elif roe >= 15:
            roe_score = 80 + (roe - 15) * 4
        elif roe >= 10:
            roe_score = 60 + (roe - 10) * 4
        elif roe > 0:
            roe_score = roe * 6
        else:
            roe_score = 0
        scores.append(min(100, roe_score))
        
        # ROA 점수
        roa = stock_data.get('roa', 0)
        if roa >= 10:
            roa_score = 100
        elif roa >= 5:
            roa_score = 70 + (roa - 5) * 6
        elif roa > 0:
            roa_score = roa * 14
        else:
            roa_score = 0
        scores.append(min(100, roa_score))
        
        # 순이익률 점수
        net_margin = stock_data.get('net_margin', 0)
        if net_margin >= 15:
            margin_score = 100
        elif net_margin >= 10:
            margin_score = 80 + (net_margin - 10) * 4
        elif net_margin >= 5:
            margin_score = 50 + (net_margin - 5) * 6
        elif net_margin > 0:
            margin_score = net_margin * 10
        else:
            margin_score = 0
        scores.append(min(100, margin_score))
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_growth_score(self, stock_data: Dict) -> float:
        """성장성 점수 계산"""
        scores = []
        
        # 매출 성장률 점수
        revenue_growth = stock_data.get('revenue_growth', 0)
        if revenue_growth >= 20:
            rev_score = 100
        elif revenue_growth >= 10:
            rev_score = 70 + (revenue_growth - 10) * 3
        elif revenue_growth >= 0:
            rev_score = 40 + revenue_growth * 3
        else:
            rev_score = max(0, 40 + revenue_growth * 2)
        scores.append(min(100, rev_score))
        
        # 순이익 성장률 점수
        earnings_growth = stock_data.get('earnings_growth', 0)
        if earnings_growth >= 25:
            earn_score = 100
        elif earnings_growth >= 15:
            earn_score = 80 + (earnings_growth - 15) * 2
        elif earnings_growth >= 5:
            earn_score = 60 + (earnings_growth - 5) * 2
        elif earnings_growth >= 0:
            earn_score = 40 + earnings_growth * 4
        else:
            earn_score = max(0, 40 + earnings_growth * 2)
        scores.append(min(100, earn_score))
        
        # EPS 성장률 점수
        eps_growth = stock_data.get('eps_growth', 0)
        if eps_growth is not None:
            if eps_growth >= 20:
                eps_score = 100
            elif eps_growth >= 10:
                eps_score = 70 + (eps_growth - 10) * 3
            elif eps_growth >= 0:
                eps_score = 40 + eps_growth * 3
            else:
                eps_score = max(0, 40 + eps_growth * 2)
            scores.append(min(100, eps_score))
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_valuation_score(self, stock_data: Dict) -> float:
        """밸류에이션 점수 계산 (낮을수록 좋음)"""
        scores = []
        
        # PER 점수
        pe_ratio = stock_data.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio <= 10:
                pe_score = 100
            elif pe_ratio <= 15:
                pe_score = 80 - (pe_ratio - 10) * 4
            elif pe_ratio <= 25:
                pe_score = 60 - (pe_ratio - 15) * 2
            else:
                pe_score = max(0, 40 - (pe_ratio - 25) * 1)
            scores.append(pe_score)
        
        # PBR 점수
        pb_ratio = stock_data.get('pb_ratio')
        if pb_ratio and pb_ratio > 0:
            if pb_ratio <= 1.0:
                pb_score = 100
            elif pb_ratio <= 1.5:
                pb_score = 80 - (pb_ratio - 1.0) * 40
            elif pb_ratio <= 2.5:
                pb_score = 60 - (pb_ratio - 1.5) * 20
            else:
                pb_score = max(0, 40 - (pb_ratio - 2.5) * 10)
            scores.append(pb_score)
        
        # PEG 비율 점수
        pe_ratio = stock_data.get('pe_ratio')
        earnings_growth = stock_data.get('earnings_growth', 0)
        if pe_ratio and pe_ratio > 0 and earnings_growth > 0:
            peg_ratio = pe_ratio / earnings_growth
            if peg_ratio <= 0.5:
                peg_score = 100
            elif peg_ratio <= 1.0:
                peg_score = 80 - (peg_ratio - 0.5) * 40
            elif peg_ratio <= 1.5:
                peg_score = 60 - (peg_ratio - 1.0) * 40
            else:
                peg_score = max(0, 40 - (peg_ratio - 1.5) * 20)
            scores.append(peg_score)
        
        return sum(scores) / len(scores) if scores else 50
    
    def calculate_momentum_score(self, stock_data: Dict) -> float:
        """모멘텀 점수 계산"""
        scores = []
        
        # 주가 모멘텀 (3개월 수익률)
        price_momentum_3m = stock_data.get('price_momentum_3m', 0)
        if price_momentum_3m is not None:
            if 5 <= price_momentum_3m <= 25:
                momentum_score = 100
            elif 0 <= price_momentum_3m < 5:
                momentum_score = 60 + price_momentum_3m * 8
            elif 25 < price_momentum_3m <= 40:
                momentum_score = 100 - (price_momentum_3m - 25) * 2
            elif price_momentum_3m > 40:
                momentum_score = max(20, 70 - (price_momentum_3m - 40) * 1)
            else:
                momentum_score = max(0, 60 + price_momentum_3m * 2)
            scores.append(momentum_score)
        
        # 거래량 증가율
        volume_growth = stock_data.get('volume_growth', 0)
        if volume_growth is not None:
            if volume_growth > 0:
                vol_score = min(100, 50 + volume_growth * 0.5)
            else:
                vol_score = max(0, 50 + volume_growth * 0.2)
            scores.append(vol_score)
        
        # 애널리스트 추천 변화
        analyst_revision = stock_data.get('analyst_revision', 0)
        if analyst_revision is not None:
            analyst_score = min(100, max(0, 50 + analyst_revision * 25))
            scores.append(analyst_score)
        
        return sum(scores) / len(scores) if scores else 50
    
    def calculate_comprehensive_score(self, stock_data: Dict, strategy_scores: Dict[str, float] = None) -> Dict:
        """종합 점수 계산"""
        try:
            # 기본 점수들 계산
        financial_health = self.calculate_financial_health_score(stock_data)
        profitability = self.calculate_profitability_score(stock_data)
        growth = self.calculate_growth_score(stock_data)
        valuation = self.calculate_valuation_score(stock_data)
        momentum = self.calculate_momentum_score(stock_data)
        
            # 기본 종합 점수 계산
            base_scores = {
                'financial_health': financial_health,
                'profitability': profitability,
                'growth': growth,
                'valuation': valuation,
                'momentum': momentum
            }
            
            # 가중치 적용
            weights = {
                'financial_health': 0.25,
                'profitability': 0.25,
                'growth': 0.20,
                'valuation': 0.20,
                'momentum': 0.10
            }
            
            comprehensive_score = sum(base_scores[key] * weights[key] for key in base_scores)
            
            # 전략 점수가 있으면 추가로 고려
            if strategy_scores:
                strategy_weights = {'buffett': 0.3, 'lynch': 0.3, 'greenblatt': 0.4}
                strategy_score = self.calculate_weighted_score(strategy_scores, strategy_weights)
                comprehensive_score = (comprehensive_score * 0.7) + (strategy_score * 0.3)
        
        return {
                'comprehensive_score': comprehensive_score,
                'base_scores': base_scores,
                'strategy_scores': strategy_scores or {},
                'confidence': self.calculate_confidence_score(strategy_scores or {}),
                'grade': self.get_score_grade(comprehensive_score),
                'recommendation': self.get_recommendation_strength(comprehensive_score, 
                                                                 self.calculate_confidence_score(strategy_scores or {}))
            }
            
        except Exception as e:
            logger.error(f"종합 점수 계산 오류: {e}")
            return {
                'comprehensive_score': 0,
                'base_scores': {},
                'strategy_scores': {},
                'confidence': 0,
                'grade': 'D',
                'recommendation': '분석 불가'
        }
    
    def rank_stocks(self, stocks_data: List[Dict]) -> List[Dict]:
        """종목들을 점수순으로 랭킹"""
        scored_stocks = []
        
        for stock in stocks_data:
            score_result = self.calculate_comprehensive_score(stock)
            stock.update(score_result)
            scored_stocks.append(stock)
        
        # 종합 점수순 정렬
        return sorted(scored_stocks, key=lambda x: x['comprehensive_score'], reverse=True)
    
    def get_score_distribution(self, stocks_data: List[Dict]) -> Dict:
        """점수 분포 통계"""
        if not stocks_data:
            return {}
        
        scores = [stock.get('comprehensive_score', 0) for stock in stocks_data]
        
        return {
            'mean': round(np.mean(scores), 2),
            'median': round(np.median(scores), 2),
            'std': round(np.std(scores), 2),
            'min': round(min(scores), 2),
            'max': round(max(scores), 2),
            'percentile_25': round(np.percentile(scores, 25), 2),
            'percentile_75': round(np.percentile(scores, 75), 2)
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """점수 가중치 업데이트"""
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.score_weights = {
                key: value / total_weight 
                for key, value in new_weights.items()
                if key in self.score_weights
            } 

# 호환성을 위한 별칭
StockScorer = CompositeScorer

# 편의 함수
def calculate_comprehensive_score(stock_data):
    """종합 점수 계산 편의 함수"""
    scorer = CompositeScorer()
    return scorer.calculate_comprehensive_score(stock_data) 