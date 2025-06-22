"""
투자 전략 종합 추천 시스템

20명의 유명 투자자 전략을 종합하여 최적의 TOP5 종목을 추천합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# 가치투자 전략
from strategies.value.buffett import BuffettStrategy
from strategies.value.graham import GrahamStrategy
from strategies.value.munger import MungerStrategy

# 성장투자 전략
from strategies.growth.lynch import LynchStrategy
from strategies.growth.oneill import ONeillStrategy
from strategies.growth.fisher import FisherStrategy
from strategies.growth.wood import WoodStrategy

# 매크로 전략
from strategies.macro.soros import SorosStrategy
from strategies.macro.dalio import DalioStrategy
from strategies.macro.druckenmiller import DruckenmillerStrategy

# 기술적분석/단타 전략
from strategies.technical.williams import WilliamsStrategy
from strategies.technical.raschke import RaschkeStrategy
from strategies.technical.livermore import LivermoreStrategy
from strategies.technical.tudor_jones import TudorJonesStrategy

# 시스템매매 전략
from strategies.systematic.dennis import DennisStrategy
from strategies.systematic.seykota import SeykotaStrategy
from strategies.systematic.henry import HenryStrategy

# 퀀트/혼합 전략
from strategies.quantitative.greenblatt import GreenblattStrategy
from strategies.quantitative.k_fisher import KFisherStrategy

# 패시브 전략
from strategies.passive.bogle import BogleStrategy

# 혁신성장 전략
from strategies.innovation.minervini import MinerviniStrategy

from recommenders.scorer import CompositeScorer

logger = logging.getLogger(__name__)

class InvestmentRecommender:
    """종합 투자 추천 시스템 - 20개 전략 통합"""
    
    def __init__(self, strategy_weights: Dict[str, float] = None):
        """
        Args:
            strategy_weights: 전략별 가중치 딕셔너리
        """
        # 모든 전략 인스턴스 생성
        self.strategies = {
            # 가치투자 (30%)
            'buffett': BuffettStrategy(),
            'graham': GrahamStrategy(),
            'munger': MungerStrategy(),
            
            # 성장투자 (25%)
            'lynch': LynchStrategy(),
            'oneill': ONeillStrategy(),
            'fisher': FisherStrategy(),
            'wood': WoodStrategy(),
            
            # 매크로 (15%)
            'soros': SorosStrategy(),
            'dalio': DalioStrategy(),
            'druckenmiller': DruckenmillerStrategy(),
            
            # 기술적분석/단타 (10%)
            'williams': WilliamsStrategy(),
            'raschke': RaschkeStrategy(),
            'livermore': LivermoreStrategy(),
            'tudor_jones': TudorJonesStrategy(),
            
            # 시스템매매 (8%)
            'dennis': DennisStrategy(),
            'seykota': SeykotaStrategy(),
            'henry': HenryStrategy(),
            
            # 퀀트/혼합 (7%)
            'greenblatt': GreenblattStrategy(),
            'k_fisher': KFisherStrategy(),
            
            # 패시브 (3%)
            'bogle': BogleStrategy(),
            
            # 혁신성장 (2%)
            'minervini': MinerviniStrategy()
        }
        
        self.scorer = CompositeScorer()
        
        # 기본 전략 가중치 (투자 스타일별)
        self.strategy_weights = strategy_weights or {
            # 가치투자 (30%)
            'buffett': 0.12,
            'graham': 0.10,
            'munger': 0.08,
            
            # 성장투자 (25%)
            'lynch': 0.08,
            'oneill': 0.06,
            'fisher': 0.06,
            'wood': 0.05,
            
            # 매크로 (15%)
            'soros': 0.06,
            'dalio': 0.05,
            'druckenmiller': 0.04,
            
            # 기술적분석/단타 (10%)
            'williams': 0.03,
            'raschke': 0.03,
            'livermore': 0.02,
            'tudor_jones': 0.02,
            
            # 시스템매매 (8%)
            'dennis': 0.03,
            'seykota': 0.03,
            'henry': 0.02,
            
            # 퀀트/혼합 (7%)
            'greenblatt': 0.04,
            'k_fisher': 0.03,
            
            # 패시브 (3%)
            'bogle': 0.03,
            
            # 혁신성장 (2%)
            'minervini': 0.02
        }
        
        # 가중치 검증
        total_weight = sum(self.strategy_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"전략 가중치 합계가 1.0이 아닙니다: {total_weight}")
    
    def analyze_all_strategies(self, stocks_data: List[Dict]) -> List[Dict]:
        """모든 전략으로 종목 분석"""
        analyzed_stocks = []
        
        for stock in stocks_data:
            stock_analysis = {
                'symbol': stock.get('symbol', ''),
                'name': stock.get('name', ''),
                'market_cap': stock.get('market_cap', 0),
                'price': stock.get('price', 0)
            }
            
            # 각 전략별 분석
            for strategy_name, strategy in self.strategies.items():
                try:
                    if strategy_name == 'greenblatt':
                        # 그린블라트 전략은 전체 주식 데이터가 필요
                        result = strategy.analyze_stock(stock, stocks_data)
                    else:
                        result = strategy.analyze_stock(stock)
                    
                    stock_analysis[strategy_name] = result
                    
                except Exception as e:
                    logger.error(f"{strategy_name} 전략 분석 오류 ({stock.get('symbol', 'Unknown')}): {e}")
                    stock_analysis[strategy_name] = {
                        'total_score': 0, 
                        'error': str(e),
                        'strategy_name': strategy_name,
                        'investment_decision': '분석 불가'
                    }
            
            analyzed_stocks.append(stock_analysis)
        
        return analyzed_stocks
    
    def calculate_composite_score(self, stock_analysis: Dict) -> Dict:
        """종합 점수 계산"""
        strategy_scores = {}
        valid_scores = []
        
        # 각 전략별 점수 수집
        for strategy_name in self.strategies.keys():
            strategy_result = stock_analysis.get(strategy_name, {})
            score = strategy_result.get('total_score', 0)
            strategy_scores[f'{strategy_name}_score'] = score
            
            # 유효한 점수만 수집 (오류가 없는 경우)
            if 'error' not in strategy_result and score > 0:
                valid_scores.append(score)
        
        # 가중 평균 계산
        composite_score = 0
        total_weight = 0
        
        for strategy_name, weight in self.strategy_weights.items():
            strategy_result = stock_analysis.get(strategy_name, {})
            if 'error' not in strategy_result:
                score = strategy_result.get('total_score', 0)
                composite_score += score * weight
                total_weight += weight
        
        # 가중치 정규화
        if total_weight > 0:
            composite_score = composite_score / total_weight * 100
        
        # 점수 신뢰도 계산
        if len(valid_scores) > 1:
            score_std = np.std(valid_scores)
            confidence = max(0, 100 - score_std)
            consistency = 100 - score_std
        else:
            confidence = 50  # 기본값
            consistency = 50
        
        return {
            **strategy_scores,
            'composite_score': composite_score,
            'confidence': confidence,
            'score_consistency': consistency,
            'valid_strategies_count': len(valid_scores)
        }
    
    def get_top_recommendations(self, stocks_data: List[Dict], top_n: int = 5) -> List[Dict]:
        """TOP N 종목 추천"""
        # 1단계: 모든 전략으로 분석
        print(f"📊 {len(self.strategies)}개 전략으로 {len(stocks_data)}개 종목 분석 중...")
        analyzed_stocks = self.analyze_all_strategies(stocks_data)
        
        # 2단계: 종합 점수 계산
        print("🔢 종합 점수 계산 중...")
        for stock in analyzed_stocks:
            composite_results = self.calculate_composite_score(stock)
            stock.update(composite_results)
        
        # 3단계: 종합 점수순 정렬
        sorted_stocks = sorted(
            analyzed_stocks, 
            key=lambda x: x.get('composite_score', 0), 
            reverse=True
        )
        
        # 4단계: TOP N 선별
        top_stocks = sorted_stocks[:top_n]
        
        # 5단계: 순위 및 추가 정보 부여
        for i, stock in enumerate(top_stocks):
            stock['rank'] = i + 1
            stock['recommendation_strength'] = self._get_recommendation_strength(stock)
            stock['dominant_strategy'] = self._determine_dominant_strategy(stock)
            stock['investment_style'] = self._determine_investment_style(stock)
        
        print(f"✅ TOP {top_n} 종목 선별 완료")
        return top_stocks
    
    def _get_recommendation_strength(self, stock: Dict) -> str:
        """추천 강도 판정"""
        composite_score = stock.get('composite_score', 0)
        confidence = stock.get('confidence', 0)
        valid_count = stock.get('valid_strategies_count', 0)
        
        if composite_score >= 80 and confidence >= 70 and valid_count >= 15:
            return "매우 강함"
        elif composite_score >= 70 and confidence >= 60 and valid_count >= 12:
            return "강함"
        elif composite_score >= 60 and confidence >= 50 and valid_count >= 10:
            return "보통"
        elif composite_score >= 50 and valid_count >= 8:
            return "약함"
        else:
            return "매우 약함"
    
    def _determine_dominant_strategy(self, stock: Dict) -> str:
        """가장 높은 점수를 준 전략 찾기"""
        max_score = 0
        dominant_strategy = "없음"
        
        for strategy_name in self.strategies.keys():
            score = stock.get(f'{strategy_name}_score', 0)
            if score > max_score:
                max_score = score
                dominant_strategy = strategy_name
        
        return dominant_strategy
    
    def _determine_investment_style(self, stock: Dict) -> str:
        """투자 스타일 판정"""
        style_scores = {
            '가치투자': 0,
            '성장투자': 0,
            '매크로': 0,
            '기술적분석': 0,
            '시스템매매': 0,
            '퀀트': 0,
            '패시브': 0,
            '혁신성장': 0
        }
        
        # 스타일별 점수 합계
        value_strategies = ['buffett', 'graham', 'munger']
        growth_strategies = ['lynch', 'oneill', 'fisher', 'wood']
        macro_strategies = ['soros', 'dalio', 'druckenmiller']
        technical_strategies = ['williams', 'raschke', 'livermore', 'tudor_jones']
        systematic_strategies = ['dennis', 'seykota', 'henry']
        quant_strategies = ['greenblatt', 'k_fisher']
        passive_strategies = ['bogle']
        innovation_strategies = ['minervini']
        
        for strategy in value_strategies:
            style_scores['가치투자'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in growth_strategies:
            style_scores['성장투자'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in macro_strategies:
            style_scores['매크로'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in technical_strategies:
            style_scores['기술적분석'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in systematic_strategies:
            style_scores['시스템매매'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in quant_strategies:
            style_scores['퀀트'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in passive_strategies:
            style_scores['패시브'] += stock.get(f'{strategy}_score', 0)
        
        for strategy in innovation_strategies:
            style_scores['혁신성장'] += stock.get(f'{strategy}_score', 0)
        
        # 최고 점수 스타일 반환
        return max(style_scores, key=style_scores.get)
    
    def generate_recommendation_report(self, recommendations: List[Dict]) -> Dict:
        """추천 보고서 생성"""
        if not recommendations:
            return {"error": "추천 종목이 없습니다."}
        
        # 전체 통계
        total_composite_score = sum(stock.get('composite_score', 0) for stock in recommendations)
        avg_composite_score = total_composite_score / len(recommendations)
        avg_confidence = sum(stock.get('confidence', 0) for stock in recommendations) / len(recommendations)
        avg_valid_strategies = sum(stock.get('valid_strategies_count', 0) for stock in recommendations) / len(recommendations)
        
        # 투자 스타일 분포
        style_distribution = {}
        for stock in recommendations:
            style = stock.get('investment_style', '기타')
            style_distribution[style] = style_distribution.get(style, 0) + 1
        
        # 지배적 전략 분포
        dominant_strategy_distribution = {}
        for stock in recommendations:
            strategy = stock.get('dominant_strategy', '없음')
            dominant_strategy_distribution[strategy] = dominant_strategy_distribution.get(strategy, 0) + 1
        
        # 전략별 평균 점수 (상위 5개만)
        strategy_avg_scores = {}
        for strategy_name in self.strategies.keys():
            scores = [stock.get(f'{strategy_name}_score', 0) for stock in recommendations]
            strategy_avg_scores[strategy_name] = sum(scores) / len(scores)
        
        # 상위 5개 전략 선별
        top_strategies = sorted(strategy_avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'summary': {
                'total_recommendations': len(recommendations),
                'average_composite_score': round(avg_composite_score, 2),
                'average_confidence': round(avg_confidence, 2),
                'average_valid_strategies': round(avg_valid_strategies, 1),
                'investment_style_distribution': style_distribution,
                'dominant_strategy_distribution': dominant_strategy_distribution
            },
            'top_performing_strategies': dict(top_strategies),
            'top_pick': recommendations[0] if recommendations else None,
            'recommendations': recommendations
        }
    
    def get_diversified_portfolio(self, stocks_data: List[Dict], portfolio_size: int = 10) -> List[Dict]:
        """다양화된 포트폴리오 구성"""
        # 기본 추천 목록 확장
        recommendations = self.get_top_recommendations(stocks_data, top_n=portfolio_size * 2)
        
        # 투자 스타일별 분산
        style_buckets = {}
        for stock in recommendations:
            style = stock.get('investment_style', '기타')
            if style not in style_buckets:
                style_buckets[style] = []
            style_buckets[style].append(stock)
        
        # 각 스타일에서 균등하게 선택
        diversified_portfolio = []
        max_per_style = max(1, portfolio_size // len(style_buckets))
        
        for style, stocks in style_buckets.items():
            selected = stocks[:max_per_style]
            diversified_portfolio.extend(selected)
        
        # 포트폴리오 크기 조정
        if len(diversified_portfolio) > portfolio_size:
            diversified_portfolio = diversified_portfolio[:portfolio_size]
        elif len(diversified_portfolio) < portfolio_size:
            # 부족한 경우 상위 종목으로 채움
            remaining = portfolio_size - len(diversified_portfolio)
            additional = [stock for stock in recommendations 
                         if stock not in diversified_portfolio][:remaining]
            diversified_portfolio.extend(additional)
        
        # 순위 재조정
        for i, stock in enumerate(diversified_portfolio):
            stock['portfolio_rank'] = i + 1
        
        return diversified_portfolio
    
    def update_strategy_weights(self, new_weights: Dict[str, float]):
        """전략 가중치 업데이트"""
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"가중치 합계가 1.0이 아닙니다: {total_weight}")
        
        self.strategy_weights.update(new_weights)
        logger.info("전략 가중치가 업데이트되었습니다.")
    
    def get_strategy_descriptions(self) -> Dict[str, str]:
        """전략 설명 반환"""
        descriptions = {}
        for strategy_name, strategy in self.strategies.items():
            descriptions[strategy_name] = {
                'name': strategy.strategy_name,
                'description': strategy.description
            }
        return descriptions
    
    def generate_recommendations(self, stocks_data: List[Dict], top_n: int = 5, 
                               ai_analysis: Dict = None) -> List[Dict]:
        """최종 투자 추천 생성"""
        try:
            # 모든 전략으로 분석
            analyzed_stocks = self.analyze_all_strategies(stocks_data)
            
            # 종합 점수 계산
            for stock in analyzed_stocks:
                composite_result = self.calculate_composite_score(stock)
                stock.update(composite_result)
                
                # AI 분석 결과가 있으면 추가
                if ai_analysis and stock.get('symbol') in ai_analysis:
                    stock['ai_analysis'] = ai_analysis[stock['symbol']]
            
            # 상위 추천 종목 선별
            recommendations = self.get_top_recommendations(analyzed_stocks, top_n)
            
            # 추가 정보 보강
            for rec in recommendations:
                rec['recommendation_reason'] = self._generate_recommendation_reason(rec)
                rec['risk_level'] = self._assess_risk_level(rec)
                rec['investment_horizon'] = self._suggest_investment_horizon(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"추천 생성 오류: {e}")
            return []
    
    def _generate_recommendation_reason(self, stock: Dict) -> str:
        """추천 이유 생성"""
        try:
            reasons = []
            
            # 주요 강점 찾기
            if stock.get('composite_score', 0) >= 80:
                reasons.append("종합점수 우수")
            
            if stock.get('confidence_score', 0) >= 70:
                reasons.append("높은 신뢰도")
            
            # 지배적 전략 기반 이유
            dominant_strategy = stock.get('dominant_strategy', '')
            if dominant_strategy == 'buffett':
                reasons.append("가치투자 매력")
            elif dominant_strategy == 'lynch':
                reasons.append("성장성 우수")
            elif dominant_strategy == 'greenblatt':
                reasons.append("퀀트 지표 양호")
            
            return ", ".join(reasons) if reasons else "종합 분석 결과"
            
        except Exception as e:
            logger.error(f"추천 이유 생성 오류: {e}")
            return "분석 완료"
    
    def _assess_risk_level(self, stock: Dict) -> str:
        """위험 수준 평가"""
        try:
            risk_score = stock.get('risk_score', 50)
            
            if risk_score <= 30:
                return "낮음"
            elif risk_score <= 50:
                return "보통"
            elif risk_score <= 70:
                return "높음"
            else:
                return "매우 높음"
                
        except Exception as e:
            logger.error(f"위험 수준 평가 오류: {e}")
            return "보통"
    
    def _suggest_investment_horizon(self, stock: Dict) -> str:
        """투자 기간 제안"""
        try:
            investment_style = stock.get('investment_style', '')
            
            if investment_style in ['가치투자', '패시브']:
                return "장기 (1년 이상)"
            elif investment_style in ['성장투자', '퀀트']:
                return "중기 (6개월-1년)"
            elif investment_style in ['기술적분석', '시스템매매']:
                return "단기 (3개월 이하)"
            else:
                return "중기 (6개월-1년)"
                
        except Exception as e:
            logger.error(f"투자 기간 제안 오류: {e}")
            return "중기 (6개월-1년)" 