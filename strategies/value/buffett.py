"""
워렌 버핏 (Warren Buffett) 투자 전략

버크셔 해서웨이 회장, 세계 최고의 가치투자자
- 저평가된 우량기업 발굴
- 경쟁우위(해자) 보유 기업 선호
- 장기 보유 전략
- "가격은 당신이 지불하는 것, 가치는 당신이 얻는 것"
"""

import logging
from typing import Dict, List
from ..common import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)

class BuffettStrategy(BaseStrategy):
    """워렌 버핏 가치투자 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "워렌 버핏 (Warren Buffett)"
        self.description = "가치투자와 우량기업 장기보유 전략"
        
        # 가중치 설정 (총합 100%)
        self.weights = {
            'competitive_moat': 30,      # 경쟁 우위 (해자)
            'management_quality': 25,    # 경영진 품질
            'financial_strength': 20,    # 재무 건전성
            'growth_prospects': 15,      # 성장 전망
            'valuation_attractiveness': 10  # 밸류에이션 매력도
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """버핏 전략으로 주식 분석"""
        try:
            scores = {}
            analysis_details = {}
            
            # 1. 경쟁 우위 분석 (30%)
            moat_score, moat_analysis = self._analyze_competitive_moat(stock)
            scores['competitive_moat'] = moat_score
            analysis_details['competitive_moat'] = moat_analysis
            
            # 2. 경영진 품질 분석 (25%)
            management_score, management_analysis = self._analyze_management_quality(stock)
            scores['management_quality'] = management_score
            analysis_details['management_quality'] = management_analysis
            
            # 3. 재무 건전성 분석 (20%)
            financial_score, financial_analysis = self._analyze_financial_strength(stock)
            scores['financial_strength'] = financial_score
            analysis_details['financial_strength'] = financial_analysis
            
            # 4. 성장 전망 분석 (15%)
            growth_score, growth_analysis = self._analyze_growth_prospects(stock)
            scores['growth_prospects'] = growth_score
            analysis_details['growth_prospects'] = growth_analysis
            
            # 5. 밸류에이션 매력도 분석 (10%)
            valuation_score, valuation_analysis = self._analyze_valuation_attractiveness(stock)
            scores['valuation_attractiveness'] = valuation_score
            analysis_details['valuation_attractiveness'] = valuation_analysis
            
            # 총점 계산
            total_score = sum(scores[key] * self.weights[key] / 100 for key in scores)
            
            # 투자 판단
            investment_decision = self._make_investment_decision(total_score)
            
            # 핵심 포인트 추출
            key_points = self._extract_key_points(scores, analysis_details)
            
            return StrategyResult(
                total_score=total_score,
                scores=scores,
                strategy_name=self.strategy_name,
                investment_decision=investment_decision,
                key_points=key_points,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"버핏 전략 분석 중 오류: {e}")
            return self._create_error_result()
    
    def _analyze_competitive_moat(self, stock) -> tuple:
        """경쟁 우위 (해자) 분석"""
        try:
            score = 50
            analysis = {}
            
            # 브랜드 파워
            if hasattr(stock, 'brand_strength'):
                brand_score = stock.brand_strength * 25
                score += brand_score
                analysis['brand_strength'] = stock.brand_strength
            
            # 시장 지배력
            if hasattr(stock, 'market_leadership'):
                market_score = stock.market_leadership * 25
                score += market_score
                analysis['market_leadership'] = stock.market_leadership
            
            # 가격 결정력
            if hasattr(stock, 'pricing_power'):
                pricing_score = stock.pricing_power * 20
                score += pricing_score
                analysis['pricing_power'] = stock.pricing_power
            
            # 진입 장벽
            if hasattr(stock, 'competitive_moat'):
                moat_score = stock.competitive_moat * 30
                score += moat_score
                analysis['competitive_moat'] = stock.competitive_moat
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"경쟁 우위 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_management_quality(self, stock) -> tuple:
        """경영진 품질 분석"""
        try:
            score = 50
            analysis = {}
            
            # 경영진 품질
            if hasattr(stock, 'management_quality'):
                management_score = stock.management_quality * 40
                score += management_score
                analysis['management_quality'] = stock.management_quality
            
            # ROE (자기자본이익률)
            if hasattr(stock, 'roe'):
                if stock.roe > 15:  # 15% 이상 우수
                    roe_score = min(30, (stock.roe / 15) * 30)
                else:
                    roe_score = (stock.roe / 15) * 20
                score += roe_score
                analysis['roe'] = stock.roe
            
            # ROA (총자산이익률)
            if hasattr(stock, 'roa'):
                if stock.roa > 5:  # 5% 이상 우수
                    roa_score = min(30, (stock.roa / 5) * 30)
                else:
                    roa_score = (stock.roa / 5) * 20
                score += roa_score
                analysis['roa'] = stock.roa
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"경영진 품질 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_financial_strength(self, stock) -> tuple:
        """재무 건전성 분석"""
        try:
            score = 50
            analysis = {}
            
            # 재무 건전성
            if hasattr(stock, 'financial_strength'):
                financial_score = stock.financial_strength * 30
                score += financial_score
                analysis['financial_strength'] = stock.financial_strength
            
            # 부채비율 (낮을수록 좋음)
            if hasattr(stock, 'debt_ratio'):
                if stock.debt_ratio < 30:  # 30% 미만 우수
                    debt_score = 30
                elif stock.debt_ratio < 50:  # 50% 미만 양호
                    debt_score = 20
                else:  # 50% 이상 위험
                    debt_score = max(0, 10 - (stock.debt_ratio - 50) * 0.2)
                score += debt_score
                analysis['debt_ratio'] = stock.debt_ratio
            
            # 유동비율 (높을수록 좋음)
            if hasattr(stock, 'current_ratio'):
                if stock.current_ratio > 2.0:  # 2.0 이상 우수
                    current_score = 20
                elif stock.current_ratio > 1.5:  # 1.5 이상 양호
                    current_score = 15
                else:  # 1.5 미만 위험
                    current_score = max(0, stock.current_ratio * 10)
                score += current_score
                analysis['current_ratio'] = stock.current_ratio
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"재무 건전성 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_growth_prospects(self, stock) -> tuple:
        """성장 전망 분석"""
        try:
            score = 50
            analysis = {}
            
            # 성장 전망
            if hasattr(stock, 'growth_prospects'):
                growth_score = stock.growth_prospects * 40
                score += growth_score
                analysis['growth_prospects'] = stock.growth_prospects
            
            # 수익 성장률 (안정적 성장 선호)
            if hasattr(stock, 'earnings_growth_rate'):
                growth_rate = stock.earnings_growth_rate
                if 5 <= growth_rate <= 15:  # 5-15% 안정적 성장
                    earnings_score = 30
                elif 0 <= growth_rate < 5:  # 저성장
                    earnings_score = growth_rate * 6
                elif 15 < growth_rate <= 25:  # 고성장 (약간 감점)
                    earnings_score = 25
                else:  # 극단적 성장률
                    earnings_score = 10
                score += earnings_score
                analysis['earnings_growth_rate'] = growth_rate
            
            # 매출 성장률
            if hasattr(stock, 'revenue_growth_rate'):
                revenue_rate = stock.revenue_growth_rate
                if revenue_rate > 0:
                    revenue_score = min(30, revenue_rate * 3)
            else:
                    revenue_score = 0
                score += revenue_score
                analysis['revenue_growth_rate'] = revenue_rate
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"성장 전망 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _analyze_valuation_attractiveness(self, stock) -> tuple:
        """밸류에이션 매력도 분석"""
        try:
            score = 50
            analysis = {}
            
            # 밸류에이션 매력도
            if hasattr(stock, 'valuation_attractiveness'):
                valuation_score = stock.valuation_attractiveness * 40
                score += valuation_score
                analysis['valuation_attractiveness'] = stock.valuation_attractiveness
            
            # PER (주가수익비율)
            if hasattr(stock, 'per'):
                if 5 <= stock.per <= 15:  # 적정 PER
                    per_score = 30
                elif stock.per < 5:  # 너무 낮음 (위험 신호)
                    per_score = 15
                elif 15 < stock.per <= 25:  # 약간 높음
                    per_score = 20
                else:  # 너무 높음
                    per_score = 5
                score += per_score
                analysis['per'] = stock.per
            
            # PBR (주가순자산비율)
            if hasattr(stock, 'pbr'):
                if stock.pbr < 1.0:  # 1.0 미만 저평가
                    pbr_score = 30
                elif stock.pbr < 2.0:  # 2.0 미만 적정
                    pbr_score = 20
                else:  # 2.0 이상 고평가
                    pbr_score = max(0, 10 - (stock.pbr - 2) * 2)
                score += pbr_score
                analysis['pbr'] = stock.pbr
            
            return min(max(score, 0), 100), analysis
            
        except Exception as e:
            logger.error(f"밸류에이션 분석 오류: {e}")
            return 50, {"error": str(e)}
    
    def _make_investment_decision(self, total_score):
        """투자 판단"""
        if total_score >= 80:
            return "강력 매수"
        elif total_score >= 70:
            return "매수"
        elif total_score >= 60:
            return "보유"
        elif total_score >= 50:
            return "관망"
        else:
            return "매도"
    
    def _extract_key_points(self, scores, analysis_details):
        """핵심 포인트 추출"""
        points = []
        
        # 최고 점수 요소
        best_factor = max(scores.items(), key=lambda x: x[1])
        points.append(f"최고 강점: {best_factor[0]} ({best_factor[1]:.1f}점)")
        
        # 개선 필요 요소
        worst_factor = min(scores.items(), key=lambda x: x[1])
        if worst_factor[1] < 60:
            points.append(f"개선 필요: {worst_factor[0]} ({worst_factor[1]:.1f}점)")
        
        # 버핏 스타일 특징
        if scores.get('competitive_moat', 0) > 70:
            points.append("경쟁 우위(해자) 우수 - 버핏 선호 스타일")
        
        if scores.get('financial_strength', 0) > 70:
            points.append("재무 건전성 우수 - 안정적 투자 대상")
        
        return points
    
    def _create_error_result(self):
        """오류 결과 생성"""
        return StrategyResult(
            total_score=0,
            scores={},
            strategy_name=self.strategy_name,
            investment_decision="분석 불가",
            key_points=["분석 중 오류 발생"],
            analysis_details={"error": "분석 실패"}
        ) 