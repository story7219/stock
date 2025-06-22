#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔮 조엘 그린블라트 CAT 전략 - 마법공식 실전 적용
AI 종목분석기 연동 버전
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics

logger = logging.getLogger(__name__)

class GreenblattCatStrategy(BaseStrategy):
    """조엘 그린블라트 CAT 전략 - 마법공식 실전 최적화"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "조엘 그린블라트 CAT"
        self.description = "마법공식 실전 적용 - PER, ROIC 순위화 점수 합산"
        
        # 실전 적용 가중치
        self.weights = {
            'earnings_yield_rank': 0.35,    # 수익률 순위 (1/PER)
            'roic_rank': 0.35,             # 자본수익률 순위
            'combined_rank': 0.20,         # 결합 순위
            'quality_filter': 0.10         # 품질 필터
        }
        
        # 마법공식 기준
        self.criteria = {
            'min_market_cap': 500,      # 최소 시가총액 500억
            'min_roic': 10,            # 최소 ROIC 10%
            'max_per': 25,             # 최대 PER 25배
            'min_revenue': 1000        # 최소 매출 1000억
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """그린블라트 CAT 전략 분석"""
        try:
            metrics = get_financial_metrics(stock)
            scores = {}
            analysis_details = {}
            
            # 기본 필터링 통과 여부
            filter_pass = self._apply_quality_filter(metrics)
            analysis_details['filter_pass'] = filter_pass
            
            if not filter_pass:
                return self._create_filtered_out_result()
            
            # 수익률 순위 (Earnings Yield = 1/PER)
            scores['earnings_yield_rank'] = self._score_earnings_yield(metrics)
            
            # ROIC 순위
            scores['roic_rank'] = self._score_roic(metrics)
            
            # 결합 순위 (마법공식 핵심)
            scores['combined_rank'] = self._score_combined_rank(metrics, scores)
            
            # 품질 필터
            scores['quality_filter'] = self._score_quality_filter(metrics)
            
            # 가중 평균 계산
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            total_score = min(max(total_score, 0), 100)
            
            # 마법공식 실전 팁 생성
            practical_tips = self._generate_practical_tips(metrics, scores)
            analysis_details['practical_tips'] = practical_tips
            
            # 투자 판단
            investment_decision = self._make_investment_decision(total_score)
            
            # 핵심 포인트
            key_points = self._extract_key_points(metrics, scores)
            
            return StrategyResult(
                total_score=total_score,
                scores=scores,
                strategy_name=self.strategy_name,
                investment_decision=investment_decision,
                key_points=key_points,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"그린블라트 CAT 전략 분석 오류: {e}")
            return self._create_error_result()
    
    def _apply_quality_filter(self, metrics: Dict) -> bool:
        """품질 필터링 적용"""
        # 시가총액 필터
        market_cap = metrics.get('market_cap', 0)
        if market_cap < self.criteria['min_market_cap'] * 100000000:  # 억원 단위
            return False
        
        # PER 필터 (너무 높으면 제외)
        per = metrics.get('per', 0)
        if per <= 0 or per > self.criteria['max_per']:
            return False
        
        # ROE 기본 필터 (ROIC 대신 ROE 사용)
        roe = metrics.get('roe', 0)
        if roe < self.criteria['min_roic']:
            return False
        
        return True
    
    def _score_earnings_yield(self, metrics: Dict) -> float:
        """수익률 순위 점수화 (1/PER)"""
        per = metrics.get('per', 0)
        
        if per <= 0:
            return 0
        
        earnings_yield = 100 / per  # 수익률 = 1/PER * 100
        
        # 수익률이 높을수록 좋음
        if earnings_yield >= 15:      # PER 6.67 이하
            return 100
        elif earnings_yield >= 12:    # PER 8.33 이하
            return 90
        elif earnings_yield >= 10:    # PER 10 이하
            return 80
        elif earnings_yield >= 8:     # PER 12.5 이하
            return 70
        elif earnings_yield >= 6:     # PER 16.67 이하
            return 60
        elif earnings_yield >= 5:     # PER 20 이하
            return 50
        elif earnings_yield >= 4:     # PER 25 이하
            return 40
        else:
            return 20
    
    def _score_roic(self, metrics: Dict) -> float:
        """ROIC 순위 점수화 (ROE로 대체)"""
        roe = metrics.get('roe', 0)  # ROIC 대신 ROE 사용
        
        # ROE가 높을수록 좋음
        if roe >= 30:
            return 100
        elif roe >= 25:
            return 90
        elif roe >= 20:
            return 80
        elif roe >= 18:
            return 75
        elif roe >= 15:
            return 70
        elif roe >= 12:
            return 60
        elif roe >= 10:
            return 50
        elif roe >= 8:
            return 40
        else:
            return 20
    
    def _score_combined_rank(self, metrics: Dict, scores: Dict) -> float:
        """결합 순위 점수화 (마법공식 핵심)"""
        earnings_yield_score = scores.get('earnings_yield_rank', 0)
        roic_score = scores.get('roic_rank', 0)
        
        # 두 점수의 평균 (마법공식은 순위 합산이지만 점수로 대체)
        combined_score = (earnings_yield_score + roic_score) / 2
        
        # 결합 점수 보너스 (둘 다 높으면 추가 점수)
        if earnings_yield_score >= 80 and roic_score >= 80:
            combined_score = min(combined_score + 15, 100)
        elif earnings_yield_score >= 70 and roic_score >= 70:
            combined_score = min(combined_score + 10, 100)
        elif earnings_yield_score >= 60 and roic_score >= 60:
            combined_score = min(combined_score + 5, 100)
        
        return combined_score
    
    def _score_quality_filter(self, metrics: Dict) -> float:
        """품질 필터 점수화"""
        score = 50
        
        # 재무 건전성
        debt_ratio = metrics.get('debt_ratio', 0)
        if debt_ratio <= 30:
            score += 25
        elif debt_ratio <= 50:
            score += 20
        elif debt_ratio <= 70:
            score += 15
        elif debt_ratio > 100:
            score -= 20
        
        # 성장성
        profit_growth = metrics.get('profit_growth', 0)
        if profit_growth >= 20:
            score += 20
        elif profit_growth >= 10:
            score += 15
        elif profit_growth >= 5:
            score += 10
        elif profit_growth < 0:
            score -= 15
        
        # 안정성 (시가총액)
        market_cap = metrics.get('market_cap', 0)
        market_cap_billion = market_cap / 100000000 if market_cap else 0
        if market_cap_billion >= 5000:
            score += 15
        elif market_cap_billion >= 1000:
            score += 10
        elif market_cap_billion >= 500:
            score += 5
        
        return min(max(score, 0), 100)
    
    def _generate_practical_tips(self, metrics: Dict, scores: Dict) -> List[str]:
        """마법공식 실전 적용 팁 생성"""
        tips = []
        
        per = metrics.get('per', 0)
        roe = metrics.get('roe', 0)
        
        # Python/Excel 활용 팁
        if scores['earnings_yield_rank'] >= 80 and scores['roic_rank'] >= 80:
            tips.append("🐍 Python으로 PER, ROE 순위화 후 상위 20개 종목 선별")
        
        # 퀀트 사이트 활용 팁
        if scores['combined_rank'] >= 75:
            tips.append("📊 네이버 증시 스크리너로 PER < 15, ROE > 15% 조건 검색")
        
        # 실전 포트폴리오 팁
        if per <= 12 and roe >= 15:
            tips.append("💼 마법공식 조건 충족, 포트폴리오 5-10% 비중 배분")
        
        # 리밸런싱 팁
        if scores['quality_filter'] >= 70:
            tips.append("🔄 연 1회 리밸런싱, 조건 이탈 종목 교체")
        
        # 장기투자 팁
        tips.append("⏰ 최소 3년 이상 장기 보유, 단기 변동성 무시")
        
        return tips
    
    def _make_investment_decision(self, total_score):
        """투자 판단"""
        if total_score >= 85:
            return "🟢 강력 매수 - 완벽한 마법공식"
        elif total_score >= 75:
            return "🔵 매수 - 우수한 가치주"
        elif total_score >= 65:
            return "🟡 관심 - 추가 검토 필요"
        elif total_score >= 55:
            return "⚪ 중립 - 조건 재확인"
        else:
            return "🔴 제외 - 기준 미달"
    
    def _extract_key_points(self, metrics: Dict, scores: Dict) -> List[str]:
        """핵심 포인트 추출"""
        points = []
        
        per = metrics.get('per', 0)
        roe = metrics.get('roe', 0)
        
        if per <= 12:
            points.append(f"✅ 낮은 PER: {per:.1f}배")
        
        if roe >= 15:
            points.append(f"✅ 높은 ROE: {roe:.1f}%")
        
        if scores.get('combined_rank', 0) >= 80:
            points.append("✅ 마법공식 상위 순위")
        
        if scores.get('earnings_yield_rank', 0) >= 80:
            points.append(f"✅ 높은 수익률: {100/per:.1f}%")
        
        if scores.get('quality_filter', 0) >= 70:
            points.append("✅ 우수한 재무 품질")
        
        # 실전 정보
        market_cap = metrics.get('market_cap', 0)
        market_cap_billion = market_cap / 100000000 if market_cap else 0
        if market_cap_billion >= 1000:
            points.append(f"✅ 대형주 안정성: {market_cap_billion:.0f}억")
        
        return points[:5]
    
    def _create_filtered_out_result(self):
        """필터링 탈락 결과"""
        return StrategyResult(
            total_score=0.0,
            scores={key: 0.0 for key in self.weights.keys()},
            strategy_name=self.strategy_name,
            investment_decision="🔴 필터링 탈락",
            key_points=["⚠️ 마법공식 기본 조건 미충족"],
            analysis_details={"filter_pass": False}
        )
    
    def _create_error_result(self):
        """오류 결과 생성"""
        return StrategyResult(
            total_score=0.0,
            scores={key: 0.0 for key in self.weights.keys()},
            strategy_name=self.strategy_name,
            investment_decision="분석 불가",
            key_points=["분석 중 오류 발생"],
            analysis_details={"error": "분석 실패"}
        )
    
    def get_ai_analyzer_integration(self) -> Dict[str, str]:
        """AI 종목분석기 연동 가이드"""
        return {
            "필터링_조건": "PER < 25, ROE > 10%, 시가총액 > 500억",
            "순위화_방법": "종목마다 PER, ROIC 순위화 → 점수 합산 → 상위 20개",
            "활용_도구": "Python, Excel, 네이버 증시 스크리너, 퀀트 사이트",
            "포트폴리오": "20-30개 종목 균등 분산, 각 3-5% 비중",
            "리밸런싱": "연 1회, 조건 이탈 종목 교체",
            "투자_기간": "최소 3년 이상 장기 보유"
        }
    
    def get_screening_guide(self) -> Dict[str, Any]:
        """스크리닝 가이드"""
        return {
            "1단계_필터": {
                "시가총액": "> 500억원",
                "PER": "5-25배",
                "ROE": "> 10%",
                "부채비율": "< 100%"
            },
            "2단계_순위화": {
                "수익률_순위": "1/PER 높은 순",
                "ROIC_순위": "ROE 높은 순",
                "결합_순위": "두 순위 합산"
            },
            "3단계_선별": {
                "상위_종목": "결합 순위 상위 20-30개",
                "균등_투자": "각 종목 동일 비중",
                "정기_교체": "연 1회 리밸런싱"
            }
        } 