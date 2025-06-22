#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 윌리엄 오닐 CAT 전략 - CAN SLIM 실전 적용
AI 종목분석기 연동 버전
"""

import logging
from typing import Dict, Any, Optional, List
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics

logger = logging.getLogger(__name__)

class ONeillCatStrategy(BaseStrategy):
    """윌리엄 오닐 CAT 전략 - 실전 적용 최적화"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "윌리엄 오닐 CAT"
        self.description = "CAN SLIM 실전 적용 - EPS 성장, 고ROE, 거래량 증가 필터링"
        
        # 실전 적용 가중치 (AI 종목분석기 최적화)
        self.weights = {
            'eps_growth_filter': 0.25,    # EPS 성장률 필터
            'high_roe_filter': 0.20,      # 고ROE 필터
            'volume_surge': 0.20,         # 거래량 증가
            'price_momentum': 0.15,       # 가격 모멘텀
            'sector_leadership': 0.10,    # 섹터 리더십
            'institutional_flow': 0.10    # 기관 자금 흐름
        }
        
        # 실전 필터링 기준
        self.filters = {
            'min_eps_growth': 25,     # 최소 EPS 성장률 25%
            'min_roe': 18,           # 최소 ROE 18%
            'min_volume_ratio': 1.5,  # 최소 거래량 비율 1.5배
            'min_market_cap': 1000   # 최소 시가총액 1000억
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """오닐 CAT 전략 분석 - 실전 필터링 적용"""
        try:
            metrics = get_financial_metrics(stock)
            scores = {}
            analysis_details = {}
            
            # 1단계: 기본 필터링 통과 여부
            filter_pass = self._apply_basic_filters(metrics)
            analysis_details['filter_pass'] = filter_pass
            
            if not filter_pass:
                return self._create_filtered_out_result()
            
            # 2단계: 세부 점수 계산
            scores['eps_growth_filter'] = self._score_eps_growth(metrics)
            scores['high_roe_filter'] = self._score_high_roe(metrics)
            scores['volume_surge'] = self._score_volume_surge(metrics)
            scores['price_momentum'] = self._score_price_momentum(metrics)
            scores['sector_leadership'] = self._score_sector_leadership(metrics)
            scores['institutional_flow'] = self._score_institutional_flow(metrics)
            
            # 가중 평균 계산
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            total_score = min(max(total_score, 0), 100)
            
            # 실전 적용 팁 생성
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
            logger.error(f"오닐 CAT 전략 분석 오류: {e}")
            return self._create_error_result()
    
    def _apply_basic_filters(self, metrics: Dict) -> bool:
        """기본 필터링 적용"""
        # EPS 성장률 필터
        eps_growth = metrics.get('profit_growth', 0)
        if eps_growth < self.filters['min_eps_growth']:
            return False
        
        # ROE 필터
        roe = metrics.get('roe', 0)
        if roe < self.filters['min_roe']:
            return False
        
        # 시가총액 필터
        market_cap = metrics.get('market_cap', 0)
        if market_cap < self.filters['min_market_cap'] * 100000000:  # 억원 단위
            return False
        
        return True
    
    def _score_eps_growth(self, metrics: Dict) -> float:
        """EPS 성장률 점수화"""
        eps_growth = metrics.get('profit_growth', 0)
        
        if eps_growth >= 50:
            return 100
        elif eps_growth >= 40:
            return 90
        elif eps_growth >= 30:
            return 80
        elif eps_growth >= 25:
            return 70
        else:
            return 50
    
    def _score_high_roe(self, metrics: Dict) -> float:
        """고ROE 점수화"""
        roe = metrics.get('roe', 0)
        
        if roe >= 30:
            return 100
        elif roe >= 25:
            return 90
        elif roe >= 20:
            return 80
        elif roe >= 18:
            return 70
        else:
            return 50
    
    def _score_volume_surge(self, metrics: Dict) -> float:
        """거래량 증가 점수화"""
        # 가격 모멘텀으로 거래량 증가 추정
        price_momentum = metrics.get('price_momentum_3m', 0)
        
        if price_momentum >= 30:
            return 100  # 강한 거래량 증가 추정
        elif price_momentum >= 20:
            return 85
        elif price_momentum >= 15:
            return 70
        elif price_momentum >= 10:
            return 60
        else:
            return 40
    
    def _score_price_momentum(self, metrics: Dict) -> float:
        """가격 모멘텀 점수화"""
        price_momentum = metrics.get('price_momentum_3m', 0)
        
        if price_momentum >= 25:
            return 100
        elif price_momentum >= 20:
            return 85
        elif price_momentum >= 15:
            return 70
        elif price_momentum >= 10:
            return 60
        else:
            return 30
    
    def _score_sector_leadership(self, metrics: Dict) -> float:
        """섹터 리더십 점수화"""
        market_cap = metrics.get('market_cap', 0)
        sector = metrics.get('sector', '')
        
        score = 50
        
        # 시가총액 기준 리더십
        market_cap_billion = market_cap / 100000000
        if market_cap_billion >= 10000:  # 10조원 이상
            score += 30
        elif market_cap_billion >= 5000:  # 5조원 이상
            score += 25
        elif market_cap_billion >= 1000:  # 1조원 이상
            score += 20
        
        # 성장 섹터 보너스
        growth_sectors = ['Technology', 'Healthcare', 'IT', '바이오', '반도체', '소프트웨어']
        if any(gs in sector for gs in growth_sectors):
            score += 20
        
        return min(score, 100)
    
    def _score_institutional_flow(self, metrics: Dict) -> float:
        """기관 자금 흐름 점수화"""
        market_cap = metrics.get('market_cap', 0)
        roe = metrics.get('roe', 0)
        
        score = 50
        
        # 대형주일수록 기관 선호
        market_cap_billion = market_cap / 100000000
        if market_cap_billion >= 5000:
            score += 25
        elif market_cap_billion >= 1000:
            score += 20
        elif market_cap_billion >= 500:
            score += 15
        
        # 높은 ROE는 기관 선호
        if roe >= 25:
            score += 25
        elif roe >= 20:
            score += 20
        
        return min(score, 100)
    
    def _generate_practical_tips(self, metrics: Dict, scores: Dict) -> List[str]:
        """실전 적용 팁 생성"""
        tips = []
        
        # 네이버 증시 활용 팁
        if scores['eps_growth_filter'] >= 80:
            tips.append("📊 네이버 증시 > 종목분석 > 실적 탭에서 EPS 성장률 확인")
        
        # AI 주가 분석기 활용 팁
        if scores['volume_surge'] >= 80:
            tips.append("🤖 AI 주가 분석기로 거래량 급증 종목 필터링 추천")
        
        # 실전 매매 팁
        price_momentum = metrics.get('price_momentum_3m', 0)
        if price_momentum >= 20:
            tips.append("💰 20일 고점 돌파 시 매수 타이밍, 10일선 이탈 시 손절")
        
        # 포트폴리오 관리 팁
        if len([s for s in scores.values() if s >= 70]) >= 4:
            tips.append("📈 CAN SLIM 조건 충족, 포트폴리오 20-30% 비중 고려")
        
        return tips
    
    def _make_investment_decision(self, total_score):
        """투자 판단"""
        if total_score >= 85:
            return "🟢 강력 매수 - CAN SLIM 완벽 충족"
        elif total_score >= 75:
            return "🔵 매수 - 우수한 성장주"
        elif total_score >= 65:
            return "🟡 관심 - 추가 모니터링"
        elif total_score >= 55:
            return "⚪ 중립 - 조건 재검토"
        else:
            return "🔴 제외 - 기준 미달"
    
    def _extract_key_points(self, metrics: Dict, scores: Dict) -> List[str]:
        """핵심 포인트 추출"""
        points = []
        
        eps_growth = metrics.get('profit_growth', 0)
        roe = metrics.get('roe', 0)
        price_momentum = metrics.get('price_momentum_3m', 0)
        
        if eps_growth >= 30:
            points.append(f"✅ 높은 EPS 성장률: {eps_growth:.1f}%")
        
        if roe >= 20:
            points.append(f"✅ 우수한 ROE: {roe:.1f}%")
        
        if price_momentum >= 15:
            points.append(f"✅ 강한 가격 모멘텀: {price_momentum:.1f}%")
        
        if scores.get('volume_surge', 0) >= 70:
            points.append("✅ 거래량 급증 신호")
        
        if scores.get('sector_leadership', 0) >= 70:
            points.append("✅ 섹터 리더십 보유")
        
        return points[:5]
    
    def _create_filtered_out_result(self):
        """필터링 탈락 결과"""
        return StrategyResult(
            total_score=0.0,
            scores={key: 0.0 for key in self.weights.keys()},
            strategy_name=self.strategy_name,
            investment_decision="🔴 필터링 탈락",
            key_points=["⚠️ 기본 조건 미충족"],
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
            "필터링_조건": "EPS 성장률 ≥ 25%, ROE ≥ 18%, 시가총액 ≥ 1000억",
            "활용_도구": "네이버 증시, AI 주가 분석기, 키움 HTS 조건검색",
            "매매_신호": "20일 고점 돌파 매수, 10일선 이탈 손절",
            "포트폴리오": "20-30개 종목 분산, 각 20-30% 비중",
            "리밸런싱": "분기별 조건 재검토 및 교체"
        } 