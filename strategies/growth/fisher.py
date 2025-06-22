#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 필립 피셔 (Philip Fisher) 투자 전략
성장주 발굴의 아버지
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics
from data.processed.data_cleaner import CleanedStockData

logger = logging.getLogger(__name__)

class FisherStrategy(BaseStrategy):
    """필립 피셔의 성장주 투자 전략"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "Philip Fisher Growth"
        self.description = "15가지 질문 기반 질적 성장주 분석"
        
        # 필립 피셔 15가지 질문 가중치
        self.weights = {
            'growth_potential': 0.25,    # 성장 잠재력
            'management_quality': 0.20,  # 경영진 품질
            'competitive_advantage': 0.20, # 경쟁 우위
            'research_development': 0.15, # 연구개발
            'financial_strength': 0.20   # 재무 건전성
        }
    
    def analyze_stock(self, stock) -> StrategyResult:
        """필립 피셔 성장주 분석"""
        try:
            # 재무 지표 추출
            metrics = get_financial_metrics(stock)
            
            scores = {}
            analysis_details = {}
            
            # 성장 잠재력 분석
            scores['growth_potential'] = self._analyze_growth_potential(metrics)
            analysis_details['growth_potential'] = scores['growth_potential']
            
            # 경영진 품질 분석
            scores['management_quality'] = self._analyze_management_quality(metrics)
            analysis_details['management_quality'] = scores['management_quality']
            
            # 경쟁 우위 분석
            scores['competitive_advantage'] = self._analyze_competitive_advantage(metrics)
            analysis_details['competitive_advantage'] = scores['competitive_advantage']
            
            # 연구개발 분석
            scores['research_development'] = self._analyze_rd_capability(metrics)
            analysis_details['research_development'] = scores['research_development']
            
            # 재무 건전성 분석
            scores['financial_strength'] = self._analyze_financial_strength(metrics)
            analysis_details['financial_strength'] = scores['financial_strength']
            
            # 가중 평균 계산
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            # 투자 판단
            investment_decision = self._get_investment_decision(total_score)
            
            # 핵심 포인트 추출
            key_points = self._get_key_points(metrics, analysis_details)
            
            logger.debug(f"Fisher 성장주 분석 ({metrics.get('symbol', 'Unknown')}): {total_score:.1f}")
            
            return StrategyResult(
                total_score=min(max(total_score, 0), 100),
                scores=scores,
                strategy_name=self.strategy_name,
                investment_decision=investment_decision,
                key_points=key_points,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"Fisher 전략 분석 오류: {e}")
            return self._create_error_result()
    
    def _analyze_growth_potential(self, metrics: Dict) -> float:
        """성장 잠재력 분석 - 피셔의 핵심"""
        score = 0.0
        
        # 매출 성장률 (지속적 성장)
        revenue_growth = metrics.get('revenue_growth')
        if revenue_growth:
            if revenue_growth >= 20:
                score += 30  # 뛰어난 성장
            elif revenue_growth >= 15:
                score += 25
            elif revenue_growth >= 10:
                score += 20
            elif revenue_growth >= 5:
                score += 15
            elif revenue_growth >= 0:
                score += 10
            else:
                score -= 10  # 매출 감소는 부정적
        
        # 이익 성장률 (수익성 개선)
        profit_growth = metrics.get('profit_growth')
        if profit_growth:
            if profit_growth >= 25:
                score += 30  # 매우 높은 이익 성장
            elif profit_growth >= 20:
                score += 25
            elif profit_growth >= 15:
                score += 20
            elif profit_growth >= 10:
                score += 15
            elif profit_growth >= 5:
                score += 10
            else:
                score -= 15  # 이익 감소
        
        # 시장 지위 (시가총액)
        market_cap = metrics.get('market_cap')
        if market_cap:
            market_cap_billion = market_cap / 100000000
            # 중견기업이 성장 잠재력 높음
            if 100 <= market_cap_billion <= 1000:
                score += 20  # 최적 성장 구간
            elif 50 <= market_cap_billion <= 2000:
                score += 15
            elif market_cap_billion >= 2000:
                score += 10  # 대형주는 성장 한계
            elif market_cap_billion >= 10:
                score += 5
        
        # ROE 성장성 (효율성 개선)
        roe = metrics.get('roe')
        if roe:
            if roe >= 20:
                score += 20  # 뛰어난 자본 효율성
            elif roe >= 15:
                score += 15
            elif roe >= 10:
                score += 10
            elif roe >= 5:
                score += 5
        
        return min(score, 100)
    
    def _analyze_management_quality(self, metrics: Dict) -> float:
        """경영진 품질 분석"""
        score = 50.0  # 기본 점수
        
        # 수익성 관리 능력 (ROE)
        roe = metrics.get('roe')
        if roe:
            if roe >= 25:
                score += 25  # 뛰어난 경영 능력
            elif roe >= 20:
                score += 20
            elif roe >= 15:
                score += 15
            elif roe >= 10:
                score += 10
            elif roe < 5:
                score -= 15  # 경영 능력 의문
        
        # 자본 배분 능력 (부채 관리)
        debt_ratio = metrics.get('debt_ratio')
        if debt_ratio:
            if debt_ratio <= 20:
                score += 20  # 보수적 재무 관리
            elif debt_ratio <= 40:
                score += 15
            elif debt_ratio <= 60:
                score += 10
            elif debt_ratio <= 100:
                score += 5
            else:
                score -= 20  # 과도한 레버리지
        
        # 성장 관리 능력 (일관된 성장)
        if metrics.get('profit_growth') and metrics.get('revenue_growth'):
            # 매출과 이익이 함께 성장
            if metrics.get('profit_growth') >= 10 and metrics.get('revenue_growth') >= 5:
                score += 15  # 균형잡힌 성장 관리
            elif metrics.get('profit_growth') >= 5:
                score += 10
        
        # 효율성 관리 (유동비율)
        current_ratio = metrics.get('current_ratio')
        if current_ratio:
            if 1.5 <= current_ratio <= 3.0:
                score += 15  # 적정 유동성 관리
            elif 1.0 <= current_ratio <= 4.0:
                score += 10
            elif current_ratio < 1.0:
                score -= 15  # 유동성 위험
        
        return min(max(score, 0), 100)
    
    def _analyze_competitive_advantage(self, metrics: Dict) -> float:
        """경쟁 우위 분석"""
        score = 50.0
        
        # 수익성 우위 (ROE 기반)
        roe = metrics.get('roe')
        if roe:
            if roe >= 20:
                score += 25  # 강한 경쟁 우위
            elif roe >= 15:
                score += 20
            elif roe >= 10:
                score += 15
            elif roe >= 5:
                score += 10
        
        # 시장 지위 (규모의 경제)
        market_cap = metrics.get('market_cap')
        if market_cap:
            market_cap_billion = market_cap / 100000000
            if market_cap_billion >= 1000:
                score += 20  # 시장 지배력
            elif market_cap_billion >= 500:
                score += 15
            elif market_cap_billion >= 100:
                score += 10
        
        # 수익성 안정성 (변동성 역산)
        volatility = metrics.get('volatility')
        if volatility:
            if volatility <= 15:
                score += 15  # 안정적 사업 모델
            elif volatility <= 25:
                score += 10
            elif volatility <= 35:
                score += 5
            elif volatility >= 50:
                score -= 10  # 불안정한 사업
        
        # 성장 지속성
        if metrics.get('profit_growth') and metrics.get('revenue_growth'):
            if metrics.get('revenue_growth') >= 10 and metrics.get('profit_growth') >= 15:
                score += 15  # 지속 가능한 성장
            elif metrics.get('revenue_growth') >= 5 and metrics.get('profit_growth') >= 10:
                score += 10
        
        # 섹터 우위 (성장 섹터)
        growth_sectors = ['Technology', 'Healthcare', 'IT', '바이오', '반도체', '소프트웨어']
        sector = metrics.get('sector')
        if sector and any(gs in sector for gs in growth_sectors):
            score += 10  # 성장 섹터 우위
        
        return min(max(score, 0), 100)
    
    def _analyze_rd_capability(self, metrics: Dict) -> float:
        """연구개발 능력 분석"""
        score = 50.0
        
        # 혁신 섹터 (R&D 집약적)
        innovation_sectors = ['Technology', 'Healthcare', 'Biotechnology', 'Software', 
                             '바이오', '반도체', '소프트웨어', '제약', 'IT']
        
        sector = metrics.get('sector')
        if sector:
            sector_match = any(gs in sector for gs in innovation_sectors)
            if sector_match:
                score += 30  # 혁신 중심 섹터
                
                # 혁신 섹터에서의 성장성
                if metrics.get('profit_growth') and metrics.get('profit_growth') >= 15:
                    score += 15  # R&D 성과
                if metrics.get('revenue_growth') and metrics.get('revenue_growth') >= 10:
                    score += 10
            else:
                # 전통 섹터도 혁신 가능
                if metrics.get('profit_growth') and metrics.get('profit_growth') >= 20:
                    score += 15  # 전통 섹터의 혁신
        
        # 투자 여력 (R&D 투자 능력)
        roe = metrics.get('roe')
        debt_ratio = metrics.get('debt_ratio')
        if roe and debt_ratio:
            if roe >= 15 and debt_ratio <= 50:
                score += 20  # 충분한 투자 여력
            elif roe >= 10 and debt_ratio <= 70:
                score += 15
        
        # 미래 성장 동력
        pe_ratio = metrics.get('pe_ratio')
        if pe_ratio:
            # 적정한 밸류에이션은 미래 성장 기대
            if 15 <= pe_ratio <= 30:
                score += 15  # 성장 기대치 반영
            elif 10 <= pe_ratio <= 40:
                score += 10
        
        return min(max(score, 0), 100)
    
    def _analyze_financial_strength(self, metrics: Dict) -> float:
        """재무 건전성 분석"""
        score = 0.0
        
        # 부채 건전성
        debt_ratio = metrics.get('debt_ratio')
        if debt_ratio:
            if debt_ratio <= 20:
                score += 25  # 매우 건전
            elif debt_ratio <= 40:
                score += 20
            elif debt_ratio <= 60:
                score += 15
            elif debt_ratio <= 100:
                score += 10
            else:
                score -= 10  # 위험 수준
        
        # 유동성
        current_ratio = metrics.get('current_ratio')
        if current_ratio:
            if current_ratio >= 2.0:
                score += 20  # 충분한 유동성
            elif current_ratio >= 1.5:
                score += 15
            elif current_ratio >= 1.0:
                score += 10
            else:
                score -= 15  # 유동성 부족
        
        # 수익성
        roe = metrics.get('roe')
        if roe:
            if roe >= 20:
                score += 25  # 뛰어난 수익성
            elif roe >= 15:
                score += 20
            elif roe >= 10:
                score += 15
            elif roe >= 5:
                score += 10
            else:
                score -= 10  # 낮은 수익성
        
        # 성장성 (미래 현금흐름)
        profit_growth = metrics.get('profit_growth')
        if profit_growth:
            if profit_growth >= 15:
                score += 20  # 강한 성장
            elif profit_growth >= 10:
                score += 15
            elif profit_growth >= 5:
                score += 10
            elif profit_growth < 0:
                score -= 15  # 성장 둔화
        
        # 배당 능력 (잉여 현금 창출)
        dividend_yield = metrics.get('dividend_yield')
        if dividend_yield:
            if 1 <= dividend_yield <= 4:
                score += 10  # 적정 배당
            elif dividend_yield > 6:
                score -= 5  # 과도한 배당 (위험 신호)
        
        return min(max(score, 0), 100)
    
    def get_strategy_summary(self, stock: CleanedStockData) -> Dict[str, Any]:
        """전략 요약 정보"""
        analysis = self.analyze_stock_detailed(stock)
        
        return {
            "전략명": self.strategy_name,
            "총점": f"{analysis['total_score']:.1f}/100",
            "성장주분석점수": {
                "성장잠재력": f"{analysis['scores']['growth_potential']:.1f}",
                "경영진품질": f"{analysis['scores']['management_quality']:.1f}",
                "경쟁우위": f"{analysis['scores']['competitive_advantage']:.1f}",
                "연구개발": f"{analysis['scores']['research_development']:.1f}",
                "재무건전성": f"{analysis['scores']['financial_strength']:.1f}"
            },
            "투자판단": self._get_investment_decision(analysis['total_score']),
            "핵심포인트": self._get_key_points(stock, analysis)
        }
    
    def _get_investment_decision(self, score: float) -> str:
        """투자 판단"""
        if score >= 80:
            return "🟢 강력매수 - 최고 성장주"
        elif score >= 70:
            return "🔵 매수 - 우수 성장주"
        elif score >= 60:
            return "🟡 관심 - 잠재 성장주"
        elif score >= 50:
            return "⚪ 중립 - 성장성 검토 필요"
        else:
            return "🔴 회피 - 성장주 부적합"
    
    def _get_key_points(self, metrics: Dict, analysis: Dict[str, Any]) -> List[str]:
        """핵심 포인트"""
        points = []
        scores = analysis['scores']
        
        if scores['growth_potential'] >= 70:
            points.append("✅ 뛰어난 성장 잠재력")
        if scores['management_quality'] >= 70:
            points.append("✅ 우수한 경영진")
        if scores['competitive_advantage'] >= 70:
            points.append("✅ 강한 경쟁 우위")
        if scores['research_development'] >= 70:
            points.append("✅ 혁신 역량 우수")
            
        profit_growth = metrics.get('profit_growth')
        if profit_growth and profit_growth >= 20:
            points.append("✅ 고성장 기업")
            
        roe = metrics.get('roe')
        if roe and roe >= 20:
            points.append("✅ 높은 자본 효율성")
        
        if scores['growth_potential'] < 50:
            points.append("⚠️ 성장 동력 부족")
        if scores['financial_strength'] < 50:
            points.append("⚠️ 재무 건전성 우려")
            
        debt_ratio = metrics.get('debt_ratio')
        if debt_ratio and debt_ratio > 80:
            points.append("⚠️ 높은 부채 비율")
            
        if profit_growth and profit_growth < 5:
            points.append("⚠️ 성장 둔화")
        
        return points[:5]  # 최대 5개 포인트
    
    def _create_error_result(self) -> StrategyResult:
        """오류 발생 시 기본 결과 반환"""
        return StrategyResult(
            total_score=0.0,
            scores={},
            strategy_name=self.strategy_name,
            investment_decision="🔴 분석 오류",
            key_points=["⚠️ 분석 중 오류 발생"],
            analysis_details={}
        ) 