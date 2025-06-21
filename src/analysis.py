#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 최적화된 분석 엔진
- 6가지 투자대가 전략 구현
- 벡터화된 계산
- 병렬 분석 처리
- 메모리 효율적 구현
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from optimized_data_processor import StockData, AnalysisResult
from performance_core import get_performance_core, performance_monitor

# 로깅 설정
logger = logging.getLogger(__name__)

class InvestmentStrategy(Enum):
    """투자 전략 열거형"""
    WILLIAM_ONEIL = "william_oneil"
    JESSE_LIVERMORE = "jesse_livermore"
    ICHIMOKU = "ichimoku"
    WARREN_BUFFETT = "warren_buffett"
    PETER_LYNCH = "peter_lynch"
    BLACKROCK = "blackrock"

@dataclass
class StrategyWeights:
    """전략별 가중치 설정"""
    technical: float = 0.3
    fundamental: float = 0.4
    momentum: float = 0.2
    risk: float = 0.1

class OptimizedAnalysisEngine:
    """🚀 최적화된 분석 엔진"""
    
    def __init__(self):
        self.core = None
        
        # 전략별 가중치 설정
        self.strategy_weights = {
            InvestmentStrategy.WILLIAM_ONEIL: StrategyWeights(
                technical=0.4, fundamental=0.3, momentum=0.2, risk=0.1
            ),
            InvestmentStrategy.JESSE_LIVERMORE: StrategyWeights(
                technical=0.5, fundamental=0.1, momentum=0.3, risk=0.1
            ),
            InvestmentStrategy.ICHIMOKU: StrategyWeights(
                technical=0.6, fundamental=0.2, momentum=0.1, risk=0.1
            ),
            InvestmentStrategy.WARREN_BUFFETT: StrategyWeights(
                technical=0.1, fundamental=0.6, momentum=0.1, risk=0.2
            ),
            InvestmentStrategy.PETER_LYNCH: StrategyWeights(
                technical=0.2, fundamental=0.5, momentum=0.2, risk=0.1
            ),
            InvestmentStrategy.BLACKROCK: StrategyWeights(
                technical=0.2, fundamental=0.4, momentum=0.2, risk=0.2
            )
        }
        
        logger.info("✅ 최적화된 분석 엔진 초기화")
    
    async def initialize(self):
        """초기화"""
        self.core = await get_performance_core()
        logger.info("✅ 분석 엔진 초기화 완료")
    
    @performance_monitor
    async def analyze_stocks(self, stocks: List[StockData], strategy: InvestmentStrategy, top_n: int = 5) -> List[AnalysisResult]:
        """주식 분석 (병렬 처리)"""
        try:
            if not stocks:
                logger.warning("⚠️ 분석할 주식 데이터가 없습니다")
                return []
            
            logger.info(f"🎯 {strategy.value} 전략으로 {len(stocks)}개 종목 분석 시작")
            
            # 병렬 분석 실행
            tasks = [self._analyze_single_stock(stock, strategy) for stock in stocks]
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 성공한 결과만 수집
            valid_results = [
                result for result in analysis_results 
                if isinstance(result, AnalysisResult) and result.score > 0
            ]
            
            # 점수 기준 정렬 및 상위 N개 선택
            sorted_results = sorted(valid_results, key=lambda x: x.score, reverse=True)
            top_results = sorted_results[:top_n]
            
            logger.info(f"✅ 분석 완료: {len(valid_results)}/{len(stocks)} 성공, TOP {len(top_results)} 선택")
            
            return top_results
            
        except Exception as e:
            logger.error(f"❌ 주식 분석 실패: {e}")
            return []
    
    async def _analyze_single_stock(self, stock: StockData, strategy: InvestmentStrategy) -> Optional[AnalysisResult]:
        """단일 주식 분석"""
        try:
            # 캐시 확인
            cache_key = f"analysis_{stock.symbol}_{strategy.value}"
            cached_result = self.core.cache.get(cache_key)
            
            if cached_result:
                logger.debug(f"📋 분석 캐시 사용: {stock.symbol}")
                return cached_result
            
            # 새로운 분석 수행
            analysis_result = await self._perform_analysis(stock, strategy)
            
            if analysis_result and analysis_result.score > 0:
                # 캐시에 저장 (10분)
                self.core.cache.set(cache_key, analysis_result, ttl=600)
                logger.debug(f"💾 분석 결과 캐시 저장: {stock.symbol}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ {stock.symbol} 분석 실패: {e}")
            return None
    
    async def _perform_analysis(self, stock: StockData, strategy: InvestmentStrategy) -> Optional[AnalysisResult]:
        """실제 분석 수행"""
        try:
            # 데이터 품질 확인
            if stock.data_quality == "POOR":
                return None
            
            # 전략별 점수 계산
            scores = await self._calculate_strategy_scores(stock, strategy)
            
            # 가중 평균 점수 계산
            weights = self.strategy_weights[strategy]
            final_score = (
                scores['technical'] * weights.technical +
                scores['fundamental'] * weights.fundamental +
                scores['momentum'] * weights.momentum +
                scores['risk'] * weights.risk
            )
            
            # 추천 등급 결정
            recommendation = self._determine_recommendation(final_score)
            
            # 분석 이유 생성
            reason = self._generate_analysis_reason(stock, strategy, scores)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(stock, scores)
            
            return AnalysisResult(
                stock_data=stock,
                score=final_score,
                recommendation=recommendation,
                reason=reason,
                strategy=strategy.value,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"❌ {stock.symbol} 분석 수행 실패: {e}")
            return None
    
    async def _calculate_strategy_scores(self, stock: StockData, strategy: InvestmentStrategy) -> Dict[str, float]:
        """전략별 점수 계산"""
        try:
            if strategy == InvestmentStrategy.WILLIAM_ONEIL:
                return await self._william_oneil_analysis(stock)
            elif strategy == InvestmentStrategy.JESSE_LIVERMORE:
                return await self._jesse_livermore_analysis(stock)
            elif strategy == InvestmentStrategy.ICHIMOKU:
                return await self._ichimoku_analysis(stock)
            elif strategy == InvestmentStrategy.WARREN_BUFFETT:
                return await self._warren_buffett_analysis(stock)
            elif strategy == InvestmentStrategy.PETER_LYNCH:
                return await self._peter_lynch_analysis(stock)
            elif strategy == InvestmentStrategy.BLACKROCK:
                return await self._blackrock_analysis(stock)
            else:
                return {'technical': 50, 'fundamental': 50, 'momentum': 50, 'risk': 50}
                
        except Exception as e:
            logger.error(f"❌ 전략 점수 계산 실패: {e}")
            return {'technical': 0, 'fundamental': 0, 'momentum': 0, 'risk': 0}
    
    async def _william_oneil_analysis(self, stock: StockData) -> Dict[str, float]:
        """윌리엄 오닐 (CAN SLIM) 분석"""
        scores = {}
        
        # Technical: RSI, MACD 기반
        technical_score = 0
        if 30 <= stock.rsi <= 70:  # 적정 RSI 범위
            technical_score += 40
        if stock.macd > 0:  # 상승 모멘텀
            technical_score += 30
        if stock.ma_trend == "BULLISH":  # 상승 추세
            technical_score += 30
        scores['technical'] = min(100, technical_score)
        
        # Fundamental: 성장성 중심
        fundamental_score = 0
        if 0 < stock.pe_ratio < 25:  # 적정 PER
            fundamental_score += 30
        if stock.roe > 15:  # 높은 ROE
            fundamental_score += 40
        if stock.debt_ratio < 0.5:  # 낮은 부채비율
            fundamental_score += 30
        scores['fundamental'] = min(100, fundamental_score)
        
        # Momentum: 상승 모멘텀
        momentum_score = 0
        if stock.change_rate > 0:  # 상승
            momentum_score += 50
        if stock.volume > 0:  # 거래량 존재
            momentum_score += 30
        if stock.bb_position > 0.5:  # 볼린저 밴드 상단
            momentum_score += 20
        scores['momentum'] = min(100, momentum_score)
        
        # Risk: 리스크 관리
        risk_score = 100 - min(100, abs(stock.change_rate) * 2)  # 변동성 기반
        scores['risk'] = max(0, risk_score)
        
        return scores
    
    async def _jesse_livermore_analysis(self, stock: StockData) -> Dict[str, float]:
        """제시 리버모어 (추세추종) 분석"""
        scores = {}
        
        # Technical: 추세 및 기술적 지표
        technical_score = 0
        if stock.ma_trend == "BULLISH":
            technical_score += 50
        if stock.rsi > 50:  # 강세
            technical_score += 25
        if stock.macd > 0:
            technical_score += 25
        scores['technical'] = min(100, technical_score)
        
        # Fundamental: 최소한의 기본 체크
        fundamental_score = 50  # 기본 점수
        if stock.pe_ratio > 0:  # 수익성 존재
            fundamental_score += 25
        if stock.market_cap > 1000000000:  # 대형주 선호
            fundamental_score += 25
        scores['fundamental'] = min(100, fundamental_score)
        
        # Momentum: 강한 모멘텀 중시
        momentum_score = 0
        if stock.change_rate > 2:  # 강한 상승
            momentum_score += 60
        elif stock.change_rate > 0:  # 상승
            momentum_score += 30
        if stock.bb_position > 0.7:  # 강한 상승 신호
            momentum_score += 40
        scores['momentum'] = min(100, momentum_score)
        
        # Risk: 추세 반전 리스크
        risk_score = 100
        if stock.rsi > 80:  # 과매수
            risk_score -= 30
        if abs(stock.change_rate) > 10:  # 과도한 변동
            risk_score -= 20
        scores['risk'] = max(0, risk_score)
        
        return scores
    
    async def _ichimoku_analysis(self, stock: StockData) -> Dict[str, float]:
        """일목산인 (균형표) 분석"""
        scores = {}
        
        # Technical: 균형 중시
        technical_score = 0
        if 40 <= stock.rsi <= 60:  # 균형 RSI
            technical_score += 40
        if stock.bb_position > 0.3 and stock.bb_position < 0.7:  # 균형 위치
            technical_score += 30
        if stock.ma_trend != "NEUTRAL":  # 명확한 추세
            technical_score += 30
        scores['technical'] = min(100, technical_score)
        
        # Fundamental: 안정성 중시
        fundamental_score = 0
        if stock.pe_ratio > 0 and stock.pe_ratio < 20:
            fundamental_score += 35
        if stock.roe > 10:
            fundamental_score += 30
        if stock.debt_ratio < 0.6:
            fundamental_score += 35
        scores['fundamental'] = min(100, fundamental_score)
        
        # Momentum: 적당한 모멘텀
        momentum_score = 0
        if -2 <= stock.change_rate <= 5:  # 적정 변화율
            momentum_score += 70
        if stock.macd != 0:  # MACD 신호 존재
            momentum_score += 30
        scores['momentum'] = min(100, momentum_score)
        
        # Risk: 균형 잡힌 리스크
        risk_score = 80  # 기본 점수
        if abs(stock.change_rate) < 3:  # 낮은 변동성
            risk_score += 20
        scores['risk'] = min(100, risk_score)
        
        return scores
    
    async def _warren_buffett_analysis(self, stock: StockData) -> Dict[str, float]:
        """워렌 버핏 (가치투자) 분석"""
        scores = {}
        
        # Technical: 최소한의 기술적 분석
        technical_score = 50  # 기본 점수
        if stock.ma_trend == "BULLISH":
            technical_score += 25
        if stock.rsi < 70:  # 과매수 아님
            technical_score += 25
        scores['technical'] = min(100, technical_score)
        
        # Fundamental: 핵심 중시
        fundamental_score = 0
        if 0 < stock.pe_ratio < 15:  # 저평가
            fundamental_score += 40
        if stock.roe > 15:  # 높은 ROE
            fundamental_score += 30
        if stock.debt_ratio < 0.3:  # 낮은 부채
            fundamental_score += 30
        scores['fundamental'] = min(100, fundamental_score)
        
        # Momentum: 장기 관점
        momentum_score = 60  # 기본 점수
        if stock.change_rate > -5:  # 큰 하락 아님
            momentum_score += 40
        scores['momentum'] = min(100, momentum_score)
        
        # Risk: 안전마진 중시
        risk_score = 90  # 높은 기본 점수
        if stock.debt_ratio > 0.5:  # 높은 부채비율
            risk_score -= 30
        if abs(stock.change_rate) > 5:  # 높은 변동성
            risk_score -= 20
        scores['risk'] = max(0, risk_score)
        
        return scores
    
    async def _peter_lynch_analysis(self, stock: StockData) -> Dict[str, float]:
        """피터 린치 (성장주) 분석"""
        scores = {}
        
        # Technical: 성장 신호
        technical_score = 0
        if stock.rsi > 50:  # 강세
            technical_score += 30
        if stock.ma_trend == "BULLISH":
            technical_score += 40
        if stock.bb_position > 0.5:  # 상승 구간
            technical_score += 30
        scores['technical'] = min(100, technical_score)
        
        # Fundamental: 성장성 중심
        fundamental_score = 0
        if 0 < stock.pe_ratio < 30:  # 성장주 적정 PER
            fundamental_score += 35
        if stock.roe > 12:  # 좋은 수익성
            fundamental_score += 35
        if stock.debt_ratio < 0.4:  # 건전한 재무
            fundamental_score += 30
        scores['fundamental'] = min(100, fundamental_score)
        
        # Momentum: 성장 모멘텀
        momentum_score = 0
        if stock.change_rate > 1:  # 상승세
            momentum_score += 50
        if stock.macd > 0:  # 상승 신호
            momentum_score += 30
        if stock.market_cap > 0:  # 시가총액 존재
            momentum_score += 20
        scores['momentum'] = min(100, momentum_score)
        
        # Risk: 성장주 리스크 관리
        risk_score = 70  # 기본 점수
        if stock.pe_ratio < 25:  # 과도한 고평가 아님
            risk_score += 30
        scores['risk'] = min(100, risk_score)
        
        return scores
    
    async def _blackrock_analysis(self, stock: StockData) -> Dict[str, float]:
        """블랙록 (기관투자) 분석"""
        scores = {}
        
        # Technical: 균형 잡힌 기술적 분석
        technical_score = 0
        if 30 <= stock.rsi <= 70:
            technical_score += 35
        if stock.ma_trend != "BEARISH":  # 하락 추세 아님
            technical_score += 35
        if stock.bb_position > 0.2:  # 극단적 하락 아님
            technical_score += 30
        scores['technical'] = min(100, technical_score)
        
        # Fundamental: 기관 투자 기준
        fundamental_score = 0
        if 0 < stock.pe_ratio < 25:
            fundamental_score += 30
        if stock.roe > 10:
            fundamental_score += 35
        if stock.debt_ratio < 0.5:
            fundamental_score += 35
        scores['fundamental'] = min(100, fundamental_score)
        
        # Momentum: 안정적 모멘텀
        momentum_score = 0
        if stock.change_rate > -3:  # 큰 하락 아님
            momentum_score += 40
        if stock.market_cap > 5000000000:  # 대형주 선호
            momentum_score += 40
        if stock.volume > 0:
            momentum_score += 20
        scores['momentum'] = min(100, momentum_score)
        
        # Risk: 리스크 관리 중시
        risk_score = 85  # 높은 기본 점수
        if abs(stock.change_rate) > 8:  # 높은 변동성
            risk_score -= 25
        if stock.debt_ratio > 0.6:  # 높은 부채
            risk_score -= 20
        scores['risk'] = max(0, risk_score)
        
        return scores
    
    def _determine_recommendation(self, score: float) -> str:
        """점수 기반 추천 등급 결정"""
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 65:
            return "BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _generate_analysis_reason(self, stock: StockData, strategy: InvestmentStrategy, scores: Dict[str, float]) -> str:
        """분석 이유 생성"""
        try:
            reasons = []
            
            # 최고 점수 영역 찾기
            max_score_area = max(scores, key=scores.get)
            max_score = scores[max_score_area]
            
            if max_score_area == 'technical' and max_score > 70:
                reasons.append(f"기술적 지표 우수 (RSI: {stock.rsi:.1f})")
            elif max_score_area == 'fundamental' and max_score > 70:
                reasons.append(f"펀더멘털 양호 (ROE: {stock.roe:.1f}%)")
            elif max_score_area == 'momentum' and max_score > 70:
                reasons.append(f"상승 모멘텀 ({stock.change_rate:+.2f}%)")
            
            # 전략별 특징 추가
            if strategy == InvestmentStrategy.WILLIAM_ONEIL:
                if stock.ma_trend == "BULLISH":
                    reasons.append("CAN SLIM 상승 추세 확인")
            elif strategy == InvestmentStrategy.WARREN_BUFFETT:
                if stock.pe_ratio > 0 and stock.pe_ratio < 15:
                    reasons.append(f"저평가 주식 (PER: {stock.pe_ratio:.1f})")
            
            # 기본 이유가 없으면 일반적인 이유 추가
            if not reasons:
                if stock.change_rate > 0:
                    reasons.append("주가 상승세")
                else:
                    reasons.append("종합 분석 결과")
            
            return ", ".join(reasons[:2])  # 최대 2개 이유
            
        except Exception as e:
            logger.error(f"❌ 분석 이유 생성 실패: {e}")
            return "종합 분석 결과"
    
    def _calculate_confidence(self, stock: StockData, scores: Dict[str, float]) -> float:
        """신뢰도 계산"""
        try:
            # 데이터 품질 기반 신뢰도
            quality_score = 1.0 if stock.data_quality == "GOOD" else 0.7
            
            # 점수 일관성 기반 신뢰도
            score_values = list(scores.values())
            score_std = np.std(score_values) if len(score_values) > 1 else 0
            consistency_score = max(0, 1 - (score_std / 50))  # 표준편차가 클수록 신뢰도 낮음
            
            # 전체 신뢰도
            confidence = (quality_score * 0.6 + consistency_score * 0.4) * 100
            
            return min(100, max(0, confidence))
            
        except Exception as e:
            logger.error(f"❌ 신뢰도 계산 실패: {e}")
            return 50.0

if __name__ == "__main__":
    async def test_analysis_engine():
        """분석 엔진 테스트"""
        print("🧪 최적화된 분석 엔진 테스트 시작...")
        
        # 테스트 데이터 생성
        test_stock = StockData(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            change_rate=2.5,
            volume=1000000,
            market_cap=2500000000000,
            rsi=55.0,
            macd=1.2,
            bb_position=0.6,
            ma_trend="BULLISH",
            pe_ratio=25.0,
            pb_ratio=3.0,
            roe=20.0,
            debt_ratio=0.3
        )
        
        engine = OptimizedAnalysisEngine()
        await engine.initialize()
        
        # 모든 전략으로 테스트
        for strategy in InvestmentStrategy:
            result = await engine._analyze_single_stock(test_stock, strategy)
            if result:
                print(f"📊 {strategy.value}: {result.score:.1f}점 ({result.recommendation}) - {result.reason}")
        
        print("✅ 테스트 완료!")
    
    asyncio.run(test_analysis_engine()) 