"""
Gemini AI 분석기 모듈
투자 전략 결과를 종합 분석하여 AI 기반 투자 추천 생성
"""
import os
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# base_interfaces에서 표준 클래스들 import
from ..core.base_interfaces import (
    IAIAnalyzer,
    StockData,
    TechnicalAnalysisResult,
    StrategyScore,
    InvestmentRecommendation,
    MarketType,
    StrategyType,
    RiskLevel,
    InvestmentPeriod,
    AIAnalysisError
)


@dataclass
class AnalysisResult:
    """분석 결과 데이터 클래스"""
    symbol: str
    strategy_scores: Dict[str, float]
    technical_data: Dict[str, Any]
    total_score: float
    reasoning: str
    confidence: float
    risk_level: str


@dataclass
class Top5Selection:
    """Top5 선정 결과 데이터 클래스"""
    selected_stocks: List[AnalysisResult]
    selection_reasoning: str
    market_analysis: str
    risk_assessment: str
    recommended_allocation: Dict[str, float]
    timestamp: datetime


class GeminiAnalyzer(IAIAnalyzer):
    """Gemini AI 기반 종합 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Gemini AI 분석기 초기화
        
        Args:
            api_key: Google Gemini API 키
        """
        self.api_key = self._initialize_api_key(api_key)
        self.is_mock = self._check_mock_mode()
        self.model = self._configure_gemini_ai()
        self.strategy_weights = self._initialize_strategy_weights()

    def _initialize_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """API 키 초기화"""
        return (
            api_key or 
            os.getenv('GOOGLE_GEMINI_API_KEY') or 
            os.getenv('GEMINI_API_KEY') or
            os.getenv('GOOGLE_AI_API_KEY')
        )
        
    def _check_mock_mode(self) -> bool:
        """Mock 모드 확인"""
        is_mock = os.getenv('IS_MOCK', 'false').lower() == 'true'
        if not self.api_key and not is_mock:
            logger.warning("Gemini API 키를 찾을 수 없습니다. Mock 모드로 실행됩니다.")
            is_mock = True
        return is_mock
        
    def _configure_gemini_ai(self) -> Optional[genai.GenerativeModel]:
        """Gemini AI 설정"""
        if not self.is_mock:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-pro')
                logger.info("Gemini AI 분석기 초기화 완료")
                return model
            except Exception as e:
                logger.warning(f"Gemini AI 초기화 실패: {e}. Mock 모드로 전환됩니다.")
                self.is_mock = True
            logger.info("Mock 모드로 Gemini AI 분석기 초기화 완료")
        return None
        
    def _initialize_strategy_weights(self) -> Dict[StrategyType, float]:
        """전략 가중치 초기화"""
        return {
            StrategyType.BENJAMIN_GRAHAM: 0.12,
            StrategyType.WARREN_BUFFETT: 0.15,
            StrategyType.PETER_LYNCH: 0.10,
            StrategyType.GEORGE_SOROS: 0.08,
            StrategyType.JAMES_SIMONS: 0.09,
            StrategyType.RAY_DALIO: 0.07,
            StrategyType.JOEL_GREENBLATT: 0.06,
            StrategyType.WILLIAM_ONEIL: 0.08,
            StrategyType.JESSE_LIVERMORE: 0.05,
            StrategyType.PAUL_TUDOR_JONES: 0.06,
            StrategyType.RICHARD_DENNIS: 0.04,
            StrategyType.ED_SEYKOTA: 0.03,
            StrategyType.LARRY_WILLIAMS: 0.03,
            StrategyType.MARTIN_SCHWARTZ: 0.02,
            StrategyType.STANLEY_DRUCKENMILLER: 0.02
        }
    
    async def analyze_recommendations(
        self,
        stocks: List[StockData],
        strategy_scores: List[StrategyScore],
        technical_results: List[TechnicalAnalysisResult]
    ) -> List[InvestmentRecommendation]:
        """
        AI 기반 종합 분석 및 추천
        
        Args:
            stocks: 주식 데이터 리스트
            strategy_scores: 전략 점수 리스트
            technical_results: 기술적 분석 결과 리스트
            
        Returns:
            List[InvestmentRecommendation]: 투자 추천 리스트
        """
        try:
            logger.info("Gemini AI 종합 분석 시작")
            stock_map, technical_map, symbol_strategy_scores = self._prepare_analysis_data(stocks, strategy_scores, technical_results)
            recommendations = await self._perform_parallel_analysis(stock_map, technical_map, symbol_strategy_scores)
            recommendations.sort(key=lambda x: x.ai_confidence, reverse=True)
            logger.info(f"총 {len(recommendations)}개 투자 추천 완료")
            return recommendations[:20]  # 상위 20개만 반환
        except Exception as e:
            logger.error(f"AI 분석 중 오류: {e}")
            raise AIAnalysisError(f"AI 분석 실패: {str(e)}")

    def _prepare_analysis_data(
        self,
        stocks: List[StockData],
        strategy_scores: List[StrategyScore],
        technical_results: List[TechnicalAnalysisResult]
    ) -> Tuple[Dict[str, StockData], Dict[str, TechnicalAnalysisResult], Dict[str, Dict[str, StrategyScore]]]:
        """분석에 필요한 데이터 준비"""
        stock_map = {stock.symbol: stock for stock in stocks}
        technical_map = {result.symbol: result for result in technical_results}
        symbol_strategy_scores = {}
        for score in strategy_scores:
            if score.symbol not in symbol_strategy_scores:
                symbol_strategy_scores[score.symbol] = {}
            symbol_strategy_scores[score.symbol][score.strategy_name] = score
        return stock_map, technical_map, symbol_strategy_scores

    async def _perform_parallel_analysis(
        self,
        stock_map: Dict[str, StockData],
        technical_map: Dict[str, TechnicalAnalysisResult],
        symbol_strategy_scores: Dict[str, Dict[str, StrategyScore]]
    ) -> List[InvestmentRecommendation]:
        """병렬로 종목 분석 수행"""
        recommendations = []
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    self._analyze_single_stock_for_recommendation,
                    stock_map[symbol],
                    symbol_strategy_scores[symbol],
                    technical_map[symbol]
                )
                for symbol in stock_map.keys()
                if symbol in technical_map and symbol in symbol_strategy_scores
            ]
            for future in asyncio.as_completed(futures):
                try:
                    result = await future
                    if result:
                        recommendations.append(result)
                except Exception as e:
                    logger.error(f"종목 분석 중 오류: {e}")
        return recommendations
    
    async def generate_market_insight(
        self,
        market: MarketType,
        recommendations: List[InvestmentRecommendation]
    ) -> Dict[str, Any]:
        """
        시장 인사이트 생성
        
        Args:
            market: 시장 유형
            recommendations: 투자 추천 리스트
            
        Returns:
            Dict[str, Any]: 시장 인사이트
        """
        try:
            logger.info(f"{market.value} 시장 인사이트 생성 시작")
            
            if not recommendations:
                return {
                    'market': market.value,
                    'total_stocks': 0,
                    'market_sentiment': '중립',
                    'key_insights': ['분석할 추천 종목이 없습니다.'],
                    'risk_assessment': '데이터 부족',
                    'sector_analysis': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            # 시장별 통계 계산
            market_recommendations = [r for r in recommendations if r.market == market]
            
            # 기본 통계
            total_stocks = len(market_recommendations)
            avg_score = sum(r.total_score for r in market_recommendations) / total_stocks if total_stocks > 0 else 0
            avg_confidence = sum(r.ai_confidence for r in market_recommendations) / total_stocks if total_stocks > 0 else 0
            
            # 위험도 분포
            risk_distribution = {}
            for rec in market_recommendations:
                risk_level = rec.risk_level.value
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            # 전략 분포
            strategy_distribution = {}
            for rec in market_recommendations:
                strategy = rec.strategy_used.value
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            
            # AI 인사이트 생성
            market_sentiment = await self._generate_market_sentiment(market, market_recommendations)
            key_insights = await self._generate_key_insights(market, market_recommendations)
            risk_assessment = self._assess_market_risk(market_recommendations)
            
            return {
                'market': market.value,
                'total_stocks': total_stocks,
                'average_score': round(avg_score, 2),
                'average_confidence': round(avg_confidence, 2),
                'market_sentiment': market_sentiment,
                'key_insights': key_insights,
                'risk_assessment': risk_assessment,
                'risk_distribution': risk_distribution,
                'strategy_distribution': strategy_distribution,
                'top_5_symbols': [r.symbol for r in market_recommendations[:5]],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"시장 인사이트 생성 중 오류: {e}")
            raise AIAnalysisError(f"시장 인사이트 생성 실패: {str(e)}")
    
    def _analyze_single_stock_for_recommendation(
        self,
        stock_data: StockData,
        strategy_scores: Dict[str, StrategyScore],
        technical_result: TechnicalAnalysisResult
    ) -> Optional[InvestmentRecommendation]:
        """
        개별 종목에 대한 투자 추천 생성
        
        Args:
            stock_data: 주식 데이터
            strategy_scores: 전략별 점수 딕셔너리
            technical_result: 기술적 분석 결과
            
        Returns:
            Optional[InvestmentRecommendation]: 투자 추천 결과
        """
        try:
            # 전략별 가중 점수 계산
            total_strategy_score = 0
            total_weight = 0
            best_strategy = None
            best_score = 0
            
            for strategy_name, score_obj in strategy_scores.items():
                # 전략 이름을 StrategyType으로 변환
                strategy_type = self._convert_strategy_name_to_type(strategy_name)
                weight = self.strategy_weights.get(strategy_type, 0.01)
                
                weighted_score = score_obj.score * weight
                total_strategy_score += weighted_score
                total_weight += weight
                
                if score_obj.score > best_score:
                    best_score = score_obj.score
                    best_strategy = strategy_type
            
            # 평균 전략 점수
            avg_strategy_score = total_strategy_score / total_weight if total_weight > 0 else 0
            
            # 기술적 점수 (기술적 분석 신뢰도 * 100)
            technical_score = technical_result.confidence * 100
            
            # 총합 점수 계산 (전략 점수 70% + 기술적 점수 30%)
            total_score = (avg_strategy_score * 0.7) + (technical_score * 0.3)
            
            # AI 신뢰도 계산
            ai_confidence = self._calculate_ai_confidence(strategy_scores, technical_result)
            
            # 가격 정보 설정
            current_price = stock_data.current_price
            
            # 기술적 분석 기반 진입/목표/손절 가격 계산
            entry_price = self._calculate_entry_price(current_price, technical_result)
            target_price = self._calculate_target_price(current_price, technical_result, total_score)
            stop_loss_price = self._calculate_stop_loss_price(current_price, technical_result)
            
            # 기대 수익률 계산
            expected_return = ((target_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            # 위험도 및 투자 기간 결정
            risk_level = self._determine_risk_level(technical_result, total_score)
            investment_period = self._determine_investment_period(technical_result, best_strategy)
            
            # 포지션 크기 계산
            position_size = self._calculate_position_size(risk_level, ai_confidence)
            
            # AI 추론 생성 (동기 함수로 변경)
            reasoning = self._generate_recommendation_reasoning_sync(
                stock_data, strategy_scores, technical_result, total_score
            )
            
            # 핵심 지표 수집
            key_indicators = self._collect_key_indicators(strategy_scores, technical_result)
            
            return InvestmentRecommendation(
                symbol=stock_data.symbol,
                name=stock_data.name,
                market=stock_data.market,
                strategy_used=best_strategy or StrategyType.WARREN_BUFFETT,
                total_score=round(total_score, 2),
                strategy_score=round(avg_strategy_score, 2),
                technical_score=round(technical_score, 2),
                ai_confidence=round(ai_confidence, 2),
                current_price=current_price,
                entry_price=round(entry_price, 2),
                target_price=round(target_price, 2),
                stop_loss_price=round(stop_loss_price, 2),
                expected_return=round(expected_return, 2),
                risk_level=risk_level,
                investment_period=investment_period,
                position_size_percent=round(position_size, 2),
                recommendation_reason=reasoning,
                key_indicators=key_indicators,
                technical_analysis=technical_result
            )
                
        except Exception as e:
            logger.error(f"종목 {stock_data.symbol} 추천 생성 중 오류: {e}")
            return None

    def _convert_strategy_name_to_type(self, strategy_name: str) -> StrategyType:
        """전략 이름을 StrategyType으로 변환"""
        name_mapping = {
            '벤저민_그레이엄': StrategyType.BENJAMIN_GRAHAM,
            '워런_버핏': StrategyType.WARREN_BUFFETT,
            '피터_린치': StrategyType.PETER_LYNCH,
            '조지_소로스': StrategyType.GEORGE_SOROS,
            '제임스_사이먼스': StrategyType.JAMES_SIMONS,
            '레이_달리오': StrategyType.RAY_DALIO,
            '조엘_그린블랫': StrategyType.JOEL_GREENBLATT,
            '윌리엄_오닐': StrategyType.WILLIAM_ONEIL,
            '제시_리버모어': StrategyType.JESSE_LIVERMORE,
            '폴_튜더_존스': StrategyType.PAUL_TUDOR_JONES,
            '리처드_데니스': StrategyType.RICHARD_DENNIS,
            '에드_세이코타': StrategyType.ED_SEYKOTA,
            '래리_윌리엄스': StrategyType.LARRY_WILLIAMS,
            '마틴_슈바르츠': StrategyType.MARTIN_SCHWARTZ,
            '스탠리_드러켄밀러': StrategyType.STANLEY_DRUCKENMILLER
        }
        return name_mapping.get(strategy_name, StrategyType.WARREN_BUFFETT)

    def _calculate_ai_confidence(
        self,
        strategy_scores: Dict[str, StrategyScore], 
        technical_result: TechnicalAnalysisResult
    ) -> float:
        """AI 신뢰도 계산"""
        # 전략 점수들의 일관성 계산
        scores = [score.score for score in strategy_scores.values()]
        if len(scores) < 2:
            strategy_consistency = 0.5
        else:
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            strategy_consistency = max(0, 1 - (score_std / (score_mean + 1)))
        
        # 기술적 분석 신뢰도
        technical_confidence = technical_result.confidence
        
        # 전략 평균 신뢰도
        avg_strategy_confidence = np.mean([score.confidence for score in strategy_scores.values()])
        
        # 종합 신뢰도 계산
        ai_confidence = (
            strategy_consistency * 0.3 +
            technical_confidence * 0.4 +
            avg_strategy_confidence * 0.3
        )
        
        return min(max(ai_confidence, 0.0), 1.0)

    def _calculate_entry_price(self, current_price: float, technical_result: TechnicalAnalysisResult) -> float:
        """진입 가격 계산"""
        # 기술적 지표를 기반으로 진입 가격 조정
        indicators = technical_result.indicators
        
        # RSI 기반 조정
        if indicators.rsi is not None:
            if indicators.rsi < 30:  # 과매도
                return current_price * 1.02  # 2% 위에서 진입
            elif indicators.rsi > 70:  # 과매수
                return current_price * 0.98  # 2% 아래에서 진입
        
        # 볼린저 밴드 기반 조정
        if indicators.bb_lower is not None and indicators.bb_upper is not None:
            if current_price <= indicators.bb_lower:
                return current_price * 1.01
            elif current_price >= indicators.bb_upper:
                return current_price * 0.99
        
        return current_price

    def _calculate_target_price(self, current_price: float, technical_result: TechnicalAnalysisResult, total_score: float) -> float:
        """목표 가격 계산"""
        # 기본 목표 수익률을 점수에 따라 조정
        base_return = 0.15  # 15% 기본 목표
        
        # 점수에 따른 목표 수익률 조정
        if total_score >= 80:
            target_return = base_return * 1.5  # 22.5%
        elif total_score >= 60:
            target_return = base_return * 1.2  # 18%
        else:
            target_return = base_return * 0.8  # 12%
        
        # 기술적 지표 기반 추가 조정
        indicators = technical_result.indicators
        if indicators.bb_upper is not None:
            bb_target = indicators.bb_upper
            calculated_target = current_price * (1 + target_return)
            return max(calculated_target, bb_target)
        
        return current_price * (1 + target_return)

    def _calculate_stop_loss_price(self, current_price: float, technical_result: TechnicalAnalysisResult) -> float:
        """손절 가격 계산"""
        indicators = technical_result.indicators
        
        # 볼린저 밴드 하단을 기준으로 손절가 설정
        if indicators.bb_lower is not None:
            return min(indicators.bb_lower, current_price * 0.92)  # 최대 8% 손실
        
        # 20일 이동평균선 기준
        if indicators.sma_20 is not None:
            return min(indicators.sma_20 * 0.95, current_price * 0.90)  # 최대 10% 손실
        
        # 기본 손절가 (5% 손실)
        return current_price * 0.95

    def _determine_risk_level(self, technical_result: TechnicalAnalysisResult, total_score: float) -> RiskLevel:
        """위험도 결정"""
        # 기술적 분석 신뢰도와 총점을 기반으로 위험도 결정
        confidence = technical_result.confidence
        
        if confidence >= 0.8 and total_score >= 70:
            return RiskLevel.LOW
        elif confidence >= 0.6 and total_score >= 50:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def _determine_investment_period(self, technical_result: TechnicalAnalysisResult, strategy_type: Optional[StrategyType]) -> InvestmentPeriod:
        """투자 기간 결정"""
        # 전략 타입에 따른 기본 투자 기간
        long_term_strategies = [
            StrategyType.WARREN_BUFFETT,
            StrategyType.BENJAMIN_GRAHAM,
            StrategyType.RAY_DALIO
        ]
        
        short_term_strategies = [
            StrategyType.JESSE_LIVERMORE,
            StrategyType.PAUL_TUDOR_JONES,
            StrategyType.RICHARD_DENNIS
        ]
        
        if strategy_type in long_term_strategies:
            return InvestmentPeriod.LONG
        elif strategy_type in short_term_strategies:
            return InvestmentPeriod.SHORT
        else:
            return InvestmentPeriod.MEDIUM

    def _calculate_position_size(self, risk_level: RiskLevel, ai_confidence: float) -> float:
        """포지션 크기 계산"""
        base_sizes = {
            RiskLevel.LOW: 8.0,
            RiskLevel.MEDIUM: 5.0,
            RiskLevel.HIGH: 2.0
        }
        
        base_size = base_sizes.get(risk_level, 3.0)
        confidence_multiplier = min(max(ai_confidence, 0.5), 1.0)
        
        return base_size * confidence_multiplier

    def _generate_recommendation_reasoning_sync(
        self,
        stock_data: StockData,
        strategy_scores: Dict[str, StrategyScore],
        technical_result: TechnicalAnalysisResult,
        total_score: float
    ) -> str:
        """투자 추천 근거 생성 (동기 버전)"""
        return self._generate_fallback_recommendation_reasoning(
            stock_data, strategy_scores, technical_result, total_score
        )

    async def _get_gemini_recommendation_reasoning(
        self,
        stock_data: StockData,
        strategy_scores: Dict[str, StrategyScore],
        technical_result: TechnicalAnalysisResult,
        total_score: float
    ) -> str:
        """Gemini AI를 사용한 추천 근거 생성"""
        try:
            # 프롬프트 구성
            prompt = f"""
다음 주식에 대한 투자 추천 근거를 한국어로 작성해주세요:

종목 정보:
- 심볼: {stock_data.symbol}
- 종목명: {stock_data.name}
- 현재가: {stock_data.current_price:,.0f}원
- 시장: {stock_data.market.value}

전략별 점수:
{chr(10).join([f'- {name}: {score.score:.1f}점 (신뢰도: {score.confidence:.2f})' for name, score in strategy_scores.items()])}

기술적 분석:
- 전체 신뢰도: {technical_result.confidence:.2f}
- RSI: {technical_result.indicators.rsi or 'N/A'}
- MACD: {technical_result.indicators.macd or 'N/A'}
- 볼린저밴드 위치: {technical_result.signals.bb_signal}
- 이동평균 추세: {technical_result.signals.ma_trend}
- 전체 추세: {technical_result.signals.overall_trend}

종합 점수: {total_score:.1f}점

위 정보를 바탕으로 투자 추천 근거를 3-4줄로 간결하게 작성해주세요.
"""
            
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            return response.text.strip()
                
        except Exception as e:
            logger.warning(f"Gemini AI 추천 근거 생성 실패: {e}")
            return self._generate_fallback_recommendation_reasoning(
                stock_data, strategy_scores, technical_result, total_score
            )

    def _generate_fallback_recommendation_reasoning(
        self,
        stock_data: StockData,
        strategy_scores: Dict[str, StrategyScore],
        technical_result: TechnicalAnalysisResult,
        total_score: float
    ) -> str:
        """Fallback 추천 근거 생성"""
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1].score)
        strategy_name = best_strategy[0]
        strategy_score = best_strategy[1].score
        
        tech_trend = technical_result.signals.overall_trend
        confidence = technical_result.confidence
        
        reasoning_parts = []
        
        # 전략 기반 근거
        if strategy_score >= 70:
            reasoning_parts.append(f"{strategy_name} 전략에서 {strategy_score:.1f}점의 높은 점수를 획득")
        elif strategy_score >= 50:
            reasoning_parts.append(f"{strategy_name} 전략에서 {strategy_score:.1f}점의 양호한 점수를 기록")
        
        # 기술적 분석 근거
        if tech_trend == "강한 상승":
            reasoning_parts.append("기술적 지표가 강한 상승 신호를 보임")
        elif tech_trend == "상승":
            reasoning_parts.append("기술적 지표가 상승 추세를 나타냄")
        elif tech_trend == "중립":
            reasoning_parts.append("기술적 지표가 중립적 상태를 유지")
        
        # 신뢰도 근거
        if confidence >= 0.7:
            reasoning_parts.append("높은 분석 신뢰도로 안정적인 투자 후보")
        elif confidence >= 0.5:
            reasoning_parts.append("적절한 분석 신뢰도를 보유")
        
        # 종합 점수 근거
        if total_score >= 70:
            reasoning_parts.append(f"종합 점수 {total_score:.1f}점으로 강력 추천")
        elif total_score >= 50:
            reasoning_parts.append(f"종합 점수 {total_score:.1f}점으로 투자 고려 가능")
        
        return ". ".join(reasoning_parts) + "."

    def _collect_key_indicators(
        self, 
        strategy_scores: Dict[str, StrategyScore], 
        technical_result: TechnicalAnalysisResult
    ) -> Dict[str, Any]:
        """핵심 지표 수집"""
        key_indicators = {}
        
        # 최고 점수 전략
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1].score)
        key_indicators['best_strategy'] = best_strategy[0]
        key_indicators['best_strategy_score'] = best_strategy[1].score
        
        # 기술적 지표
        indicators = technical_result.indicators
        if indicators.rsi is not None:
            key_indicators['rsi'] = indicators.rsi
        if indicators.macd is not None:
            key_indicators['macd'] = indicators.macd
        if indicators.bb_upper is not None and indicators.bb_lower is not None:
            key_indicators['bb_position'] = 'upper' if technical_result.signals.bb_signal == '상단 돌파' else 'lower' if technical_result.signals.bb_signal == '하단 터치' else 'middle'
        
        # 신호
        key_indicators['overall_trend'] = technical_result.signals.overall_trend
        key_indicators['ma_trend'] = technical_result.signals.ma_trend
        
        return key_indicators

    async def _generate_market_sentiment(self, market: MarketType, recommendations: List[InvestmentRecommendation]) -> str:
        """시장 심리 분석"""
        if not recommendations:
            return "중립"
        
        avg_score = sum(r.total_score for r in recommendations) / len(recommendations)
        avg_confidence = sum(r.ai_confidence for r in recommendations) / len(recommendations)
        
        if avg_score >= 70 and avg_confidence >= 0.7:
            return "매우 긍정적"
        elif avg_score >= 60 and avg_confidence >= 0.6:
            return "긍정적"
        elif avg_score >= 40 and avg_confidence >= 0.4:
            return "보통"
        elif avg_score >= 30:
            return "부정적"
        else:
            return "매우 부정적"

    async def _generate_key_insights(self, market: MarketType, recommendations: List[InvestmentRecommendation]) -> List[str]:
        """주요 인사이트 생성"""
        insights = []
        
        if not recommendations:
            return ["분석 가능한 종목이 없습니다."]
        
        # 평균 점수 분석
        avg_score = sum(r.total_score for r in recommendations) / len(recommendations)
        insights.append(f"{market.value} 시장 평균 점수: {avg_score:.1f}점")
        
        # 전략 분포 분석
        strategy_counts = {}
        for rec in recommendations:
            strategy = rec.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            top_strategy = max(strategy_counts.items(), key=lambda x: x[1])
            insights.append(f"가장 유효한 전략: {top_strategy[0]} ({top_strategy[1]}개 종목)")
        
        # 위험도 분석
        risk_counts = {}
        for rec in recommendations:
            risk = rec.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        total_count = len(recommendations)
        if risk_counts.get('LOW', 0) / total_count > 0.3:
            insights.append("저위험 종목이 30% 이상으로 안정적 투자 환경")
        elif risk_counts.get('HIGH', 0) / total_count > 0.5:
            insights.append("고위험 종목 비중이 높아 신중한 접근 필요")
        
        # 수익률 분석
        avg_return = sum(r.expected_return for r in recommendations) / len(recommendations)
        insights.append(f"평균 기대 수익률: {avg_return:.1f}%")
        
        return insights

    def _assess_market_risk(self, recommendations: List[InvestmentRecommendation]) -> str:
        """시장 위험도 평가"""
        if not recommendations:
            return "평가 불가"
            
        # 위험도 분포 계산
        risk_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        total_risk_score = sum(risk_scores.get(r.risk_level.value, 2) for r in recommendations)
        avg_risk_score = total_risk_score / len(recommendations)
        
        # AI 신뢰도 평균
        avg_confidence = sum(r.ai_confidence for r in recommendations) / len(recommendations)
        
        # 종합 위험도 판단
        if avg_risk_score <= 1.5 and avg_confidence >= 0.7:
            return "낮음 - 안정적인 투자 환경"
        elif avg_risk_score <= 2.0 and avg_confidence >= 0.5:
            return "보통 - 적절한 위험 관리 필요"
        elif avg_risk_score <= 2.5:
            return "높음 - 신중한 투자 접근 권장"
        else:
            return "매우 높음 - 보수적 투자 전략 필요"

    async def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """개별 종목 분석 (테스트용 메서드)"""
        try:
            logger.info(f"{symbol} 개별 종목 분석 시작")
            
            # Mock 데이터로 분석 결과 생성
            analysis_result = {
                'symbol': symbol,
                'recommendation': 'BUY',
                'confidence': 0.75,
                'target_price': 180.0,
                'current_price': 150.0,
                'expected_return': 20.0,
                'risk_level': 'MEDIUM',
                'reasoning': f"{symbol} 종목은 기술적/전략적 분석 결과 매수 추천합니다."
            }
            
            logger.info(f"{symbol} 개별 종목 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"{symbol} 개별 종목 분석 중 오류: {e}")
            raise AIAnalysisError(f"{symbol} 분석 실패: {str(e)}")


# 유틸리티 함수들
def format_currency(amount: float, currency: str = 'KRW') -> str:
    """통화 포맷팅"""
    if currency == 'KRW':
        return f"{amount:,.0f}원"
    elif currency == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_portfolio_metrics(
    selected_stocks: List[AnalysisResult],
    allocation: Dict[str, float]
) -> Dict[str, float]:
    """포트폴리오 메트릭 계산"""
    if not selected_stocks:
        return {}
        
    total_score = sum(stock.total_score for stock in selected_stocks)
    avg_score = total_score / len(selected_stocks)
    
    avg_confidence = sum(stock.confidence for stock in selected_stocks) / len(selected_stocks)
        
    return {
        'portfolio_score': avg_score,
        'portfolio_confidence': avg_confidence,
        'diversification_score': len(selected_stocks) * 20,  # 간단한 다각화 점수
        'total_allocation': sum(allocation.values())
    } 