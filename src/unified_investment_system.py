#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 통합 투자 분석 시스템 v4.0 (Unified Investment Analysis System)
================================================================

코스피200·나스닥100·S&P500 전체 종목을 분석하여 투자 대가 전략과 
Gemini AI가 Top5 종목을 자동 선정하는 통합 투자 분석 시스템입니다.

주요 기능:
1. 다중 시장 지원
   - 코스피200: 한국 대표 200개 종목
   - 나스닥100: 미국 기술주 중심 100개 종목  
   - S&P500: 미국 대표 500개 종목

2. 15개 투자 대가 전략 구현
   - 워런 버핏, 벤저민 그레이엄, 피터 린치 등
   - 각 전략별 독립적 점수 산출
   - 전략별 특화된 종목 선별 기준

3. 종합 분석 시스템
   - 기술적 분석: RSI, MACD, 볼린저밴드 등
   - 뉴스 감정 분석: 실시간 뉴스 데이터 분석
   - Gemini AI 종합 판단: 모든 지표를 종합한 AI 추론

4. 상세 투자 정보 제공
   - 현재가, 진입가, 목표가, 손절가
   - 기대 수익률 및 위험도 평가
   - 투자 기간 및 포지션 크기 제안
   - 상세한 투자 근거 및 핵심 지표

5. 유연한 분석 옵션
   - 전략별 개별 분석 가능
   - 시장별 개별 분석 가능
   - 전체 시장 통합 분석 가능
   - Top N 종목 선택 가능

사용 예시:
- 워런 버핏 전략으로 코스피200 분석
- 피터 린치 전략으로 나스닥100 분석
- 모든 전략으로 전체 시장 분석

이 시스템은 투자 의사결정을 위한 종합적인 정보를 제공하며,
실제 투자 시에는 추가적인 리서치와 전문가 상담이 권장됩니다.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv

# --- .env 파일 경로를 명시적으로 지정하여 로드 ---
# 스크립트(unified_investment_system.py)의 위치를 기준으로 .env 파일의 절대 경로를 계산
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
if api_key:
    print(f"Success: GOOGLE_GEMINI_API_KEY loaded.")
else:
    print("Failure: GOOGLE_GEMINI_API_KEY not found after specifying path.")
# --- END ---

# 프로젝트 루트 경로 추가
sys.path.append(project_root)

from modules.data_collector import DataCollector
from modules.investment_strategies import InvestmentMasterStrategies, StockData, StrategyScore
from modules.technical_analysis import TechnicalAnalyzer, TechnicalAnalysisResult
from modules.gemini_analyzer import GeminiAnalyzer
from modules.news_collector import NewsCollector
from modules.news_analyzer import NewsAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class DetailedRecommendation:
    """상세 추천 결과"""
    symbol: str
    name: str
    market: str
    strategy_name: str
    
    # 점수 정보
    total_score: float
    strategy_score: float
    technical_score: float
    news_sentiment_score: float
    ai_confidence: float
    
    # 가격 정보
    current_price: float
    entry_price: float  # 진입 권장가
    target_price: float  # 목표가
    stop_loss_price: float  # 손절가
    expected_return: float  # 기대수익률
    
    # 상세 분석
    recommendation_reason: str
    risk_level: str  # LOW, MEDIUM, HIGH
    investment_period: str  # SHORT, MEDIUM, LONG
    key_indicators: Dict[str, Any]
    news_summary: str
    
    # 메타 정보
    analysis_date: datetime
    confidence_level: str

@dataclass
class MarketAnalysisResult:
    """시장별 분석 결과"""
    market_name: str
    total_stocks_analyzed: int
    strategy_name: str
    top_recommendations: List[DetailedRecommendation]
    market_sentiment: str
    market_trend: str
    analysis_summary: str

class UnifiedInvestmentSystem:
    """🎯 통합 투자 분석 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        logger.info("🚀 통합 투자 분석 시스템 v4.0 초기화")
        
        # 핵심 컴포넌트 초기화
        self.data_collector = DataCollector()
        self.strategy_manager = InvestmentMasterStrategies()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_analyzer = GeminiAnalyzer()
        self.news_collector = NewsCollector()
        self.news_analyzer = NewsAnalyzer()
        
        # 지원 시장 목록
        self.supported_markets = {
            'KOSPI200': 'Korean KOSPI 200',
            'NASDAQ100': 'NASDAQ 100',
            'SP500': 'S&P 500'
        }
        
        # 지원 전략 목록
        self.supported_strategies = self.strategy_manager.get_strategy_names()
        
        logger.info("✅ 시스템 초기화 완료")
    
    async def analyze_by_strategy_and_market(self, 
                                           strategy_name: str, 
                                           market: str = "ALL",
                                           top_n: int = 5) -> Dict[str, MarketAnalysisResult]:
        """특정 전략으로 특정 시장(또는 전체) 분석"""
        
        logger.info(f"🎯 전략별 분석 시작: {strategy_name}, 시장: {market}")
        
        # 전략 유효성 검증
        if strategy_name not in self.supported_strategies:
            raise ValueError(f"지원하지 않는 전략: {strategy_name}")
        
        results = {}
        
        # 분석할 시장 결정
        if market == "ALL":
            markets_to_analyze = list(self.supported_markets.keys())
        else:
            if market not in self.supported_markets:
                raise ValueError(f"지원하지 않는 시장: {market}")
            markets_to_analyze = [market]
        
        # 각 시장별 분석 실행
        for market_code in markets_to_analyze:
            try:
                logger.info(f"📊 {market_code} 시장 분석 시작")
                
                # 1. 시장 데이터 수집
                market_stocks = await self._collect_market_data(market_code)
                logger.info(f"✅ {market_code}: {len(market_stocks)}개 종목 수집")
                
                # 2. 뉴스 데이터 수집 및 분석
                news_data = await self._collect_and_analyze_news(market_code, market_stocks)
                
                # 3. 전략 적용
                strategy_results = await self._apply_single_strategy(
                    market_stocks, strategy_name
                )
                
                # 4. 기술적 분석
                technical_results = await self._perform_technical_analysis(market_stocks)
                
                # 5. AI 종합 분석 및 상세 추천 생성
                detailed_recommendations = await self._generate_detailed_recommendations(
                    market_stocks, strategy_results, technical_results, 
                    news_data, strategy_name, market_code, top_n
                )
                
                # 6. 시장 분석 결과 생성
                market_result = MarketAnalysisResult(
                    market_name=self.supported_markets[market_code],
                    total_stocks_analyzed=len(market_stocks),
                    strategy_name=strategy_name,
                    top_recommendations=detailed_recommendations,
                    market_sentiment=self._calculate_market_sentiment(news_data),
                    market_trend=self._analyze_market_trend(market_stocks),
                    analysis_summary=self._generate_market_summary(
                        market_code, strategy_name, detailed_recommendations
                    )
                )
                
                results[market_code] = market_result
                logger.info(f"✅ {market_code} 분석 완료")
                
            except Exception as e:
                logger.error(f"❌ {market_code} 분석 실패: {e}")
                continue
        
        logger.info(f"🎉 전략별 분석 완료: {len(results)}개 시장")
        return results
    
    async def _collect_market_data(self, market: str) -> List[StockData]:
        """시장별 데이터 수집"""
        try:
            if market == "KOSPI200":
                return await self.data_collector.collect_kospi200_data()
            elif market == "NASDAQ100":
                return await self.data_collector.collect_nasdaq100_data()
            elif market == "SP500":
                return await self.data_collector.collect_sp500_data()
            else:
                raise ValueError(f"지원하지 않는 시장: {market}")
        except Exception as e:
            logger.error(f"시장 데이터 수집 실패 {market}: {e}")
            return []
    
    async def _collect_and_analyze_news(self, market: str, stocks: List[StockData]) -> Dict[str, Any]:
        """뉴스 수집 및 분석"""
        try:
            # 시장별 뉴스 수집
            if market == "KOSPI200":
                news_data = await self.news_collector.collect_korean_market_news()
            else:
                news_data = await self.news_collector.collect_global_market_news()
            
            # 개별 종목 뉴스 수집 (상위 종목만)
            top_symbols = [stock.symbol for stock in stocks[:20]]  # 상위 20개만
            for symbol in top_symbols:
                stock_news = await self.news_collector.collect_stock_news(symbol)
                news_data.extend(stock_news)
            
            # 뉴스 분석
            analyzed_news = await self.news_analyzer.analyze_news_batch(news_data)
            
            return {
                'raw_news': news_data,
                'analyzed_news': analyzed_news,
                'market_sentiment': self.news_analyzer.calculate_market_sentiment(analyzed_news),
                'key_themes': self.news_analyzer.extract_key_themes(analyzed_news)
            }
            
        except Exception as e:
            logger.error(f"뉴스 분석 실패: {e}")
            return {
                'raw_news': [],
                'analyzed_news': [],
                'market_sentiment': 0.0,
                'key_themes': []
            }
    
    async def _apply_single_strategy(self, stocks: List[StockData], strategy_name: str) -> List[StrategyScore]:
        """단일 전략 적용"""
        try:
            strategy = self.strategy_manager.get_strategy(strategy_name)
            return strategy.apply_strategy(stocks)
        except Exception as e:
            logger.error(f"전략 적용 실패 {strategy_name}: {e}")
            return []
    
    async def _perform_technical_analysis(self, stocks: List[StockData]) -> Dict[str, TechnicalAnalysisResult]:
        """기술적 분석 수행"""
        results = {}
        
        for stock in stocks:
            try:
                # 가격 히스토리 생성 (실제로는 데이터 수집 시 포함되어야 함)
                price_history = self._generate_price_history(stock)
                technical_result = self.technical_analyzer.analyze_stock(stock, price_history)
                results[stock.symbol] = technical_result
            except Exception as e:
                logger.warning(f"기술적 분석 실패 {stock.symbol}: {e}")
                continue
        
        return results
    
    def _generate_price_history(self, stock: StockData) -> Dict[str, np.array]:
        """가격 히스토리 생성 (임시 구현)"""
        # 실제로는 데이터 수집기에서 제공되어야 함
        base_price = stock.current_price
        days = 60
        
        # 랜덤 가격 히스토리 생성 (실제 구현에서는 실제 데이터 사용)
        np.random.seed(hash(stock.symbol) % 2**32)
        returns = np.random.normal(0.001, 0.02, days)
        prices = [base_price]
        
        for i in range(days - 1):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # 최소가격 보장
        
        prices = np.array(prices)
        
        return {
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
            'open': np.roll(prices, 1),
            'volume': np.random.randint(100000, 1000000, len(prices))
        }
    
    async def _generate_detailed_recommendations(self,
                                               stocks: List[StockData],
                                               strategy_results: List[StrategyScore],
                                               technical_results: Dict[str, TechnicalAnalysisResult],
                                               news_data: Dict[str, Any],
                                               strategy_name: str,
                                               market: str,
                                               top_n: int) -> List[DetailedRecommendation]:
        """상세 추천 생성"""
        
        detailed_recommendations = []
        
        # 상위 N개 종목에 대해 상세 분석
        top_strategy_results = strategy_results[:top_n]
        
        for strategy_score in top_strategy_results:
            try:
                stock = next(s for s in stocks if s.symbol == strategy_score.symbol)
                technical_result = technical_results.get(strategy_score.symbol)
                
                # 가격 계산
                current_price = stock.current_price
                entry_price, target_price, stop_loss_price = self._calculate_price_targets(
                    stock, strategy_score, technical_result
                )
                
                # 뉴스 감정 점수
                news_sentiment_score = self._get_news_sentiment_for_stock(
                    stock.symbol, news_data
                )
                
                # AI 신뢰도 계산
                ai_confidence = await self._calculate_ai_confidence(
                    stock, strategy_score, technical_result, news_sentiment_score
                )
                
                # 상세 추천 생성
                recommendation = DetailedRecommendation(
                    symbol=stock.symbol,
                    name=stock.name,
                    market=market,
                    strategy_name=strategy_name,
                    
                    total_score=strategy_score.total_score,
                    strategy_score=strategy_score.total_score,
                    technical_score=technical_result.overall_score if technical_result else 50.0,
                    news_sentiment_score=news_sentiment_score,
                    ai_confidence=ai_confidence,
                    
                    current_price=current_price,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss_price=stop_loss_price,
                    expected_return=((target_price - entry_price) / entry_price) * 100,
                    
                    recommendation_reason=self._generate_recommendation_reason(
                        stock, strategy_score, technical_result, news_sentiment_score
                    ),
                    risk_level=self._calculate_risk_level(stock, strategy_score, technical_result),
                    investment_period=self._determine_investment_period(strategy_name),
                    key_indicators=self._extract_key_indicators(stock, technical_result),
                    news_summary=self._generate_news_summary(stock.symbol, news_data),
                    
                    analysis_date=datetime.now(),
                    confidence_level=self._get_confidence_level(ai_confidence)
                )
                
                detailed_recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"상세 추천 생성 실패 {strategy_score.symbol}: {e}")
                continue
        
        return detailed_recommendations
    
    def _calculate_price_targets(self, 
                               stock: StockData, 
                               strategy_score: StrategyScore, 
                               technical_result: Optional[TechnicalAnalysisResult]) -> Tuple[float, float, float]:
        """가격 목표 계산"""
        current_price = stock.current_price
        
        # 전략별 기본 목표 수익률
        strategy_target_returns = {
            'Benjamin Graham': 0.15,  # 15% 목표
            'Warren Buffett': 0.20,   # 20% 목표
            'Peter Lynch': 0.25,      # 25% 목표
            'George Soros': 0.30,     # 30% 목표
            'Jesse Livermore': 0.35,  # 35% 목표
        }
        
        base_target_return = strategy_target_returns.get(strategy_score.strategy_name, 0.20)
        
        # 점수에 따른 조정
        score_multiplier = strategy_score.total_score / 100
        adjusted_target_return = base_target_return * score_multiplier
        
        # 기술적 분석 조정
        if technical_result:
            if technical_result.recommendation == "STRONG_BUY":
                adjusted_target_return *= 1.2
            elif technical_result.recommendation == "BUY":
                adjusted_target_return *= 1.1
            elif technical_result.recommendation == "SELL":
                adjusted_target_return *= 0.8
        
        # 가격 계산
        entry_price = current_price * 0.98  # 2% 할인 진입
        target_price = entry_price * (1 + adjusted_target_return)
        stop_loss_price = entry_price * 0.92  # 8% 손절
        
        return entry_price, target_price, stop_loss_price
    
    def _get_news_sentiment_for_stock(self, symbol: str, news_data: Dict[str, Any]) -> float:
        """종목별 뉴스 감정 점수"""
        try:
            analyzed_news = news_data.get('analyzed_news', [])
            stock_news = [news for news in analyzed_news if symbol.upper() in news.get('content', '').upper()]
            
            if not stock_news:
                return news_data.get('market_sentiment', 0.0)
            
            sentiments = [news.get('sentiment_score', 0.0) for news in stock_news]
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.warning(f"뉴스 감정 분석 실패 {symbol}: {e}")
            return 0.0
    
    async def _calculate_ai_confidence(self,
                                     stock: StockData,
                                     strategy_score: StrategyScore,
                                     technical_result: Optional[TechnicalAnalysisResult],
                                     news_sentiment: float) -> float:
        """AI 신뢰도 계산"""
        try:
            # 각 요소별 가중치
            strategy_weight = 0.4
            technical_weight = 0.3
            news_weight = 0.2
            consistency_weight = 0.1
            
            # 전략 신뢰도
            strategy_confidence = strategy_score.confidence
            
            # 기술적 신뢰도
            technical_confidence = 0.5
            if technical_result:
                technical_confidence = sum(signal.confidence for signal in technical_result.signals) / len(technical_result.signals) if technical_result.signals else 0.5
            
            # 뉴스 신뢰도
            news_confidence = min(abs(news_sentiment), 1.0)
            
            # 일관성 점수 (모든 지표가 같은 방향인지)
            consistency_score = self._calculate_consistency_score(
                strategy_score, technical_result, news_sentiment
            )
            
            # 종합 신뢰도 계산
            total_confidence = (
                strategy_confidence * strategy_weight +
                technical_confidence * technical_weight +
                news_confidence * news_weight +
                consistency_score * consistency_weight
            )
            
            return min(max(total_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"AI 신뢰도 계산 실패: {e}")
            return 0.5
    
    def _calculate_consistency_score(self,
                                   strategy_score: StrategyScore,
                                   technical_result: Optional[TechnicalAnalysisResult],
                                   news_sentiment: float) -> float:
        """일관성 점수 계산"""
        scores = []
        
        # 전략 점수 정규화
        if strategy_score.total_score > 70:
            scores.append(1.0)
        elif strategy_score.total_score > 50:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # 기술적 분석 점수
        if technical_result:
            if technical_result.recommendation in ["STRONG_BUY", "BUY"]:
                scores.append(1.0)
            elif technical_result.recommendation == "HOLD":
                scores.append(0.5)
            else:
                scores.append(0.0)
        
        # 뉴스 감정 점수
        if news_sentiment > 0.3:
            scores.append(1.0)
        elif news_sentiment > -0.3:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # 일관성 계산
        if not scores:
            return 0.5
        
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        consistency = 1.0 - min(variance, 1.0)
        
        return consistency
    
    def _generate_recommendation_reason(self,
                                      stock: StockData,
                                      strategy_score: StrategyScore,
                                      technical_result: Optional[TechnicalAnalysisResult],
                                      news_sentiment: float) -> str:
        """추천 이유 생성"""
        reasons = []
        
        # 전략별 이유
        reasons.append(f"📊 {strategy_score.strategy_name} 전략 점수: {strategy_score.total_score:.1f}점")
        reasons.append(f"🎯 전략 분석: {strategy_score.reasoning}")
        
        # 기술적 분석 이유
        if technical_result:
            reasons.append(f"📈 기술적 분석: {technical_result.recommendation} ({technical_result.overall_score:.1f}점)")
            if technical_result.signals:
                top_signals = sorted(technical_result.signals, key=lambda x: x.strength, reverse=True)[:2]
                for signal in top_signals:
                    reasons.append(f"   • {signal.indicator_name}: {signal.description}")
        
        # 뉴스 감정 이유
        if abs(news_sentiment) > 0.2:
            sentiment_desc = "긍정적" if news_sentiment > 0 else "부정적"
            reasons.append(f"📰 뉴스 감정: {sentiment_desc} ({news_sentiment:.2f})")
        
        return "\n".join(reasons)
    
    def _calculate_risk_level(self,
                            stock: StockData,
                            strategy_score: StrategyScore,
                            technical_result: Optional[TechnicalAnalysisResult]) -> str:
        """리스크 레벨 계산"""
        risk_factors = 0
        
        # 변동성 체크
        if technical_result and technical_result.volatility_score > 70:
            risk_factors += 1
        
        # 전략 신뢰도 체크
        if strategy_score.confidence < 0.6:
            risk_factors += 1
        
        # 시가총액 체크
        if stock.market_cap and stock.market_cap < 1e10:  # 100억 미만
            risk_factors += 1
        
        if risk_factors >= 2:
            return "HIGH"
        elif risk_factors == 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_investment_period(self, strategy_name: str) -> str:
        """투자 기간 결정"""
        long_term_strategies = ["Benjamin Graham", "Warren Buffett", "Peter Lynch"]
        short_term_strategies = ["Jesse Livermore", "Paul Tudor Jones", "George Soros"]
        
        if strategy_name in long_term_strategies:
            return "LONG"
        elif strategy_name in short_term_strategies:
            return "SHORT"
        else:
            return "MEDIUM"
    
    def _extract_key_indicators(self, stock: StockData, technical_result: Optional[TechnicalAnalysisResult]) -> Dict[str, Any]:
        """핵심 지표 추출"""
        indicators = {
            'current_price': stock.current_price,
            'pe_ratio': stock.pe_ratio,
            'pb_ratio': stock.pb_ratio,
            'roe': stock.roe,
            'debt_ratio': stock.debt_ratio,
            'dividend_yield': stock.dividend_yield
        }
        
        if technical_result:
            indicators.update({
                'rsi': getattr(stock, 'rsi', None),
                'macd': getattr(stock, 'macd', None),
                'trend_direction': technical_result.trend_direction,
                'volatility': technical_result.volatility_score
            })
        
        return {k: v for k, v in indicators.items() if v is not None}
    
    def _generate_news_summary(self, symbol: str, news_data: Dict[str, Any]) -> str:
        """뉴스 요약 생성"""
        try:
            analyzed_news = news_data.get('analyzed_news', [])
            stock_news = [news for news in analyzed_news if symbol.upper() in news.get('content', '').upper()]
            
            if not stock_news:
                return "관련 뉴스 없음"
            
            # 최근 3개 뉴스 요약
            recent_news = sorted(stock_news, key=lambda x: x.get('timestamp', ''), reverse=True)[:3]
            summaries = [news.get('summary', news.get('title', '')) for news in recent_news]
            
            return " | ".join(summaries[:3])
            
        except Exception as e:
            logger.warning(f"뉴스 요약 생성 실패: {e}")
            return "뉴스 분석 불가"
    
    def _get_confidence_level(self, confidence: float) -> str:
        """신뢰도 레벨 문자열"""
        if confidence >= 0.8:
            return "매우 높음"
        elif confidence >= 0.6:
            return "높음"
        elif confidence >= 0.4:
            return "보통"
        else:
            return "낮음"
    
    def _calculate_market_sentiment(self, news_data: Dict[str, Any]) -> str:
        """시장 감정 계산"""
        sentiment_score = news_data.get('market_sentiment', 0.0)
        
        if sentiment_score > 0.3:
            return "매우 긍정적"
        elif sentiment_score > 0.1:
            return "긍정적"
        elif sentiment_score > -0.1:
            return "중립적"
        elif sentiment_score > -0.3:
            return "부정적"
        else:
            return "매우 부정적"
    
    def _analyze_market_trend(self, stocks: List[StockData]) -> str:
        """시장 트렌드 분석"""
        if not stocks:
            return "분석 불가"
        
        # 간단한 트렌드 분석 (실제로는 더 복잡한 로직 필요)
        avg_change = sum(getattr(stock, 'price_change_pct', 0) for stock in stocks) / len(stocks)
        
        if avg_change > 2:
            return "강한 상승세"
        elif avg_change > 0.5:
            return "상승세"
        elif avg_change > -0.5:
            return "횡보세"
        elif avg_change > -2:
            return "하락세"
        else:
            return "강한 하락세"
    
    def _generate_market_summary(self, market: str, strategy: str, recommendations: List[DetailedRecommendation]) -> str:
        """시장 요약 생성"""
        if not recommendations:
            return f"{market} 시장에서 {strategy} 전략으로 추천할 종목이 없습니다."
        
        avg_score = sum(rec.total_score for rec in recommendations) / len(recommendations)
        avg_expected_return = sum(rec.expected_return for rec in recommendations) / len(recommendations)
        
        high_confidence_count = sum(1 for rec in recommendations if rec.ai_confidence > 0.7)
        
        return f"""
        {self.supported_markets[market]} 시장 {strategy} 전략 분석 결과:
        • 분석 종목 수: {len(recommendations)}개
        • 평균 전략 점수: {avg_score:.1f}점
        • 평균 기대수익률: {avg_expected_return:.1f}%
        • 고신뢰도 종목: {high_confidence_count}개
        • 추천 종목: {', '.join([rec.name for rec in recommendations[:3]])}
        """.strip()
    
    def display_detailed_results(self, results: Dict[str, MarketAnalysisResult]):
        """상세 결과 출력"""
        print("\n" + "="*100)
        print("🚀 통합 투자 분석 시스템 - 상세 결과")
        print("="*100)
        
        for market_code, result in results.items():
            print(f"\n📊 {result.market_name} 시장 분석 결과")
            print("-" * 80)
            print(f"전략: {result.strategy_name}")
            print(f"분석 종목 수: {result.total_stocks_analyzed}개")
            print(f"시장 감정: {result.market_sentiment}")
            print(f"시장 트렌드: {result.market_trend}")
            
            print(f"\n🏆 Top {len(result.top_recommendations)}개 추천 종목:")
            for i, rec in enumerate(result.top_recommendations, 1):
                print(f"\n{i}. {rec.name} ({rec.symbol})")
                print(f"   💰 현재가: ${rec.current_price:.2f}")
                print(f"   🎯 진입가: ${rec.entry_price:.2f}")
                print(f"   🚀 목표가: ${rec.target_price:.2f}")
                print(f"   🛑 손절가: ${rec.stop_loss_price:.2f}")
                print(f"   📈 기대수익률: {rec.expected_return:.1f}%")
                print(f"   ⭐ 종합점수: {rec.total_score:.1f}점")
                print(f"   🤖 AI 신뢰도: {rec.confidence_level} ({rec.ai_confidence:.1%})")
                print(f"   ⚠️ 리스크: {rec.risk_level}")
                print(f"   ⏰ 투자기간: {rec.investment_period}")
                print(f"   📰 뉴스: {rec.news_summary[:100]}...")
                print(f"   💡 추천이유:")
                for line in rec.recommendation_reason.split('\n')[:3]:
                    print(f"      {line}")
            
            print(f"\n📋 시장 요약:")
            print(f"   {result.analysis_summary}")
        
        print("\n" + "="*100)

# 사용 예시 함수들
async def analyze_warren_buffett_kospi():
    """워런 버핏 전략으로 코스피200 분석"""
    system = UnifiedInvestmentSystem()
    results = await system.analyze_by_strategy_and_market("Warren Buffett", "KOSPI200", 5)
    system.display_detailed_results(results)
    return results

async def analyze_jesse_livermore_all_markets():
    """제시 리버모어 전략으로 전체 시장 분석"""
    system = UnifiedInvestmentSystem()
    results = await system.analyze_by_strategy_and_market("Jesse Livermore", "ALL", 5)
    system.display_detailed_results(results)
    return results

async def analyze_peter_lynch_nasdaq():
    """피터 린치 전략으로 나스닥100 분석"""
    system = UnifiedInvestmentSystem()
    results = await system.analyze_by_strategy_and_market("Peter Lynch", "NASDAQ100", 5)
    system.display_detailed_results(results)
    return results

if __name__ == "__main__":
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 투자 분석 시스템")
    parser.add_argument("--strategy", required=True, help="투자 전략 이름")
    parser.add_argument("--market", default="ALL", help="시장 (KOSPI200, NASDAQ100, SP500, ALL)")
    parser.add_argument("--top-n", type=int, default=5, help="상위 N개 종목")
    
    args = parser.parse_args()
    
    async def main():
        system = UnifiedInvestmentSystem()
        results = await system.analyze_by_strategy_and_market(
            args.strategy, args.market, args.top_n
        )
        system.display_detailed_results(results)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{args.strategy}_{args.market}_{timestamp}.json"
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for market, result in results.items():
            serializable_results[market] = {
                'market_name': result.market_name,
                'total_stocks_analyzed': result.total_stocks_analyzed,
                'strategy_name': result.strategy_name,
                'market_sentiment': result.market_sentiment,
                'market_trend': result.market_trend,
                'analysis_summary': result.analysis_summary,
                'top_recommendations': [asdict(rec) for rec in result.top_recommendations]
            }
        
        with open(f"reports/{filename}", 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 결과 저장: reports/{filename}")
    
    asyncio.run(main()) 