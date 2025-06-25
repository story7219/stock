#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 투자 대가 독립실행 시스템
각 투자 대가별로 독립적으로 실행 가능한 시스템
지수별(코스피200, 나스닥100, S&P500) 독립 실행 지원
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import json
from pathlib import Path

from modules.investment_strategies import (
    InvestmentMasterStrategies, 
    StockData, 
    StrategyScore,
    BenjaminGrahamStrategy,
    WarrenBuffettStrategy,
    PeterLynchStrategy,
    GeorgeSorosStrategy,
    JamesSimonsStrategy,
    RayDalioStrategy,
    JoelGreenblattStrategy,
    WilliamONeilStrategy,
    JesseLivermoreStrategy,
    PaulTudorJonesStrategy,
    RichardDennisStrategy,
    EdSeykotaStrategy,
    LarryWilliamsStrategy,
    MartinSchwartzStrategy,
    StanleyDruckenmillerStrategy
)
from modules.data_collector import DataCollector
from modules.gemini_analyzer import GeminiAnalyzer
from modules.technical_analysis import TechnicalAnalyzer
from modules.world_chart_experts import WorldChartExperts

logger = logging.getLogger(__name__)

@dataclass
class MarketRecommendation:
    """시장별 추천 결과"""
    symbol: str
    name: str
    market: str
    strategy_name: str
    current_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    expected_return: float
    score: float
    rank: int
    reasoning: str
    confidence: float
    risk_level: str
    investment_period: str

@dataclass
class StrategyResult:
    """전략 실행 결과"""
    strategy_name: str
    market: str
    total_analyzed: int
    recommendations: List[MarketRecommendation]
    execution_time: float
    timestamp: datetime
    gemini_analysis: str

class IndependentStrategyRunner:
    """투자 대가 독립실행 시스템"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.gemini_analyzer = GeminiAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.chart_experts = WorldChartExperts()
        self.investment_strategies = InvestmentMasterStrategies()
        
        # 투자 대가 전략 매핑
        self.strategy_classes = {
            'benjamin_graham': BenjaminGrahamStrategy,
            'warren_buffett': WarrenBuffettStrategy,
            'peter_lynch': PeterLynchStrategy,
            'george_soros': GeorgeSorosStrategy,
            'james_simons': JamesSimonsStrategy,
            'ray_dalio': RayDalioStrategy,
            'joel_greenblatt': JoelGreenblattStrategy,
            'william_oneil': WilliamONeilStrategy,
            'jesse_livermore': JesseLivermoreStrategy,
            'paul_tudor_jones': PaulTudorJonesStrategy,
            'richard_dennis': RichardDennisStrategy,
            'ed_seykota': EdSeykotaStrategy,
            'larry_williams': LarryWilliamsStrategy,
            'martin_schwartz': MartinSchwartzStrategy,
            'stanley_druckenmiller': StanleyDruckenmillerStrategy
        }
        
        # 지수별 설정
        self.market_configs = {
            'kospi200': {
                'name': '코스피200',
                'symbols_file': 'data/kospi200_symbols.json',
                'market_code': 'KRX',
                'currency': 'KRW'
            },
            'nasdaq100': {
                'name': '나스닥100',
                'symbols_file': 'data/nasdaq100_symbols.json',
                'market_code': 'NASDAQ',
                'currency': 'USD'
            },
            'sp500': {
                'name': 'S&P500',
                'symbols_file': 'data/sp500_symbols.json',
                'market_code': 'NYSE',
                'currency': 'USD'
            }
        }
        
        self.results_cache = {}
        
    async def run_single_strategy(self, 
                                 strategy_name: str, 
                                 market: str = 'all',
                                 top_n: int = 5) -> Dict[str, StrategyResult]:
        """
        단일 투자 대가 전략 독립 실행
        
        Args:
            strategy_name: 투자 대가 이름 (예: 'jesse_livermore')
            market: 시장 ('kospi200', 'nasdaq100', 'sp500', 'all')
            top_n: 추천 종목 수
            
        Returns:
            Dict[str, StrategyResult]: 시장별 결과
        """
        logger.info(f"🎯 {strategy_name} 전략 독립 실행 시작 - 시장: {market}")
        
        if strategy_name not in self.strategy_classes:
            raise ValueError(f"지원하지 않는 전략: {strategy_name}")
        
        results = {}
        markets_to_run = [market] if market != 'all' else list(self.market_configs.keys())
        
        for market_key in markets_to_run:
            if market_key not in self.market_configs:
                logger.warning(f"지원하지 않는 시장: {market_key}")
                continue
                
            try:
                result = await self._execute_strategy_for_market(
                    strategy_name, market_key, top_n
                )
                results[market_key] = result
                
            except Exception as e:
                logger.error(f"{strategy_name} - {market_key} 실행 실패: {e}")
                
        return results
    
    async def run_market_strategy(self,
                                 market: str,
                                 strategy_name: str = 'all',
                                 top_n: int = 5) -> Dict[str, StrategyResult]:
        """
        특정 시장에서 전략 실행
        
        Args:
            market: 시장 코드
            strategy_name: 전략 이름 ('all'이면 모든 전략)
            top_n: 추천 종목 수
            
        Returns:
            Dict[str, StrategyResult]: 전략별 결과
        """
        logger.info(f"📈 {market} 시장 전략 실행 - 전략: {strategy_name}")
        
        if market not in self.market_configs:
            raise ValueError(f"지원하지 않는 시장: {market}")
        
        results = {}
        strategies_to_run = [strategy_name] if strategy_name != 'all' else list(self.strategy_classes.keys())
        
        for strategy_key in strategies_to_run:
            if strategy_key not in self.strategy_classes:
                logger.warning(f"지원하지 않는 전략: {strategy_key}")
                continue
                
            try:
                result = await self._execute_strategy_for_market(
                    strategy_key, market, top_n
                )
                results[strategy_key] = result
                
            except Exception as e:
                logger.error(f"{strategy_key} - {market} 실행 실패: {e}")
                
        return results
    
    async def _execute_strategy_for_market(self,
                                         strategy_name: str,
                                         market: str,
                                         top_n: int) -> StrategyResult:
        """시장별 전략 실행"""
        start_time = datetime.now()
        
        # 1. 시장 데이터 수집
        stocks_data = await self._collect_market_data(market)
        logger.info(f"{market} 데이터 수집 완료: {len(stocks_data)}개 종목")
        
        # 2. 기술적 분석 적용
        stocks_with_technical = await self._apply_technical_analysis(stocks_data)
        
        # 3. 전략 적용
        strategy_class = self.strategy_classes[strategy_name]
        strategy = strategy_class()
        strategy_scores = strategy.apply_strategy(stocks_with_technical)
        
        # 4. 상위 종목 선별
        top_scores = strategy_scores[:top_n]
        
        # 5. Gemini AI 분석
        gemini_analysis = await self._get_gemini_analysis(
            strategy_name, market, top_scores, stocks_with_technical
        )
        
        # 6. 추천 결과 생성
        recommendations = await self._generate_recommendations(
            top_scores, market, strategy_name
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return StrategyResult(
            strategy_name=strategy_name,
            market=market,
            total_analyzed=len(stocks_data),
            recommendations=recommendations,
            execution_time=execution_time,
            timestamp=datetime.now(),
            gemini_analysis=gemini_analysis
        )
    
    async def _collect_market_data(self, market: str) -> List[StockData]:
        """시장별 데이터 수집"""
        market_config = self.market_configs[market]
        
        try:
            if market == 'kospi200':
                return await self.data_collector.get_kospi200_data()
            elif market == 'nasdaq100':
                return await self.data_collector.get_nasdaq100_data()
            elif market == 'sp500':
                return await self.data_collector.get_sp500_data()
            else:
                raise ValueError(f"지원하지 않는 시장: {market}")
                
        except Exception as e:
            logger.error(f"{market} 데이터 수집 실패: {e}")
            return []
    
    async def _apply_technical_analysis(self, stocks: List[StockData]) -> List[StockData]:
        """기술적 분석 적용"""
        enhanced_stocks = []
        
        for stock in stocks:
            try:
                # 차트 전문가 분석 적용
                chart_analysis = await self.chart_experts.analyze_stock(stock.symbol)
                
                # 기술적 지표 계산
                technical_data = await self.technical_analyzer.calculate_indicators(stock.symbol)
                
                # StockData 업데이트
                stock.rsi = technical_data.get('rsi')
                stock.macd = technical_data.get('macd')
                stock.moving_avg_20 = technical_data.get('ma20')
                stock.moving_avg_60 = technical_data.get('ma60')
                stock.bollinger_upper = technical_data.get('bb_upper')
                stock.bollinger_lower = technical_data.get('bb_lower')
                stock.volume_ratio = technical_data.get('volume_ratio')
                
                enhanced_stocks.append(stock)
                
            except Exception as e:
                logger.warning(f"{stock.symbol} 기술적 분석 실패: {e}")
                enhanced_stocks.append(stock)
        
        return enhanced_stocks
    
    async def _get_gemini_analysis(self,
                                  strategy_name: str,
                                  market: str,
                                  top_scores: List[StrategyScore],
                                  all_stocks: List[StockData]) -> str:
        """Gemini AI 분석"""
        try:
            analysis_prompt = f"""
            {strategy_name} 전략으로 {market} 시장 분석 결과:
            
            상위 {len(top_scores)}개 종목:
            {[f"{score.symbol}({score.name}): {score.total_score:.1f}점" for score in top_scores]}
            
            각 종목에 대한 상세 분석과 투자 판단을 제공해주세요:
            1. 선정 이유
            2. 진입 타이밍
            3. 목표가 및 손절가
            4. 리스크 요인
            5. 투자 기간 추천
            """
            
            return await self.gemini_analyzer.analyze_investment_opportunity(
                analysis_prompt, top_scores
            )
            
        except Exception as e:
            logger.error(f"Gemini 분석 실패: {e}")
            return f"{strategy_name} 전략 기반 {market} 시장 분석 결과입니다."
    
    async def _generate_recommendations(self,
                                      scores: List[StrategyScore],
                                      market: str,
                                      strategy_name: str) -> List[MarketRecommendation]:
        """추천 결과 생성"""
        recommendations = []
        
        for i, score in enumerate(scores):
            try:
                # 가격 정보 계산
                current_price = await self._get_current_price(score.symbol)
                entry_price, target_price, stop_loss = await self._calculate_prices(
                    score.symbol, current_price, strategy_name
                )
                
                expected_return = ((target_price - entry_price) / entry_price) * 100
                risk_level = self._assess_risk_level(score.confidence)
                investment_period = self._get_investment_period(strategy_name)
                
                recommendation = MarketRecommendation(
                    symbol=score.symbol,
                    name=score.name,
                    market=market,
                    strategy_name=strategy_name,
                    current_price=current_price,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_return=expected_return,
                    score=score.total_score,
                    rank=i + 1,
                    reasoning=score.reasoning,
                    confidence=score.confidence,
                    risk_level=risk_level,
                    investment_period=investment_period
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"{score.symbol} 추천 생성 실패: {e}")
        
        return recommendations
    
    async def _get_current_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            # 실제 구현에서는 실시간 가격 API 호출
            return 100.0  # 임시값
        except Exception:
            return 100.0
    
    async def _calculate_prices(self, 
                              symbol: str, 
                              current_price: float, 
                              strategy_name: str) -> Tuple[float, float, float]:
        """진입가, 목표가, 손절가 계산"""
        try:
            # 전략별 가격 계산 로직
            if strategy_name in ['jesse_livermore', 'paul_tudor_jones']:
                # 단기 트레이딩 전략
                entry_price = current_price * 0.98  # 2% 하락 시 진입
                target_price = current_price * 1.15  # 15% 목표
                stop_loss = current_price * 0.92    # 8% 손절
            elif strategy_name in ['warren_buffett', 'benjamin_graham']:
                # 장기 가치투자 전략
                entry_price = current_price * 0.95  # 5% 하락 시 진입
                target_price = current_price * 1.30  # 30% 목표
                stop_loss = current_price * 0.85    # 15% 손절
            else:
                # 기본 설정
                entry_price = current_price * 0.97
                target_price = current_price * 1.20
                stop_loss = current_price * 0.90
            
            return entry_price, target_price, stop_loss
            
        except Exception:
            return current_price * 0.97, current_price * 1.20, current_price * 0.90
    
    def _assess_risk_level(self, confidence: float) -> str:
        """리스크 레벨 평가"""
        if confidence >= 0.8:
            return "낮음"
        elif confidence >= 0.6:
            return "보통"
        else:
            return "높음"
    
    def _get_investment_period(self, strategy_name: str) -> str:
        """투자 기간 추천"""
        long_term_strategies = ['warren_buffett', 'benjamin_graham', 'peter_lynch']
        short_term_strategies = ['jesse_livermore', 'larry_williams', 'martin_schwartz']
        
        if strategy_name in long_term_strategies:
            return "장기 (1년 이상)"
        elif strategy_name in short_term_strategies:
            return "단기 (1-3개월)"
        else:
            return "중기 (3-12개월)"
    
    def save_results(self, results: Dict[str, StrategyResult], filename: str = None):
        """결과 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_results_{timestamp}.json"
        
        results_dir = Path("reports/strategy_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = {}
        for market, result in results.items():
            serializable_results[market] = {
                'strategy_name': result.strategy_name,
                'market': result.market,
                'total_analyzed': result.total_analyzed,
                'execution_time': result.execution_time,
                'timestamp': result.timestamp.isoformat(),
                'gemini_analysis': result.gemini_analysis,
                'recommendations': [asdict(rec) for rec in result.recommendations]
            }
        
        filepath = results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장 완료: {filepath}")
    
    def generate_report(self, results: Dict[str, StrategyResult]) -> str:
        """결과 리포트 생성"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🎯 투자 대가 전략 독립실행 결과 리포트")
        report_lines.append("=" * 80)
        
        for market, result in results.items():
            market_name = self.market_configs[market]['name']
            report_lines.append(f"\n📈 {market_name} ({market.upper()}) - {result.strategy_name}")
            report_lines.append("-" * 60)
            report_lines.append(f"분석 종목 수: {result.total_analyzed}개")
            report_lines.append(f"실행 시간: {result.execution_time:.2f}초")
            report_lines.append(f"실행 시각: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            report_lines.append(f"\n🏆 TOP {len(result.recommendations)} 추천 종목:")
            for rec in result.recommendations:
                report_lines.append(f"\n{rec.rank}. {rec.name} ({rec.symbol})")
                report_lines.append(f"   현재가: {rec.current_price:,.0f}")
                report_lines.append(f"   진입가: {rec.entry_price:,.0f}")
                report_lines.append(f"   목표가: {rec.target_price:,.0f}")
                report_lines.append(f"   손절가: {rec.stop_loss:,.0f}")
                report_lines.append(f"   기대수익률: {rec.expected_return:.1f}%")
                report_lines.append(f"   점수: {rec.score:.1f}/100")
                report_lines.append(f"   신뢰도: {rec.confidence:.1%}")
                report_lines.append(f"   리스크: {rec.risk_level}")
                report_lines.append(f"   투자기간: {rec.investment_period}")
                report_lines.append(f"   추천이유: {rec.reasoning[:100]}...")
            
            report_lines.append(f"\n🤖 Gemini AI 분석:")
            report_lines.append(result.gemini_analysis[:500] + "...")
        
        return "\n".join(report_lines)

# 편의 함수들
async def run_livermore_strategy(market: str = 'all', top_n: int = 5) -> Dict[str, StrategyResult]:
    """제시 리버모어 전략 실행"""
    runner = IndependentStrategyRunner()
    return await runner.run_single_strategy('jesse_livermore', market, top_n)

async def run_buffett_strategy(market: str = 'all', top_n: int = 5) -> Dict[str, StrategyResult]:
    """워런 버핏 전략 실행"""
    runner = IndependentStrategyRunner()
    return await runner.run_single_strategy('warren_buffett', market, top_n)

async def run_kospi200_analysis(strategy: str = 'all', top_n: int = 5) -> Dict[str, StrategyResult]:
    """코스피200 분석"""
    runner = IndependentStrategyRunner()
    return await runner.run_market_strategy('kospi200', strategy, top_n)

async def run_nasdaq100_analysis(strategy: str = 'all', top_n: int = 5) -> Dict[str, StrategyResult]:
    """나스닥100 분석"""
    runner = IndependentStrategyRunner()
    return await runner.run_market_strategy('nasdaq100', strategy, top_n)

async def run_sp500_analysis(strategy: str = 'all', top_n: int = 5) -> Dict[str, StrategyResult]:
    """S&P500 분석"""
    runner = IndependentStrategyRunner()
    return await runner.run_market_strategy('sp500', strategy, top_n)

if __name__ == "__main__":
    # 사용 예시
    async def main():
        runner = IndependentStrategyRunner()
        
        # 제시 리버모어 전략으로 모든 시장 분석
        print("🎯 제시 리버모어 전략 실행...")
        results = await run_livermore_strategy('all', 5)
        
        # 결과 출력
        report = runner.generate_report(results)
        print(report)
        
        # 결과 저장
        runner.save_results(results)
    
    asyncio.run(main()) 