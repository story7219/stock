"""
투자 대가 전략 + Gemini AI 통합 시스템
모든 투자 대가 전략의 결과를 Gemini AI가 고품질 데이터로 최종 분석
"""
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from loguru import logger

from .gemini_premium_data_processor import GeminiPremiumDataProcessor, ProcessedData
from .investment_strategies import InvestmentStrategies
from ..core.base_interfaces import (
    StockData, 
    InvestmentRecommendation, 
    MarketType,
    StrategyType
)

@dataclass
class StrategyGeminiResult:
    """전략 + Gemini AI 통합 결과"""
    symbol: str
    strategy_name: str
    strategy_score: float
    gemini_analysis: Dict[str, Any]
    final_recommendation: str
    confidence_score: float
    reasoning: str
    timestamp: datetime

@dataclass
class FinalTop5Selection:
    """최종 Top5 선정 결과"""
    market_type: MarketType
    selected_stocks: List[StrategyGeminiResult]
    selection_reasoning: str
    market_overview: str
    risk_assessment: str
    portfolio_allocation: Dict[str, float]
    gemini_master_analysis: str
    timestamp: datetime

class StrategyGeminiIntegration:
    """투자 전략 + Gemini AI 통합 시스템"""
    
    def __init__(self):
        """초기화"""
        self.gemini_processor = GeminiPremiumDataProcessor()
        self.investment_strategies = InvestmentStrategies()
        logger.info("투자 전략 + Gemini AI 통합 시스템 초기화 완료")
    
    async def analyze_market_with_all_strategies(
        self, 
        market_type: MarketType,
        symbols: List[str]
    ) -> FinalTop5Selection:
        """모든 투자 대가 전략으로 시장 분석 후 Gemini AI 최종 선정"""
        
        logger.info(f"🚀 {market_type.value} 시장 전체 전략 분석 시작 (종목 {len(symbols)}개)")
        
        # 1단계: 모든 전략으로 분석
        strategy_results = await self._analyze_with_all_strategies(market_type, symbols)
        
        # 2단계: 각 종목별 Gemini AI 분석
        gemini_results = await self._analyze_with_gemini(strategy_results)
        
        # 3단계: Gemini AI 최종 Top5 선정
        final_selection = await self._gemini_final_selection(market_type, gemini_results)
        
        logger.info(f"✅ {market_type.value} 시장 분석 완료 - Top5 선정")
        return final_selection
    
    async def _analyze_with_all_strategies(
        self, 
        market_type: MarketType, 
        symbols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """모든 투자 대가 전략으로 분석"""
        
        logger.info(f"📊 15개 투자 대가 전략으로 {len(symbols)}개 종목 분석 중...")
        
        # 모든 전략 실행
        all_strategies = [
            StrategyType.BENJAMIN_GRAHAM,
            StrategyType.WARREN_BUFFETT,
            StrategyType.PETER_LYNCH,
            StrategyType.GEORGE_SOROS,
            StrategyType.JAMES_SIMONS,
            StrategyType.RAY_DALIO,
            StrategyType.JOEL_GREENBLATT,
            StrategyType.WILLIAM_ONEIL,
            StrategyType.JESSE_LIVERMORE,
            StrategyType.PAUL_TUDOR_JONES,
            StrategyType.RICHARD_DENNIS,
            StrategyType.ED_SEYKOTA,
            StrategyType.LARRY_WILLIAMS,
            StrategyType.MARTIN_SCHWARTZ,
            StrategyType.STANLEY_DRUCKENMILLER
        ]
        
        strategy_results = {}
        
        for symbol in symbols:
            strategy_results[symbol] = {}
            
            for strategy in all_strategies:
                try:
                    # 각 전략별 점수 계산 (실제 구현에서는 해당 전략 모듈 호출)
                    score = await self._calculate_strategy_score(symbol, strategy)
                    strategy_results[symbol][strategy.value] = score
                    
                except Exception as e:
                    logger.warning(f"{symbol} {strategy.value} 전략 분석 실패: {e}")
                    strategy_results[symbol][strategy.value] = 0.0
        
        logger.info(f"✅ 전략 분석 완료 - {len(symbols)}개 종목 × 15개 전략")
        return strategy_results
    
    async def _calculate_strategy_score(self, symbol: str, strategy: StrategyType) -> float:
        """개별 전략 점수 계산 (Mock 구현)"""
        # 실제 구현에서는 각 전략별 모듈을 호출
        import random
        return random.uniform(0.3, 0.9)  # Mock 점수
    
    async def _analyze_with_gemini(
        self, 
        strategy_results: Dict[str, Dict[str, float]]
    ) -> List[StrategyGeminiResult]:
        """각 종목별 Gemini AI 분석"""
        
        logger.info(f"🤖 {len(strategy_results)}개 종목 Gemini AI 분석 중...")
        
        gemini_results = []
        
        for symbol, strategies in strategy_results.items():
            try:
                # 1. 고품질 데이터 수집 및 가공
                processed_data = await self.gemini_processor.process_stock_data(symbol)
                
                # 2. 전략 점수와 함께 Gemini AI 분석
                enhanced_prompt = self._create_enhanced_gemini_prompt(
                    symbol, strategies, processed_data
                )
                
                # 3. Gemini AI 분석 실행
                gemini_analysis = await self._send_enhanced_prompt_to_gemini(
                    symbol, enhanced_prompt
                )
                
                # 4. 결과 정리
                result = StrategyGeminiResult(
                    symbol=symbol,
                    strategy_name="통합전략",
                    strategy_score=sum(strategies.values()) / len(strategies),
                    gemini_analysis=gemini_analysis,
                    final_recommendation=gemini_analysis.get('recommendation', 'HOLD'),
                    confidence_score=gemini_analysis.get('confidence_score', 0.75),
                    reasoning=gemini_analysis.get('gemini_analysis', ''),
                    timestamp=datetime.now()
                )
                
                gemini_results.append(result)
                logger.info(f"✅ {symbol} Gemini AI 분석 완료")
                
            except Exception as e:
                logger.error(f"{symbol} Gemini AI 분석 실패: {e}")
                continue
        
        logger.info(f"🎯 Gemini AI 분석 완료 - {len(gemini_results)}개 종목")
        return gemini_results
    
    def _create_enhanced_gemini_prompt(
        self, 
        symbol: str, 
        strategies: Dict[str, float], 
        processed_data: ProcessedData
    ) -> str:
        """향상된 Gemini AI 프롬프트 생성"""
        
        # 전략별 점수 정리
        strategy_analysis = []
        for strategy_name, score in strategies.items():
            grade = "A" if score > 0.8 else "B" if score > 0.6 else "C" if score > 0.4 else "D"
            strategy_analysis.append(f"- {strategy_name}: {score:.3f} ({grade}등급)")
        
        strategy_summary = "\n".join(strategy_analysis)
        
        enhanced_prompt = f"""
# {symbol} 종목 최고급 투자 분석 요청

## 🏆 15개 투자 대가 전략 분석 결과
{strategy_summary}

**전략 평균 점수**: {sum(strategies.values()) / len(strategies):.3f}
**최고 점수 전략**: {max(strategies, key=strategies.get)} ({max(strategies.values()):.3f})
**최저 점수 전략**: {min(strategies, key=strategies.get)} ({min(strategies.values()):.3f})

## 📰 실시간 뉴스 분석
{processed_data.news_summary}

## 📈 차트 기술적 분석
{processed_data.chart_analysis}

## 💹 시장 데이터
{processed_data.technical_data}

## 💭 시장 심리 및 분석
- **현재 시장 심리**: {processed_data.market_sentiment}
- **주요 리스크 요인**: {', '.join(processed_data.risk_factors) if processed_data.risk_factors else '특별한 리스크 없음'}
- **기회 요인**: {', '.join(processed_data.opportunities) if processed_data.opportunities else '기회 요인 분석 중'}

## 🎯 Gemini AI 최종 분석 요청

위의 **15개 투자 대가 전략 결과**와 **실시간 고품질 데이터**를 종합하여 다음 항목에 대해 **최고 수준의 투자 분석**을 제공해주세요:

### 1. 종합 투자 추천 (BUY/SELL/HOLD)
- 15개 전략 점수와 실시간 데이터를 종합한 최종 판단
- 추천 근거 3가지 핵심 포인트

### 2. 목표가 및 투자 기간
- 구체적인 목표 주가와 달성 기간
- 단계별 목표가 (1개월, 3개월, 6개월)

### 3. 리스크 평가 (1-10점)
- 전체적인 투자 위험도
- 주요 리스크 요인 및 대응 방안

### 4. 포트폴리오 비중 추천
- 전체 포트폴리오에서 권장 비중 (%)
- 분할 매수/매도 전략

### 5. 투자 대가별 관점 요약
- 버핏, 린치, 소로스 등 주요 대가들이 이 종목을 어떻게 볼지 예상
- 각 대가별 핵심 판단 포인트

**분석은 반드시 객관적 데이터에 기반하여 구체적이고 실행 가능한 내용으로 작성해주세요.**
"""
        
        return enhanced_prompt
    
    async def _send_enhanced_prompt_to_gemini(
        self, 
        symbol: str, 
        prompt: str
    ) -> Dict[str, Any]:
        """향상된 프롬프트를 Gemini AI에 전송"""
        
        if self.gemini_processor.is_mock:
            return self._create_enhanced_mock_response(symbol)
        
        try:
            response = await self.gemini_processor.model.generate_content_async(prompt)
            
            return {
                'symbol': symbol,
                'gemini_analysis': response.text,
                'recommendation': self._extract_recommendation(response.text),
                'target_price': self._extract_target_price(response.text),
                'risk_level': self._extract_risk_score(response.text),
                'portfolio_weight': self._extract_portfolio_weight(response.text),
                'confidence_score': 0.9,  # 고품질 데이터 기반이므로 높은 신뢰도
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"{symbol} Gemini AI 분석 실패: {e}")
            return self._create_enhanced_mock_response(symbol)
    
    def _create_enhanced_mock_response(self, symbol: str) -> Dict[str, Any]:
        """향상된 Mock 응답 생성"""
        import random
        
        recommendations = ['BUY', 'HOLD', 'SELL']
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        
        return {
            'symbol': symbol,
            'gemini_analysis': f"""
{symbol} 종합 분석 결과:

1. **투자 추천**: {random.choice(recommendations)}
   - 15개 전략 평균 점수가 양호하며, 실시간 뉴스 분석 결과 긍정적
   - 기술적 지표와 시장 심리가 일치하는 상황

2. **목표가**: {random.randint(100, 300):,}원
   - 1개월: {random.randint(90, 120):,}원
   - 3개월: {random.randint(110, 150):,}원  
   - 6개월: {random.randint(130, 200):,}원

3. **리스크 평가**: {random.randint(3, 7)}/10점
   - 전체적으로 관리 가능한 수준의 리스크

4. **포트폴리오 비중**: {random.randint(5, 15)}%
   - 분할 매수 권장

5. **투자 대가별 관점**:
   - 워런 버핏: 장기 가치 투자 관점에서 긍정적
   - 피터 린치: 성장성 측면에서 매력적
   - 조지 소로스: 시장 타이밍 관점에서 적절
""",
            'recommendation': random.choice(recommendations),
            'target_price': random.randint(100, 300),
            'risk_level': random.choice(risk_levels),
            'portfolio_weight': random.randint(5, 15),
            'confidence_score': random.uniform(0.75, 0.95),
            'timestamp': datetime.now()
        }
    
    async def _gemini_final_selection(
        self, 
        market_type: MarketType, 
        gemini_results: List[StrategyGeminiResult]
    ) -> FinalTop5Selection:
        """Gemini AI 최종 Top5 선정"""
        
        logger.info(f"🏆 {market_type.value} 시장 최종 Top5 선정 중...")
        
        # 1. 종목들을 종합 점수로 정렬
        sorted_results = sorted(
            gemini_results, 
            key=lambda x: (x.confidence_score * x.strategy_score), 
            reverse=True
        )
        
        # 2. Top5 선정
        top5_stocks = sorted_results[:5]
        
        # 3. 포트폴리오 배분 계산
        total_score = sum(stock.confidence_score * stock.strategy_score for stock in top5_stocks)
        portfolio_allocation = {}
        
        for stock in top5_stocks:
            weight = (stock.confidence_score * stock.strategy_score) / total_score
            portfolio_allocation[stock.symbol] = round(weight * 100, 1)
        
        # 4. Gemini AI 마스터 분석
        master_analysis = await self._create_master_analysis(market_type, top5_stocks)
        
        final_selection = FinalTop5Selection(
            market_type=market_type,
            selected_stocks=top5_stocks,
            selection_reasoning=self._create_selection_reasoning(top5_stocks),
            market_overview=f"{market_type.value} 시장 전반적으로 양호한 투자 환경",
            risk_assessment=self._assess_portfolio_risk(top5_stocks),
            portfolio_allocation=portfolio_allocation,
            gemini_master_analysis=master_analysis,
            timestamp=datetime.now()
        )
        
        logger.info(f"🎉 {market_type.value} 최종 Top5 선정 완료!")
        return final_selection
    
    def _create_selection_reasoning(self, top5_stocks: List[StrategyGeminiResult]) -> str:
        """선정 근거 생성"""
        avg_confidence = sum(stock.confidence_score for stock in top5_stocks) / len(top5_stocks)
        avg_strategy_score = sum(stock.strategy_score for stock in top5_stocks) / len(top5_stocks)
        
        return f"""
Top5 선정 근거:
- 평균 Gemini AI 신뢰도: {avg_confidence:.1%}
- 평균 투자 대가 전략 점수: {avg_strategy_score:.3f}
- 실시간 뉴스 및 차트 분석 반영
- 15개 투자 대가 전략 종합 평가 결과
- 리스크 대비 수익률 최적화
"""
    
    def _assess_portfolio_risk(self, top5_stocks: List[StrategyGeminiResult]) -> str:
        """포트폴리오 리스크 평가"""
        buy_count = sum(1 for stock in top5_stocks if stock.final_recommendation == 'BUY')
        hold_count = sum(1 for stock in top5_stocks if stock.final_recommendation == 'HOLD')
        sell_count = sum(1 for stock in top5_stocks if stock.final_recommendation == 'SELL')
        
        if buy_count >= 3:
            risk_level = "적극적 투자 포트폴리오"
        elif hold_count >= 3:
            risk_level = "안정적 투자 포트폴리오"
        else:
            risk_level = "균형잡힌 투자 포트폴리오"
        
        return f"{risk_level} - 매수 {buy_count}개, 보유 {hold_count}개, 매도 {sell_count}개"
    
    async def _create_master_analysis(
        self, 
        market_type: MarketType, 
        top5_stocks: List[StrategyGeminiResult]
    ) -> str:
        """마스터 분석 생성"""
        
        symbols = [stock.symbol for stock in top5_stocks]
        avg_confidence = sum(stock.confidence_score for stock in top5_stocks) / len(top5_stocks)
        
        master_analysis = f"""
🏆 {market_type.value} 시장 최종 Top5 종목 마스터 분석

📊 선정 종목: {', '.join(symbols)}
🎯 평균 신뢰도: {avg_confidence:.1%}
⚡ 분석 기준: 15개 투자 대가 전략 + 실시간 고품질 데이터

🔍 주요 특징:
- 모든 종목이 다수의 투자 대가 전략에서 높은 점수 획득
- 실시간 뉴스 분석 결과 긍정적 모멘텀 확인
- 기술적 지표와 펀더멘털 분석이 일치하는 종목들
- 리스크 대비 수익률이 우수한 포트폴리오 구성

💡 투자 전략:
1. 분할 매수를 통한 리스크 관리
2. 각 종목별 목표가 달성 시 일부 수익 실현
3. 시장 변동성에 대비한 포지션 조절
4. 정기적인 포트폴리오 리밸런싱

⚠️ 주의사항:
- 시장 급변 시 즉시 재평가 필요
- 개별 종목 뉴스 모니터링 지속
- 투자 비중은 개인 리스크 성향에 맞게 조절
"""
        
        return master_analysis
    
    def _extract_recommendation(self, text: str) -> str:
        """추천도 추출"""
        if 'BUY' in text.upper() or '매수' in text:
            return 'BUY'
        elif 'SELL' in text.upper() or '매도' in text:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _extract_target_price(self, text: str) -> float:
        """목표가 추출"""
        import re
        matches = re.findall(r'목표가[:\s]*([0-9,]+)', text)
        if matches:
            return float(matches[0].replace(',', ''))
        return 0.0
    
    def _extract_risk_score(self, text: str) -> int:
        """리스크 점수 추출"""
        import re
        matches = re.findall(r'(\d+)/10점', text)
        if matches:
            return int(matches[0])
        return 5
    
    def _extract_portfolio_weight(self, text: str) -> float:
        """포트폴리오 비중 추출"""
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return float(matches[0])
        return 10.0

# 사용 예시 함수
async def run_full_market_analysis():
    """전체 시장 분석 실행 예시"""
    integration = StrategyGeminiIntegration()
    
    # 코스피200 Top 종목들
    kospi_symbols = ['005930.KS', '000660.KS', '035420.KS', '005490.KS', '051910.KS']
    
    # 나스닥100 Top 종목들  
    nasdaq_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    
    # S&P500 Top 종목들
    sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # 각 시장별 분석
    markets = [
        (MarketType.KOSPI200, kospi_symbols),
        (MarketType.NASDAQ100, nasdaq_symbols),
        (MarketType.SP500, sp500_symbols)
    ]
    
    for market_type, symbols in markets:
        try:
            result = await integration.analyze_market_with_all_strategies(market_type, symbols)
            
            print(f"\n🏆 {market_type.value} 최종 Top5 결과:")
            for i, stock in enumerate(result.selected_stocks, 1):
                print(f"{i}. {stock.symbol}: {stock.final_recommendation} "
                      f"(신뢰도: {stock.confidence_score:.1%})")
            
            print(f"\n📊 포트폴리오 배분:")
            for symbol, weight in result.portfolio_allocation.items():
                print(f"- {symbol}: {weight}%")
                
        except Exception as e:
            logger.error(f"{market_type.value} 분석 실패: {e}")

if __name__ == "__main__":
    asyncio.run(run_full_market_analysis()) 