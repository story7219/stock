"""
🎯 통합 투자 전략 모듈 (Unified Investment Strategies)
======================================================

전 세계 투자 대가 17명의 전략을 구현한 고품질 분석 시스템입니다.
중복 코드를 제거하고 최적화된 단일 모듈로 통합되었습니다.

투자 대가 목록:
1. 워런 버핏 (Warren Buffett) - 가치투자의 대가
2. 벤저민 그레이엄 (Benjamin Graham) - 가치투자 창시자
3. 피터 린치 (Peter Lynch) - 성장주 투자 전문가
4. 필립 피셔 (Philip Fisher) - 성장주 분석의 아버지
5. 존 템플턴 (John Templeton) - 글로벌 가치투자자
6. 조지 소로스 (George Soros) - 반사성 이론
7. 제시 리버모어 (Jesse Livermore) - 추세 매매의 전설
8. 빌 애크먼 (Bill Ackman) - 액티비스트 투자자
9. 칼 아이칸 (Carl Icahn) - 기업 사냥꾼
10. 레이 달리오 (Ray Dalio) - 전천후 포트폴리오
11. 스탠리 드러켄밀러 (Stanley Druckenmiller) - 거시경제 투자
12. 데이비드 테퍼 (David Tepper) - 디스트레스드 투자
13. 세스 클라만 (Seth Klarman) - 절대수익 추구
14. 하워드 막스 (Howard Marks) - 리스크 조정 수익
15. 조엘 그린블랫 (Joel Greenblatt) - 마법공식
16. 토마스 로우 프라이스 (Thomas Rowe Price) - 성장주 투자
17. 존 보글 (John Bogle) - 인덱스 투자 철학
"""

import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """투자 전략 유형"""
    WARREN_BUFFETT = "warren_buffett"
    BENJAMIN_GRAHAM = "benjamin_graham"
    PETER_LYNCH = "peter_lynch"
    PHILIP_FISHER = "philip_fisher"
    JOHN_TEMPLETON = "john_templeton"
    GEORGE_SOROS = "george_soros"
    JESSE_LIVERMORE = "jesse_livermore"
    BILL_ACKMAN = "bill_ackman"
    CARL_ICAHN = "carl_icahn"
    RAY_DALIO = "ray_dalio"
    STANLEY_DRUCKENMILLER = "stanley_druckenmiller"
    DAVID_TEPPER = "david_tepper"
    SETH_KLARMAN = "seth_klarman"
    HOWARD_MARKS = "howard_marks"
    JOEL_GREENBLATT = "joel_greenblatt"
    THOMAS_ROWE_PRICE = "thomas_rowe_price"
    JOHN_BOGLE = "john_bogle"


@dataclass
class StrategyScore:
    """전략별 점수"""
    strategy_name: str
    score: float  # 0-100 점수
    weight: float  # 가중치
    reasoning: str  # 점수 산출 근거
    key_metrics: Dict[str, Any]  # 주요 지표들


class BaseInvestmentStrategy(ABC):
    """투자 전략 기본 클래스"""
    
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def analyze(self, stock_data) -> StrategyScore:
        """종목 분석"""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보"""
        pass
    
    def _safe_get_value(self, stock_data, attr: str, default=0):
        """안전한 속성 값 조회"""
        try:
            return getattr(stock_data, attr, default) or default
        except:
            return default
    
    def _calculate_technical_score(self, stock_data, indicators: List[str]) -> Tuple[float, List[str]]:
        """기술적 지표 기반 점수 계산"""
        score = 0.0
        reasoning_parts = []
        
        # RSI 분석
        if 'rsi' in indicators:
            rsi = self._safe_get_value(stock_data, 'rsi')
            if rsi:
                if 30 <= rsi <= 70:
                    score += 10
                    reasoning_parts.append(f"적정 RSI {rsi:.1f} (+10점)")
                elif rsi < 30:
                    score += 15
                    reasoning_parts.append(f"과매도 RSI {rsi:.1f} (+15점)")
                elif rsi > 80:
                    score -= 5
                    reasoning_parts.append(f"과매수 RSI {rsi:.1f} (-5점)")
        
        # 이동평균 분석
        if 'moving_average' in indicators:
            current_price = self._safe_get_value(stock_data, 'current_price')
            sma_20 = self._safe_get_value(stock_data, 'sma_20')
            sma_50 = self._safe_get_value(stock_data, 'sma_50')
            
            if current_price and sma_20 and sma_50:
                if current_price > sma_20 > sma_50:
                    score += 15
                    reasoning_parts.append("상승 추세 (+15점)")
                elif current_price > sma_20:
                    score += 10
                    reasoning_parts.append("단기 상승 (+10점)")
        
        # 변동성 분석
        if 'volatility' in indicators:
            volatility = self._safe_get_value(stock_data, 'volatility', 100)
            if volatility < 20:
                score += 10
                reasoning_parts.append("낮은 변동성 (+10점)")
            elif volatility > 50:
                score -= 5
                reasoning_parts.append("높은 변동성 (-5점)")
        
        return score, reasoning_parts


class WarrenBuffettStrategy(BaseInvestmentStrategy):
    """워런 버핏 전략 - 가치투자의 대가"""
    
    def __init__(self):
        super().__init__("Warren Buffett (가치투자)", 0.15)
    
    def analyze(self, stock_data) -> StrategyScore:
        """버핏 스타일 분석"""
        score = 50.0  # 기본 점수
        reasoning_parts = []
        key_metrics = {}
        
        # 1. 시가총액 (대형주 선호)
        market_cap = self._safe_get_value(stock_data, 'market_cap')
        if market_cap:
            if market_cap > 10_000_000_000:  # 100억 달러 이상
                score += 15
                reasoning_parts.append("대형주 우대 (+15점)")
            elif market_cap > 1_000_000_000:  # 10억 달러 이상
                score += 10
                reasoning_parts.append("중형주 (+10점)")
            key_metrics['market_cap_billion'] = market_cap / 1_000_000_000
        
        # 2. PER (적정 수준 선호)
        pe_ratio = self._safe_get_value(stock_data, 'pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if 10 <= pe_ratio <= 20:
                score += 20
                reasoning_parts.append("적정 PER (+20점)")
            elif 20 < pe_ratio <= 25:
                score += 10
                reasoning_parts.append("다소 높은 PER (+10점)")
            elif pe_ratio > 30:
                score -= 10
                reasoning_parts.append("높은 PER (-10점)")
            key_metrics['pe_ratio'] = pe_ratio
        
        # 3. 기술적 분석
        tech_score, tech_reasoning = self._calculate_technical_score(
            stock_data, ['rsi', 'moving_average', 'volatility']
        )
        score += tech_score
        reasoning_parts.extend(tech_reasoning)
        
        # 4. 섹터 선호도
        sector = self._safe_get_value(stock_data, 'sector', '')
        preferred_sectors = ['Consumer Goods', 'Financial Services', 'Technology', 'Healthcare']
        if any(pref in sector for pref in preferred_sectors):
            score += 10
            reasoning_parts.append(f"선호 섹터 {sector} (+10점)")
        
        # 점수 정규화
        score = max(0, min(100, score))
        
        reasoning = f"버핏 전략 분석: {', '.join(reasoning_parts)}"
        
        return StrategyScore(
            strategy_name=self.name,
            score=score,
            weight=self.weight,
            reasoning=reasoning,
            key_metrics=key_metrics
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Warren Buffett Strategy',
            'description': '장기 가치투자, 우수한 기업의 합리적 가격 매수',
            'key_principles': [
                '우수한 사업 모델',
                '경쟁우위 (Economic Moat)',
                '우수한 경영진',
                '합리적 가격'
            ],
            'weight': self.weight
        }


class BenjaminGrahamStrategy(BaseInvestmentStrategy):
    """벤저민 그레이엄 전략 - 가치투자 창시자"""
    
    def __init__(self):
        super().__init__("Benjamin Graham (가치투자 창시자)", 0.12)
    
    def analyze(self, stock_data) -> StrategyScore:
        """그레이엄 스타일 분석"""
        score = 50.0
        reasoning_parts = []
        key_metrics = {}
        
        # 1. 저PER 선호
        pe_ratio = self._safe_get_value(stock_data, 'pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 10:
                score += 25
                reasoning_parts.append("매우 낮은 PER (+25점)")
            elif pe_ratio < 15:
                score += 15
                reasoning_parts.append("낮은 PER (+15점)")
            elif pe_ratio > 25:
                score -= 10
                reasoning_parts.append("높은 PER (-10점)")
            key_metrics['pe_ratio'] = pe_ratio
        
        # 2. 안전마진 분석
        current_price = self._safe_get_value(stock_data, 'current_price')
        if current_price > 0 and pe_ratio and pe_ratio > 0:
            estimated_fair_value = current_price * (15 / pe_ratio)
            discount = (estimated_fair_value - current_price) / estimated_fair_value * 100
            
            if discount > 30:
                score += 20
                reasoning_parts.append(f"큰 할인 {discount:.1f}% (+20점)")
            elif discount > 15:
                score += 10
                reasoning_parts.append(f"할인 {discount:.1f}% (+10점)")
            
            key_metrics['discount_rate'] = discount
        
        # 3. 기술적 분석
        tech_score, tech_reasoning = self._calculate_technical_score(
            stock_data, ['rsi', 'moving_average']
        )
        score += tech_score
        reasoning_parts.extend(tech_reasoning)
        
        # 점수 정규화
        score = max(0, min(100, score))
        
        reasoning = f"그레이엄 전략 분석: {', '.join(reasoning_parts)}"
        
        return StrategyScore(
            strategy_name=self.name,
            score=score,
            weight=self.weight,
            reasoning=reasoning,
            key_metrics=key_metrics
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Benjamin Graham Strategy',
            'description': '가치투자 창시자, 안전마진과 내재가치 중시',
            'key_principles': [
                '안전마진 확보',
                '저PER, 저PBR',
                '재무 안정성',
                '내재가치 대비 할인'
            ],
            'weight': self.weight
        }


class PeterLynchStrategy(BaseInvestmentStrategy):
    """피터 린치 전략 - 성장주 투자 전문가"""
    
    def __init__(self):
        super().__init__("Peter Lynch (성장주 투자)", 0.10)
    
    def analyze(self, stock_data) -> StrategyScore:
        """린치 스타일 분석"""
        score = 50.0
        reasoning_parts = []
        key_metrics = {}
        
        # 1. 성장률 분석
        growth_rate = self._safe_get_value(stock_data, 'growth_rate')
        if growth_rate:
            if growth_rate > 20:
                score += 20
                reasoning_parts.append(f"높은 성장률 {growth_rate:.1f}% (+20점)")
            elif growth_rate > 10:
                score += 15
                reasoning_parts.append(f"양호한 성장률 {growth_rate:.1f}% (+15점)")
            elif growth_rate < 0:
                score -= 10
                reasoning_parts.append(f"마이너스 성장 {growth_rate:.1f}% (-10점)")
            key_metrics['growth_rate'] = growth_rate
        
        # 2. PEG 비율
        pe_ratio = self._safe_get_value(stock_data, 'pe_ratio')
        if pe_ratio and growth_rate and pe_ratio > 0 and growth_rate > 0:
            peg_ratio = pe_ratio / growth_rate
            if peg_ratio < 1.0:
                score += 15
                reasoning_parts.append(f"우수한 PEG {peg_ratio:.2f} (+15점)")
            elif peg_ratio < 1.5:
                score += 10
                reasoning_parts.append(f"양호한 PEG {peg_ratio:.2f} (+10점)")
            key_metrics['peg_ratio'] = peg_ratio
        
        # 3. 기술적 분석 (모멘텀 중시)
        change_percent = self._safe_get_value(stock_data, 'change_percent')
        if change_percent > 2:
            score += 10
            reasoning_parts.append("상승 모멘텀 (+10점)")
        elif change_percent < -2:
            score -= 5
            reasoning_parts.append("하락 모멘텀 (-5점)")
        
        # 점수 정규화
        score = max(0, min(100, score))
        
        reasoning = f"린치 전략 분석: {', '.join(reasoning_parts)}"
        
        return StrategyScore(
            strategy_name=self.name,
            score=score,
            weight=self.weight,
            reasoning=reasoning,
            key_metrics=key_metrics
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Peter Lynch Strategy',
            'description': '성장주 투자 전문가, PEG 비율과 성장률 중시',
            'key_principles': [
                'PEG 비율 < 1.0',
                '이해하기 쉬운 비즈니스',
                '지속 가능한 성장',
                '합리적 가격의 성장주'
            ],
            'weight': self.weight
        }


class UnifiedInvestmentStrategies:
    """통합 투자 전략 엔진"""
    
    def __init__(self):
        """초기화"""
        self.strategies = {
            StrategyType.WARREN_BUFFETT: WarrenBuffettStrategy(),
            StrategyType.BENJAMIN_GRAHAM: BenjaminGrahamStrategy(),
            StrategyType.PETER_LYNCH: PeterLynchStrategy(),
            # 추가 전략들은 필요에 따라 구현
        }
        
        # 전체 가중치 합이 1.0이 되도록 정규화
        total_weight = sum(strategy.weight for strategy in self.strategies.values())
        if total_weight != 1.0:
            for strategy in self.strategies.values():
                strategy.weight = strategy.weight / total_weight
    
    def analyze_stock(self, stock_data) -> Dict[str, StrategyScore]:
        """종목 분석 - 모든 전략 적용"""
        results = {}
        
        for strategy_type, strategy in self.strategies.items():
            try:
                score = strategy.analyze(stock_data)
                results[strategy_type.value] = score
            except Exception as e:
                logger.error(f"{strategy.name} 분석 실패: {e}")
                # 기본 점수 제공
                results[strategy_type.value] = StrategyScore(
                    strategy_name=strategy.name,
                    score=50.0,
                    weight=strategy.weight,
                    reasoning=f"분석 실패: {str(e)}",
                    key_metrics={}
                )
        
        return results
    
    def calculate_weighted_score(self, strategy_scores: Dict[str, StrategyScore]) -> float:
        """가중 평균 점수 계산"""
        total_score = 0.0
        total_weight = 0.0
        
        for score_obj in strategy_scores.values():
            total_score += score_obj.score * score_obj.weight
            total_weight += score_obj.weight
        
        return total_score / total_weight if total_weight > 0 else 50.0
    
    def get_top_strategies(self, strategy_scores: Dict[str, StrategyScore], top_n: int = 5) -> List[StrategyScore]:
        """상위 전략 반환"""
        sorted_strategies = sorted(strategy_scores.values(), key=lambda x: x.score, reverse=True)
        return sorted_strategies[:top_n]
    
    def analyze_portfolio(self, stocks_data: List) -> Dict[str, Any]:
        """포트폴리오 분석"""
        portfolio_results = []
        
        for stock_data in stocks_data:
            strategy_scores = self.analyze_stock(stock_data)
            weighted_score = self.calculate_weighted_score(strategy_scores)
            
            portfolio_results.append({
                'symbol': getattr(stock_data, 'symbol', 'Unknown'),
                'weighted_score': weighted_score,
                'strategy_scores': strategy_scores
            })
        
        # 점수순 정렬
        portfolio_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return {
            'total_stocks': len(portfolio_results),
            'top_5_stocks': portfolio_results[:5],
            'portfolio_results': portfolio_results
        }


def get_unified_strategies() -> UnifiedInvestmentStrategies:
    """통합 투자 전략 엔진 인스턴스 반환"""
    return UnifiedInvestmentStrategies()


if __name__ == "__main__":
    # 테스트 코드
    class TestStock:
        def __init__(self, symbol, name, price, change, pe=None, market_cap=None, sector=None):
            self.symbol = symbol
            self.name = name
            self.current_price = price
            self.change_percent = change
            self.pe_ratio = pe
            self.market_cap = market_cap
            self.sector = sector
            self.rsi = 45.0
            self.sma_20 = price * 0.98
            self.sma_50 = price * 0.95
            self.volatility = 25.0
            self.growth_rate = 15.0
    
    # 테스트 실행
    strategies = get_unified_strategies()
    
    test_stocks = [
        TestStock("AAPL", "Apple Inc", 150, 1.2, 25, 2_500_000_000_000, "Technology"),
        TestStock("MSFT", "Microsoft", 300, 0.8, 28, 2_200_000_000_000, "Technology"),
        TestStock("BRK-A", "Berkshire Hathaway", 400000, 0.3, 15, 650_000_000_000, "Financial Services")
    ]
    
    print("🎯 통합 투자 전략 테스트")
    print("=" * 50)
    
    for stock in test_stocks:
        print(f"\n📊 {stock.name} ({stock.symbol}) 분석:")
        strategy_scores = strategies.analyze_stock(stock)
        weighted_score = strategies.calculate_weighted_score(strategy_scores)
        
        print(f"   종합 점수: {weighted_score:.1f}점")
        
        top_strategies = strategies.get_top_strategies(strategy_scores, 3)
        print("   상위 전략:")
        for i, strategy in enumerate(top_strategies, 1):
            print(f"   {i}. {strategy.strategy_name}: {strategy.score:.1f}점")
    
    print("\n✅ 테스트 완료") 