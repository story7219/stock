"""
척후병 매수 전략 관리
5개 후보 → 4개 척후병 → 3일 오디션 → 2개 최종 선정
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .base_strategy import BaseStrategy, StrategySignal

@dataclass
class ScoutConfig:
    """척후병 전략 설정"""
    enabled: bool = True
    candidate_count: int = 5  # 후보 종목 수
    scout_count: int = 4      # 척후병 매수 수
    final_count: int = 2      # 최종 선정 수
    scout_shares: int = 1     # 척후병당 매수 주식 수
    evaluation_period: int = 3  # 오디션 기간 (일)
    
    # 상태 추적
    evaluation_start: Optional[datetime] = None
    candidates: List[str] = field(default_factory=list)
    scout_positions: Dict[str, Dict] = field(default_factory=dict)

class ScoutStrategyManager(BaseStrategy):
    """척후병 전략 관리자"""
    
    def __init__(self, config: ScoutConfig = None):
        super().__init__("척후병 전략")
        self.config = config or ScoutConfig()
        self.quality_stocks = [
            '005930', '000660', '035420', '005490', '051910', 
            '035720', '006400', '028260', '068270', '207940'
        ]
    
    async def analyze(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """척후병 전략 분석"""
        if not self.config.enabled:
            return None
        
        # 현재 단계 확인
        current_phase = self._get_current_phase()
        
        if current_phase == "CANDIDATE_SELECTION":
            return await self._analyze_candidate_selection(stock_code, market_data)
        elif current_phase == "SCOUT_BUYING":
            return await self._analyze_scout_buying(stock_code, market_data)
        elif current_phase == "EVALUATION":
            return await self._analyze_evaluation(stock_code, market_data)
        elif current_phase == "FINAL_SELECTION":
            return await self._analyze_final_selection(stock_code, market_data)
        
        return None
    
    def _get_current_phase(self) -> str:
        """현재 척후병 전략 단계 확인"""
        if not self.config.candidates:
            return "CANDIDATE_SELECTION"
        elif not self.config.scout_positions:
            return "SCOUT_BUYING"
        elif self.config.evaluation_start:
            if datetime.now() < self.config.evaluation_start + timedelta(days=self.config.evaluation_period):
                return "EVALUATION"
            else:
                return "FINAL_SELECTION"
        return "CANDIDATE_SELECTION"
    
    async def _analyze_candidate_selection(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """후보 종목 선정 분석"""
        if stock_code in self.quality_stocks and len(self.config.candidates) < self.config.candidate_count:
            # AI 추천 점수나 기술적 분석 점수 계산
            score = await self._calculate_candidate_score(stock_code, market_data)
            
            if score > 0.6:  # 임계값 이상
                return StrategySignal(
                    action="CANDIDATE",
                    confidence=score,
                    reason=f"후보 종목 선정 (점수: {score:.2f})",
                    priority=1,
                    metadata={"phase": "candidate_selection", "score": score}
                )
        return None
    
    async def _analyze_scout_buying(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """척후병 매수 분석"""
        if stock_code in self.config.candidates[:self.config.scout_count]:
            return StrategySignal(
                action="BUY",
                confidence=0.8,
                reason="척후병 매수",
                priority=1,
                quantity=self.config.scout_shares,
                metadata={"phase": "scout_buying"}
            )
        return None
    
    async def _analyze_evaluation(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """오디션 기간 평가"""
        if stock_code in self.config.scout_positions:
            # 성과 평가 로직
            performance = await self._evaluate_scout_performance(stock_code, market_data)
            
            return StrategySignal(
                action="HOLD",
                confidence=0.7,
                reason=f"오디션 진행 중 (성과: {performance:.2f}%)",
                priority=2,
                metadata={"phase": "evaluation", "performance": performance}
            )
        return None
    
    async def _analyze_final_selection(self, stock_code: str, market_data: Dict) -> Optional[StrategySignal]:
        """최종 선정 분석"""
        # 상위 2개 종목 선정 로직
        top_performers = await self._select_top_performers()
        
        if stock_code in top_performers:
            return StrategySignal(
                action="FINAL_SELECT",
                confidence=0.9,
                reason="최종 선정 - 피보나치 전략 적용 대상",
                priority=1,
                metadata={"phase": "final_selection", "rank": top_performers.index(stock_code) + 1}
            )
        else:
            return StrategySignal(
                action="SELL",
                confidence=0.8,
                reason="오디션 탈락 - 매도",
                priority=2,
                metadata={"phase": "final_selection"}
            )
    
    async def _calculate_candidate_score(self, stock_code: str, market_data: Dict) -> float:
        """후보 종목 점수 계산"""
        try:
            score = 0.0
            
            # 기본 점수 (우량주)
            if stock_code in self.quality_stocks:
                score += 0.3
            
            # 기술적 분석 점수
            technical_score = market_data.get('technical_score', 0.5)
            score += technical_score * 0.4
            
            # AI 추천 점수
            ai_score = market_data.get('ai_score', 0.5)
            score += ai_score * 0.3
            
            return min(1.0, score)
        except Exception as e:
            logging.error(f"❌ 후보 점수 계산 오류 ({stock_code}): {e}")
            return 0.0
    
    async def _evaluate_scout_performance(self, stock_code: str, market_data: Dict) -> float:
        """척후병 성과 평가"""
        try:
            position = self.config.scout_positions.get(stock_code, {})
            if not position:
                return 0.0
            
            buy_price = position.get('buy_price', 0)
            current_price = market_data.get('current_price', 0)
            
            if buy_price > 0:
                return ((current_price - buy_price) / buy_price) * 100
            return 0.0
        except Exception as e:
            logging.error(f"❌ 척후병 성과 평가 오류 ({stock_code}): {e}")
            return 0.0
    
    async def _select_top_performers(self) -> List[str]:
        """상위 성과 종목 선정"""
        try:
            performances = []
            for stock_code in self.config.scout_positions:
                # 각 종목의 성과 계산 (구현 필요)
                performance = 0.0  # 실제 성과 계산 로직
                performances.append((stock_code, performance))
            
            # 성과순 정렬하여 상위 2개 선정
            performances.sort(key=lambda x: x[1], reverse=True)
            return [stock for stock, _ in performances[:self.config.final_count]]
        except Exception as e:
            logging.error(f"❌ 상위 성과자 선정 오류: {e}")
            return []
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        return {
            "name": self.name,
            "enabled": self.config.enabled,
            "current_phase": self._get_current_phase(),
            "candidates_count": len(self.config.candidates),
            "scout_positions_count": len(self.config.scout_positions),
            "evaluation_start": self.config.evaluation_start,
            "config": {
                "candidate_count": self.config.candidate_count,
                "scout_count": self.config.scout_count,
                "final_count": self.config.final_count,
                "evaluation_period": self.config.evaluation_period
            }
        }
    
    def start_evaluation(self):
        """오디션 시작"""
        self.config.evaluation_start = datetime.now()
        logging.info(f"🎬 척후병 오디션 시작 - {self.config.evaluation_period}일간")
    
    def add_scout_position(self, stock_code: str, buy_price: int, quantity: int):
        """척후병 포지션 추가"""
        self.config.scout_positions[stock_code] = {
            'buy_price': buy_price,
            'quantity': quantity,
            'buy_time': datetime.now()
        }
        logging.info(f"🎖️ 척후병 포지션 추가: {stock_code} @ {buy_price:,}원 {quantity}주") 