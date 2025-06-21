"""
🚀 최적화된 스캘핑 시스템
- ATR, 멀티타임프레임, 모멘텀 분석 통합
- API 호출 최적화 및 일일 한도 관리
- 실시간 신호 생성 및 자동 매매
- v1.1.0 (2024-07-26): 리팩토링 및 구조 개선
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 내부 모듈 import
from .atr_analyzer import ATRAnalyzer, ATRData
from .multi_timeframe_analyzer import MultiTimeframeAnalyzer, MultiTimeFrameSignal
from .momentum_scorer import MomentumScorer, MomentumData

logger = logging.getLogger(__name__)

# --- 데이터 클래스 정의 ---

@dataclass
class ScalpingSignal:
    """통합 스캘핑 신호 정보를 담는 데이터 클래스"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PositionInfo:
    """보유 포지션 정보를 담는 데이터 클래스"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    profit_loss: float
    profit_loss_rate: float

@dataclass
class SystemConfig:
    """시스템 설정을 담는 데이터 클래스"""
    target_symbols: List[str]
    max_positions: int
    risk_per_trade: float
    analysis_interval: int = 60  # 초
    cache_duration_sec: int = 30
    batch_size: int = 10

@dataclass
class SystemState:
    """시스템의 동적 상태를 담는 데이터 클래스"""
    is_running: bool = False
    active_positions: Dict[str, PositionInfo] = field(default_factory=dict)
    signal_cache: Dict[str, ScalpingSignal] = field(default_factory=dict)
    last_analysis_time: Dict[str, datetime] = field(default_factory=dict)

# --- 메인 시스템 클래스 ---

class OptimizedScalpingSystem:
    """
    여러 분석 모듈을 통합하고 API 호출을 최적화하여
    자동 스캘핑 매매를 수행하는 실행 엔진입니다.
    """
    
    def __init__(self, core_trader: Any, daily_api_limit: int = 5000):
        self.trader = core_trader
        self.daily_api_limit = daily_api_limit
        
        # 분석 모듈 초기화
        self._initialize_analyzers()
        
        # 시스템 상태 및 설정
        self.state = SystemState()
        self.config: Optional[SystemConfig] = None

        # 비동기 및 병렬 처리를 위한 Executor
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info("🚀 최적화된 스캘핑 시스템 초기화 완료 (v1.1.0)")

    def _initialize_analyzers(self):
        """분석 모듈들을 초기화합니다."""
        self.atr_analyzer = ATRAnalyzer(optimal_atr_min=0.5, optimal_atr_max=3.0)
        self.momentum_scorer = MomentumScorer(short_period=5, medium_period=20)
        self.multi_analyzer = MultiTimeframeAnalyzer()
        
    # --- Public Methods: 시스템 제어 ---
    
    def start_scalping(self, target_symbols: List[str], 
                      max_concurrent_positions: int = 3,
                      risk_per_trade: float = 0.02) -> None:
        """스캘핑 시스템을 시작합니다."""
        if self.state.is_running:
            logger.warning("⚠️ 시스템이 이미 실행 중입니다.")
            return
        
        self.config = SystemConfig(
            target_symbols=target_symbols,
            max_positions=max_concurrent_positions,
            risk_per_trade=risk_per_trade
        )
        self.state.is_running = True
        
        logger.info(f"🎯 스캘핑 시작: {len(target_symbols)}개 종목, 최대 {max_concurrent_positions}개 포지션")
        
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            logger.info("⏹️ 사용자의 요청으로 시스템을 중단합니다.")
        except Exception as e:
            logger.critical(f"❌ 메인 루프에서 심각한 오류 발생: {e}", exc_info=True)
        finally:
            self.stop_scalping()
    
    def stop_scalping(self) -> None:
        """스캘핑 시스템을 중지합니다."""
        if not self.state.is_running:
            return
        logger.info("🛑 스캘핑 시스템을 중지합니다...")
        self.state.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("✅ 시스템이 안전하게 중지되었습니다.")

    def get_system_status(self) -> Dict[str, Any]:
        """시스템의 현재 상태를 반환합니다."""
        return {
            "is_running": self.state.is_running,
            "config": self.config,
            "active_positions_count": len(self.state.active_positions),
            "active_positions": self.state.active_positions,
            "api_calls_remaining": self.trader.daily_counter.get_remaining_calls()
        }

    # --- Private Methods: 메인 루프 및 워크플로우 ---

    async def _main_loop(self) -> None:
        """시스템의 메인 이벤트 루프"""
        logger.info("🔄 메인 스캘핑 루프 시작")
        while self.state.is_running:
            loop_start_time = time.monotonic()
            try:
                if not self._is_api_limit_ok():
                    logger.warning("🚨 일일 API 한도에 도달하여 5분간 대기합니다.")
                    await asyncio.sleep(300)
                    continue

                await self._update_active_positions()
                
                promising_symbols = await self._find_promising_symbols()
                
                signals = await self._analyze_symbols_in_parallel(promising_symbols)
                
                await self._execute_new_trades(signals)
                
            except Exception as e:
                logger.error(f"❌ 메인 루프 중 오류 발생: {e}", exc_info=True)
            
            # 루프 실행 시간 제어
            elapsed = time.monotonic() - loop_start_time
            sleep_time = max(0, self.config.analysis_interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def _find_promising_symbols(self) -> List[str]:
        """거래 기회가 있을 만한 유망 종목을 탐색합니다."""
        try:
            loop = asyncio.get_running_loop()
            # `get_top_ranking_stocks`는 동기 함수이므로 `run_in_executor` 사용
            top_stocks = await loop.run_in_executor(
                self.executor, self.trader.get_top_ranking_stocks, 50
            )
            if not top_stocks:
                logger.warning("⚠️ 랭킹 데이터를 가져오지 못했습니다.")
                return self.config.target_symbols[:10]

            # 거래량 및 변동성 기반 필터링
            promising = [
                s['symbol'] for s in top_stocks
                if s.get('volume_rate', 0) > 120 and abs(s.get('change_rate', 0)) > 1.0
            ]
            logger.info(f"🔍 {len(promising)}개의 유망 종목 발견")
            return promising[:20] # 분석 부하를 줄이기 위해 상위 20개로 제한
        except Exception as e:
            logger.error(f"❌ 유망 종목 탐색 실패: {e}", exc_info=True)
            return self.config.target_symbols[:10]

    async def _analyze_symbols_in_parallel(self, symbols: List[str]) -> List[ScalpingSignal]:
        """여러 종목을 병렬로 분석하여 거래 신호를 생성합니다."""
        tasks = [self._analyze_one_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_signals = []
        for res in results:
            if isinstance(res, ScalpingSignal):
                valid_signals.append(res)
            elif isinstance(res, Exception):
                logger.warning(f"⚠️ 종목 분석 중 오류 발생: {res}")
        
        return valid_signals

    async def _execute_new_trades(self, signals: List[ScalpingSignal]) -> None:
        """생성된 신호를 바탕으로 새로운 거래를 실행합니다."""
        # 신뢰도 순으로 정렬
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        for signal in signals:
            # 포지션 수 확인
            if len(self.state.active_positions) >= self.config.max_positions:
                logger.info("💼 최대 포지션 수에 도달하여 추가 진입을 보류합니다.")
                break
            
            # 이미 포지션이 있거나, 신뢰도가 낮으면 건너뛰기
            if signal.symbol in self.state.active_positions or signal.confidence < 75:
                        continue
                
            await self._execute_signal(signal)
            # 동시 주문 방지를 위한 짧은 대기
            await asyncio.sleep(1)

    # --- Private Methods: 개별 종목 분석 ---

    async def _analyze_one_symbol(self, symbol: str) -> Optional[ScalpingSignal]:
        """단일 종목을 분석하여 거래 신호를 반환합니다. (병렬 처리용)"""
        # 캐시 확인
        if self._is_cached(symbol):
            return self.state.signal_cache.get(symbol)

        loop = asyncio.get_running_loop()
        try:
            # 데이터 가져오기 (동기 I/O는 executor에서 실행)
            price_data = await loop.run_in_executor(self.executor, self.trader.get_current_price, symbol)
            if not price_data: return None

            ohlcv_data = await loop.run_in_executor(self.executor, self.trader.get_ohlcv, symbol, "1m", 100)
            if ohlcv_data is None or len(ohlcv_data['close']) < 50: return None

            # 분석 실행
            atr_result = self.atr_analyzer.analyze_volatility(
                symbol, ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
            )
            momentum_result = self.momentum_scorer.calculate_batch_momentum(
                symbol, ohlcv_data['close'], ohlcv_data['volume']
            )
            
            # 신호 종합
            signal = self._synthesize_signal(symbol, price_data['price'], atr_result, momentum_result)
            if signal:
                self.state.signal_cache[symbol] = signal
                self.state.last_analysis_time[symbol] = datetime.now()
            return signal
        except Exception as e:
            logger.error(f"❌ {symbol} 분석 실패: {e}", exc_info=True)
            return None
    
    def _synthesize_signal(self, symbol: str, price: float, atr: Optional[ATRData], 
                           momentum: Optional[MomentumData]) -> Optional[ScalpingSignal]:
        """여러 분석 결과를 종합하여 최종 거래 신호를 생성합니다."""
        if not all([atr, momentum]):
            return None
    
        confidence = (atr.scalping_suitability * 0.6) + (momentum.combined_score * 0.4)
        if confidence < 70:
            return None
    
        action = "BUY" if momentum.momentum_direction == "BULLISH" else "SELL"
        
        # 진입/손절/익절가 설정
        entry = price
        stop_loss = entry - atr.atr_value if action == "BUY" else entry + atr.atr_value
        take_profit = entry + atr.atr_value * 1.5 if action == "BUY" else entry - atr.atr_value * 1.5

        position_size = self._calculate_position_size(entry, stop_loss, confidence)
            
            return ScalpingSignal(
                symbol=symbol,
                action=action,
            confidence=round(confidence, 2),
            entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
            details={'atr': atr, 'momentum': momentum}
        )

    # --- Private Methods: 거래 실행 및 관리 ---
    
    async def _execute_signal(self, signal: ScalpingSignal) -> None:
        """거래 신호를 실행하여 주문을 제출합니다."""
        logger.info(f"🚀 신호 실행: {signal.symbol} {signal.action} (신뢰도: {signal.confidence}%)")
        try:
            loop = asyncio.get_running_loop()
            order_result = await loop.run_in_executor(
                self.executor,
                self.trader.place_order,
                    symbol=signal.symbol,
                order_type="market",
                    side=signal.action.lower(),
                quantity=signal.position_size,
                price=signal.entry_price
            )
            if order_result and order_result.get('status') == 'filled':
                logger.info(f"✅ 주문 체결: {signal.symbol} {signal.quantity}주 {signal.action}")
                # 포지션 정보 업데이트는 _update_active_positions에서 처리
            else:
                logger.error(f"❌ 주문 실패: {signal.symbol}, 사유: {order_result.get('message')}")
        except Exception as e:
            logger.error(f"❌ 주문 실행 중 오류: {signal.symbol}, {e}", exc_info=True)

    async def _update_active_positions(self) -> None:
        """현재 보유 포지션을 업데이트하고 청산 조건을 확인합니다."""
        try:
            loop = asyncio.get_running_loop()
            positions_data = await loop.run_in_executor(self.executor, self.trader.get_positions)
            
            updated_positions = {}
            for pos in positions_data:
                # 포지션 정보 객체 생성
                p_info = PositionInfo(**pos)
                updated_positions[p_info.symbol] = p_info
                
                # 청산 조건 확인
                self._check_and_close_position(p_info)

            self.state.active_positions = updated_positions
        except Exception as e:
            logger.error(f"❌ 포지션 업데이트 실패: {e}", exc_info=True)

    def _check_and_close_position(self, pos: PositionInfo):
        """개별 포지션의 손절/익절 조건을 확인하고 청산 주문을 실행합니다."""
        signal = self.state.signal_cache.get(pos.symbol)
        if not signal: return

        should_close = False
        close_reason = ""

        if pos.profit_loss_rate <= -self.config.risk_per_trade:
            should_close, close_reason = True, "손절매"
        elif signal.take_profit and pos.current_price >= signal.take_profit:
            should_close, close_reason = True, "이익 실현"

        if should_close:
            logger.info(f"🏁 포지션 청산: {pos.symbol} ({close_reason})")
            # 비동기 루프가 아니므로 executor로 청산 주문 제출
            self.executor.submit(
                self.trader.place_order,
                symbol=pos.symbol,
                order_type="market",
                side="sell" if signal.action == "BUY" else "buy",
                quantity=pos.quantity
            )
            # 청산 후 캐시에서 제거
            self.state.signal_cache.pop(pos.symbol, None)

    # --- 유틸리티 메서드 ---

    def _is_cached(self, symbol: str) -> bool:
        """분석 결과가 유효한 캐시 기간 내에 있는지 확인합니다."""
        last_time = self.state.last_analysis_time.get(symbol)
        if not last_time:
            return False
        return datetime.now() < last_time + timedelta(seconds=self.config.cache_duration_sec)

    def _is_api_limit_ok(self) -> bool:
        """API 호출 한도가 충분한지 확인합니다."""
        remaining = self.trader.daily_counter.get_remaining_calls()
        # 95% 이상 사용 시 중단
        return remaining > self.daily_api_limit * 0.05

    def _calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float) -> float:
        """계좌 잔고와 리스크를 고려하여 적절한 포지션 크기를 계산합니다."""
        try:
            balance = self.trader.get_balance()
            if not balance or balance.cash == 0:
                return 0.01

            risk_amount_per_trade = balance.cash * self.config.risk_per_trade
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0: return 0.01

            size = risk_amount_per_trade / risk_per_share
            
            # 신뢰도에 따라 포지션 크기 조절
            size *= (confidence / 100)
            
            # 최소 거래 단위에 맞게 조절 (예: 0.01 단위)
            return max(0.01, round(size, 2))
        except Exception as e:
            logger.error(f"❌ 포지션 크기 계산 실패: {e}")
            return 0.01 