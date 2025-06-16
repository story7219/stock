"""
🎯 고급 자동매매 트레이더 - 핵심 로직 (import 수정)
기존 전략 로직을 그대로 유지하면서 최적화
"""

import asyncio
import logging
import sys
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 절대 import로 변경
from config import config
from strategies import ScoutStrategy, FibonacciStrategy, TechnicalAnalyzer
from core.data_manager import DataManager
from core.order_executor import OrderExecutor
from core.notifier import TelegramNotifier

@dataclass
class TradingState:
    """매매 상태 관리"""
    scout_candidates: List[str] = field(default_factory=list)
    scout_positions: Dict[str, dict] = field(default_factory=dict)
    final_selections: List[str] = field(default_factory=list)
    fibonacci_positions: Dict[str, dict] = field(default_factory=dict)
    current_phase: str = "SCOUT"  # SCOUT, FIBONACCI, MONITORING

class AdvancedTrader:
    """고급 자동매매 트레이더"""
    
    def __init__(self):
        # 로거 설정 (버퍼 분리 오류 방지)
        self.logger = self._setup_safe_logger()
        self.config = config
        
        # 실행 상태 관리 (running 속성 추가)
        self.running = False
        self.is_initialized = False
        
        # 핵심 컴포넌트 초기화
        self.data_manager = None
        self.order_executor = None
        self.notifier = None
        
        # 전략 컴포넌트 초기화
        self.scout_strategy = None
        self.fibonacci_strategy = None
        self.technical_analyzer = None
        
        # 상태 관리
        self.state = TradingState()
    
    def _setup_safe_logger(self) -> logging.Logger:
        """안전한 로거 설정"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        
        if not logger.handlers:
            # 콘솔 핸들러 설정
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 포맷터 설정
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        
        return logger
    
    async def initialize(self):
        """시스템 초기화"""
        if self.is_initialized:
            return
            
        self.logger.info("🔧 시스템 초기화 중...")
        
        try:
            # 핵심 컴포넌트 초기화
            self.data_manager = DataManager()
            self.order_executor = OrderExecutor()
            self.notifier = TelegramNotifier()
            
            # 전략 컴포넌트 초기화 (기존 로직 유지)
            self.scout_strategy = ScoutStrategy()
            self.fibonacci_strategy = FibonacciStrategy()
            self.technical_analyzer = TechnicalAnalyzer()
            
            # 각 컴포넌트 초기화
            await self.data_manager.initialize()
            await self.order_executor.initialize()
            await self.notifier.initialize()
            
            # 전략 초기화
            await self.scout_strategy.initialize()
            await self.fibonacci_strategy.initialize()
            await self.technical_analyzer.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            raise
    
    async def run(self):
        """메인 실행 루프 (오류 수정)"""
        if not self.is_initialized:
            await self.initialize()
            
        self.running = True  # running 속성 설정
        self.logger.info("🚀 자동매매 시작")
        
        try:
            while self.running:
                current_time = datetime.now().time()
                
                # 장중 시간 체크 (9:00 ~ 15:30)
                if self._is_market_hours(current_time):
                    await self._execute_trading_cycle()
                else:
                    self.logger.info("⏰ 장외 시간 - 대기 중...")
                    await asyncio.sleep(300)  # 5분 대기
                
                # 시스템 상태 체크
                if not self.running:
                    break
                    
                await asyncio.sleep(60)  # 1분 간격 체크
                
        except KeyboardInterrupt:
            self.logger.info("👋 사용자에 의해 중단됨")
        except Exception as e:
            self.logger.error(f"❌ 실행 중 오류: {e}")
            if self.notifier:
                try:
                    await self.notifier.send_error_alert(str(e))
                except:
                    pass  # 알림 실패해도 계속 진행
        finally:
            await self._cleanup()
    
    async def _cleanup(self):
        """시스템 정리"""
        self.running = False
        self.logger.info("🧹 시스템 정리 중...")
        
        try:
            # 각 컴포넌트 정리
            if self.data_manager:
                await self.data_manager.cleanup()
            if self.order_executor:
                await self.order_executor.cleanup()
            if self.notifier:
                await self.notifier.cleanup()
                
        except Exception as e:
            self.logger.error(f"❌ 정리 중 오류: {e}")
        finally:
            self.logger.info("🔚 시스템 종료 완료")
    
    async def _execute_trading_cycle(self):
        """매매 사이클 실행 (기존 로직 유지)"""
        try:
            if self.state.current_phase == "SCOUT":
                await self._execute_scout_phase()
            elif self.state.current_phase == "FIBONACCI":
                await self._execute_fibonacci_phase()
            elif self.state.current_phase == "MONITORING":
                await self._execute_monitoring_phase()
                
        except Exception as e:
            self.logger.error(f"❌ 매매 사이클 오류: {e}")
    
    async def _execute_scout_phase(self):
        """척후병 단계 실행 (디버깅 강화)"""
        self.logger.info("🔍 척후병 단계 실행")
        
        try:
            # 1. 후보 종목 선정
            if not self.state.scout_candidates:
                candidates = await self.scout_strategy.select_candidates()
                self.state.scout_candidates = candidates[:self.config.scout_candidates]
                self.logger.info(f"📋 척후병 후보: {self.state.scout_candidates}")
            
            # 2. 척후병 매수 실행 (디버깅 강화)
            for symbol in self.state.scout_candidates:
                if symbol not in self.state.scout_positions:
                    self.logger.info(f"🛒 {symbol} 매수 시도 중...")
                    
                    # 실제 매수 실행
                    success = await self.order_executor.buy_market_order(symbol, 1)
                    
                    if success:
                        current_price = await self.data_manager.get_current_price(symbol)
                        self.state.scout_positions[symbol] = {
                            'quantity': 1,
                            'entry_time': datetime.now(),
                            'entry_price': current_price
                        }
                        
                        # 성공 알림
                        await self.notifier.send_trade_alert(f"✅ 척후병 매수 성공: {symbol}")
                        self.logger.info(f"✅ {symbol} 매수 완료")
                    else:
                        # 실패 알림
                        await self.notifier.send_trade_alert(f"❌ 척후병 매수 실패: {symbol}")
                        self.logger.error(f"❌ {symbol} 매수 실패")
                    
                    # 다음 주문까지 1초 대기
                    await asyncio.sleep(1)
            
            # 3. 3일 오디션 후 최종 선정
            if await self._is_scout_period_complete():
                await self._select_final_candidates()
                
        except Exception as e:
            self.logger.error(f"❌ 척후병 단계 오류: {e}")
    
    async def _execute_fibonacci_phase(self):
        """피보나치 단계 실행 (기존 로직 유지)"""
        self.logger.info("📈 피보나치 단계 실행")
        
        try:
            for symbol in self.state.final_selections:
                # 시장 상황 분석
                market_condition = await self.technical_analyzer.analyze_market_condition(symbol)
                
                # 매수 신호 확인 (기존 우선순위 로직 유지)
                buy_signal = await self._check_fibonacci_buy_signals(symbol, market_condition)
                
                if buy_signal:
                    await self._execute_fibonacci_buy(symbol, buy_signal)
                    
        except Exception as e:
            self.logger.error(f"❌ 피보나치 단계 오류: {e}")
    
    async def _execute_monitoring_phase(self):
        """모니터링 단계 실행"""
        self.logger.info("👁️ 모니터링 단계 실행")
        
        try:
            # 포지션 모니터링 및 손절/익절 체크
            for symbol, position in self.state.fibonacci_positions.items():
                await self._monitor_position(symbol, position)
                
        except Exception as e:
            self.logger.error(f"❌ 모니터링 단계 오류: {e}")
    
    async def _check_fibonacci_buy_signals(self, symbol: str, market_condition: dict) -> Optional[dict]:
        """피보나치 매수 신호 체크 (기존 우선순위 로직 유지)"""
        try:
            # 1순위: 추세전환 매수
            trend_reversal = await self.technical_analyzer.check_trend_reversal(symbol)
            if trend_reversal.get('signal', False):
                return {'type': 'trend_reversal', 'priority': 1, 'data': trend_reversal}
            
            # 2순위: 눌림목 매수 (상승 추세에서)
            if market_condition.get('trend') == 'uptrend':
                pullback = await self.technical_analyzer.check_pullback_buy(symbol)
                if pullback.get('signal', False):
                    return {'type': 'pullback', 'priority': 2, 'data': pullback}
            
            # 3순위: 전고점 돌파 매수
            breakout = await self.technical_analyzer.check_breakout_buy(symbol)
            if breakout.get('signal', False):
                return {'type': 'breakout', 'priority': 3, 'data': breakout}
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 매수 신호 체크 오류 ({symbol}): {e}")
            return None
    
    async def _execute_fibonacci_buy(self, symbol: str, buy_signal: dict):
        """피보나치 분할매수 실행 (기존 로직 유지)"""
        try:
            # 다음 피보나치 수량 계산
            current_position = self.state.fibonacci_positions.get(symbol, {'total_quantity': 0})
            next_quantity = self.fibonacci_strategy.get_next_quantity(
                current_position.get('total_quantity', 0)
            )
            
            if next_quantity > 0:
                success = await self.order_executor.buy_market_order(symbol, next_quantity)
                if success:
                    # 포지션 업데이트
                    if symbol not in self.state.fibonacci_positions:
                        self.state.fibonacci_positions[symbol] = {
                            'total_quantity': 0,
                            'buy_orders': [],
                            'avg_price': 0
                        }
                    
                    current_price = await self.data_manager.get_current_price(symbol)
                    self.state.fibonacci_positions[symbol]['buy_orders'].append({
                        'quantity': next_quantity,
                        'price': current_price,
                        'time': datetime.now(),
                        'signal_type': buy_signal['type']
                    })
                    
                    # 평균 단가 재계산
                    self._update_average_price(symbol)
                    
                    await self.notifier.send_trade_alert(
                        f"📈 피보나치 매수: {symbol} {next_quantity}주 ({buy_signal['type']})"
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ 피보나치 매수 오류 ({symbol}): {e}")
    
    def _update_average_price(self, symbol: str):
        """평균 단가 업데이트"""
        try:
            position = self.state.fibonacci_positions[symbol]
            buy_orders = position.get('buy_orders', [])
            
            if buy_orders:
                total_cost = sum(order['quantity'] * order['price'] for order in buy_orders)
                total_quantity = sum(order['quantity'] for order in buy_orders)
                
                position['total_quantity'] = total_quantity
                position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
                
        except Exception as e:
            self.logger.error(f"❌ 평균 단가 계산 오류 ({symbol}): {e}")
    
    async def _monitor_position(self, symbol: str, position: dict):
        """포지션 모니터링"""
        try:
            current_price = await self.data_manager.get_current_price(symbol)
            avg_price = position.get('avg_price', 0)
            
            if avg_price > 0:
                profit_rate = (current_price - avg_price) / avg_price * 100
                
                # 손절 체크 (-10%)
                if profit_rate <= -10:
                    await self._execute_stop_loss(symbol, position)
                
                # 익절 체크 (+20%)
                elif profit_rate >= 20:
                    await self._execute_take_profit(symbol, position)
                    
        except Exception as e:
            self.logger.error(f"❌ 포지션 모니터링 오류 ({symbol}): {e}")
    
    async def _execute_stop_loss(self, symbol: str, position: dict):
        """손절 실행"""
        try:
            quantity = position.get('total_quantity', 0)
            if quantity > 0:
                success = await self.order_executor.sell_market_order(symbol, quantity)
                
                if success:
                    del self.state.fibonacci_positions[symbol]
                    await self.notifier.send_trade_alert(f"🔻 손절 매도: {symbol} {quantity}주")
                    
        except Exception as e:
            self.logger.error(f"❌ 손절 실행 오류 ({symbol}): {e}")
    
    async def _execute_take_profit(self, symbol: str, position: dict):
        """익절 실행"""
        try:
            quantity = position.get('total_quantity', 0)
            if quantity > 0:
                success = await self.order_executor.sell_market_order(symbol, quantity)
                
                if success:
                    del self.state.fibonacci_positions[symbol]
                    await self.notifier.send_trade_alert(f"🔺 익절 매도: {symbol} {quantity}주")
                    
        except Exception as e:
            self.logger.error(f"❌ 익절 실행 오류 ({symbol}): {e}")
    
    async def _is_scout_period_complete(self) -> bool:
        """척후병 기간 완료 체크 (3일)"""
        try:
            if not self.state.scout_positions:
                return False
            
            oldest_entry = min(
                pos['entry_time'] for pos in self.state.scout_positions.values()
            )
            
            return (datetime.now() - oldest_entry).days >= 3
            
        except Exception as e:
            self.logger.error(f"❌ 척후병 기간 체크 오류: {e}")
            return False
    
    async def _select_final_candidates(self):
        """최종 후보 선정 (상위 2개)"""
        try:
            performances = {}
            
            for symbol, position in self.state.scout_positions.items():
                current_price = await self.data_manager.get_current_price(symbol)
                entry_price = position.get('entry_price', 0)
                
                if entry_price > 0:
                    performance = (current_price - entry_price) / entry_price * 100
                    performances[symbol] = performance
            
            if performances:
                # 성과 순으로 정렬하여 상위 2개 선정
                sorted_candidates = sorted(
                    performances.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                self.state.final_selections = [symbol for symbol, _ in sorted_candidates[:2]]
                self.state.current_phase = "FIBONACCI"
                
                self.logger.info(f"🎯 최종 선정: {self.state.final_selections}")
                await self.notifier.send_trade_alert(
                    f"🎯 최종 선정 완료: {', '.join(self.state.final_selections)}"
                )
                
                # 탈락 종목 매도
                for symbol in self.state.scout_candidates:
                    if symbol not in self.state.final_selections:
                        await self.order_executor.sell_market_order(symbol, 1)
                        
        except Exception as e:
            self.logger.error(f"❌ 최종 후보 선정 오류: {e}")
    
    def _is_market_hours(self, current_time: time) -> bool:
        """장중 시간 체크"""
        try:
            market_open = time(9, 0)
            market_close = time(15, 30)
            return market_open <= current_time <= market_close
        except:
            return False
    
    async def stop(self):
        """시스템 중지"""
        self.running = False
        self.logger.info("🛑 시스템 중지 요청") 