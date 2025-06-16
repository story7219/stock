"""
🚀 최적화된 스캘핑 시스템
- ATR, 멀티타임프레임, 모멘텀 분석 통합
- API 호출 최적화 및 일일 한도 관리
- 실시간 신호 생성 및 자동 매매
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from .atr_analyzer import ATRAnalyzer, ATRData
from .multi_timeframe_analyzer import MultiTimeframeAnalyzer, MultiTimeFrameSignal, TimeFrame
from .momentum_scorer import MomentumScorer, MomentumData

logger = logging.getLogger(__name__)

@dataclass
class ScalpingSignal:
    """통합 스캘핑 신호"""
    symbol: str
    action: str  # BUY, SELL, HOLD, WAIT
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    # 세부 분석 결과
    atr_data: Optional[ATRData]
    momentum_data: Optional[MomentumData]
    timeframe_signal: Optional[MultiTimeFrameSignal]
    
    # 메타데이터
    signal_strength: float
    risk_level: str  # LOW, MEDIUM, HIGH
    expected_duration: str  # SHORT, MEDIUM, LONG
    timestamp: datetime

class OptimizedScalpingSystem:
    """API 최적화된 고급 스캘핑 시스템"""
    
    def __init__(self, core_trader, daily_api_limit: int = 5000):
        """
        최적화된 스캘핑 시스템 초기화
        
        Args:
            core_trader: CoreTrader 인스턴스
            daily_api_limit: 일일 API 호출 한도
        """
        self.trader = core_trader
        self.daily_api_limit = daily_api_limit
        
        # 분석 모듈들 초기화
        self.atr_analyzer = ATRAnalyzer(
            optimal_atr_min=0.5,  # 스캘핑 최적 ATR 범위
            optimal_atr_max=3.0,
            period=14
        )
        
        self.multi_analyzer = MultiTimeframeAnalyzer()
        
        self.momentum_scorer = MomentumScorer(
            short_period=5,   # 스캘핑용 단기 설정
            medium_period=20,
            long_period=50
        )
        
        # 캐시 및 상태 관리
        self.price_cache: Dict[str, Dict] = {}
        self.signal_cache: Dict[str, ScalpingSignal] = {}
        self.last_analysis_time: Dict[str, datetime] = {}
        
        # API 호출 최적화 설정
        self.batch_size = 10  # 배치 처리 크기
        self.cache_duration = 30  # 캐시 유지 시간 (초)
        self.analysis_interval = 60  # 분석 간격 (초)
        
        # 실행 상태
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("🚀 최적화된 스캘핑 시스템 초기화 완료")
    
    def start_scalping(self, target_symbols: List[str], 
                      max_concurrent_positions: int = 3,
                      risk_per_trade: float = 0.02) -> None:
        """
        스캘핑 시작
        
        Args:
            target_symbols: 대상 종목 리스트
            max_concurrent_positions: 최대 동시 포지션 수
            risk_per_trade: 거래당 리스크 비율 (0.02 = 2%)
        """
        if self.is_running:
            logger.warning("⚠️ 스캘핑 시스템이 이미 실행 중입니다")
            return
        
        self.is_running = True
        self.target_symbols = target_symbols
        self.max_positions = max_concurrent_positions
        self.risk_per_trade = risk_per_trade
        
        logger.info(f"🎯 스캘핑 시작: {len(target_symbols)}개 종목, "
                   f"최대 {max_concurrent_positions}개 포지션")
        
        try:
            # 메인 스캘핑 루프 실행
            asyncio.run(self._scalping_main_loop())
        except KeyboardInterrupt:
            logger.info("⏹️ 사용자에 의한 중단")
        except Exception as e:
            logger.error(f"❌ 스캘핑 실행 중 오류: {e}")
        finally:
            self.stop_scalping()
    
    async def _scalping_main_loop(self) -> None:
        """메인 스캘핑 루프"""
        logger.info("🔄 메인 스캘핑 루프 시작")
        
        while self.is_running:
            try:
                # API 한도 확인
                if not self._check_api_limit():
                    logger.warning("⚠️ 일일 API 한도 근접 - 대기 모드")
                    await asyncio.sleep(300)  # 5분 대기
                    continue
                
                # 1. 종목 선별 및 분석
                analysis_start = time.time()
                promising_symbols = await self._find_promising_symbols()
                analysis_time = time.time() - analysis_start
                
                logger.info(f"📊 분석 완료: {len(promising_symbols)}개 유망 종목 "
                           f"(소요: {analysis_time:.1f}초)")
                
                # 2. 신호 생성 및 실행
                if promising_symbols:
                    await self._process_trading_signals(promising_symbols)
                
                # 3. 기존 포지션 관리
                await self._manage_existing_positions()
                
                # 4. 대기 시간 (API 효율화)
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"❌ 메인 루프 오류: {e}")
                await asyncio.sleep(30)  # 오류 시 30초 대기
    
    def _check_api_limit(self) -> bool:
        """API 한도 확인"""
        try:
            remaining = self.trader.daily_counter.get_remaining_calls()
            if isinstance(remaining, float) and remaining == float('inf'):
                return True
            
            # 80% 사용 시 경고, 90% 사용 시 중단
            used_ratio = (self.daily_api_limit - remaining) / self.daily_api_limit
            
            if used_ratio >= 0.9:
                return False
            elif used_ratio >= 0.8:
                logger.warning(f"⚠️ API 사용량 80% 초과: {used_ratio:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API 한도 확인 실패: {e}")
            return True  # 확인 실패 시 계속 진행
    
    async def _find_promising_symbols(self) -> List[str]:
        """유망 종목 발굴"""
        try:
            # 랭킹 기반 종목 선별
            top_stocks = self.trader.get_top_ranking_stocks(top_n=50)
            if not top_stocks:
                logger.warning("⚠️ 랭킹 데이터 조회 실패")
                return self.target_symbols[:10]  # 기본 종목 사용
            
            # 빠른 선별을 위한 기본 필터링
            filtered_symbols = []
            for stock in top_stocks:
                try:
                    # 기본 조건 확인
                    if (stock.get('volume_rate', 0) > 100 and  # 거래량 증가
                        abs(stock.get('change_rate', 0)) > 0.5 and  # 최소 변동률
                        stock.get('price', 0) > 1000):  # 최소 가격
                        
                        filtered_symbols.append(stock['symbol'])
                        
                except Exception as e:
                    logger.debug(f"⚠️ {stock.get('symbol', 'Unknown')} 필터링 실패: {e}")
                    continue
            
            logger.info(f"🔍 1차 필터링: {len(filtered_symbols)}개 종목 선별")
            return filtered_symbols[:20]  # 상위 20개로 제한
            
        except Exception as e:
            logger.error(f"❌ 유망 종목 발굴 실패: {e}")
            return self.target_symbols[:10]
    
    async def _process_trading_signals(self, symbols: List[str]) -> None:
        """매매 신호 처리"""
        try:
            # 배치 처리로 API 효율화
            batches = [symbols[i:i+self.batch_size] 
                      for i in range(0, len(symbols), self.batch_size)]
            
            for batch in batches:
                # 동시 분석 (멀티스레딩)
                analysis_tasks = []
                for symbol in batch:
                    task = self.executor.submit(self._analyze_symbol, symbol)
                    analysis_tasks.append((symbol, task))
                
                # 결과 수집
                for symbol, task in analysis_tasks:
                    try:
                        signal = task.result(timeout=10)
                        if signal and signal.action in ['BUY', 'SELL']:
                            await self._execute_signal(signal)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ {symbol} 분석/실행 실패: {e}")
                        continue
                
                # 배치 간 잠시 대기 (API 부하 분산)
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"❌ 매매 신호 처리 실패: {e}")
    
    def _analyze_symbol(self, symbol: str) -> Optional[ScalpingSignal]:
        """종목 분석 (스레드에서 실행)"""
        try:
            # 캐시 확인
            if self._is_analysis_cached(symbol):
                return self.signal_cache.get(symbol)
            
            # 현재가 조회
            price_data = self.trader.get_current_price(symbol)
            if not price_data:
                return None
            
            current_price = price_data['price']
            
            # 간단한 모멘텀 체크 (API 절약)
            momentum_signal = self._quick_momentum_check(symbol, current_price)
            if not momentum_signal:
                return None
            
            # 상세 분석 (필요한 경우만)
            atr_data = self._get_atr_analysis(symbol, current_price)
            
            # 통합 신호 생성
            signal = self._generate_integrated_signal(
                symbol, current_price, momentum_signal, atr_data
            )
            
            # 캐시에 저장
            if signal:
                self.signal_cache[symbol] = signal
                self.last_analysis_time[symbol] = datetime.now()
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ {symbol} 분석 실패: {e}")
            return None
    
    def _is_analysis_cached(self, symbol: str) -> bool:
        """분석 결과가 캐시되어 있는지 확인"""
        if symbol not in self.last_analysis_time:
            return False
        
        time_diff = datetime.now() - self.last_analysis_time[symbol]
        return time_diff.seconds < self.cache_duration
    
    def _quick_momentum_check(self, symbol: str, current_price: float) -> Optional[Dict]:
        """빠른 모멘텀 체크 (최소 API 사용)"""
        try:
            # 캐시된 가격 히스토리 사용
            if symbol not in self.price_cache:
                self.price_cache[symbol] = {'prices': [], 'timestamps': []}
            
            # 현재 가격 추가
            cache = self.price_cache[symbol]
            cache['prices'].append(current_price)
            cache['timestamps'].append(datetime.now())
            
            # 최근 10개만 유지 (메모리 효율화)
            if len(cache['prices']) > 10:
                cache['prices'] = cache['prices'][-10:]
                cache['timestamps'] = cache['timestamps'][-10:]
            
            # 최소 5개 데이터 필요
            if len(cache['prices']) < 5:
                return None
            
            # 간단한 모멘텀 계산
            prices = cache['prices']
            short_ma = sum(prices[-3:]) / 3  # 3기간 평균
            long_ma = sum(prices[-5:]) / 5   # 5기간 평균
            
            momentum = (short_ma - long_ma) / long_ma * 100
            
            # 임계값 확인
            if abs(momentum) < 0.5:  # 0.5% 미만은 무시
                return None
            
            return {
                'momentum': momentum,
                'direction': 'BULLISH' if momentum > 0 else 'BEARISH',
                'strength': 'STRONG' if abs(momentum) > 2 else 'MODERATE'
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} 빠른 모멘텀 체크 실패: {e}")
            return None
    
    def _get_atr_analysis(self, symbol: str, current_price: float) -> Optional[ATRData]:
        """ATR 분석 (필요한 경우만 수행)"""
        try:
            # 간단한 ATR 추정 (정확한 계산보다 속도 우선)
            if symbol not in self.price_cache:
                return None
            
            prices = self.price_cache[symbol]['prices']
            if len(prices) < 5:
                return None
            
            # 간단한 변동성 계산
            recent_prices = prices[-5:]
            max_price = max(recent_prices)
            min_price = min(recent_prices)
            
            atr_estimate = (max_price - min_price) / 2
            atr_percentage = (atr_estimate / current_price) * 100
            
            # ATR 적합성 점수 계산
            if 0.5 <= atr_percentage <= 3.0:
                suitability = 85  # 최적 범위
            elif atr_percentage < 0.5:
                suitability = 40  # 너무 낮음
            else:
                suitability = 60  # 높은 변동성
            
            return ATRData(
                symbol=symbol,
                atr_value=atr_estimate,
                atr_percentage=atr_percentage,
                volatility_level='MEDIUM',
                scalping_suitability=suitability,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} ATR 분석 실패: {e}")
            return None
    
    def _generate_integrated_signal(self, 
                                  symbol: str,
                                  current_price: float,
                                  momentum_signal: Dict,
                                  atr_data: Optional[ATRData]) -> Optional[ScalpingSignal]:
        """통합 신호 생성"""
        try:
            # 기본 신호 강도 계산
            momentum_score = abs(momentum_signal['momentum'])
            atr_score = atr_data.scalping_suitability if atr_data else 50
            
            # 종합 신뢰도 계산
            confidence = (momentum_score * 3 + atr_score) / 4  # 모멘텀 가중치 높음
            confidence = min(95, max(30, confidence))
            
            # 최소 신뢰도 확인
            if confidence < 60:
                return None
            
            # 액션 결정
            if momentum_signal['direction'] == 'BULLISH' and confidence > 70:
                action = 'BUY'
            elif momentum_signal['direction'] == 'BEARISH' and confidence > 70:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            if action == 'HOLD':
                return None
            
            # 가격 수준 계산
            atr_value = atr_data.atr_value if atr_data else current_price * 0.02
            
            if action == 'BUY':
                entry_price = current_price
                stop_loss = current_price - (atr_value * 1.5)
                take_profit = current_price + (atr_value * 2.0)
            else:  # SELL
                entry_price = current_price
                stop_loss = current_price + (atr_value * 1.5)
                take_profit = current_price - (atr_value * 2.0)
            
            # 포지션 크기 계산
            position_size = self._calculate_position_size(
                current_price, stop_loss, confidence
            )
            
            return ScalpingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                atr_data=atr_data,
                momentum_data=None,  # 간소화된 버전에서는 생략
                timeframe_signal=None,  # 간소화된 버전에서는 생략
                signal_strength=confidence,
                risk_level='MEDIUM',
                expected_duration='SHORT',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} 통합 신호 생성 실패: {e}")
            return None
    
    def _calculate_position_size(self, 
                               entry_price: float,
                               stop_loss: float,
                               confidence: float) -> float:
        """포지션 크기 계산"""
        try:
            # 계좌 잔고 조회
            balance = self.trader.get_balance()
            if not balance:
                return 0
            
            available_cash = balance.cash * 0.9  # 90%만 사용
            
            # 리스크 기반 포지션 크기
            risk_amount = available_cash * self.risk_per_trade
            
            # 신뢰도 기반 조정
            confidence_multiplier = confidence / 100
            adjusted_risk = risk_amount * confidence_multiplier
            
            # 손실 폭 계산
            loss_per_share = abs(entry_price - stop_loss)
            if loss_per_share == 0:
                return 0
            
            # 주식 수 계산
            shares = int(adjusted_risk / loss_per_share)
            
            # 최대 투자 한도 확인
            max_investment = available_cash * 0.3  # 한 종목에 최대 30%
            max_shares = int(max_investment / entry_price)
            
            return min(shares, max_shares)
            
        except Exception as e:
            logger.error(f"❌ 포지션 크기 계산 실패: {e}")
            return 0
    
    async def _execute_signal(self, signal: ScalpingSignal) -> None:
        """신호 실행"""
        try:
            # 포지션 한도 확인
            current_positions = len(self._get_current_positions())
            if current_positions >= self.max_positions:
                logger.info(f"⚠️ 최대 포지션 수 도달: {current_positions}/{self.max_positions}")
                return
            
            # 포지션 크기 확인
            if signal.position_size <= 0:
                logger.warning(f"⚠️ {signal.symbol} 포지션 크기 부적절: {signal.position_size}")
                return
            
            # 주문 실행
            logger.info(f"📈 {signal.symbol} {signal.action} 신호 실행")
            logger.info(f"   진입가: {signal.entry_price:,}원")
            logger.info(f"   수량: {signal.position_size:,}주")
            logger.info(f"   신뢰도: {signal.confidence:.1f}%")
            
            # 실제 주문 (모의투자에서만 실행)
            if self.trader.is_mock:
                order_result = self.trader.execute_order(
                    symbol=signal.symbol,
                    side=signal.action.lower(),
                    quantity=int(signal.position_size),
                    price=0,  # 시장가
                    log_payload={'signal_confidence': signal.confidence}
                )
                
                if order_result and order_result.get('success'):
                    logger.info(f"✅ {signal.symbol} 주문 성공")
                else:
                    logger.error(f"❌ {signal.symbol} 주문 실패")
            else:
                logger.info("📝 실전투자 모드가 아니므로 주문 시뮬레이션만 수행")
                
        except Exception as e:
            logger.error(f"❌ {signal.symbol} 신호 실행 실패: {e}")
    
    def _get_current_positions(self) -> List[Dict]:
        """현재 보유 포지션 조회"""
        try:
            balance = self.trader.get_balance()
            if not balance:
                return []
            
            positions = []
            for symbol, position in balance.positions.items():
                if position.get('quantity', 0) > 0:
                    positions.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'avg_price': position['avg_price'],
                        'current_price': position['current_price'],
                        'profit_loss': position['profit_loss']
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"❌ 현재 포지션 조회 실패: {e}")
            return []
    
    async def _manage_existing_positions(self) -> None:
        """기존 포지션 관리"""
        try:
            positions = self._get_current_positions()
            if not positions:
                return
            
            logger.info(f"📊 포지션 관리: {len(positions)}개 포지션")
            
            for position in positions:
                symbol = position['symbol']
                current_price = position['current_price']
                avg_price = position['avg_price']
                profit_loss_rate = (current_price - avg_price) / avg_price * 100
                
                # 손절/익절 체크
                if profit_loss_rate <= -3.0:  # 3% 손실
                    logger.warning(f"🔴 {symbol} 손절 검토: {profit_loss_rate:.1f}%")
                    # 손절 로직 구현
                elif profit_loss_rate >= 5.0:  # 5% 수익
                    logger.info(f"🟢 {symbol} 익절 검토: {profit_loss_rate:.1f}%")
                    # 익절 로직 구현
                
        except Exception as e:
            logger.error(f"❌ 포지션 관리 실패: {e}")
    
    def stop_scalping(self) -> None:
        """스캘핑 중지"""
        if not self.is_running:
            return
        
        logger.info("🛑 스캘핑 시스템 중지 중...")
        self.is_running = False
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)
        
        # 캐시 정리
        self.price_cache.clear()
        self.signal_cache.clear()
        self.last_analysis_time.clear()
        
        logger.info("✅ 스캘핑 시스템 중지 완료")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            positions = self._get_current_positions()
            api_remaining = self.trader.daily_counter.get_remaining_calls()
            
            return {
                'is_running': self.is_running,
                'current_positions': len(positions),
                'max_positions': self.max_positions,
                'cached_symbols': len(self.price_cache),
                'api_calls_remaining': api_remaining,
                'api_usage_percent': ((self.daily_api_limit - api_remaining) / self.daily_api_limit * 100) 
                    if isinstance(api_remaining, int) else 0,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 시스템 상태 조회 실패: {e}")
            return {'error': str(e)} 