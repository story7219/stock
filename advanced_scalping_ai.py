"""
🚀 고급 AI 스캘핑 & 데이트레이딩 시스템 
깃허브 분석 결과를 바탕으로 한 고급 기법 적용

주요 기능:
1. ATR 기반 동적 스캘핑 전략 (Alpaca 스타일)
2. 멀티타임프레임 분석 (1분, 5분, 15분)
3. 실시간 모멘텀 스코어링
4. 고급 호가창 분석 (Depth Analysis)
5. 시장 마이크로스트럭처 분석
6. AI 기반 진입/청산 타이밍
"""
import asyncio
import logging
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional, Callable
import numpy as np
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class AdvancedScalpingAI:
    """🎯 고급 스캘핑 & 데이트레이딩 AI 시스템"""
    
    def __init__(self, trader):
        """초기화 - CoreTrader와 연동"""
        self.trader = trader
        self.is_monitoring = False
        self.price_buffer = {}  # 실시간 가격 버퍼 (심볼별 최근 100개)
        self.volume_buffer = {}  # 거래량 버퍼
        self.momentum_scores = {}  # 모멘텀 점수 추적
        self.atr_values = {}  # ATR 값들
        
        # 콜백 함수들
        self.scalping_signal_callback = None
        self.risk_alert_callback = None
        
    # === 🎯 ATR 기반 동적 스캘핑 ===
    def calculate_atr_scalping_signals(self, symbol: str, period: int = 14) -> Dict:
        """ATR 기반 동적 스캘핑 신호 생성 (깃허브 분석된 전략 적용)"""
        try:
            logger.info(f"🔥 {symbol} ATR 스캘핑 분석 시작...")
            
            # 1분봉 데이터 수집
            minute_data = self._get_minute_bars(symbol, count=period + 5)
            if not minute_data or len(minute_data) < period:
                return None
            
            # ATR 계산
            atr = self._calculate_atr(minute_data, period)
            current_price = minute_data[-1]['close']
            
            # 동적 진입/청산 레벨 계산 (Alpaca 스타일)
            entry_threshold = atr * 0.5  # ATR의 50%에서 진입
            stop_loss = atr * 1.5       # ATR의 150%에서 손절
            take_profit = atr * 2.0     # ATR의 200%에서 익절
            
            # 현재 변동성 상태 분석
            volatility_state = self._analyze_volatility_state(minute_data, atr)
            
            # 모멘텀 방향성 분석
            momentum_direction = self._calculate_momentum_direction(minute_data)
            
            # 스캘핑 적합성 점수 (0-100)
            scalping_suitability = self._calculate_scalping_suitability(
                atr, current_price, volatility_state, momentum_direction
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'atr': round(atr, 2),
                'entry_threshold': round(entry_threshold, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'volatility_state': volatility_state,  # high/medium/low
                'momentum_direction': momentum_direction,  # bullish/bearish/neutral
                'scalping_suitability': scalping_suitability,  # 0-100 점수
                'recommended_action': self._get_atr_recommendation(
                    scalping_suitability, momentum_direction, volatility_state
                )
            }
            
        except Exception as e:
            logger.error(f"❌ ATR 스캘핑 분석 실패 {symbol}: {e}")
            return None
    
    # === 📊 멀티타임프레임 분석 ===
    def multi_timeframe_analysis(self, symbol: str) -> Dict:
        """멀티타임프레임 종합 분석 (1분, 5분, 15분)"""
        try:
            logger.info(f"📊 {symbol} 멀티타임프레임 분석...")
            
            analysis_results = {}
            timeframes = [
                ('1min', 1, 20),   # 1분봉 20개
                ('5min', 5, 12),   # 5분봉 12개 (1시간)
                ('15min', 15, 8)   # 15분봉 8개 (2시간)
            ]
            
            for tf_name, interval, count in timeframes:
                try:
                    bars = self._get_minute_bars(symbol, interval, count)
                    if not bars or len(bars) < 5:
                        continue
                    
                    tf_analysis = {
                        'trend_direction': self._calculate_trend_direction(bars),
                        'momentum_strength': self._calculate_momentum_strength(bars),
                        'support_resistance': self._find_support_resistance(bars),
                        'volume_profile': self._analyze_volume_profile(bars),
                        'breakout_potential': self._assess_breakout_potential(bars)
                    }
                    
                    analysis_results[tf_name] = tf_analysis
                    
                except Exception as e:
                    logger.warning(f"⚠️ {tf_name} 분석 실패: {e}")
                    continue
            
            # 타임프레임 간 일치도 분석
            consistency_score = self._calculate_timeframe_consistency(analysis_results)
            
            # 종합 판단
            overall_signal = self._generate_multi_tf_signal(analysis_results, consistency_score)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe_analysis': analysis_results,
                'consistency_score': consistency_score,
                'overall_signal': overall_signal,
                'confidence_level': self._calculate_confidence_level(analysis_results, consistency_score)
            }
            
        except Exception as e:
            logger.error(f"❌ 멀티타임프레임 분석 실패 {symbol}: {e}")
            return None
    
    # === ⚡ 실시간 모멘텀 스코어링 ===
    def real_time_momentum_scoring(self, symbol: str) -> Dict:
        """실시간 모멘텀 점수 계산 및 추적"""
        try:
            current_price = self._get_current_price(symbol)
            if not current_price:
                return None
            
            # 가격 버퍼에 추가 (최근 100개 유지)
            if symbol not in self.price_buffer:
                self.price_buffer[symbol] = deque(maxlen=100)
            self.price_buffer[symbol].append({
                'price': current_price,
                'timestamp': datetime.now()
            })
            
            if len(self.price_buffer[symbol]) < 10:
                return None
            
            # 다양한 시간 단위 모멘텀 계산
            momentum_1min = self._calculate_price_momentum(symbol, minutes=1)
            momentum_5min = self._calculate_price_momentum(symbol, minutes=5)
            momentum_15min = self._calculate_price_momentum(symbol, minutes=15)
            
            # 가속도 계산 (모멘텀의 변화율)
            acceleration = self._calculate_momentum_acceleration(symbol)
            
            # 거래량 가중 모멘텀
            volume_weighted_momentum = self._calculate_volume_weighted_momentum(symbol)
            
            # 종합 모멘텀 스코어 (0-100)
            composite_momentum = self._calculate_composite_momentum(
                momentum_1min, momentum_5min, momentum_15min, 
                acceleration, volume_weighted_momentum
            )
            
            # 모멘텀 등급 결정
            momentum_grade = self._get_momentum_grade(composite_momentum)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'momentum_1min': momentum_1min,
                'momentum_5min': momentum_5min,
                'momentum_15min': momentum_15min,
                'acceleration': acceleration,
                'volume_weighted_momentum': volume_weighted_momentum,
                'composite_momentum': composite_momentum,
                'momentum_grade': momentum_grade,
                'trading_signal': self._get_momentum_trading_signal(composite_momentum, momentum_grade)
            }
            
        except Exception as e:
            logger.error(f"❌ 실시간 모멘텀 스코어링 실패 {symbol}: {e}")
            return None
    
    # === 🏗️ 고급 호가창 분석 ===
    def advanced_depth_analysis(self, symbol: str) -> Dict:
        """고급 호가창 심도 분석 (Level 2 데이터 활용)"""
        try:
            logger.info(f"🏗️ {symbol} 고급 호가창 분석...")
            
            # 전체 호가창 데이터 수집
            depth_data = self._get_full_orderbook(symbol)
            if not depth_data:
                return None
            
            # 호가창 불균형 분석
            imbalance_analysis = self._calculate_orderbook_imbalance(depth_data)
            
            # 큰 주문 감지 (Iceberg 주문 등)
            large_order_detection = self._detect_large_orders(depth_data)
            
            # 호가창 압력 분석
            pressure_analysis = self._analyze_order_pressure(depth_data)
            
            # 스프레드 분석
            spread_analysis = self._analyze_bid_ask_spread(depth_data)
            
            # 호가창 유동성 분석
            liquidity_analysis = self._analyze_market_liquidity(depth_data)
            
            # 프로 vs 아마추어 매매 패턴 감지
            smart_money_analysis = self._detect_smart_money_flow(depth_data)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'imbalance_analysis': imbalance_analysis,
                'large_order_detection': large_order_detection,
                'pressure_analysis': pressure_analysis,
                'spread_analysis': spread_analysis,
                'liquidity_analysis': liquidity_analysis,
                'smart_money_analysis': smart_money_analysis,
                'overall_depth_score': self._calculate_overall_depth_score(
                    imbalance_analysis, pressure_analysis, liquidity_analysis
                ),
                'recommended_action': self._get_depth_based_recommendation(
                    imbalance_analysis, pressure_analysis, smart_money_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"❌ 고급 호가창 분석 실패 {symbol}: {e}")
            return None
    
    # === 🎯 AI 기반 진입/청산 타이밍 ===
    def ai_entry_exit_timing(self, symbol: str) -> Dict:
        """AI 기반 최적 진입/청산 타이밍 분석"""
        try:
            logger.info(f"🎯 {symbol} AI 진입/청산 타이밍 분석...")
            
            # 여러 분석 결과 통합
            atr_signals = self.calculate_atr_scalping_signals(symbol)
            momentum_scores = self.real_time_momentum_scoring(symbol)
            depth_analysis = self.advanced_depth_analysis(symbol)
            
            if not all([atr_signals, momentum_scores, depth_analysis]):
                return None
            
            # 가중치 기반 종합 점수 계산
            entry_score = self._calculate_entry_score(
                atr_signals, momentum_scores, depth_analysis
            )
            
            exit_score = self._calculate_exit_score(
                atr_signals, momentum_scores, depth_analysis
            )
            
            # 리스크 조정 점수
            risk_adjusted_scores = self._apply_risk_adjustment(entry_score, exit_score, symbol)
            
            # 최종 AI 추천
            ai_recommendation = self._generate_ai_recommendation(
                risk_adjusted_scores, atr_signals, momentum_scores
            )
            
            # 확신도 계산
            confidence_level = self._calculate_ai_confidence(
                atr_signals, momentum_scores, depth_analysis
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'entry_score': entry_score,
                'exit_score': exit_score,
                'risk_adjusted_entry': risk_adjusted_scores['entry'],
                'risk_adjusted_exit': risk_adjusted_scores['exit'],
                'ai_recommendation': ai_recommendation,
                'confidence_level': confidence_level,
                'execution_priority': self._calculate_execution_priority(
                    entry_score, confidence_level, momentum_scores
                ),
                'optimal_position_size': self._calculate_optimal_position_size(
                    symbol, confidence_level, atr_signals
                )
            }
            
        except Exception as e:
            logger.error(f"❌ AI 진입/청산 타이밍 분석 실패 {symbol}: {e}")
            return None
    
    # === 🚀 실시간 멀티 심볼 모니터링 ===
    async def start_advanced_monitoring(self, symbols: List[str], 
                                      signal_callback: Callable = None,
                                      risk_callback: Callable = None):
        """고급 실시간 모니터링 시작"""
        try:
            logger.info(f"🚀 고급 실시간 모니터링 시작: {len(symbols)}개 종목")
            
            self.is_monitoring = True
            self.scalping_signal_callback = signal_callback
            self.risk_alert_callback = risk_callback
            
            # 각 심볼별 비동기 모니터링 태스크 생성
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._monitor_symbol(symbol))
                tasks.append(task)
            
            # 시장 전체 상황 모니터링 태스크
            market_task = asyncio.create_task(self._monitor_market_conditions())
            tasks.append(market_task)
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"❌ 고급 모니터링 실패: {e}")
            self.is_monitoring = False
    
    async def _monitor_symbol(self, symbol: str):
        """개별 심볼 모니터링"""
        while self.is_monitoring:
            try:
                # AI 분석 실행
                ai_analysis = self.ai_entry_exit_timing(symbol)
                
                if ai_analysis and self.scalping_signal_callback:
                    self.scalping_signal_callback(symbol, ai_analysis)
                
                # 리스크 체크
                risk_alert = self._check_risk_conditions(symbol, ai_analysis)
                if risk_alert and self.risk_alert_callback:
                    self.risk_alert_callback(symbol, risk_alert)
                
                # 2초 대기 (API 한도 고려)
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"⚠️ {symbol} 모니터링 오류: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_market_conditions(self):
        """시장 전체 상황 모니터링"""
        while self.is_monitoring:
            try:
                market_health = self._assess_market_health()
                
                if market_health['risk_level'] == 'HIGH':
                    logger.warning("🚨 시장 고위험 상황 감지!")
                    if self.risk_alert_callback:
                        self.risk_alert_callback('MARKET', market_health)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.warning(f"⚠️ 시장 상황 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    def stop_advanced_monitoring(self):
        """고급 모니터링 중지"""
        self.is_monitoring = False
        logger.info("🔴 고급 모니터링 중지")
    
    # === 🛠️ 핵심 헬퍼 메서드들 ===
    def _get_minute_bars(self, symbol: str, interval: int = 1, count: int = 20) -> List[Dict]:
        """분봉 데이터 수집"""
        try:
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice", 
                    headers={"tr_id": "FHKST03010200"}, 
                    params={
                        "fid_etc_cls_code": "",
                        "fid_cond_mrkt_div_code": "J",
                        "fid_input_iscd": symbol,
                        "fid_input_hour_1": str(interval),
                        "fid_pw_data_incu_yn": "Y"
                    }
                )
                
                if res and res.get('rt_cd') == '0':
                    items = res.get('output2', [])[:count]
                    
                    bars = []
                    for item in items:
                        bars.append({
                            'timestamp': item.get('stck_cntg_hour', ''),
                            'open': float(item.get('stck_oprc', 0)),
                            'high': float(item.get('stck_hgpr', 0)),
                            'low': float(item.get('stck_lwpr', 0)),
                            'close': float(item.get('stck_prpr', 0)),
                            'volume': int(item.get('cntg_vol', 0))
                        })
                    
                    return bars
                
                return []
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 분봉 데이터 수집 실패: {e}")
            return []
    
    def _calculate_atr(self, bars: List[Dict], period: int = 14) -> float:
        """Average True Range 계산"""
        if len(bars) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        else:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            price_info = self.trader.get_current_price(symbol)
            return float(price_info.get('price', 0)) if price_info else None
        except:
            return None
    
    def _get_full_orderbook(self, symbol: str) -> Optional[Dict]:
        """전체 호가창 데이터 수집"""
        try:
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", 
                    headers={"tr_id": "FHKST01010200"}, 
                    params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol}
                )
                
                if res and res.get('rt_cd') == '0':
                    return res.get('output1', {})
                return None
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 호가창 데이터 수집 실패: {e}")
            return None
    
    # === 🧮 분석 메서드들 (기본 구현) ===
    def _analyze_volatility_state(self, bars, atr):
        """변동성 상태 분석"""
        if atr > 2000:
            return "high"
        elif atr > 1000:
            return "medium"
        else:
            return "low"
    
    def _calculate_momentum_direction(self, bars):
        """모멘텀 방향성 계산"""
        if len(bars) < 3:
            return "neutral"
        
        recent_prices = [bar['close'] for bar in bars[-3:]]
        if recent_prices[-1] > recent_prices[0]:
            return "bullish"
        elif recent_prices[-1] < recent_prices[0]:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_scalping_suitability(self, atr, price, volatility, momentum):
        """스캘핑 적합성 점수"""
        score = 50  # 기본 점수
        
        # 변동성 점수
        if volatility == "high":
            score += 20
        elif volatility == "medium":
            score += 10
        
        # 모멘텀 점수
        if momentum in ["bullish", "bearish"]:
            score += 15
        
        # ATR 기반 점수
        if price > 0 and atr / price > 0.02:  # 2% 이상 변동성
            score += 15
        
        return min(100, max(0, score))
    
    def _get_atr_recommendation(self, suitability, momentum, volatility):
        """ATR 기반 추천"""
        if suitability > 70 and momentum == "bullish":
            return "STRONG_BUY"
        elif suitability > 70 and momentum == "bearish":
            return "STRONG_SELL"
        elif suitability > 50:
            if momentum == "bullish":
                return "BUY"
            elif momentum == "bearish":
                return "SELL"
            else:
                return "WATCH"
        else:
            return "HOLD"
    
    def _calculate_trend_direction(self, bars):
        """추세 방향 분석"""
        if len(bars) < 5:
            return "SIDEWAYS"
        
        closes = [bar['close'] for bar in bars]
        if closes[-1] > closes[0] * 1.02:
            return "UP"
        elif closes[-1] < closes[0] * 0.98:
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def _calculate_momentum_strength(self, bars):
        """모멘텀 강도 분석"""
        if len(bars) < 2:
            return 0.0
        
        price_change = (bars[-1]['close'] - bars[0]['close']) / bars[0]['close']
        return abs(price_change) * 100  # 백분율로 변환
    
    def _find_support_resistance(self, bars):
        """지지/저항 분석"""
        if len(bars) < 5:
            return {"support": 0, "resistance": 0}
        
        lows = [bar['low'] for bar in bars]
        highs = [bar['high'] for bar in bars]
        
        return {
            "support": min(lows),
            "resistance": max(highs)
        }
    
    def _analyze_volume_profile(self, bars):
        """거래량 프로파일 분석"""
        if len(bars) < 2:
            return 0.0
        
        volumes = [bar['volume'] for bar in bars]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1] if volumes else 0
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _assess_breakout_potential(self, bars):
        """브레이크아웃 잠재력 평가"""
        if len(bars) < 5:
            return 0.0
        
        # 최근 5개 봉의 범위 대비 현재 위치
        highs = [bar['high'] for bar in bars[-5:]]
        lows = [bar['low'] for bar in bars[-5:]]
        current = bars[-1]['close']
        
        high_range = max(highs)
        low_range = min(lows)
        
        if high_range == low_range:
            return 0.0
        
        # 0~100 범위로 정규화
        position = (current - low_range) / (high_range - low_range) * 100
        
        # 80% 이상이면 상향 돌파 가능성, 20% 이하면 하향 돌파 가능성
        if position > 80:
            return position
        elif position < 20:
            return -position
        else:
            return 0.0
    
    # === 📊 스텁 메서드들 (추후 확장 가능) ===
    def _calculate_price_momentum(self, symbol: str, minutes: int) -> float:
        """가격 모멘텀 계산 (현재는 간단 구현)"""
        if symbol not in self.price_buffer or len(self.price_buffer[symbol]) < 5:
            return 0.0
        
        # 간단한 모멘텀 계산 (최근 vs 이전 가격)
        recent_prices = list(self.price_buffer[symbol])[-5:]
        if len(recent_prices) < 2:
            return 0.0
        
        price_change = (recent_prices[-1]['price'] - recent_prices[0]['price']) / recent_prices[0]['price']
        return price_change * 100  # 백분율로 반환
    
    def _calculate_momentum_acceleration(self, symbol: str) -> float:
        return self._calculate_price_momentum(symbol, 1) * 0.5  # 간단 구현
    
    def _calculate_volume_weighted_momentum(self, symbol: str) -> float:
        return self._calculate_price_momentum(symbol, 5) * 0.8  # 간단 구현
    
    def _calculate_composite_momentum(self, m1, m5, m15, acc, vol_weighted):
        return (m1 + m5 + m15 + acc + vol_weighted) / 5
    
    def _get_momentum_grade(self, composite_momentum: float) -> str:
        if composite_momentum > 80:
            return "A+"
        elif composite_momentum > 60:
            return "A"
        elif composite_momentum > 40:
            return "B"
        else:
            return "C"
    
    def _get_momentum_trading_signal(self, composite_momentum: float, momentum_grade: str) -> str:
        if composite_momentum > 70:
            return "STRONG_BUY"
        elif composite_momentum > 30:
            return "BUY"
        elif composite_momentum < -30:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_orderbook_imbalance(self, depth_data):
        return 50.0  # 기본값
    
    def _detect_large_orders(self, depth_data):
        return False  # 기본값
    
    def _analyze_order_pressure(self, depth_data):
        return 50.0  # 기본값
    
    def _analyze_bid_ask_spread(self, depth_data):
        return 50.0  # 기본값
    
    def _analyze_market_liquidity(self, depth_data):
        return 50.0  # 기본값
    
    def _detect_smart_money_flow(self, depth_data):
        return False  # 기본값
    
    def _calculate_overall_depth_score(self, imbalance, pressure, liquidity):
        return (imbalance + pressure + liquidity) / 3
    
    def _get_depth_based_recommendation(self, imbalance, pressure, smart_money):
        return "HOLD"  # 기본값
    
    def _calculate_entry_score(self, atr_signals, momentum_scores, depth_analysis):
        # 간단한 가중치 계산
        atr_score = atr_signals.get('scalping_suitability', 50)
        momentum_score = momentum_scores.get('composite_momentum', 50)
        depth_score = depth_analysis.get('overall_depth_score', 50)
        
        return (atr_score * 0.4 + momentum_score * 0.4 + depth_score * 0.2)
    
    def _calculate_exit_score(self, atr_signals, momentum_scores, depth_analysis):
        # 진입 점수의 역순으로 계산
        entry_score = self._calculate_entry_score(atr_signals, momentum_scores, depth_analysis)
        return 100 - entry_score
    
    def _apply_risk_adjustment(self, entry_score, exit_score, symbol):
        # 간단한 리스크 조정 (10% 할인)
        return {
            'entry': entry_score * 0.9,
            'exit': exit_score * 0.9
        }
    
    def _generate_ai_recommendation(self, risk_adjusted_scores, atr_signals, momentum_scores):
        entry_score = risk_adjusted_scores['entry']
        
        if entry_score > 80:
            return "STRONG_BUY"
        elif entry_score > 60:
            return "BUY"
        elif entry_score < 20:
            return "SELL"
        elif entry_score < 40:
            return "WEAK_SELL"
        else:
            return "HOLD"
    
    def _calculate_ai_confidence(self, atr_signals, momentum_scores, depth_analysis):
        # 각 분석의 일관성을 기반으로 확신도 계산
        scores = [
            atr_signals.get('scalping_suitability', 50),
            momentum_scores.get('composite_momentum', 50),
            depth_analysis.get('overall_depth_score', 50)
        ]
        
        # 표준편차가 낮을수록 높은 확신도
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        confidence = max(0, 100 - std_dev * 2)
        
        return confidence
    
    def _calculate_execution_priority(self, entry_score, confidence_level, momentum_scores):
        return (entry_score + confidence_level) / 2
    
    def _calculate_optimal_position_size(self, symbol, confidence_level, atr_signals):
        # 확신도에 비례한 포지션 사이즈 (0-100%)
        return min(100, confidence_level) / 100
    
    def _calculate_timeframe_consistency(self, analysis_results):
        if not analysis_results:
            return 0.0
        
        # 간단한 일치도 계산
        trend_directions = []
        for tf_data in analysis_results.values():
            trend_directions.append(tf_data.get('trend_direction', 'SIDEWAYS'))
        
        # 동일한 방향성이 많을수록 높은 일치도
        if not trend_directions:
            return 0.0
        
        most_common = max(set(trend_directions), key=trend_directions.count)
        consistency = trend_directions.count(most_common) / len(trend_directions) * 100
        
        return consistency
    
    def _generate_multi_tf_signal(self, analysis_results, consistency_score):
        if consistency_score > 70:
            return "HIGH_CONFIDENCE"
        elif consistency_score > 50:
            return "MEDIUM_CONFIDENCE"
        else:
            return "LOW_CONFIDENCE"
    
    def _calculate_confidence_level(self, analysis_results, consistency_score):
        return consistency_score
    
    def _check_risk_conditions(self, symbol, ai_analysis):
        if not ai_analysis:
            return None
        
        confidence = ai_analysis.get('confidence_level', 0)
        if confidence < 30:
            return {
                'risk_type': 'LOW_CONFIDENCE',
                'message': f'{symbol} 분석 확신도 부족: {confidence}%'
            }
        
        return None
    
    def _assess_market_health(self):
        # 간단한 시장 건강도 체크
        return {'risk_level': 'LOW'} 