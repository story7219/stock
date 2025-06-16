"""
전략 실행 관리자 - 완전 리팩토링 버전
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

from .base_strategy import BaseStrategy, StrategySignal
from .scout_strategy import ScoutStrategyManager
from .fibonacci_strategy import FibonacciStrategyManager
from .technical_analyzer import TechnicalAnalyzer

class ExecutionMode(Enum):
    """실행 모드"""
    CONSERVATIVE = "conservative"  # 보수적
    BALANCED = "balanced"         # 균형
    AGGRESSIVE = "aggressive"     # 공격적

@dataclass
class ExecutionConfig:
    """실행 설정"""
    mode: ExecutionMode = ExecutionMode.BALANCED
    max_positions: int = 10
    max_daily_trades: int = 20
    risk_per_trade: float = 0.02  # 거래당 리스크 2%
    confidence_threshold: float = 0.7
    
    # 실행 제한
    daily_trade_count: int = 0
    last_reset_date: Optional[datetime] = None

@dataclass
class ExecutionResult:
    """실행 결과"""
    timestamp: datetime
    signals_analyzed: int
    signals_executed: int
    total_investment: int
    success_rate: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class AdvancedStrategyExecutor:
    """고급 전략 실행 관리자"""
    
    def __init__(self, api_client, portfolio_manager, telegram_notifier, 
                 config: ExecutionConfig = None):
        self.api_client = api_client
        self.portfolio_manager = portfolio_manager
        self.telegram_notifier = telegram_notifier
        self.config = config or ExecutionConfig()
        
        # 전략 관리자들 초기화
        self.scout_strategy = ScoutStrategyManager()
        self.fibonacci_strategy = FibonacciStrategyManager()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # 활성 전략 목록
        self.active_strategies: List[BaseStrategy] = [
            self.scout_strategy,
            self.fibonacci_strategy
        ]
        
        # 실행 이력 관리
        self.execution_history: List[ExecutionResult] = []
        self.signal_cache: Dict[str, Tuple[StrategySignal, datetime]] = {}
        
        # 성과 추적
        self.performance_tracker = PerformanceTracker()
        
        logging.info(f"🚀 고급 전략 실행기 초기화 완료 - 모드: {self.config.mode.value}")
    
    async def execute_comprehensive_analysis(self, stock_codes: List[str]) -> ExecutionResult:
        """종합적인 전략 분석 및 실행"""
        start_time = datetime.now()
        
        try:
            # 일일 거래 제한 확인
            self._check_daily_limits()
            
            # 시장 상황 분석
            market_context = await self._analyze_market_context()
            
            # 전략별 신호 수집
            all_signals = await self._collect_strategy_signals(stock_codes, market_context)
            
            # 신호 검증 및 필터링
            validated_signals = await self._validate_and_filter_signals(all_signals)
            
            # 리스크 관리 적용
            risk_adjusted_signals = self._apply_risk_management(validated_signals)
            
            # 포트폴리오 최적화
            optimized_signals = self._optimize_portfolio_allocation(risk_adjusted_signals)
            
            # 실행
            execution_results = await self._execute_signals(optimized_signals)
            
            # 결과 분석 및 저장
            result = self._create_execution_result(
                start_time, all_signals, execution_results
            )
            
            # 성과 업데이트
            await self._update_performance_metrics(result)
            
            # 알림 전송
            await self._send_execution_notification(result)
            
            return result
            
        except Exception as e:
            logging.error(f"❌ 종합 분석 실행 중 오류: {e}", exc_info=True)
            raise
    
    def _check_daily_limits(self):
        """일일 거래 제한 확인"""
        today = datetime.now().date()
        
        if self.config.last_reset_date != today:
            self.config.daily_trade_count = 0
            self.config.last_reset_date = today
            logging.info("🔄 일일 거래 카운터 리셋")
        
        if self.config.daily_trade_count >= self.config.max_daily_trades:
            raise Exception(f"일일 거래 한도 초과: {self.config.daily_trade_count}/{self.config.max_daily_trades}")
    
    async def _analyze_market_context(self) -> Dict[str, Any]:
        """시장 상황 종합 분석"""
        try:
            # 주요 지수 분석
            kospi_data = await self._get_index_data("0001")  # KOSPI
            kosdaq_data = await self._get_index_data("1001")  # KOSDAQ
            
            # 시장 심리 분석
            market_sentiment = self._analyze_market_sentiment(kospi_data, kosdaq_data)
            
            # 섹터 분석
            sector_analysis = await self._analyze_sector_rotation()
            
            # 글로벌 시장 영향
            global_impact = await self._analyze_global_market_impact()
            
            return {
                "market_sentiment": market_sentiment,
                "sector_analysis": sector_analysis,
                "global_impact": global_impact,
                "kospi_trend": self._calculate_trend_strength(kospi_data),
                "kosdaq_trend": self._calculate_trend_strength(kosdaq_data),
                "volatility_index": self._calculate_volatility_index(kospi_data),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logging.error(f"❌ 시장 상황 분석 오류: {e}")
            return {"market_sentiment": "NEUTRAL", "timestamp": datetime.now()}
    
    async def _collect_strategy_signals(self, stock_codes: List[str], 
                                      market_context: Dict) -> List[StrategySignal]:
        """모든 전략에서 신호 수집"""
        all_signals = []
        
        # 병렬 처리로 성능 최적화
        tasks = []
        for stock_code in stock_codes:
            task = self._analyze_single_stock(stock_code, market_context)
            tasks.append(task)
        
        # 배치 처리 (10개씩)
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"❌ 종목 분석 오류: {result}")
                elif result:
                    all_signals.extend(result)
        
        logging.info(f"📊 총 {len(all_signals)}개 신호 수집 완료")
        return all_signals
    
    async def _analyze_single_stock(self, stock_code: str, 
                                  market_context: Dict) -> List[StrategySignal]:
        """개별 종목 분석"""
        try:
            # 시장 데이터 수집
            market_data = await self._collect_enhanced_market_data(stock_code, market_context)
            if not market_data:
                return []
            
            # 각 전략별 분석
            signals = []
            for strategy in self.active_strategies:
                if not strategy.enabled:
                    continue
                
                try:
                    signal = await strategy.analyze(stock_code, market_data)
                    if signal and strategy.is_signal_valid(signal):
                        # 시장 상황에 따른 신뢰도 조정
                        adjusted_signal = self._adjust_signal_for_market_context(
                            signal, market_context
                        )
                        signals.append(adjusted_signal)
                        
                except Exception as e:
                    logging.error(f"❌ {strategy.name} 분석 오류 ({stock_code}): {e}")
            
            return signals
            
        except Exception as e:
            logging.error(f"❌ {stock_code} 분석 중 오류: {e}")
            return []
    
    async def _validate_and_filter_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """신호 검증 및 필터링"""
        validated_signals = []
        
        for signal in signals:
            # 기본 검증
            if not self._is_signal_valid(signal):
                continue
            
            # 신뢰도 임계값 확인
            if signal.confidence < self.config.confidence_threshold:
                continue
            
            # 중복 신호 제거
            if self._is_duplicate_signal(signal):
                continue
            
            # 시장 상황 적합성 확인
            if not self._is_signal_suitable_for_market(signal):
                continue
            
            validated_signals.append(signal)
        
        logging.info(f"✅ {len(validated_signals)}개 신호 검증 통과")
        return validated_signals
    
    def _apply_risk_management(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """리스크 관리 적용"""
        risk_adjusted = []
        total_risk = 0.0
        
        # 신뢰도 순으로 정렬
        sorted_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
        
        for signal in sorted_signals:
            # 개별 거래 리스크 계산
            trade_risk = self._calculate_trade_risk(signal)
            
            # 총 리스크 한도 확인
            if total_risk + trade_risk > self.config.risk_per_trade * len(signals):
                logging.warning(f"⚠️ 리스크 한도 초과로 신호 제외: {signal.reason}")
                continue
            
            # 포지션 크기 조정
            adjusted_signal = self._adjust_position_size(signal, trade_risk)
            risk_adjusted.append(adjusted_signal)
            total_risk += trade_risk
        
        logging.info(f"🛡️ 리스크 관리 적용 완료 - 총 리스크: {total_risk:.2%}")
        return risk_adjusted
    
    async def _execute_signals(self, signals: List[StrategySignal]) -> List[Dict]:
        """신호 실행"""
        execution_results = []
        
        for signal in signals:
            try:
                result = await self._execute_single_signal(signal)
                execution_results.append(result)
                
                # 실행 간격 (API 제한 고려)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"❌ 신호 실행 오류: {e}")
                execution_results.append({
                    'signal': signal,
                    'status': 'error',
                    'error': str(e)
                })
        
        return execution_results
    
    async def _send_execution_notification(self, result: ExecutionResult):
        """실행 결과 알림"""
        message = f"""
🎯 <b>고급 전략 실행 완료</b>

⏰ 실행 시간: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📊 분석 신호: {result.signals_analyzed}개
✅ 실행 신호: {result.signals_executed}개
💰 총 투자금액: {result.total_investment:,}원
📈 성공률: {result.success_rate:.1%}

🎯 <b>모드:</b> {self.config.mode.value.upper()}
📋 <b>일일 거래:</b> {self.config.daily_trade_count}/{self.config.max_daily_trades}

💡 <b>성과 지표:</b>
{self._format_performance_metrics(result.performance_metrics)}
"""
        
        await self.telegram_notifier.send_message(message)

class PerformanceTracker:
    """성과 추적기"""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = {}
        self.strategy_performance = {}
    
    def record_trade(self, signal: StrategySignal, result: Dict):
        """거래 기록"""
        trade_record = {
            'timestamp': datetime.now(),
            'strategy': signal.metadata.get('strategy_name'),
            'stock_code': signal.metadata.get('stock_code'),
            'action': signal.action,
            'quantity': signal.quantity,
            'confidence': signal.confidence,
            'result': result
        }
        self.trades.append(trade_record)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """성과 지표 계산"""
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        successful_trades = sum(1 for t in self.trades if t['result'].get('status') == 'success')
        
        return {
            'total_trades': total_trades,
            'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'avg_confidence': sum(t['confidence'] for t in self.trades) / total_trades,
            'strategy_breakdown': self._get_strategy_breakdown()
        }
    
    def _get_strategy_breakdown(self) -> Dict[str, Dict]:
        """전략별 성과 분석"""
        breakdown = {}
        for trade in self.trades:
            strategy = trade['strategy']
            if strategy not in breakdown:
                breakdown[strategy] = {'total': 0, 'success': 0}
            
            breakdown[strategy]['total'] += 1
            if trade['result'].get('status') == 'success':
                breakdown[strategy]['success'] += 1
        
        # 성공률 계산
        for strategy, data in breakdown.items():
            data['success_rate'] = data['success'] / data['total'] if data['total'] > 0 else 0
        
        return breakdown 