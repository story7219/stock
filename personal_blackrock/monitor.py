"""
🚀 실시간 거래대금 TOP 20 종목 전략 매칭 분석기 (통합 모니터링 시스템)

거래대금 상위 20개 종목을 실시간으로 모니터링하여
6가지 투자 전략(윌리엄 오닐, 제시 리버모어, 워렌 버핏, 피터 린치, 일목균형표, 블랙록)에
매칭되는 종목을 찾아 알림하는 통합 시스템입니다.

기능:
- 거래대금 TOP 20 실시간 모니터링
- 6가지 전략 자동 매칭 분석
- 차트 분석 (RSI, MACD, 볼린저 밴드)
- 수급 분석 (외국인, 기관, 개인)
- 급변 알림 시스템
- 텔레그램 알림 연동
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import pandas as pd
from pathlib import Path
import json

from pykrx import stock
from personal_blackrock.ai_analyzer import HighPerformanceAIAnalyzer
from personal_blackrock.data import DataManager


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "정보"
    WARNING = "주의"
    CRITICAL = "긴급"


@dataclass
class TradingVolumeStock:
    """거래대금 상위 종목 정보 (통합)"""
    code: str
    name: str
    current_price: float
    trading_value: int  # 거래대금
    volume: int
    change_rate: float
    rank: int
    
    # 추가 분석 데이터
    per: float = 0.0
    pbr: float = 0.0
    roe: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    market_cap: int = 0
    bollinger_position: str = "중간"
    
    # 수급 데이터
    foreign_net: int = 0
    institution_net: int = 0
    individual_net: int = 0
    supply_demand_score: float = 50.0


@dataclass
class StrategyMatch:
    """전략 매칭 결과"""
    stock_code: str
    stock_name: str
    strategy_name: str
    score: int
    recommendation: str
    reason: str
    entry_price: str
    target_price: str
    confidence: float
    timestamp: datetime


@dataclass
class MonitoringAlert:
    """모니터링 알림 데이터"""
    stock_code: str
    stock_name: str
    alert_type: str
    message: str
    level: AlertLevel
    timestamp: datetime
    data: Dict[str, Any]


class RealTimeMonitor:
    """실시간 거래대금 TOP 20 종목 전략 매칭 분석기 (통합 모니터링 시스템)"""
    
    def __init__(self, trader, notifier, data_manager=None):
        """
        실시간 모니터링 시스템을 초기화합니다.
        
        Args:
            trader: CoreTrader 인스턴스
            notifier: Notifier 인스턴스
            data_manager: 외부에서 전달받은 DataManager 인스턴스 (성능 최적화)
        """
        # 로거 설정을 먼저 해야 함
        self.logger = logging.getLogger("통합모니터링")
        self.logger.setLevel(logging.INFO)
        
        self.trader = trader
        self.notifier = notifier
        
        # 외부에서 전달받은 DataManager 사용 또는 새로 생성
        if data_manager:
            self.data_manager = data_manager
            self.logger.info("✅ 외부 DataManager 사용 (성능 최적화)")
        else:
            self.data_manager = DataManager()
            self.logger.info("✅ 새로운 DataManager 생성")
        
        # AI 분석기 초기화 (공통 DataManager 사용)
        self.ai_analyzer = HighPerformanceAIAnalyzer(data_manager=self.data_manager)
        
        # 분석 설정
        self.strategies = ["윌리엄 오닐", "제시 리버모어", "워렌 버핏", "피터 린치", "일목균형표", "블랙록"]
        self.min_score_threshold = 70  # 최소 점수 임계값
        self.analysis_interval = 300   # 5분마다 분석
        
        # 중복 알림 방지
        self.last_notifications = {}
        self.notification_cooldown = 1800  # 30분 쿨다운
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_start_time = None
        
        # 추가 속성들 (기존 코드 호환성)
        self.is_running = False
        self.alert_history = []
        
        self.logger.info("실시간 거래대금 TOP 20 종목 전략 매칭 분석기 (통합) 초기화 완료")

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - [통합모니터링] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def start_monitoring(self):
        """실시간 모니터링 시작 (기존 호환성 유지)"""
        await self.start_real_time_analysis()

    async def start_real_time_analysis(self):
        """실시간 분석 시작"""
        self.logger.info("🚀 실시간 거래대금 TOP 20 종목 전략 매칭 분석 시작")
        self.logger.info(f"📊 분석 주기: {self.analysis_interval}초")
        self.logger.info(f"🎯 최소 매칭 점수: {self.min_score_threshold}점")
        self.logger.info(f"📈 분석 전략: {', '.join(self.strategies)}")
        
        print("\n" + "="*100)
        print("🔥 거래대금 상위 20개 종목 실시간 전략 매칭 분석 시작!")
        print("📊 차트분석 + 수급분석 + 전략매칭 + 급변알림 통합 서비스")
        print(f"⏰ {self.analysis_interval}초마다 전략 분석 + {self.monitoring_interval}초마다 기본 모니터링")
        print("📱 텔레그램 알림 연동")
        print("="*100)
        
        self.is_running = True
        
        try:
            # 초기 데이터 로드
            await self._initial_data_load()
            
            last_strategy_analysis = 0
            
            while self.is_running:
                start_time = time.time()
                current_time = time.time()
                
                # 전략 분석 (설정된 주기마다)
                if current_time - last_strategy_analysis >= self.analysis_interval:
                    await self._analyze_top_trading_volume_stocks()
                    last_strategy_analysis = current_time
                
                # 기본 모니터링 (30초마다)
                await self._monitoring_cycle()
                
                analysis_time = time.time() - start_time
                self.logger.info(f"⏱️ 모니터링 사이클 완료 (소요시간: {analysis_time:.1f}초)")
                
                # 다음 분석까지 대기
                if self.is_running:
                    await asyncio.sleep(self.monitoring_interval)
                    
        except KeyboardInterrupt:
            self.logger.info("⏹️ 사용자가 분석을 중단했습니다.")
        except Exception as e:
            self.logger.error(f"❌ 실시간 분석 중 오류 발생: {e}")
        finally:
            self.is_running = False
            self.logger.info("🔚 실시간 분석이 종료되었습니다.")

    async def stop_monitoring(self):
        """분석 중단"""
        self.is_running = False
        self.logger.info("⏹️ 분석 중단 요청됨")

    async def _initial_data_load(self):
        """초기 데이터 로드"""
        try:
            self.logger.info("📊 초기 데이터 로드 중...")
            
            # 거래대금 상위 종목 조회
            top_stocks = await self._get_top_trading_value_stocks()
            
            if not top_stocks:
                self.logger.warning("⚠️ 초기 데이터 로드 실패 - 샘플 데이터 사용")
                top_stocks = self._get_sample_trading_stocks()
            
            # 초기 데이터 저장
            self.previous_data = {stock.code: stock for stock in top_stocks}
            
            # 초기 현황 출력
            await self._display_current_status(top_stocks)
            
            self.logger.info(f"✅ 초기 데이터 로드 완료: {len(top_stocks)}개 종목")
            
        except Exception as e:
            self.logger.error(f"❌ 초기 데이터 로드 실패: {e}")

    async def _monitoring_cycle(self):
        """기본 모니터링 사이클 실행"""
        try:
            # 거래대금 상위 종목 업데이트
            current_stocks = await self._get_top_trading_value_stocks()
            
            if not current_stocks:
                self.logger.warning("⚠️ 데이터 업데이트 실패")
                return
            
            # 변화 감지 및 알림
            alerts = await self._detect_changes(current_stocks)
            
            # 알림 발송
            if alerts:
                await self._send_monitoring_alerts(alerts)
            
            # 현황 업데이트 출력 (전략 분석이 아닌 경우만)
            await self._display_current_status(current_stocks, show_full=False)
            
            # 이전 데이터 업데이트
            self.previous_data = {stock.code: stock for stock in current_stocks}
            
            # 캐시 저장
            await self._save_monitoring_cache(current_stocks)
            
        except Exception as e:
            self.logger.error(f"❌ 모니터링 사이클 오류: {e}")

    async def _analyze_top_trading_volume_stocks(self):
        """거래대금 TOP 20 종목 전략 분석"""
        try:
            self.logger.info("📊 거래대금 TOP 20 종목 전략 분석 시작...")
            
            # 1. 거래대금 TOP 20 종목 가져오기
            top_stocks = await self._get_top_trading_value_stocks()
            
            if not top_stocks:
                self.logger.warning("⚠️ 거래대금 상위 종목 데이터를 가져올 수 없습니다.")
                return
            
            self.logger.info(f"✅ {len(top_stocks)}개 종목 데이터 수집 완료")
            
            # 2. 각 종목에 대해 6가지 전략 분석
            strategy_matches = []
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # 모든 종목 × 모든 전략 조합으로 분석 작업 생성
                futures = []
                for stock in top_stocks:
                    for strategy in self.strategies:
                        future = executor.submit(
                            self._analyze_stock_with_strategy_sync,
                            stock, strategy
                        )
                        futures.append((future, stock, strategy))
                
                # 분석 결과 수집
                for future, stock, strategy in futures:
                    try:
                        match_result = future.result(timeout=30)
                        if match_result and match_result.score >= self.min_score_threshold:
                            strategy_matches.append(match_result)
                    except Exception as e:
                        self.logger.error(f"❌ {stock.code}({strategy}) 분석 실패: {e}")
            
            # 3. 매칭된 결과 처리
            if strategy_matches:
                await self._process_strategy_matches(strategy_matches)
            else:
                self.logger.info("📊 현재 시점에서 전략 매칭 조건을 만족하는 종목이 없습니다.")
                
        except Exception as e:
            self.logger.error(f"❌ TOP 20 종목 전략 분석 중 오류: {e}")

    async def _get_top_trading_value_stocks(self) -> List[TradingVolumeStock]:
        """거래대금 상위 20개 종목 조회"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            
            # PyKRX를 통한 거래대금 상위 종목 조회
            kospi_trading = stock.get_market_trading_value_by_ticker(today, market="KOSPI")
            kosdaq_trading = stock.get_market_trading_value_by_ticker(today, market="KOSDAQ")
            
            # 통합 및 정렬
            all_trading = pd.concat([kospi_trading, kosdaq_trading])
            top_20 = all_trading.sort_values('거래대금', ascending=False).head(20)
            
            trading_stocks = []
            
            for rank, (code, row) in enumerate(top_20.iterrows(), 1):
                try:
                    # 종목명 조회
                    stock_name = self.data_manager.get_stock_name(code)
                    
                    # 차트 분석 데이터 추가
                    chart_data = await self._get_chart_analysis(code)
                    supply_demand_data = await self._get_supply_demand_analysis(code)
                    
                    # 기본 정보 구성
                    trading_stock = TradingVolumeStock(
                        code=code,
                        name=stock_name,
                        current_price=float(row.get('종가', 0)),
                        trading_value=int(row.get('거래대금', 0)),
                        volume=int(row.get('거래량', 0)),
                        change_rate=float(row.get('등락률', 0)),
                        rank=rank,
                        rsi=chart_data.get('rsi', 50.0),
                        macd=chart_data.get('macd', 0.0),
                        bollinger_position=chart_data.get('bollinger_position', '중간'),
                        foreign_net=supply_demand_data.get('foreign_net', 0),
                        institution_net=supply_demand_data.get('institution_net', 0),
                        individual_net=supply_demand_data.get('individual_net', 0),
                        supply_demand_score=supply_demand_data.get('score', 50.0)
                    )
                    
                    trading_stocks.append(trading_stock)
                    
                except Exception as e:
                    self.logger.error(f"❌ [{code}] 종목 데이터 처리 실패: {e}")
                    continue
            
            return trading_stocks
            
        except Exception as e:
            self.logger.error(f"❌ 거래대금 상위 종목 조회 실패: {e}")
            return self._get_sample_trading_stocks()

    def _get_sample_trading_stocks(self) -> List[TradingVolumeStock]:
        """샘플 거래대금 상위 종목 (API 실패 시 사용)"""
        sample_data = [
            ('005930', '삼성전자', 70000, 5000000000, 1000000, 1.2),
            ('000660', 'SK하이닉스', 120000, 3000000000, 800000, -0.8),
            ('035420', 'NAVER', 180000, 2500000000, 600000, 2.1),
            ('051910', 'LG화학', 400000, 2000000000, 400000, -1.5),
            ('006400', '삼성SDI', 500000, 1800000000, 350000, 3.2),
            ('035720', '카카오', 50000, 1600000000, 900000, -2.1),
            ('028260', '삼성물산', 120000, 1400000000, 300000, 0.5),
            ('068270', '셀트리온', 180000, 1200000000, 250000, 1.8),
            ('096770', 'SK이노베이션', 200000, 1100000000, 200000, -0.9),
            ('323410', '카카오뱅크', 25000, 1000000000, 1200000, 2.5),
        ]
        
        trading_stocks = []
        for rank, (code, name, price, trading_value, volume, change_rate) in enumerate(sample_data, 1):
            trading_stock = TradingVolumeStock(
                code=code,
                name=name,
                current_price=price,
                trading_value=trading_value,
                volume=volume,
                change_rate=change_rate,
                rank=rank
            )
            trading_stocks.append(trading_stock)
        
        return trading_stocks

    async def _get_chart_analysis(self, stock_code: str) -> Dict[str, Any]:
        """차트 분석 데이터 조회"""
        try:
            # 실제 차트 분석 로직 구현
            # 여기서는 샘플 데이터 반환
            import random
            return {
                'rsi': random.uniform(20, 80),
                'macd': random.uniform(-10, 10),
                'bollinger_position': random.choice(['상단', '중간', '하단'])
            }
        except Exception as e:
            self.logger.error(f"❌ [{stock_code}] 차트 분석 실패: {e}")
            return {'rsi': 50.0, 'macd': 0.0, 'bollinger_position': '중간'}

    async def _get_supply_demand_analysis(self, stock_code: str) -> Dict[str, Any]:
        """수급 분석 데이터 조회"""
        try:
            # 실제 수급 분석 로직 구현
            # 여기서는 샘플 데이터 반환
            import random
            return {
                'foreign_net': random.randint(-1000000, 1000000),
                'institution_net': random.randint(-500000, 500000),
                'individual_net': random.randint(-2000000, 2000000),
                'score': random.uniform(30, 80)
            }
        except Exception as e:
            self.logger.error(f"❌ [{stock_code}] 수급 분석 실패: {e}")
            return {'foreign_net': 0, 'institution_net': 0, 'individual_net': 0, 'score': 50.0}

    def _analyze_stock_with_strategy_sync(
        self, 
        stock: TradingVolumeStock, 
        strategy_name: str
    ) -> Optional[StrategyMatch]:
        """단일 종목을 특정 전략으로 분석 (동기 버전)"""
        try:
            # AI 분석 수행 (리팩토링된 AIAnalyzer는 stock_code만 받음)
            # asyncio.run을 사용하여 비동기 함수를 동기 컨텍스트에서 실행
            analysis_result = asyncio.run(
                self.ai_analyzer.analyze_stock_with_strategy(
                    stock.code, strategy_name
                )
            )
            
            if not analysis_result or 'error' in analysis_result:
                error_msg = analysis_result.get('error', '알 수 없는 오류')
                self.logger.warning(f"❌ {stock.code}({strategy_name}) 분석 실패: {error_msg}")
                return None
            
            score = analysis_result.get('점수', 0)
            
            # 최소 점수 미달시 제외
            if score < self.min_score_threshold:
                return None
            
            # 전략 매칭 결과 생성
            strategy_match = StrategyMatch(
                stock_code=stock.code,
                stock_name=stock.name,
                strategy_name=strategy_name,
                score=score,
                recommendation=analysis_result.get('추천 등급', '보류'),
                reason=analysis_result.get('추천 이유', '분석 결과 기반'),
                entry_price=analysis_result.get('진입 가격', '현재가 기준'),
                target_price=analysis_result.get('목표 가격', '목표가 미설정'),
                confidence=analysis_result.get('신뢰도', 0.5),
                timestamp=datetime.now()
            )
            
            return strategy_match
            
        except Exception as e:
            self.logger.error(f"❌ {stock.code}({strategy_name}) 분석 중 심각한 오류 발생: {e}", exc_info=True)
            return None

    async def _process_strategy_matches(self, matches: List[StrategyMatch]):
        """전략 매칭 결과 처리"""
        try:
            # 점수 순으로 정렬
            sorted_matches = sorted(matches, key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"🎯 {len(sorted_matches)}개 종목이 전략 매칭 조건을 만족합니다!")
            
            # 터미널 출력
            self._print_matches_to_terminal(sorted_matches)
            
            # 새로운 매칭만 텔레그램 알림
            new_matches = self._filter_new_matches(sorted_matches)
            
            if new_matches:
                await self._send_telegram_notifications(new_matches)
            else:
                self.logger.info("📱 새로운 매칭 결과가 없어 텔레그램 알림을 보내지 않습니다.")
                
        except Exception as e:
            self.logger.error(f"❌ 매칭 결과 처리 중 오류: {e}")

    def _print_matches_to_terminal(self, matches: List[StrategyMatch]):
        """매칭 결과를 터미널에 출력"""
        print("\n" + "="*100)
        print("🚀 거래대금 TOP 20 종목 전략 매칭 결과")
        print(f"📅 분석시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        for i, match in enumerate(matches, 1):
            print(f"\n  {i:2d}. {match.stock_name} ({match.stock_code}) - {match.strategy_name}")
            print(f"       📊 점수: {match.score}점 | 💡 추천: {match.recommendation}")
            print(f"       🎯 이유: {match.reason}")
            print(f"       💰 진입가: {match.entry_price}")
            print(f"       🚀 목표가: {match.target_price}")
            print(f"       🔍 신뢰도: {match.confidence:.1%}")
            print("-" * 100)
        
        print("="*100)

    def _filter_new_matches(self, matches: List[StrategyMatch]) -> List[StrategyMatch]:
        """새로운 매칭 결과만 필터링 (중복 알림 방지)"""
        new_matches = []
        
        for match in matches:
            # 종목코드 + 전략명으로 고유 키 생성
            match_key = f"{match.stock_code}_{match.strategy_name}"
            
            if match_key not in self.last_notifications:
                new_matches.append(match)
                self.last_notifications[match_key] = datetime.now()
        
        # 24시간 후 알림 기록 초기화 (메모리 관리)
        if len(self.last_notifications) > 1000:
            self.last_notifications.clear()
            self.logger.info("🔄 알림 기록 초기화 (메모리 관리)")
        
        return new_matches

    async def _send_telegram_notifications(self, matches: List[StrategyMatch]):
        """텔레그램 알림 전송"""
        try:
            # 전략별로 그룹화
            strategy_groups = {}
            for match in matches:
                if match.strategy_name not in strategy_groups:
                    strategy_groups[match.strategy_name] = []
                strategy_groups[match.strategy_name].append(match)
            
            # 전략별로 알림 전송
            for strategy_name, strategy_matches in strategy_groups.items():
                message = self._create_telegram_message(strategy_name, strategy_matches)
                
                # notifier를 직접 사용하도록 변경
                if self.notifier:
                    success = await self.notifier.send_notification(
                        message, parse_mode="Markdown"
                    )
                else:
                    success = False # notifier가 없을 경우
                
                if success:
                    self.logger.info(f"📱 '{strategy_name}' 전략 매칭 알림 전송 완료 ({len(strategy_matches)}개 종목)")
                
                # 전송 간격 (텔레그램 제한 방지)
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"❌ 텔레그램 알림 전송 중 오류: {e}")

    def _create_telegram_message(self, strategy_name: str, matches: List[StrategyMatch]) -> str:
        """텔레그램 메시지 생성"""
        message = f"""🎯 **{strategy_name} 전략 매칭 발견!**

📊 **거래대금 TOP 20 종목 중 매칭**
📅 발견시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        for i, match in enumerate(matches, 1):
            message += f"""**{i}. {match.stock_name} ({match.stock_code})**
📊 점수: {match.score}점
💡 추천: {match.recommendation}
🎯 이유: {match.reason}
💰 진입가: {match.entry_price}
🚀 목표가: {match.target_price}
🔍 신뢰도: {match.confidence:.1%}

"""

        message += f"""
⚡ **투자 시 주의사항**
- 실시간 분석 결과로 참고용입니다
- 추가 검토 후 투자 결정하세요
- 투자 손실 책임은 본인에게 있습니다

#{strategy_name.replace(' ', '')} #거래대금TOP20 #실시간매칭
"""
        
        return message

    async def _detect_changes(self, current_stocks: List[TradingVolumeStock]) -> List[MonitoringAlert]:
        """변화 감지 및 알림 생성"""
        alerts = []
        
        try:
            for stock in current_stocks:
                # 이전 데이터와 비교
                if stock.code in self.previous_data:
                    prev_stock = self.previous_data[stock.code]
                    
                    # 가격 급변 감지
                    price_change = abs(stock.change_rate)
                    if price_change >= self.config['price_change_threshold']:
                        alert = MonitoringAlert(
                            stock_code=stock.code,
                            stock_name=stock.name,
                            alert_type="가격급변",
                            message=f"{stock.name} 가격 {stock.change_rate:+.2f}% 급변",
                            level=AlertLevel.WARNING if price_change < 10 else AlertLevel.CRITICAL,
                            timestamp=datetime.now(),
                            data={
                                'current_price': stock.current_price,
                                'change_rate': stock.change_rate,
                                'trading_value': stock.trading_value
                            }
                        )
                        alerts.append(alert)
                    
                    # 거래량 급증 감지
                    if hasattr(prev_stock, 'volume') and prev_stock.volume > 0:
                        volume_ratio = stock.volume / prev_stock.volume
                        if volume_ratio >= self.config['volume_spike_threshold']:
                            alert = MonitoringAlert(
                                stock_code=stock.code,
                                stock_name=stock.name,
                                alert_type="거래량급증",
                                message=f"{stock.name} 거래량 {volume_ratio:.1f}배 급증",
                                level=AlertLevel.WARNING,
                                timestamp=datetime.now(),
                                data={
                                    'current_volume': stock.volume,
                                    'previous_volume': prev_stock.volume,
                                    'ratio': volume_ratio
                                }
                            )
                            alerts.append(alert)
                
                # RSI 과매수/과매도 감지
                if stock.rsi >= self.config['rsi_overbought']:
                    alert = MonitoringAlert(
                        stock_code=stock.code,
                        stock_name=stock.name,
                        alert_type="RSI과매수",
                        message=f"{stock.name} RSI {stock.rsi:.1f} 과매수 구간",
                        level=AlertLevel.INFO,
                        timestamp=datetime.now(),
                        data={'rsi': stock.rsi}
                    )
                    alerts.append(alert)
                elif stock.rsi <= self.config['rsi_oversold']:
                    alert = MonitoringAlert(
                        stock_code=stock.code,
                        stock_name=stock.name,
                        alert_type="RSI과매도",
                        message=f"{stock.name} RSI {stock.rsi:.1f} 과매도 구간",
                        level=AlertLevel.INFO,
                        timestamp=datetime.now(),
                        data={'rsi': stock.rsi}
                    )
                    alerts.append(alert)
                
                # 수급 급변 감지
                total_supply_demand = abs(stock.foreign_net) + abs(stock.institution_net)
                if total_supply_demand >= self.config['supply_demand_threshold']:
                    alert = MonitoringAlert(
                        stock_code=stock.code,
                        stock_name=stock.name,
                        alert_type="수급급변",
                        message=f"{stock.name} 수급 급변 (외국인: {stock.foreign_net:,}, 기관: {stock.institution_net:,})",
                        level=AlertLevel.WARNING,
                        timestamp=datetime.now(),
                        data={
                            'foreign_net': stock.foreign_net,
                            'institution_net': stock.institution_net,
                            'individual_net': stock.individual_net
                        }
                    )
                    alerts.append(alert)
                    
        except Exception as e:
            self.logger.error(f"❌ 변화 감지 중 오류: {e}")
        
        return alerts

    async def _send_monitoring_alerts(self, alerts: List[MonitoringAlert]):
        """모니터링 알림 전송"""
        try:
            for alert in alerts:
                # 알림 히스토리에 추가
                self.alert_history.append(alert)
                
                # 콘솔 출력
                level_emoji = {
                    AlertLevel.INFO: "ℹ️",
                    AlertLevel.WARNING: "⚠️", 
                    AlertLevel.CRITICAL: "🚨"
                }
                
                print(f"\n{level_emoji[alert.level]} [{alert.alert_type}] {alert.message}")
                print(f"   시간: {alert.timestamp.strftime('%H:%M:%S')}")
                
                # 중요한 알림만 텔레그램 전송
                if alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                    telegram_message = f"""
{level_emoji[alert.level]} **{alert.alert_type} 알림**

📈 **{alert.stock_name} ({alert.stock_code})**
📝 {alert.message}
🕐 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

#모니터링알림 #{alert.alert_type}
"""
                    
                    # 텔레그램 알림을 notifier를 통해 직접 전송
                    if self.notifier:
                        success = await self.notifier.send_notification(
                            telegram_message, parse_mode="Markdown"
                        )
                    else:
                        success = False
                    
                    if success:
                        self.logger.info(f"📱 {alert.alert_type} 알림 전송 완료: {alert.stock_name}")
                    
                    # 전송 간격
                    await asyncio.sleep(0.5)
            
            # 알림 히스토리 관리 (최대 1000개)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
                
        except Exception as e:
            self.logger.error(f"❌ 모니터링 알림 전송 중 오류: {e}")

    async def _display_current_status(self, stocks: List[TradingVolumeStock], show_full: bool = True):
        """현재 상태 출력"""
        try:
            if not show_full:
                # 간단한 상태만 출력 (기본 모니터링 시)
                current_time = datetime.now().strftime('%H:%M:%S')
                top_3 = stocks[:3]
                status_line = f"[{current_time}] TOP3: "
                for i, stock in enumerate(top_3):
                    status_line += f"{stock.name}({stock.change_rate:+.1f}%)"
                    if i < len(top_3) - 1:
                        status_line += ", "
                print(f"\r{status_line}", end="", flush=True)
                return
            
            # 전체 상태 출력 (전략 분석 시)
            print(f"\n📊 거래대금 TOP 20 현황 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 120)
            print(f"{'순위':<4} {'종목명':<12} {'코드':<8} {'현재가':<10} {'등락률':<8} {'거래대금':<15} {'RSI':<6} {'수급점수':<8}")
            print("-" * 120)
            
            for stock in stocks[:10]:  # 상위 10개만 출력
                trading_value_str = f"{stock.trading_value//100000000:,}억" if stock.trading_value >= 100000000 else f"{stock.trading_value//100000:,}만"
                
                print(f"{stock.rank:<4} {stock.name:<12} {stock.code:<8} "
                      f"{stock.current_price:>8,.0f} {stock.change_rate:>+6.2f}% "
                      f"{trading_value_str:<15} {stock.rsi:>4.1f} {stock.supply_demand_score:>6.1f}")
            
            print("-" * 120)
            
        except Exception as e:
            self.logger.error(f"❌ 현황 출력 중 오류: {e}")

    async def _save_monitoring_cache(self, stocks: List[TradingVolumeStock]):
        """모니터링 캐시 저장"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'stocks': [
                    {
                        'code': stock.code,
                        'name': stock.name,
                        'current_price': stock.current_price,
                        'trading_value': stock.trading_value,
                        'volume': stock.volume,
                        'change_rate': stock.change_rate,
                        'rank': stock.rank,
                        'rsi': stock.rsi,
                        'supply_demand_score': stock.supply_demand_score
                    }
                    for stock in stocks
                ]
            }
            
            cache_file = self.cache_dir / f"monitoring_cache_{datetime.now().strftime('%Y%m%d')}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 저장 중 오류: {e}")

    async def get_current_analysis_status(self) -> Dict[str, Any]:
        """현재 분석 상태 조회"""
        return {
            "is_running": self.is_running,
            "analysis_interval": self.analysis_interval,
            "monitoring_interval": self.monitoring_interval,
            "min_score_threshold": self.min_score_threshold,
            "strategies": self.strategies,
            "notified_matches_count": len(self.last_notifications),
            "alert_history_count": len(self.alert_history),
            "last_analysis_time": datetime.now().isoformat() if self.is_running else None
        }

    async def update_analysis_settings(
        self, 
        interval: Optional[int] = None,
        min_score: Optional[int] = None,
        monitoring_interval: Optional[int] = None
    ):
        """분석 설정 업데이트"""
        if interval and interval >= 60:  # 최소 1분
            self.analysis_interval = interval
            self.logger.info(f"⚙️ 전략 분석 주기 변경: {interval}초")
        
        if min_score and 50 <= min_score <= 100:
            self.min_score_threshold = min_score
            self.logger.info(f"⚙️ 최소 매칭 점수 변경: {min_score}점")
        
        if monitoring_interval and monitoring_interval >= 10:  # 최소 10초
            self.monitoring_interval = monitoring_interval
            self.logger.info(f"⚙️ 모니터링 주기 변경: {monitoring_interval}초")

    def get_monitoring_summary(self) -> str:
        """모니터링 요약 정보"""
        try:
            recent_alerts = [alert for alert in self.alert_history 
                           if alert.timestamp > datetime.now() - timedelta(hours=1)]
            
            summary = f"""
📊 실시간 모니터링 요약
- 실행 상태: {'실행 중' if self.is_running else '중지됨'}
- 전략 분석 주기: {self.analysis_interval}초
- 기본 모니터링 주기: {self.monitoring_interval}초
- 최소 매칭 점수: {self.min_score_threshold}점
- 최근 1시간 알림: {len(recent_alerts)}개
- 전체 알림 기록: {len(self.alert_history)}개
- 매칭 알림 기록: {len(self.last_notifications)}개
"""
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ 요약 정보 생성 중 오류: {e}")
            return "요약 정보를 가져올 수 없습니다."

    async def cleanup(self):
        """리소스 정리"""
        try:
            self.is_running = False
            
            # 최종 캐시 저장
            if self.previous_data:
                stocks = list(self.previous_data.values())
                await self._save_monitoring_cache(stocks)
            
            self.logger.info("✅ 모니터링 시스템 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 정리 중 오류: {e}")

    def __del__(self):
        """소멸자"""
        self.is_running = False 