                        important_disclosures.append(disclosure)
                        important_news.append(news)
            important_disclosures = []
            important_news = []
from core.config import config
from core.logger import get_logger
from data.dart import DARTMonitor, DisclosureAlert
from data.futures import DerivativesDataCollector
from data.news import NewsCollector
from data.preprocessing import DataPreprocessor
from data.realtime import RealtimeDataManager, RealtimeConfig
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from typing import Dict, List, Optional, Any, Callable
import asyncio
import json
import numpy as np
import pandas as pd
import time
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 모니터링 시스템
개장 전 뉴스/공시 스캔, 장중 급등/급락주 감지, 테마주 동반상승 패턴 감지
"""



logger = get_logger(__name__)
console = Console()

@dataclass
class MarketAlert:
    """시장 알림"""
    alert_type: str
    symbol: str
    message: str
    priority: int  # 1-5 (5가 가장 높음)
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThemePattern:
    """테마주 패턴"""
    theme_name: str
    symbols: List[str]
    correlation: float
    avg_change: float
    volume_surge: float
    timestamp: datetime

class RealtimeMonitor:
    """실시간 모니터링 시스템"""

    def __init__(self):
        self.realtime_manager = RealtimeDataManager()
        self.dart_monitor = DARTMonitor()
        self.derivatives_collector = DerivativesDataCollector()
        self.news_collector = NewsCollector()
        self.preprocessor = DataPreprocessor()

        self.alerts: List[MarketAlert] = []
        self.theme_patterns: List[ThemePattern] = []
        self.running = False

        # 콜백 등록
        self.dart_monitor.add_callback(self.on_disclosure_alert)

        # 모니터링 설정
        self.monitored_symbols = []
        self.theme_keywords = []
        self.alert_callbacks: List[Callable] = []

    async def start_monitoring(self, symbols: List[str] = None, theme_keywords: List[str] = None):
        """모니터링 시작"""
        try:
            self.monitored_symbols = symbols or []
            self.theme_keywords = theme_keywords or []

            self.running = True
            logger.info("실시간 모니터링 시작")

            # 개장 전 스캔 (8:30-9:00)
            await self.pre_market_scan()

            # 장중 모니터링 시작
            await self.start_intraday_monitoring()

        except Exception as e:
            logger.error(f"모니터링 시작 실패: {e}")
            raise

    async def stop_monitoring(self):
        """모니터링 중지"""
        self.running = False

        # 모든 수집기 중지
        await self.realtime_manager.stop_all()
        await self.dart_monitor.stop_monitoring()
        await self.derivatives_collector.stop()

        logger.info("실시간 모니터링 중지")

    async def pre_market_scan(self):
        """개장 전 스캔 (8:30-9:00)"""
        try:
            console.print(Panel("🔍 개장 전 스캔 시작 (8:30-9:00)", style="bold blue"))

            # 뉴스 스캔
            await self.scan_news()

            # 공시 스캔
            await self.scan_disclosures()

            # 테마 분석
            await self.analyze_themes()

            console.print(Panel("✅ 개장 전 스캔 완료", style="bold green"))

        except Exception as e:
            logger.error(f"개장 전 스캔 오류: {e}")

    async def start_intraday_monitoring(self):
        """장중 모니터링 시작"""
        try:
            console.print(Panel("📈 장중 실시간 모니터링 시작", style="bold green"))

            # 실시간 데이터 수집 시작
            if self.monitored_symbols:
                await self.realtime_manager.add_symbols(self.monitored_symbols)
                await self.realtime_manager.start_all()

            # DART 모니터링 시작
            await self.dart_monitor.start_monitoring(
                corps=self.monitored_symbols,
                keywords=self.theme_keywords
            )

            # 파생상품 모니터링 시작
            await self.derivatives_collector.start(
                futures_symbols=config.trading.FUTURES_SYMBOLS,
                options_symbols=config.trading.OPTIONS_SYMBOLS
            )

            # 실시간 대시보드 시작
            await self.start_dashboard()

        except Exception as e:
            logger.error(f"장중 모니터링 시작 오류: {e}")

    async def scan_news(self):
        """뉴스 스캔"""
        try:
            console.print("📰 뉴스 스캔 중...")

            # 최근 뉴스 수집
            news_data = await self.news_collector.get_recent_news(hours=24)

            # 중요 키워드 필터링
            for news in news_data:
                for keyword in self.theme_keywords:
                    if keyword in news.title or keyword in news.content:
                        break

            # 알림 생성
            for news in important_news[:10]:  # 상위 10개만
                alert = MarketAlert(
                    alert_type='news',
                    symbol='',
                    message=f"중요 뉴스: {news.title}",
                    priority=3,
                    timestamp=datetime.now(),
                    data={'news': news}
                )
                self.alerts.append(alert)

            console.print(f"📰 뉴스 스캔 완료: {len(important_news)}개 중요 뉴스 발견")

        except Exception as e:
            logger.error(f"뉴스 스캔 오류: {e}")

    async def scan_disclosures(self):
        """공시 스캔"""
        try:
            console.print("📋 공시 스캔 중...")

            # 최근 공시 조회
            recent_disclosures = self.dart_monitor.get_recent_disclosures(hours=24)

            # 중요 공시 필터링
            for disclosure in recent_disclosures:
                for keyword in self.theme_keywords:
                    if keyword in disclosure.report_nm:
                        break

            # 알림 생성
            for disclosure in important_disclosures[:10]:  # 상위 10개만
                alert = MarketAlert(
                    alert_type='disclosure',
                    symbol=disclosure.stock_code,
                    message=f"중요 공시: {disclosure.corp_name} - {disclosure.report_nm}",
                    priority=4,
                    timestamp=datetime.now(),
                    data={'disclosure': disclosure}
                )
                self.alerts.append(alert)

            console.print(f"📋 공시 스캔 완료: {len(important_disclosures)}개 중요 공시 발견")

        except Exception as e:
            logger.error(f"공시 스캔 오류: {e}")

    async def analyze_themes(self):
        """테마 분석"""
        try:
            console.print("🎯 테마 분석 중...")

            # 테마별 종목 그룹핑
            theme_groups = {}
            for keyword in self.theme_keywords:
                theme_groups[keyword] = []
                # 실제로는 종목 데이터베이스에서 테마별 종목을 조회해야 함
                # 여기서는 예시로 빈 리스트 사용

            # 테마 상관관계 분석
            for theme, symbols in theme_groups.items():
                if len(symbols) >= config.trading.THEME_MIN_STOCKS:
                    pattern = await self.analyze_theme_correlation(theme, symbols)
                    if pattern:
                        self.theme_patterns.append(pattern)

            console.print(f"🎯 테마 분석 완료: {len(self.theme_patterns)}개 테마 패턴 발견")

        except Exception as e:
            logger.error(f"테마 분석 오류: {e}")

    async def analyze_theme_correlation(self, theme: str, symbols: List[str]) -> Optional[ThemePattern]:
        """테마 상관관계 분석"""
        try:
            # 실제로는 각 종목의 가격 데이터를 수집하여 상관관계를 계산해야 함
            # 여기서는 예시로 더미 데이터 사용

            if len(symbols) < config.trading.THEME_MIN_STOCKS:
                return None

            # 가격 변화율 계산 (실제로는 실시간 데이터 사용)
            price_changes = np.random.normal(0, 0.02, len(symbols))  # 예시 데이터

            # 상관관계 계산
            correlation_matrix = np.corrcoef(price_changes)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])

            # 테마 패턴 생성
            if avg_correlation >= config.trading.THEME_CORRELATION_THRESHOLD:
                pattern = ThemePattern(
                    theme_name=theme,
                    symbols=symbols,
                    correlation=avg_correlation,
                    avg_change=np.mean(price_changes),
                    volume_surge=np.random.uniform(1.5, 3.0),  # 예시 데이터
                    timestamp=datetime.now()
                )
                return pattern

            return None

        except Exception as e:
            logger.error(f"테마 상관관계 분석 오류: {e}")
            return None

    async def on_disclosure_alert(self, alert: DisclosureAlert):
        """공시 알림 콜백"""
        try:
            market_alert = MarketAlert(
                alert_type='disclosure',
                symbol=alert.disclosure.stock_code,
                message=f"공시 알림: {alert.disclosure.corp_name} - {alert.disclosure.report_nm}",
                priority=alert.priority,
                timestamp=datetime.now(),
                data={'disclosure_alert': alert}
            )

            self.alerts.append(market_alert)

            # 콜백 실행
            for callback in self.alert_callbacks:
                await callback(market_alert)

            console.print(f"📋 [bold red]공시 알림[/bold red]: {alert.disclosure.corp_name}")

        except Exception as e:
            logger.error(f"공시 알림 처리 오류: {e}")

    async def detect_volume_surge(self):
        """거래량 급증 감지"""
        try:
            for collector_key, collector in self.realtime_manager.collectors.items():
                volume_alerts = collector.get_alerts('volume')

                for alert in volume_alerts:
                    market_alert = MarketAlert(
                        alert_type='volume_surge',
                        symbol=alert.symbol,
                        message=f"거래량 급증: {alert.symbol} {alert.volume_ratio:.2f}배",
                        priority=3,
                        timestamp=datetime.now(),
                        data={'volume_alert': alert}
                    )

                    self.alerts.append(market_alert)

                    # 콜백 실행
                    for callback in self.alert_callbacks:
                        await callback(market_alert)

                    console.print(f"📊 [bold yellow]거래량 급증[/bold yellow]: {alert.symbol}")

        except Exception as e:
            logger.error(f"거래량 급증 감지 오류: {e}")

    async def detect_price_breakout(self):
        """가격 급등/급락 감지"""
        try:
            for collector_key, collector in self.realtime_manager.collectors.items():
                price_alerts = collector.get_alerts('price')

                for alert in price_alerts:
                    market_alert = MarketAlert(
                        alert_type='price_breakout',
                        symbol=alert.symbol,
                        message=f"가격 급변: {alert.symbol} {alert.price_change_pct:.2f}%",
                        priority=4 if abs(alert.price_change_pct) > 10 else 3,
                        timestamp=datetime.now(),
                        data={'price_alert': alert}
                    )

                    self.alerts.append(market_alert)

                    # 콜백 실행
                    for callback in self.alert_callbacks:
                        await callback(market_alert)

                    color = "bold red" if alert.price_change_pct > 0 else "bold green"
                    console.print(f"💰 [{color}]가격 급변[/{color}]: {alert.symbol}")

        except Exception as e:
            logger.error(f"가격 급등/급락 감지 오류: {e}")

    async def start_dashboard(self):
        """실시간 대시보드 시작"""
        try:
            layout = Layout()

            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )

            layout["main"].split_row(
                Layout(name="alerts", ratio=2),
                Layout(name="themes", ratio=1)
            )

            with Live(layout, refresh_per_second=1, screen=True):
                while self.running:
                    # 대시보드 업데이트
                    layout["header"].update(self.create_header())
                    layout["alerts"].update(self.create_alerts_table())
                    layout["themes"].update(self.create_themes_table())
                    layout["footer"].update(self.create_footer())

                    # 알림 감지
                    await self.detect_volume_surge()
                    await self.detect_price_breakout()

                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"대시보드 시작 오류: {e}")

    def create_header(self) -> Panel:
        """헤더 패널 생성"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_status = "🟢 장중" if self.is_market_open() else "🔴 장마감"

        header_text = f"📈 실시간 모니터링 시스템 | {current_time} | {market_status}"
        return Panel(header_text, style="bold blue")

    def create_alerts_table(self) -> Table:
        """알림 테이블 생성"""
        table = Table(title="🚨 실시간 알림")
        table.add_column("시간", style="cyan")
        table.add_column("유형", style="magenta")
        table.add_column("종목", style="green")
        table.add_column("메시지", style="white")
        table.add_column("우선순위", style="yellow")

        # 최근 알림 20개만 표시
        recent_alerts = sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:20]

        for alert in recent_alerts:
            time_str = alert.timestamp.strftime("%H:%M:%S")
            priority_stars = "★" * alert.priority

            table.add_row(
                time_str,
                alert.alert_type,
                alert.symbol,
                alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
                priority_stars
            )

        return table

    def create_themes_table(self) -> Table:
        """테마 테이블 생성"""
        table = Table(title="🎯 테마 패턴")
        table.add_column("테마", style="cyan")
        table.add_column("상관관계", style="magenta")
        table.add_column("평균변화", style="green")
        table.add_column("거래량증가", style="yellow")

        # 최근 테마 패턴 10개만 표시
        recent_patterns = sorted(self.theme_patterns, key=lambda x: x.timestamp, reverse=True)[:10]

        for pattern in recent_patterns:
            table.add_row(
                pattern.theme_name,
                f"{pattern.correlation:.3f}",
                f"{pattern.avg_change:.2%}",
                f"{pattern.volume_surge:.1f}배"
            )

        return table

    def create_footer(self) -> Panel:
        """푸터 패널 생성"""
        stats = f"총 알림: {len(self.alerts)} | 테마 패턴: {len(self.theme_patterns)} | 모니터링 종목: {len(self.monitored_symbols)}"
        return Panel(stats, style="bold green")

    def is_market_open(self) -> bool:
        """장 개장 여부 확인"""
        now = datetime.now()
        current_time = now.time()

        # 평일 9:00-15:30
        market_start = datetime.strptime(config.trading.MARKET_OPEN_TIME, "%H:%M").time()
        market_end = datetime.strptime(config.trading.MARKET_CLOSE_TIME, "%H:%M").time()

        return (now.weekday() < 5 and  # 평일
                market_start <= current_time <= market_end)

    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def get_recent_alerts(self, hours: int = 1) -> List[MarketAlert]:
        """최근 알림 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]

    def get_alerts_by_type(self, alert_type: str) -> List[MarketAlert]:
        """알림 유형별 조회"""
        return [alert for alert in self.alerts if alert.alert_type == alert_type]

    def get_high_priority_alerts(self, min_priority: int = 4) -> List[MarketAlert]:
        """고우선순위 알림 조회"""
        return [alert for alert in self.alerts if alert.priority >= min_priority]

class MarketAnalyzer:
    """시장 분석기"""

    def __init__(self, monitor: RealtimeMonitor):
        self.monitor = monitor

    async def analyze_market_sentiment(self) -> Dict[str, Any]:
        """시장 심리 분석"""
        try:
            # 최근 알림 분석
            recent_alerts = self.monitor.get_recent_alerts(hours=1)

            # 알림 유형별 분류
            alert_counts = {}
            for alert in recent_alerts:
                alert_counts[alert.alert_type] = alert_counts.get(alert.alert_type, 0) + 1

            # 시장 심리 점수 계산
            sentiment_score = 0
            if 'price_breakout' in alert_counts:
                sentiment_score += alert_counts['price_breakout'] * 2
            if 'volume_surge' in alert_counts:
                sentiment_score += alert_counts['volume_surge']
            if 'disclosure' in alert_counts:
                sentiment_score += alert_counts['disclosure'] * 3

            # 심리 상태 판단
            if sentiment_score > 20:
                sentiment = "매우 활발"
            elif sentiment_score > 10:
                sentiment = "활발"
            elif sentiment_score > 5:
                sentiment = "보통"
            else:
                sentiment = "침체"

            return {
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'alert_counts': alert_counts,
                'total_alerts': len(recent_alerts)
            }

        except Exception as e:
            logger.error(f"시장 심리 분석 오류: {e}")
            return {}

    async def analyze_theme_performance(self) -> List[Dict[str, Any]]:
        """테마 성과 분석"""
        try:
            theme_performance = []

            for pattern in self.monitor.theme_patterns:
                performance = {
                    'theme_name': pattern.theme_name,
                    'correlation': pattern.correlation,
                    'avg_change': pattern.avg_change,
                    'volume_surge': pattern.volume_surge,
                    'symbol_count': len(pattern.symbols),
                    'timestamp': pattern.timestamp
                }
                theme_performance.append(performance)

            # 상관관계 순으로 정렬
            theme_performance.sort(key=lambda x: x['correlation'], reverse=True)

            return theme_performance

        except Exception as e:
            logger.error(f"테마 성과 분석 오류: {e}")
            return []

    async def generate_market_report(self) -> str:
        """시장 리포트 생성"""
        try:
            sentiment_analysis = await self.analyze_market_sentiment()
            theme_performance = await self.analyze_theme_performance()

            report = f"""
📊 시장 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

🎯 시장 심리
- 심리 점수: {sentiment_analysis.get('sentiment_score', 0)}
- 심리 상태: {sentiment_analysis.get('sentiment', '알 수 없음')}
- 총 알림: {sentiment_analysis.get('total_alerts', 0)}개

📈 테마 성과 (상위 5개)
"""

            for i, theme in enumerate(theme_performance[:5]):
                report += f"{i+1}. {theme['theme_name']}: 상관관계 {theme['correlation']:.3f}, 변화율 {theme['avg_change']:.2%}\n"

            return report

        except Exception as e:
            logger.error(f"시장 리포트 생성 오류: {e}")
            return "리포트 생성 실패"

