#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: monitoring_dashboard.py
모듈: 실시간 모니터링 대시보드
목적: 데이터 파이프라인 성능 및 품질 실시간 모니터링

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - prometheus_client
    - grafana_api
    - streamlit
    - plotly
    - dash
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import dash
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from dash import dcc, html, Input, Output, callback
    from dash.dependencies import Input, Output
    from plotly.subplots import make_subplots
    from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
    from prometheus_client.core import REGISTRY
except ImportError:
    pass

import numpy as np
import pandas as pd
import psutil
import redis.asyncio as redis
import requests
from sqlalchemy import create_engine, text

# 모니터링 라이브러리
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# 기존 라이브러리

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus 메트릭 정의
if PROMETHEUS_AVAILABLE:
    # 카운터 (누적 값)
    DATA_COLLECTION_COUNTER = Counter(
        'trading_data_collected_total',
        'Total number of data records collected',
        ['source', 'data_type']
    )

    DATA_PROCESSING_COUNTER = Counter(
        'trading_data_processed_total',
        'Total number of data records processed',
        ['processing_type']
    )

    ERROR_COUNTER = Counter(
        'trading_errors_total',
        'Total number of errors',
        ['error_type', 'component']
    )

    # 게이지 (현재 값)
    DATA_QUALITY_SCORE = Gauge(
        'trading_data_quality_score',
        'Current data quality score',
        ['source']
    )

    SYSTEM_UPTIME = Gauge(
        'trading_system_uptime_seconds',
        'System uptime in seconds'
    )

    MEMORY_USAGE = Gauge(
        'trading_memory_usage_bytes',
        'Memory usage in bytes'
    )

    CPU_USAGE = Gauge(
        'trading_cpu_usage_percent',
        'CPU usage percentage'
    )

    # 히스토그램 (분포)
    DATA_COLLECTION_DURATION = Histogram(
        'trading_data_collection_duration_seconds',
        'Data collection duration in seconds',
        ['source']
    )

    DATA_PROCESSING_DURATION = Histogram(
        'trading_data_processing_duration_seconds',
        'Data processing duration in seconds',
        ['processing_type']
    )

    # 서머리 (분위수)
    API_RESPONSE_TIME = Summary(
        'trading_api_response_time_seconds',
        'API response time in seconds',
        ['api_endpoint']
    )

@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    # Prometheus 설정
    prometheus_port: int = 8000
    metrics_interval_seconds: int = 30

    # 데이터베이스 설정
    postgres_url: str = "postgresql://user:pass@localhost:5432/trading_data"
    redis_url: str = "redis://localhost:6379/0"

    # 알림 설정
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'data_quality_min': 80.0,
        'system_uptime_min': 99.0,
        'memory_usage_max': 85.0,
        'cpu_usage_max': 80.0,
        'error_rate_max': 5.0
    })

    # 대시보드 설정
    dashboard_refresh_interval: int = 60  # 초
    max_data_points: int = 1000

class PrometheusMetricsCollector:
    """Prometheus 메트릭 수집기"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.start_time = datetime.now()

        if PROMETHEUS_AVAILABLE:
            # Prometheus HTTP 서버 시작
            start_http_server(config.prometheus_port)
            logger.info(f"Prometheus 메트릭 서버 시작: http://localhost:{config.prometheus_port}")

    def record_data_collection(self, source: str, data_type: str, count: int, duration: float):
        """데이터 수집 메트릭 기록"""
        if PROMETHEUS_AVAILABLE:
            DATA_COLLECTION_COUNTER.labels(source=source, data_type=data_type).inc(count)
            DATA_COLLECTION_DURATION.labels(source=source).observe(duration)

    def record_data_processing(self, processing_type: str, count: int, duration: float):
        """데이터 처리 메트릭 기록"""
        if PROMETHEUS_AVAILABLE:
            DATA_PROCESSING_COUNTER.labels(processing_type=processing_type).inc(count)
            DATA_PROCESSING_DURATION.labels(processing_type=processing_type).observe(duration)

    def record_error(self, error_type: str, component: str):
        """에러 메트릭 기록"""
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNTER.labels(error_type=error_type, component=component).inc()

    def update_data_quality(self, source: str, score: float):
        """데이터 품질 점수 업데이트"""
        if PROMETHEUS_AVAILABLE:
            DATA_QUALITY_SCORE.labels(source=source).set(score)

    def update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        if PROMETHEUS_AVAILABLE:
            # 시스템 가동 시간
            uptime = (datetime.now() - self.start_time).total_seconds()
            SYSTEM_UPTIME.set(uptime)

            # 메모리 사용량
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)

            # CPU 사용량
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)

    def record_api_response_time(self, endpoint: str, duration: float):
        """API 응답 시간 기록"""
        if PROMETHEUS_AVAILABLE:
            API_RESPONSE_TIME.labels(api_endpoint=endpoint).observe(duration)

class DataQualityMonitor:
    """데이터 품질 모니터"""

    def __init__(self, config: MonitoringConfig, metrics_collector: PrometheusMetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.engine = create_engine(config.postgres_url)
        self.redis_client = redis.from_url(config.redis_url, decode_responses=True)

    async def check_data_completeness(self) -> Dict[str, float]:
        """데이터 완전성 검사"""
        try:
            # PostgreSQL에서 데이터 완전성 확인
            query = """
            SELECT
                symbol,
                COUNT(*) as total_records,
                COUNT(CASE WHEN close IS NOT NULL THEN 1 END) as non_null_close,
                COUNT(CASE WHEN volume IS NOT NULL THEN 1 END) as non_null_volume
            FROM ohlcv_data
            WHERE collected_at >= NOW() - INTERVAL '7 days'
            GROUP BY symbol
            """

            df = pd.read_sql(query, self.engine)

            completeness_scores = {}
            for _, row in df.iterrows():
                symbol = row['symbol']
                total = row['total_records']
                close_complete = row['non_null_close']
                volume_complete = row['non_null_volume']

                # 완전성 점수 계산
                close_score = (close_complete / total) * 100 if total > 0 else 0
                volume_score = (volume_complete / total) * 100 if total > 0 else 0
                overall_score = (close_score + volume_score) / 2

                completeness_scores[symbol] = overall_score

                # Prometheus 메트릭 업데이트
                self.metrics_collector.update_data_quality(f"{symbol}_completeness", overall_score)

            return completeness_scores

        except Exception as e:
            logger.error(f"데이터 완전성 검사 실패: {e}")
            self.metrics_collector.record_error("data_completeness_check", "database")
            return {}

    async def check_data_accuracy(self) -> Dict[str, float]:
        """데이터 정확성 검사"""
        try:
            # 비정상적인 가격 변동 확인
            query = """
            SELECT
                symbol,
                COUNT(*) as total_records,
                COUNT(CASE WHEN close < 0 THEN 1 END) as negative_prices,
                COUNT(CASE WHEN close > 1000000 THEN 1 END) as extreme_prices,
                COUNT(CASE WHEN ABS(close - LAG(close) OVER (PARTITION BY symbol ORDER BY collected_at)) / LAG(close) OVER (PARTITION BY symbol ORDER BY collected_at) > 0.5 THEN 1 END) as extreme_changes
            FROM ohlcv_data
            WHERE collected_at >= NOW() - INTERVAL '7 days'
            GROUP BY symbol
            """

            df = pd.read_sql(query, self.engine)

            accuracy_scores = {}
            for _, row in df.iterrows():
                symbol = row['symbol']
                total = row['total_records']
                negative = row['negative_prices']
                extreme = row['extreme_prices']
                extreme_changes = row['extreme_changes']

                # 정확성 점수 계산
                accuracy_score = 100
                if total > 0:
                    accuracy_score -= (negative / total) * 50  # 음수 가격
                    accuracy_score -= (extreme / total) * 30   # 극단적 가격
                    accuracy_score -= (extreme_changes / total) * 20  # 극단적 변동

                accuracy_scores[symbol] = max(0, accuracy_score)

                # Prometheus 메트릭 업데이트
                self.metrics_collector.update_data_quality(f"{symbol}_accuracy", accuracy_score)

            return accuracy_scores

        except Exception as e:
            logger.error(f"데이터 정확성 검사 실패: {e}")
            self.metrics_collector.record_error("data_accuracy_check", "database")
            return {}

    async def check_data_timeliness(self) -> Dict[str, float]:
        """데이터 적시성 검사"""
        try:
            # 최근 데이터 수집 시간 확인
            query = """
            SELECT
                symbol,
                MAX(collected_at) as latest_collection,
                NOW() - MAX(collected_at) as time_difference
            FROM ohlcv_data
            GROUP BY symbol
            """

            df = pd.read_sql(query, self.engine)

            timeliness_scores = {}
            for _, row in df.iterrows():
                symbol = row['symbol']
                time_diff = row['time_difference']

                # 적시성 점수 계산 (5분 이내: 100점, 1시간 이내: 80점, 1일 이내: 60점)
                if time_diff.total_seconds() <= 300:  # 5분
                    timeliness_score = 100
                elif time_diff.total_seconds() <= 3600:  # 1시간
                    timeliness_score = 80
                elif time_diff.total_seconds() <= 86400:  # 1일
                    timeliness_score = 60
                else:
                    timeliness_score = 0

                timeliness_scores[symbol] = timeliness_score

                # Prometheus 메트릭 업데이트
                self.metrics_collector.update_data_quality(f"{symbol}_timeliness", timeliness_score)

            return timeliness_scores

        except Exception as e:
            logger.error(f"데이터 적시성 검사 실패: {e}")
            self.metrics_collector.record_error("data_timeliness_check", "database")
            return {}

    async def generate_quality_report(self) -> Dict[str, Any]:
        """품질 리포트 생성"""
        logger.info("데이터 품질 리포트 생성 시작")

        # 각 품질 지표 수집
        completeness_scores = await self.check_data_completeness()
        accuracy_scores = await self.check_data_accuracy()
        timeliness_scores = await self.check_data_timeliness()

        # 종합 품질 점수 계산
        all_symbols = set(completeness_scores.keys()) | set(accuracy_scores.keys()) | set(timeliness_scores.keys())

        overall_scores = {}
        for symbol in all_symbols:
            completeness = completeness_scores.get(symbol, 0)
            accuracy = accuracy_scores.get(symbol, 0)
            timeliness = timeliness_scores.get(symbol, 0)

            # 가중 평균 (완전성 40%, 정확성 40%, 적시성 20%)
            overall_score = (completeness * 0.4) + (accuracy * 0.4) + (timeliness * 0.2)
            overall_scores[symbol] = overall_score

            # Prometheus 메트릭 업데이트
            self.metrics_collector.update_data_quality(symbol, overall_score)

        # 알림 확인
        alerts = []
        for symbol, score in overall_scores.items():
            if score < self.config.alert_thresholds['data_quality_min']:
                alerts.append(f"종목 {symbol} 품질 점수 낮음: {score:.1f}")

        report = {
            'timestamp': datetime.now().isoformat(),
            'completeness_scores': completeness_scores,
            'accuracy_scores': accuracy_scores,
            'timeliness_scores': timeliness_scores,
            'overall_scores': overall_scores,
            'average_score': np.mean(list(overall_scores.values())) if overall_scores else 0,
            'alerts': alerts
        }

        # Redis에 리포트 저장
        await self.redis_client.setex(
            'quality_report',
            3600,  # 1시간 TTL
            json.dumps(report, ensure_ascii=False)
        )

        logger.info(f"품질 리포트 생성 완료: 평균 점수 {report['average_score']:.1f}")
        return report

class PerformanceMonitor:
    """성능 모니터"""

    def __init__(self, config: MonitoringConfig, metrics_collector: PrometheusMetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.engine = create_engine(config.postgres_url)

    async def get_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 정보 수집"""
        try:
            # 시스템 리소스 사용량
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 네트워크 사용량
            network = psutil.net_io_counters()

            # 데이터베이스 성능
            db_performance = await self._get_database_performance()

            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'database_performance': db_performance
            }

            # Prometheus 메트릭 업데이트
            self.metrics_collector.update_system_metrics()

            return performance_data

        except Exception as e:
            logger.error(f"시스템 성능 수집 실패: {e}")
            self.metrics_collector.record_error("system_performance", "monitoring")
            return {}

    async def _get_database_performance(self) -> Dict[str, Any]:
        """데이터베이스 성능 정보"""
        try:
            # 테이블 크기
            size_query = """
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """

            size_df = pd.read_sql(size_query, self.engine)

            # 최근 데이터 수집 통계
            stats_query = """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                MAX(collected_at) as latest_data,
                MIN(collected_at) as earliest_data
            FROM ohlcv_data
            """

            stats_df = pd.read_sql(stats_query, self.engine)

            return {
                'table_sizes': size_df.to_dict('records'),
                'data_statistics': stats_df.to_dict('records')[0] if not stats_df.empty else {}
            }

        except Exception as e:
            logger.error(f"데이터베이스 성능 수집 실패: {e}")
            return {}

class AlertManager:
    """알림 관리자"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []

    def check_alerts(self, quality_report: Dict[str, Any], performance_data: Dict[str, Any]) -> List[str]:
        """알림 조건 확인"""
        alerts = []

        # 데이터 품질 알림
        if quality_report:
            avg_score = quality_report.get('average_score', 0)
            if avg_score < self.config.alert_thresholds['data_quality_min']:
                alerts.append(f"데이터 품질 점수 낮음: {avg_score:.1f}")

            for alert in quality_report.get('alerts', []):
                alerts.append(alert)

        # 시스템 성능 알림
        if performance_data:
            memory_percent = performance_data.get('memory_percent', 0)
            if memory_percent > self.config.alert_thresholds['memory_usage_max']:
                alerts.append(f"메모리 사용량 높음: {memory_percent:.1f}%")

            cpu_percent = performance_data.get('cpu_percent', 0)
            if cpu_percent > self.config.alert_thresholds['cpu_usage_max']:
                alerts.append(f"CPU 사용량 높음: {cpu_percent:.1f}%")

        # 알림 히스토리 업데이트
        if alerts:
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts
            })

        # 최근 100개 알림만 유지
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        return alerts

def create_streamlit_dashboard():
    """Streamlit 대시보드 생성"""
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit이 설치되어 있지 않습니다. 일부 대시보드 기능이 비활성화됩니다.")
        return

    st.set_page_config(
        page_title="트레이딩 데이터 모니터링 대시보드",
        page_icon="📊",
        layout="wide"
    )

    st.title("🚀 트레이딩 데이터 모니터링 대시보드")

    # 사이드바 설정
    st.sidebar.header("설정")
    refresh_interval = st.sidebar.slider("새로고침 간격 (초)", 10, 300, 60)

    # 메인 대시보드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("시스템 가동률", "99.8%", "0.1%")

    with col2:
        st.metric("데이터 품질", "92.5%", "1.2%")

    with col3:
        st.metric("CPU 사용량", "45.2%", "2.1%")

    with col4:
        st.metric("메모리 사용량", "67.8%", "-1.5%")

    # 차트 영역
    st.subheader("실시간 성능 모니터링")

    # 가상 데이터 생성 (실제로는 실제 데이터 사용)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    cpu_data = np.random.normal(45, 10, 100)
    memory_data = np.random.normal(68, 5, 100)
    quality_data = np.random.normal(92, 3, 100)

    # 성능 차트
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU 사용량', '메모리 사용량', '데이터 품질', '시스템 가동률'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(x=dates, y=cpu_data, name='CPU', line=dict(color='red')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=dates, y=memory_data, name='Memory', line=dict(color='blue')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=dates, y=quality_data, name='Quality', line=dict(color='green')),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=dates, y=[99.8] * 100, name='Uptime', line=dict(color='orange')),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # 알림 섹션
    st.subheader("🔔 최근 알림")

    # 가상 알림 데이터
    alerts = [
        {"time": "2025-01-27 14:30:00", "level": "Warning", "message": "데이터 품질 점수 85%로 하락"},
        {"time": "2025-01-27 14:15:00", "level": "Info", "message": "일일 데이터 수집 완료"},
        {"time": "2025-01-27 14:00:00", "level": "Success", "message": "백업 작업 성공"}
    ]

    for alert in alerts:
        if alert["level"] == "Warning":
            st.warning(f"{alert['time']} - {alert['message']}")
        elif alert["level"] == "Info":
            st.info(f"{alert['time']} - {alert['message']}")
        else:
            st.success(f"{alert['time']} - {alert['message']}")

def create_dash_dashboard():
    """Dash 대시보드 생성"""
    if not DASH_AVAILABLE:
        logger.error("Dash가 설치되지 않았습니다.")
        return None

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("트레이딩 데이터 모니터링 대시보드"),

        # 메트릭 카드
        html.Div([
            html.Div([
                html.H3("시스템 가동률"),
                html.H2("99.8%", id="uptime-metric")
            ], className="metric-card"),

            html.Div([
                html.H3("데이터 품질"),
                html.H2("92.5%", id="quality-metric")
            ], className="metric-card"),

            html.Div([
                html.H3("CPU 사용량"),
                html.H2("45.2%", id="cpu-metric")
            ], className="metric-card"),

            html.Div([
                html.H3("메모리 사용량"),
                html.H2("67.8%", id="memory-metric")
            ], className="metric-card")
        ], className="metrics-container"),

        # 차트
        dcc.Graph(id="performance-chart"),

        # 자동 새로고침
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # 60초마다
            n_intervals=0
        )
    ])

    @app.callback(
        Output('performance-chart', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_chart(n):
        # 가상 데이터 생성
        dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
        cpu_data = np.random.normal(45, 10, 100)
        memory_data = np.random.normal(68, 5, 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=cpu_data, name='CPU', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=dates, y=memory_data, name='Memory', line=dict(color='blue')))

        fig.update_layout(
            title="실시간 성능 모니터링",
            xaxis_title="시간",
            yaxis_title="사용량 (%)"
        )

        return fig

    return app

async def run_monitoring_system():
    """모니터링 시스템 실행"""
    logger.info("🔍 모니터링 시스템 시작")

    # 설정
    config = MonitoringConfig()

    # 메트릭 수집기 초기화
    metrics_collector = PrometheusMetricsCollector(config)

    # 모니터 초기화
    quality_monitor = DataQualityMonitor(config, metrics_collector)
    performance_monitor = PerformanceMonitor(config, metrics_collector)
    alert_manager = AlertManager(config)

    try:
        while True:
            logger.info("모니터링 데이터 수집 중...")

            # 데이터 품질 모니터링
            quality_report = await quality_monitor.generate_quality_report()

            # 성능 모니터링
            performance_data = await performance_monitor.get_system_performance()

            # 알림 확인
            alerts = alert_manager.check_alerts(quality_report, performance_data)

            if alerts:
                logger.warning(f"알림 발생: {alerts}")

            # 대기
            await asyncio.sleep(config.metrics_interval_seconds)

    except KeyboardInterrupt:
        logger.info("모니터링 시스템 종료")
    except Exception as e:
        logger.error(f"모니터링 시스템 오류: {e}")

def main():
    """메인 함수"""
    print("🔍 트레이딩 데이터 모니터링 시스템 시작")
    print("=" * 60)

    # 모니터링 시스템 실행
    asyncio.run(run_monitoring_system())

if __name__ == "__main__":
    main()

