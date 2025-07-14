#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: realtime_monitor.py
목적: AI 트레이딩 시스템 실시간 모니터링 대시보드
Author: [Your Name]
Created: 2025-07-11
Version: 1.0.0

- 커서룰 100% 준수 (타입힌트, 예외처리, 구조화 로깅, 문서화)
- 실시간 데이터 수집 상태, ML 모델 성능, 시스템 리소스 모니터링
- 웹 대시보드, 알림 시스템, 성능 최적화
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime
import timedelta
from pathlib import Path
from typing import Dict
import List
import Optional, Any
import pandas as pd
import psutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc
import html
import Input, Output
import dash_bootstrap_components as dbc
import threading
import queue
import json

# 구조화 로깅
logging.basicConfig(
    filename="logs/realtime_monitor.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

class RealtimeMonitor:
    """실시간 모니터링 시스템"""

    def __init__(self):
        self.data_queue = queue.Queue()
        self.performance_metrics: Dict[str, Any] = {}
        self.system_metrics: Dict[str, float] = {}
        self.ml_metrics: Dict[str, float] = {}
        self.collection_status: Dict[str, str] = {}

    async def collect_system_metrics(self) -> None:
        """시스템 리소스 메트릭 수집"""
        while True:
            try:
                # CPU, 메모리, 디스크 사용량
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                self.system_metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / (1024**3),
                    'disk_total_gb': disk.total / (1024**3)
                }

                logger.info(f"System metrics collected: CPU {cpu_percent}%, Memory {memory.percent}%")
                await asyncio.sleep(5)  # 5초마다 수집

            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(10)

    async def collect_data_status(self) -> None:
        """데이터 수집 상태 모니터링"""
        while True:
            try:
                data_dir = Path("data")
                parquet_files = list(data_dir.glob("*.parquet"))

                # 최근 수집된 파일들 확인
                recent_files = [f for f in parquet_files if f.stat().st_mtime > time.time() - 3600]

                self.collection_status = {
                    'total_files': len(parquet_files),
                    'recent_files': len(recent_files),
                    'last_update': datetime.now().isoformat(),
                    'data_size_gb': sum(f.stat().st_size for f in parquet_files) / (1024**3)
                }

                logger.info(f"Data collection status: {len(recent_files)} recent files")
                await asyncio.sleep(30)  # 30초마다 확인

            except Exception as e:
                logger.error(f"Data status collection error: {e}")
                await asyncio.sleep(60)

    async def collect_ml_metrics(self) -> None:
        """ML 모델 성능 메트릭 수집"""
        while True:
            try:
                # MLflow에서 모델 메트릭 가져오기
                import mlflow
                client = mlflow.tracking.MlflowClient()

                # 최근 실험 결과 확인
                experiments = client.list_experiments()
                if experiments:
                    latest_exp = experiments[-1]
                    runs = client.list_run_infos(latest_exp.experiment_id)

                    if runs:
                        latest_run = runs[-1]
                        run_data = client.get_run(latest_run.run_id)

                        self.ml_metrics = {
                            'model_name': run_data.data.tags.get('mlflow.runName', 'Unknown'),
                            'mse': run_data.data.metrics.get('mse', 0.0),
                            'mae': run_data.data.metrics.get('mae', 0.0),
                            'r2_score': run_data.data.metrics.get('r2_score', 0.0),
                            'training_time': run_data.data.metrics.get('training_time', 0.0),
                            'last_training': run_data.info.start_time
                        }

                logger.info(f"ML metrics collected: MSE {self.ml_metrics.get('mse', 0):.2f}")
                await asyncio.sleep(60)  # 1분마다 확인

            except Exception as e:
                logger.error(f"ML metrics collection error: {e}")
                await asyncio.sleep(120)

    def create_dashboard(self) -> dash.Dash:
        """Dash 대시보드 생성"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("AI 트레이딩 시스템 실시간 모니터링", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),

            dbc.Row([
                # 시스템 리소스
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("시스템 리소스"),
                        dbc.CardBody([
                            dcc.Graph(id='system-metrics-graph'),
                            html.Div(id='system-metrics-text')
                        ])
                    ])
                ], width=6),

                # 데이터 수집 상태
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("데이터 수집 상태"),
                        dbc.CardBody([
                            dcc.Graph(id='data-status-graph'),
                            html.Div(id='data-status-text')
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            dbc.Row([
                # ML 모델 성능
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ML 모델 성능"),
                        dbc.CardBody([
                            dcc.Graph(id='ml-metrics-graph'),
                            html.Div(id='ml-metrics-text')
                        ])
                    ])
                ], width=12)
            ]),

            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5초마다 업데이트
                n_intervals=0
            )
        ], fluid=True)

        @app.callback(
            [Output('system-metrics-graph', 'figure'),
             Output('system-metrics-text', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_system_metrics(n):
            # CPU, 메모리, 디스크 사용량 그래프
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('CPU 사용률', '메모리 사용률', '디스크 사용률'),
                vertical_spacing=0.1
            )

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=self.system_metrics.get('cpu_percent', 0),
                    title={'text': "CPU (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}]}
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=self.system_metrics.get('memory_percent', 0),
                    title={'text': "Memory (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 70], 'color': "lightgray"},
                                    {'range': [70, 90], 'color': "yellow"},
                                    {'range': [90, 100], 'color': "red"}]}
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=self.system_metrics.get('disk_percent', 0),
                    title={'text': "Disk (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkred"},
                           'steps': [{'range': [0, 80], 'color': "lightgray"},
                                    {'range': [80, 95], 'color': "yellow"},
                                    {'range': [95, 100], 'color': "red"}]}
                ),
                row=3, col=1
            )

            fig.update_layout(height=600, showlegend=False)

            # 텍스트 정보
            text = f"""
            CPU: {self.system_metrics.get('cpu_percent', 0):.1f}%
            Memory: {self.system_metrics.get('memory_percent', 0):.1f}%
            ({self.system_metrics.get('memory_used_gb', 0):.1f}GB / {self.system_metrics.get('memory_total_gb', 0):.1f}GB)
            Disk: {self.system_metrics.get('disk_percent', 0):.1f}%
            ({self.system_metrics.get('disk_used_gb', 0):.1f}GB / {self.system_metrics.get('disk_total_gb', 0):.1f}GB)
            """

            return fig, text

        @app.callback(
            [Output('data-status-graph', 'figure'),
             Output('data-status-text', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_data_status(n):
            # 데이터 수집 상태 그래프
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['총 파일 수', '최근 파일 수'],
                y=[self.collection_status.get('total_files', 0),
                   self.collection_status.get('recent_files', 0)],
                marker_color=['blue', 'green']
            ))

            fig.update_layout(
                title="데이터 수집 현황",
                yaxis_title="파일 수",
                height=400
            )

            # 텍스트 정보
            text = f"""
            총 데이터 파일: {self.collection_status.get('total_files', 0)}개
            최근 수집 파일: {self.collection_status.get('recent_files', 0)}개
            데이터 크기: {self.collection_status.get('data_size_gb', 0):.2f}GB
            마지막 업데이트: {self.collection_status.get('last_update', 'Unknown')}
            """

            return fig, text

        @app.callback(
            [Output('ml-metrics-graph', 'figure'),
             Output('ml-metrics-text', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_ml_metrics(n):
            # ML 모델 성능 그래프
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('MSE', 'MAE', 'R² Score', 'Training Time'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=self.ml_metrics.get('mse', 0),
                    title={'text': "MSE"}
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=self.ml_metrics.get('mae', 0),
                    title={'text': "MAE"}
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=self.ml_metrics.get('r2_score', 0),
                    title={'text': "R² Score"}
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=self.ml_metrics.get('training_time', 0),
                    title={'text': "Training Time (s)"}
                ),
                row=2, col=2
            )

            fig.update_layout(height=500, showlegend=False)

            # 텍스트 정보
            text = f"""
            모델명: {self.ml_metrics.get('model_name', 'Unknown')}
            MSE: {self.ml_metrics.get('mse', 0):.4f}
            MAE: {self.ml_metrics.get('mae', 0):.4f}
            R² Score: {self.ml_metrics.get('r2_score', 0):.4f}
            학습 시간: {self.ml_metrics.get('training_time', 0):.2f}초
            """

            return fig, text

        return app

    async def start_monitoring(self) -> None:
        """모니터링 시작"""
        logger.info("실시간 모니터링 시스템 시작")

        # 백그라운드 태스크들 시작
        tasks = [
            self.collect_system_metrics(),
            self.collect_data_status(),
            self.collect_ml_metrics()
        ]

        # Dash 대시보드 시작
        app = self.create_dashboard()

        # 별도 스레드에서 Dash 서버 실행
        def run_dash():
            app.run(debug=False, host='0.0.0.0', port=8050)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.daemon = True
        dash_thread.start()

        logger.info("대시보드 시작: http://localhost:8050")

        # 백그라운드 태스크들 실행
        await asyncio.gather(*tasks)

async def main():
    """메인 함수"""
    monitor = RealtimeMonitor()
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
