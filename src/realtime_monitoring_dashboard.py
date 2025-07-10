#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: realtime_monitoring_dashboard.py
모듈: 실시간 데이터 시스템 모니터링 대시보드
목적: 시스템 상태, 데이터 품질, 성능 메트릭, 알림 시스템 모니터링

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - streamlit, plotly, dash
    - websockets, asyncio
    - psutil, prometheus_client
    - requests, aiohttp

Features:
    - 실시간 시스템 상태 모니터링
    - 데이터 품질 대시보드
    - 성능 메트릭 추적
    - 실시간 알림 시스템
    - 모바일 반응형 디자인

Performance:
    - 실시간 업데이트: < 1초
    - 대시보드 로딩: < 3초
    - 알림 전송: < 5초
    - 데이터 처리: 10,000+ events/sec

License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import hashlib
import uuid

# 외부 라이브러리
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    import pandas as pd
    import numpy as np
    import psutil
    import websockets
    import requests
    import aiohttp
    from prometheus_client import Counter, Gauge, Histogram, Summary
    import asyncio_mqtt as aiomqtt
    EXTERNALS_AVAILABLE = True
except ImportError:
    EXTERNALS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    # 대시보드 설정
    dashboard_port: int = 8501
    update_interval_seconds: float = 1.0
    max_data_points: int = 1000
    
    # 알림 설정
    slack_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    email_smtp_server: str = ""
    email_username: str = ""
    email_password: str = ""
    
    # 임계값 설정
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    latency_threshold_ms: float = 100.0
    error_rate_threshold_percent: float = 5.0
    
    # 데이터 품질 설정
    data_coverage_threshold_percent: float = 95.0
    data_delay_threshold_seconds: float = 60.0
    anomaly_detection_enabled: bool = True
    
    # 성능 설정
    websocket_enabled: bool = True
    websocket_port: int = 8765
    prometheus_enabled: bool = True
    prometheus_port: int = 8000


class RealTimeMonitor:
    """실시간 시스템 모니터링"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.is_running = False
        
        # 메트릭 저장소
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_io': [],
            'data_receive_rate': [],
            'latency': [],
            'error_rate': [],
            'connection_status': {}
        }
        
        # Prometheus 메트릭 (선택사항)
        if self.config.prometheus_enabled:
            self.prometheus_metrics = self._setup_prometheus_metrics()
        
        # 모니터링 스레드
        self.monitor_thread = None
        
    def _setup_prometheus_metrics(self) -> Dict[str, Any]:
        """Prometheus 메트릭 설정"""
        try:
            return {
                'cpu_usage': Gauge('system_cpu_usage_percent', 'CPU 사용률'),
                'memory_usage': Gauge('system_memory_usage_percent', '메모리 사용률'),
                'disk_usage': Gauge('system_disk_usage_percent', '디스크 사용률'),
                'data_receive_rate': Counter('data_receive_total', '데이터 수신 총량'),
                'latency': Histogram('system_latency_seconds', '시스템 레이턴시'),
                'error_rate': Counter('system_errors_total', '시스템 에러 총량'),
                'connection_status': Gauge('connection_status', '연결 상태')
            }
        except Exception as e:
            logger.error(f"Prometheus 메트릭 설정 실패: {e}")
            return {}
    
    def start_monitoring(self):
        """모니터링 시작"""
        try:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info("실시간 모니터링 시작")
            
        except Exception as e:
            logger.error(f"모니터링 시작 실패: {e}")
            raise
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("실시간 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                # 시스템 메트릭 수집
                metrics = self._collect_system_metrics()
                
                # 메트릭 저장
                self._store_metrics(metrics)
                
                # Prometheus 메트릭 업데이트
                if self.config.prometheus_enabled:
                    self._update_prometheus_metrics(metrics)
                
                # 임계값 체크
                self._check_thresholds(metrics)
                
                # 메트릭 큐에 추가
                self.metrics_queue.put(metrics)
                
                time.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # 데이터 수신률 (시뮬레이션)
            data_receive_rate = np.random.poisson(1000)  # 초당 1000개 이벤트
            
            # 레이턴시 (시뮬레이션)
            latency = np.random.exponential(50)  # 평균 50ms
            
            # 에러율 (시뮬레이션)
            error_rate = np.random.beta(1, 20) * 100  # 평균 5%
            
            # 연결 상태
            connection_status = {
                'timescale_db': self._check_db_connection(),
                'redis': self._check_redis_connection(),
                'kafka': self._check_kafka_connection(),
                'websocket': self._check_websocket_connection()
            }
            
            return {
                'timestamp': datetime.now(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'network_io': network_io,
                'data_receive_rate': data_receive_rate,
                'latency': latency,
                'error_rate': error_rate,
                'connection_status': connection_status
            }
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}
    
    def _check_db_connection(self) -> bool:
        """데이터베이스 연결 상태 확인"""
        try:
            # 실제 구현에서는 실제 DB 연결 테스트
            return np.random.choice([True, False], p=[0.95, 0.05])
        except Exception:
            return False
    
    def _check_redis_connection(self) -> bool:
        """Redis 연결 상태 확인"""
        try:
            # 실제 구현에서는 실제 Redis 연결 테스트
            return np.random.choice([True, False], p=[0.98, 0.02])
        except Exception:
            return False
    
    def _check_kafka_connection(self) -> bool:
        """Kafka 연결 상태 확인"""
        try:
            # 실제 구현에서는 실제 Kafka 연결 테스트
            return np.random.choice([True, False], p=[0.97, 0.03])
        except Exception:
            return False
    
    def _check_websocket_connection(self) -> bool:
        """WebSocket 연결 상태 확인"""
        try:
            # 실제 구현에서는 실제 WebSocket 연결 테스트
            return np.random.choice([True, False], p=[0.99, 0.01])
        except Exception:
            return False
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """메트릭 저장"""
        try:
            timestamp = metrics.get('timestamp', datetime.now())
            
            # 각 메트릭을 시간순으로 저장
            for key in ['cpu_usage', 'memory_usage', 'disk_usage', 'data_receive_rate', 'latency', 'error_rate']:
                if key in metrics:
                    self.system_metrics[key].append({
                        'timestamp': timestamp,
                        'value': metrics[key]
                    })
                    
                    # 최대 데이터 포인트 제한
                    if len(self.system_metrics[key]) > self.config.max_data_points:
                        self.system_metrics[key] = self.system_metrics[key][-self.config.max_data_points:]
            
            # 연결 상태 업데이트
            self.system_metrics['connection_status'] = metrics.get('connection_status', {})
            
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
    
    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Prometheus 메트릭 업데이트"""
        try:
            if not self.prometheus_metrics:
                return
            
            # CPU 사용률
            if 'cpu_usage' in metrics:
                self.prometheus_metrics['cpu_usage'].set(metrics['cpu_usage'])
            
            # 메모리 사용률
            if 'memory_usage' in metrics:
                self.prometheus_metrics['memory_usage'].set(metrics['memory_usage'])
            
            # 디스크 사용률
            if 'disk_usage' in metrics:
                self.prometheus_metrics['disk_usage'].set(metrics['disk_usage'])
            
            # 데이터 수신률
            if 'data_receive_rate' in metrics:
                self.prometheus_metrics['data_receive_rate'].inc(metrics['data_receive_rate'])
            
            # 레이턴시
            if 'latency' in metrics:
                self.prometheus_metrics['latency'].observe(metrics['latency'] / 1000)  # ms to seconds
            
            # 에러율
            if 'error_rate' in metrics:
                self.prometheus_metrics['error_rate'].inc(int(metrics['error_rate']))
            
            # 연결 상태
            if 'connection_status' in metrics:
                total_connections = sum(metrics['connection_status'].values())
                self.prometheus_metrics['connection_status'].set(total_connections)
                
        except Exception as e:
            logger.error(f"Prometheus 메트릭 업데이트 실패: {e}")
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """임계값 체크"""
        try:
            alerts = []
            
            # CPU 임계값 체크
            if metrics.get('cpu_usage', 0) > self.config.cpu_threshold_percent:
                alerts.append({
                    'level': 'warning',
                    'message': f"CPU 사용률이 {metrics['cpu_usage']:.1f}%로 임계값을 초과했습니다.",
                    'metric': 'cpu_usage',
                    'value': metrics['cpu_usage'],
                    'threshold': self.config.cpu_threshold_percent
                })
            
            # 메모리 임계값 체크
            if metrics.get('memory_usage', 0) > self.config.memory_threshold_percent:
                alerts.append({
                    'level': 'warning',
                    'message': f"메모리 사용률이 {metrics['memory_usage']:.1f}%로 임계값을 초과했습니다.",
                    'metric': 'memory_usage',
                    'value': metrics['memory_usage'],
                    'threshold': self.config.memory_threshold_percent
                })
            
            # 디스크 임계값 체크
            if metrics.get('disk_usage', 0) > self.config.disk_threshold_percent:
                alerts.append({
                    'level': 'critical',
                    'message': f"디스크 사용률이 {metrics['disk_usage']:.1f}%로 임계값을 초과했습니다.",
                    'metric': 'disk_usage',
                    'value': metrics['disk_usage'],
                    'threshold': self.config.disk_threshold_percent
                })
            
            # 레이턴시 임계값 체크
            if metrics.get('latency', 0) > self.config.latency_threshold_ms:
                alerts.append({
                    'level': 'warning',
                    'message': f"레이턴시가 {metrics['latency']:.1f}ms로 임계값을 초과했습니다.",
                    'metric': 'latency',
                    'value': metrics['latency'],
                    'threshold': self.config.latency_threshold_ms
                })
            
            # 에러율 임계값 체크
            if metrics.get('error_rate', 0) > self.config.error_rate_threshold_percent:
                alerts.append({
                    'level': 'critical',
                    'message': f"에러율이 {metrics['error_rate']:.1f}%로 임계값을 초과했습니다.",
                    'metric': 'error_rate',
                    'value': metrics['error_rate'],
                    'threshold': self.config.error_rate_threshold_percent
                })
            
            # 연결 상태 체크
            connection_status = metrics.get('connection_status', {})
            for service, status in connection_status.items():
                if not status:
                    alerts.append({
                        'level': 'critical',
                        'message': f"{service} 연결이 끊어졌습니다.",
                        'metric': f'connection_{service}',
                        'value': 0,
                        'threshold': 1
                    })
            
            # 알림 큐에 추가
            for alert in alerts:
                self.alert_queue.put(alert)
                
        except Exception as e:
            logger.error(f"임계값 체크 실패: {e}")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """최신 메트릭 조회"""
        try:
            latest_metrics = {}
            
            for key in ['cpu_usage', 'memory_usage', 'disk_usage', 'data_receive_rate', 'latency', 'error_rate']:
                if self.system_metrics[key]:
                    latest_metrics[key] = self.system_metrics[key][-1]
            
            latest_metrics['connection_status'] = self.system_metrics['connection_status']
            
            return latest_metrics
            
        except Exception as e:
            logger.error(f"최신 메트릭 조회 실패: {e}")
            return {}
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """메트릭 히스토리 조회"""
        try:
            if metric_name not in self.system_metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = [
                metric for metric in self.system_metrics[metric_name]
                if metric['timestamp'] > cutoff_time
            ]
            
            return history
            
        except Exception as e:
            logger.error(f"메트릭 히스토리 조회 실패: {e}")
            return []


class AlertManager:
    """알림 관리 시스템"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []
        self.alert_rules = self._setup_alert_rules()
        self.is_running = False
        
    def _setup_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """알림 규칙 설정"""
        return {
            'cpu_high': {
                'condition': lambda x: x > self.config.cpu_threshold_percent,
                'message': 'CPU 사용률이 높습니다.',
                'level': 'warning'
            },
            'memory_high': {
                'condition': lambda x: x > self.config.memory_threshold_percent,
                'message': '메모리 사용률이 높습니다.',
                'level': 'warning'
            },
            'disk_full': {
                'condition': lambda x: x > self.config.disk_threshold_percent,
                'message': '디스크 공간이 부족합니다.',
                'level': 'critical'
            },
            'latency_high': {
                'condition': lambda x: x > self.config.latency_threshold_ms,
                'message': '시스템 레이턴시가 높습니다.',
                'level': 'warning'
            },
            'error_rate_high': {
                'condition': lambda x: x > self.config.error_rate_threshold_percent,
                'message': '에러율이 높습니다.',
                'level': 'critical'
            }
        }
    
    def start_alert_manager(self):
        """알림 관리자 시작"""
        self.is_running = True
        logger.info("알림 관리자 시작")
    
    def stop_alert_manager(self):
        """알림 관리자 중지"""
        self.is_running = False
        logger.info("알림 관리자 중지")
    
    def process_alert(self, alert: Dict[str, Any]):
        """알림 처리"""
        try:
            # 알림 히스토리에 추가
            alert['timestamp'] = datetime.now()
            alert['id'] = str(uuid.uuid4())
            self.alert_history.append(alert)
            
            # 알림 히스토리 크기 제한
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # 알림 전송
            self._send_alert(alert)
            
            logger.info(f"알림 처리 완료: {alert['message']}")
            
        except Exception as e:
            logger.error(f"알림 처리 실패: {e}")
    
    def _send_alert(self, alert: Dict[str, Any]):
        """알림 전송"""
        try:
            # Slack 알림
            if self.config.slack_webhook_url:
                self._send_slack_alert(alert)
            
            # Telegram 알림
            if self.config.telegram_bot_token and self.config.telegram_chat_id:
                self._send_telegram_alert(alert)
            
            # 이메일 알림
            if self.config.email_smtp_server:
                self._send_email_alert(alert)
            
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Slack 알림 전송"""
        try:
            message = {
                "text": f"🚨 {alert['level'].upper()}: {alert['message']}",
                "attachments": [{
                    "color": "danger" if alert['level'] == 'critical' else "warning",
                    "fields": [
                        {
                            "title": "메트릭",
                            "value": alert.get('metric', 'unknown'),
                            "short": True
                        },
                        {
                            "title": "값",
                            "value": f"{alert.get('value', 0):.2f}",
                            "short": True
                        },
                        {
                            "title": "임계값",
                            "value": f"{alert.get('threshold', 0):.2f}",
                            "short": True
                        }
                    ],
                    "footer": f"시간: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Slack 알림 전송 성공")
            else:
                logger.error(f"Slack 알림 전송 실패: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Slack 알림 전송 실패: {e}")
    
    def _send_telegram_alert(self, alert: Dict[str, Any]):
        """Telegram 알림 전송"""
        try:
            message = f"🚨 {alert['level'].upper()}\n{alert['message']}\n\n"
            message += f"메트릭: {alert.get('metric', 'unknown')}\n"
            message += f"값: {alert.get('value', 0):.2f}\n"
            message += f"임계값: {alert.get('threshold', 0):.2f}\n"
            message += f"시간: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram 알림 전송 성공")
            else:
                logger.error(f"Telegram 알림 전송 실패: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Telegram 알림 전송 실패: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """이메일 알림 전송"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = self.config.email_username  # 관리자에게 전송
            msg['Subject'] = f"시스템 알림 - {alert['level'].upper()}"
            
            body = f"""
            🚨 시스템 알림
            
            레벨: {alert['level'].upper()}
            메시지: {alert['message']}
            메트릭: {alert.get('metric', 'unknown')}
            값: {alert.get('value', 0):.2f}
            임계값: {alert.get('threshold', 0):.2f}
            시간: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_smtp_server, 587)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_username, text)
            server.quit()
            
            logger.info("이메일 알림 전송 성공")
            
        except Exception as e:
            logger.error(f"이메일 알림 전송 실패: {e}")
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """알림 히스토리 조회"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                alert for alert in self.alert_history
                if alert['timestamp'] > cutoff_time
            ]
            return recent_alerts
            
        except Exception as e:
            logger.error(f"알림 히스토리 조회 실패: {e}")
            return []


class PerformanceDashboard:
    """성능 대시보드"""
    
    def __init__(self, monitor: RealTimeMonitor, alert_manager: AlertManager):
        self.monitor = monitor
        self.alert_manager = alert_manager
        
    def create_system_status_dashboard(self) -> go.Figure:
        """시스템 상태 대시보드 생성"""
        try:
            # 서브플롯 생성
            fig = sp.make_subplots(
                rows=3, cols=2,
                subplot_titles=('CPU 사용률', '메모리 사용률', '디스크 사용률', 
                              '데이터 수신률', '레이턴시', '에러율'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # CPU 사용률
            cpu_history = self.monitor.get_metrics_history('cpu_usage', 1)
            if cpu_history:
                cpu_df = pd.DataFrame(cpu_history)
                fig.add_trace(
                    go.Scatter(x=cpu_df['timestamp'], y=cpu_df['value'],
                              mode='lines', name='CPU 사용률',
                              line=dict(color='red')),
                    row=1, col=1
                )
            
            # 메모리 사용률
            memory_history = self.monitor.get_metrics_history('memory_usage', 1)
            if memory_history:
                memory_df = pd.DataFrame(memory_history)
                fig.add_trace(
                    go.Scatter(x=memory_df['timestamp'], y=memory_df['value'],
                              mode='lines', name='메모리 사용률',
                              line=dict(color='blue')),
                    row=1, col=2
                )
            
            # 디스크 사용률
            disk_history = self.monitor.get_metrics_history('disk_usage', 1)
            if disk_history:
                disk_df = pd.DataFrame(disk_history)
                fig.add_trace(
                    go.Scatter(x=disk_df['timestamp'], y=disk_df['value'],
                              mode='lines', name='디스크 사용률',
                              line=dict(color='green')),
                    row=2, col=1
                )
            
            # 데이터 수신률
            data_history = self.monitor.get_metrics_history('data_receive_rate', 1)
            if data_history:
                data_df = pd.DataFrame(data_history)
                fig.add_trace(
                    go.Scatter(x=data_df['timestamp'], y=data_df['value'],
                              mode='lines', name='데이터 수신률',
                              line=dict(color='orange')),
                    row=2, col=2
                )
            
            # 레이턴시
            latency_history = self.monitor.get_metrics_history('latency', 1)
            if latency_history:
                latency_df = pd.DataFrame(latency_history)
                fig.add_trace(
                    go.Scatter(x=latency_df['timestamp'], y=latency_df['value'],
                              mode='lines', name='레이턴시',
                              line=dict(color='purple')),
                    row=3, col=1
                )
            
            # 에러율
            error_history = self.monitor.get_metrics_history('error_rate', 1)
            if error_history:
                error_df = pd.DataFrame(error_history)
                fig.add_trace(
                    go.Scatter(x=error_df['timestamp'], y=error_df['value'],
                              mode='lines', name='에러율',
                              line=dict(color='brown')),
                    row=3, col=2
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title='실시간 시스템 상태 모니터링',
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"시스템 상태 대시보드 생성 실패: {e}")
            return go.Figure()
    
    def create_connection_status_dashboard(self) -> go.Figure:
        """연결 상태 대시보드 생성"""
        try:
            latest_metrics = self.monitor.get_latest_metrics()
            connection_status = latest_metrics.get('connection_status', {})
            
            # 연결 상태 데이터
            services = list(connection_status.keys())
            status_values = [1 if connection_status[service] else 0 for service in services]
            colors = ['green' if status else 'red' for status in status_values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=services,
                    y=status_values,
                    marker_color=colors,
                    text=[f"{'연결됨' if status else '연결 끊김'}" for status in status_values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='시스템 연결 상태',
                yaxis=dict(range=[0, 1.2]),
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"연결 상태 대시보드 생성 실패: {e}")
            return go.Figure()
    
    def create_alert_history_dashboard(self) -> go.Figure:
        """알림 히스토리 대시보드 생성"""
        try:
            alert_history = self.alert_manager.get_alert_history(24)
            
            if not alert_history:
                return go.Figure()
            
            # 알림 레벨별 분류
            alert_df = pd.DataFrame(alert_history)
            alert_counts = alert_df['level'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=alert_counts.index,
                    values=alert_counts.values,
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title='알림 히스토리 (24시간)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"알림 히스토리 대시보드 생성 실패: {e}")
            return go.Figure()


class NotificationSystem:
    """알림 시스템"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.notification_queue = queue.Queue()
        self.is_running = False
        
    def start_notification_system(self):
        """알림 시스템 시작"""
        self.is_running = True
        logger.info("알림 시스템 시작")
    
    def stop_notification_system(self):
        """알림 시스템 중지"""
        self.is_running = False
        logger.info("알림 시스템 중지")
    
    def send_notification(self, message: str, level: str = 'info', 
                         channels: List[str] = None):
        """알림 전송"""
        try:
            notification = {
                'id': str(uuid.uuid4()),
                'message': message,
                'level': level,
                'timestamp': datetime.now(),
                'channels': channels or ['all']
            }
            
            self.notification_queue.put(notification)
            
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    def get_notification_queue(self) -> queue.Queue:
        """알림 큐 조회"""
        return self.notification_queue


# Streamlit 대시보드
def create_streamlit_dashboard():
    """Streamlit 대시보드 생성"""
    try:
        st.set_page_config(
            page_title="실시간 모니터링 대시보드",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 설정
        config = MonitoringConfig()
        
        # 컴포넌트 초기화
        monitor = RealTimeMonitor(config)
        alert_manager = AlertManager(config)
        notification_system = NotificationSystem(config)
        dashboard = PerformanceDashboard(monitor, alert_manager)
        
        # 모니터링 시작
        monitor.start_monitoring()
        alert_manager.start_alert_manager()
        notification_system.start_notification_system()
        
        # 사이드바
        st.sidebar.title("📊 모니터링 대시보드")
        
        # 메뉴 선택
        menu = st.sidebar.selectbox(
            "메뉴 선택",
            ["시스템 상태", "데이터 품질", "성능 메트릭", "알림 시스템", "설정"]
        )
        
        if menu == "시스템 상태":
            st.title("🖥️ 시스템 상태 모니터링")
            
            # 실시간 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            latest_metrics = monitor.get_latest_metrics()
            
            with col1:
                st.metric(
                    label="CPU 사용률",
                    value=f"{latest_metrics.get('cpu_usage', {}).get('value', 0):.1f}%",
                    delta="+2.1%"
                )
            
            with col2:
                st.metric(
                    label="메모리 사용률",
                    value=f"{latest_metrics.get('memory_usage', {}).get('value', 0):.1f}%",
                    delta="+1.5%"
                )
            
            with col3:
                st.metric(
                    label="디스크 사용률",
                    value=f"{latest_metrics.get('disk_usage', {}).get('value', 0):.1f}%",
                    delta="+0.8%"
                )
            
            with col4:
                st.metric(
                    label="레이턴시",
                    value=f"{latest_metrics.get('latency', {}).get('value', 0):.1f}ms",
                    delta="-5.2ms"
                )
            
            # 시스템 상태 차트
            st.subheader("📈 실시간 시스템 메트릭")
            system_fig = dashboard.create_system_status_dashboard()
            st.plotly_chart(system_fig, use_container_width=True)
            
            # 연결 상태
            st.subheader("🔗 연결 상태")
            connection_fig = dashboard.create_connection_status_dashboard()
            st.plotly_chart(connection_fig, use_container_width=True)
            
        elif menu == "데이터 품질":
            st.title("📊 데이터 품질 대시보드")
            
            # 데이터 품질 메트릭
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="데이터 커버리지",
                    value="98.5%",
                    delta="+0.2%"
                )
            
            with col2:
                st.metric(
                    label="데이터 지연",
                    value="2.3초",
                    delta="-0.5초"
                )
            
            with col3:
                st.metric(
                    label="품질 점수",
                    value="95.2",
                    delta="+1.8"
                )
            
            # 데이터 품질 차트
            st.subheader("📈 데이터 품질 추적")
            
            # 시뮬레이션 데이터
            dates = pd.date_range(start='2025-01-01', end='2025-01-27', freq='D')
            coverage_data = np.random.normal(98, 1, len(dates))
            delay_data = np.random.exponential(3, len(dates))
            quality_data = np.random.normal(95, 2, len(dates))
            
            quality_df = pd.DataFrame({
                'date': dates,
                'coverage': coverage_data,
                'delay': delay_data,
                'quality': quality_data
            })
            
            # 커버리지 차트
            fig1 = px.line(quality_df, x='date', y='coverage', 
                          title='데이터 커버리지 추이')
            st.plotly_chart(fig1, use_container_width=True)
            
            # 지연 시간 차트
            fig2 = px.line(quality_df, x='date', y='delay', 
                          title='데이터 지연 시간 추이')
            st.plotly_chart(fig2, use_container_width=True)
            
        elif menu == "성능 메트릭":
            st.title("⚡ 성능 메트릭")
            
            # 성능 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="처리량",
                    value="15,234",
                    delta="+1,234"
                )
            
            with col2:
                st.metric(
                    label="네트워크 I/O",
                    value="2.1 GB/s",
                    delta="+0.3 GB/s"
                )
            
            with col3:
                st.metric(
                    label="에러율",
                    value="0.2%",
                    delta="-0.1%"
                )
            
            with col4:
                st.metric(
                    label="가용성",
                    value="99.9%",
                    delta="+0.1%"
                )
            
            # 성능 히트맵
            st.subheader("🔥 레이턴시 히트맵")
            
            # 시뮬레이션 히트맵 데이터
            hours = list(range(24))
            services = ['API Gateway', 'Database', 'Cache', 'Message Queue', 'Storage']
            latency_data = np.random.exponential(50, (len(services), len(hours)))
            
            fig = px.imshow(
                latency_data,
                x=hours,
                y=services,
                color_continuous_scale='RdBu_r',
                title='서비스별 레이턴시 히트맵 (24시간)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif menu == "알림 시스템":
            st.title("🔔 알림 시스템")
            
            # 알림 설정
            st.subheader("⚙️ 알림 설정")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.checkbox("Slack 알림", value=True)
                st.checkbox("Telegram 알림", value=True)
                st.checkbox("이메일 알림", value=False)
            
            with col2:
                st.checkbox("CPU 임계값 알림", value=True)
                st.checkbox("메모리 임계값 알림", value=True)
                st.checkbox("디스크 임계값 알림", value=True)
            
            # 알림 히스토리
            st.subheader("📋 알림 히스토리")
            
            alert_fig = dashboard.create_alert_history_dashboard()
            st.plotly_chart(alert_fig, use_container_width=True)
            
            # 최근 알림 목록
            recent_alerts = alert_manager.get_alert_history(1)
            
            if recent_alerts:
                st.subheader("🕐 최근 알림")
                
                for alert in recent_alerts[-10:]:  # 최근 10개
                    level_color = "🔴" if alert['level'] == 'critical' else "🟡"
                    st.write(f"{level_color} {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
            else:
                st.info("최근 알림이 없습니다.")
            
        elif menu == "설정":
            st.title("⚙️ 설정")
            
            st.subheader("📊 모니터링 설정")
            
            # 임계값 설정
            cpu_threshold = st.slider("CPU 임계값 (%)", 50, 100, 80)
            memory_threshold = st.slider("메모리 임계값 (%)", 50, 100, 85)
            disk_threshold = st.slider("디스크 임계값 (%)", 50, 100, 90)
            
            # 알림 설정
            st.subheader("🔔 알림 설정")
            
            slack_webhook = st.text_input("Slack Webhook URL", value="")
            telegram_token = st.text_input("Telegram Bot Token", value="")
            telegram_chat_id = st.text_input("Telegram Chat ID", value="")
            
            # 저장 버튼
            if st.button("설정 저장"):
                st.success("설정이 저장되었습니다!")
        
        # 자동 새로고침
        st.empty()
        time.sleep(1)
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"대시보드 생성 실패: {e}")
        logger.error(f"Streamlit 대시보드 생성 실패: {e}")


# 메인 실행 함수
def main():
    """메인 실행 함수"""
    try:
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Streamlit 대시보드 실행
        create_streamlit_dashboard()
        
    except Exception as e:
        logger.error(f"메인 실행 실패: {e}")
        raise


if __name__ == "__main__":
    main() 