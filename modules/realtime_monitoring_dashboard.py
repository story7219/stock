#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: realtime_monitoring_dashboard.py
모듈: 실시간 모니터링 대시보드
목적: 실시간 시스템 상태 모니터링

Author: Monitoring Engineer
Created: 2025-07-10
Version: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import psutil
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger(__name__)

class SystemMonitor:
    """시스템 모니터링"""

    def __init__(self):
        self.logger = logger
        self.metrics_history = []
        self.max_history = 1000

    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)

            # 메모리 사용률
            memory = psutil.virtual_memory()

            # 디스크 사용률
            disk = psutil.disk_usage('/')

            # 네트워크 통계
            network = psutil.net_io_counters()

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }

            # 히스토리 저장
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

            return metrics

        except Exception as e:
            self.logger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}

    def get_metrics_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """메트릭 히스토리 조회"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time]:
:
class TradingMonitor:
    """거래 모니터링"""

    def __init__(self):
        self.logger = logger
        self.trading_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'current_balance': 10000000.0,  # 1천만원 시작
            'signals_generated': 0,
            'last_trade_time': None
        }

    def update_trade_metrics(self, trade_result: Dict[str, Any]):
        """거래 메트릭 업데이트"""
        self.trading_metrics['total_trades'] += 1

        if trade_result.get('success', False):
            self.trading_metrics['successful_trades'] += 1
            self.trading_metrics['total_profit'] += trade_result.get('profit', 0.0)
        else:
            self.trading_metrics['failed_trades'] += 1

        self.trading_metrics['current_balance'] = trade_result.get('new_balance', self.trading_metrics['current_balance'])
        self.trading_metrics['last_trade_time'] = datetime.now().isoformat()

    def update_signal_metrics(self, signal_count: int = 1):
        """신호 메트릭 업데이트"""
        self.trading_metrics['signals_generated'] += signal_count

    def get_trading_metrics(self) -> Dict[str, Any]:
        """거래 메트릭 조회"""
        metrics = self.trading_metrics.copy()

        # 성공률 계산
        if metrics['total_trades'] > 0:
            metrics['success_rate'] = (metrics['successful_trades'] / metrics['total_trades']) * 100
        else:
            metrics['success_rate'] = 0.0

        # 수익률 계산
        initial_balance = 10000000.0
        metrics['profit_rate'] = ((metrics['current_balance'] - initial_balance) / initial_balance) * 100

        return metrics

class WebSocketManager:
    """WebSocket 연결 관리"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logger

    async def connect(self, websocket: WebSocket):
        """WebSocket 연결"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket 연결됨. 총 연결: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """WebSocket 연결 해제"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket 연결 해제됨. 총 연결: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """모든 연결에 메시지 브로드캐스트"""
        if self.active_connections:
            message_json = json.dumps(message)
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    self.logger.error(f"WebSocket 메시지 전송 실패: {e}")
                    self.disconnect(connection)

class RealtimeMonitoringDashboard:
    """실시간 모니터링 대시보드"""

    def __init__(self):
        self.app = FastAPI(title="Trading System Monitor", version="1.0.0")
        self.system_monitor = SystemMonitor()
        self.trading_monitor = TradingMonitor()
        self.websocket_manager = WebSocketManager()
        self.logger = logger

        self.setup_routes()

    def setup_routes(self):
        """라우트 설정"""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """대시보드 HTML 페이지"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading System Monitor</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .metric-card {
                        border: 1px solid #ddd;
                        padding: 15px;
                        margin: 10px;
                        border-radius: 5px;
                        display: inline-block;
                        width: 200px;
                    }
                    .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
                    .metric-label { color: #666; margin-bottom: 5px; }
                    .status-good { color: #28a745; }
                    .status-warning { color: #ffc107; }
                    .status-danger { color: #dc3545; }
                </style>
            </head>
            <body>
                <h1>🚀 Trading System Monitor</h1>

                <h2>💻 System Metrics</h2>
                <div id="system-metrics">
                    <div class="metric-card">
                        <div class="metric-label">CPU Usage</div>
                        <div class="metric-value" id="cpu-usage">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value" id="memory-usage">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Disk Usage</div>
                        <div class="metric-value" id="disk-usage">-</div>
                    </div>
                </div>

                <h2>📈 Trading Metrics</h2>
                <div id="trading-metrics">
                    <div class="metric-card">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value" id="total-trades">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Success Rate</div>
                        <div class="metric-value" id="success-rate">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Profit Rate</div>
                        <div class="metric-value" id="profit-rate">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Current Balance</div>
                        <div class="metric-value" id="current-balance">-</div>
                    </div>
                </div>

                <script>
                    const ws = new WebSocket('ws://localhost:8000/ws');

                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);

                        // System metrics
                        if (data.system) {:
                            document.getElementById('cpu-usage').textContent = data.system.cpu.percent + '%';:
                            document.getElementById('memory-usage').textContent = data.system.memory.percent + '%';:
                            document.getElementById('disk-usage').textContent = data.system.disk.percent.toFixed(1) + '%';:
                        }:
:
                        // Trading metrics:
                        if (data.trading) {:
                            document.getElementById('total-trades').textContent = data.trading.total_trades;:
                            document.getElementById('success-rate').textContent = data.trading.success_rate.toFixed(1) + '%';:
                            document.getElementById('profit-rate').textContent = data.trading.profit_rate.toFixed(2) + '%';:
                            document.getElementById('current-balance').textContent = '₩' + data.trading.current_balance.toLocaleString();:
                        }:
                    };:
:
                    ws.onerror = function(error) {:
                        console.error('WebSocket error:', error);
                    };

                    ws.onclose = function() {
                        console.log('WebSocket connection closed');
                    };
                </script>
            </body>
            </html>
            """

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 엔드포인트"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # 실시간 메트릭 전송
                    system_metrics = self.system_monitor.get_system_metrics()
                    trading_metrics = self.trading_monitor.get_trading_metrics()

                    message = {
                        'timestamp': datetime.now().isoformat(),
                        'system': system_metrics,
                        'trading': trading_metrics
                    }

                    await self.websocket_manager.broadcast(message)
                    await asyncio.sleep(5)  # 5초마다 업데이트

            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)

        @self.app.get("/api/metrics")
        async def get_metrics():
            """API 메트릭 조회"""
            return {
                'system': self.system_monitor.get_system_metrics(),
                'trading': self.trading_monitor.get_trading_metrics()
            }

        @self.app.get("/api/metrics/history")
        async def get_metrics_history(minutes: int = 60):
            """메트릭 히스토리 조회"""
            return {
                'system_history': self.system_monitor.get_metrics_history(minutes),
                'trading_metrics': self.trading_monitor.get_trading_metrics()
            }

    def update_trading_metrics(self, trade_result: Dict[str, Any]):
        """거래 메트릭 업데이트"""
        self.trading_monitor.update_trade_metrics(trade_result)

    def update_signal_metrics(self, signal_count: int = 1):
        """신호 메트릭 업데이트"""
        self.trading_monitor.update_signal_metrics(signal_count)

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """대시보드 시작"""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        self.logger.info(f"모니터링 대시보드 시작: http://{host}:{port}")
        await server.serve()

async def main():
    """메인 함수"""
    dashboard = RealtimeMonitoringDashboard()
    await dashboard.start()

if __name__ == "__main__":
    asyncio.run(main())
