from __future__ import annotations
from agile_trading_strategy import AgileTradingStrategy, AgileSignal, SmallCapOpportunity, ThemeOpportunity
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional, Any, Set
import asyncio
import json
import logging
import uvicorn
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: agile_dashboard.py
모듈: 소액 투자 민첩성 대시보드
목적: 소액 투자 민첩성 전략 전용 실시간 대시보드

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - fastapi, uvicorn, websockets
    - agile_trading_strategy
"""





logger = logging.getLogger(__name__)

class ConnectionManager:
    """WebSocket 연결 관리"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.mobile_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, is_mobile: bool = False):
        await websocket.accept()
        if is_mobile:
            self.mobile_connections.append(websocket)
        else:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)} desktop, {len(self.mobile_connections)} mobile")

    def disconnect(self, websocket: WebSocket, is_mobile: bool = False):
        if is_mobile:
            self.mobile_connections.remove(websocket)
        else:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)} desktop, {len(self.mobile_connections)} mobile")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, is_mobile: bool = False):
        connections = self.mobile_connections if is_mobile else self.active_connections
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                if connection in connections:
                    connections.remove(connection)


@dataclass
class AgileDashboardState:
    total_opportunities: int
    active_signals: int
    instant_entries: int
    fast_exits: int
    small_cap_opportunities: int
    theme_opportunities: int
    news_opportunities: int
    total_pnl: float
    win_rate: float
    avg_hold_time: float
    last_update: datetime = datetime.now()


class AgileDashboard:
    # ... (rest of the code)
