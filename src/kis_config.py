import sys
from typing import Literal
import asyncio
import json
import os
import websockets

class KISConfig:
    def __init__(self, mode: Literal["live", "mock"] = None):
        self.mode = mode or os.getenv("KIS_MODE", "live")
        if self.mode == "live":
            self.app_key = os.getenv("LIVE_KIS_APP_KEY")
            self.app_secret = os.getenv("LIVE_KIS_APP_SECRET")
            self.account_no = os.getenv("LIVE_KIS_ACCOUNT_NUMBER")
            # 모의계좌 URL 사용
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.app_key = os.getenv("MOCK_KIS_APP_KEY")
            self.app_secret = os.getenv("MOCK_KIS_APP_SECRET")
            self.account_no = os.getenv("MOCK_KIS_ACCOUNT_NUMBER")
            self.base_url = "https://openapivts.koreainvestment.com:29443"

    def is_live(self) -> bool:
        return self.mode == "live"

    def info(self) -> str:
        return f"[KIS MODE: {self.mode.upper()}] 계좌: {self.account_no}"

# 실전 주문 안전장치
class OrderSafety:
    @staticmethod
    def confirm_live_order() -> bool:
        print("\n[경고] 실전 계좌로 주문을 실행하려고 합니다!\n")
        answer = input("정말로 실전 주문을 진행하시겠습니까? (yes/NO): ").strip().lower()
        if answer != "yes":
            print("주문이 취소되었습니다.")
            sys.exit(0)
        return True

# 실시간 대시보드 연동용 WebSocket 송신기 (FastAPI/Plotly 연동 예시)

class DashboardBroadcaster:
    def __init__(self, ws_url: str = "ws://localhost:8000/ws"):
        self.ws_url = ws_url

    async def send(self, data: dict):
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps(data))

    async def send_signal(self, signal: str, confidence: float, mode: str):
        await self.send({
            "signal": signal,
            "confidence": confidence,
            "mode": mode
        })

