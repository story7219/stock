from .kis_config import KISConfig, OrderSafety, DashboardBroadcaster
from typing import Dict, Any
import aiohttp

class KISTrader:
    def __init__(self, config: KISConfig, dashboard: DashboardBroadcaster = None):
        self.config = config
        self.dashboard = dashboard

    async def get_ohlcv(self, symbol: str, interval: str = "1m", count: int = 1000) -> list[dict]:
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        headers = {"appkey": self.config.app_key, "appsecret": self.config.app_secret}
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
            "fid_input_time_unit": interval,
            "fid_input_date_1": "20220101",
            "fid_input_date_2": "20240101",
            "fid_org_adj_prc": "0"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()
                return data.get("output2", [])

    async def send_order(self, symbol: str, qty: int, price: float, side: str) -> Dict[str, Any]:
        if self.config.is_live():
            OrderSafety.confirm_live_order()
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        headers = {"appkey": self.config.app_key, "appsecret": self.config.app_secret}
        payload = {
            "CANO": self.config.account_no.split('-')[0],
            "ACNT_PRDT_CD": self.config.account_no.split('-')[1],
            "PDNO": symbol,
            "ORD_DVSN": "01" if side == "BUY" else "02",
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price)
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                result = await resp.json()
                # 실시간 대시보드 연동
                if self.dashboard:
                    await self.dashboard.send_signal(side, 1.0, self.config.mode)
                return result

