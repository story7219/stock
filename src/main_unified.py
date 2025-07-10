from kis_config import KISConfig, DashboardBroadcaster
from kis_trader import KISTrader
import asyncio

async def main():
    kis_config = KISConfig()  # KIS_MODE=live/mock 자동 인식
    dashboard = DashboardBroadcaster()
    trader = KISTrader(kis_config, dashboard)

    print(kis_config.info())

    # 예시: 실시간 데이터 조회
    ohlcv = await trader.get_ohlcv("005930", interval="1m", count=100)
    print(f"시세 데이터(샘플): {ohlcv[:2]}")

    # 예시: 주문 (실전이면 안전장치 발동)
    # result = await trader.send_order("005930", qty=1, price=80000, side="BUY")
    # print(result)

    # 실시간 대시보드에 신호 전송 예시
    await dashboard.send_signal("BUY", 0.95, kis_config.mode)

if __name__ == "__main__":
    asyncio.run(main())

