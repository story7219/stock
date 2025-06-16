"""
🚀 월요일 자동 실행
"""
import asyncio
import schedule
import time
from datetime import datetime
from trade import full_auto_trading

def is_trading_day():
    """거래일 확인"""
    today = datetime.now().weekday()
    return 0 <= today <= 4  # 월~금

async def monday_morning():
    """월요일 아침 자동 실행"""
    if is_trading_day():
        print("🌅 월요일 아침 자동매매 시작!")
        result = await full_auto_trading()
        print(f"📊 결과: {result}")
    else:
        print("📅 오늘은 휴장일입니다")

# 월요일 오전 9시 자동 실행 예약
schedule.every().monday.at("09:00").do(lambda: asyncio.run(monday_morning()))

if __name__ == "__main__":
    print("⏰ 월요일 자동 실행 대기 중...")
    while True:
        schedule.run_pending()
        time.sleep(60) 