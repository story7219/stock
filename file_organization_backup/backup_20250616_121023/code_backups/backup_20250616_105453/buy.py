"""
🛒 매수 테스트
"""

import asyncio
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class Trader:
    def __init__(self):
        self.url = "https://openapivts.koreainvestment.com:29443"
        self.key = os.getenv('MOCK_KIS_APP_KEY')
        self.secret = os.getenv('MOCK_KIS_APP_SECRET')
        self.account = os.getenv('MOCK_KIS_ACCOUNT_NUMBER')
        self.token = None
    
    async def get_token(self):
        data = {"grant_type": "client_credentials", "appkey": self.key, "appsecret": self.secret}
        r = requests.post(f"{self.url}/oauth2/tokenP", json=data)
        if r.status_code == 200:
            self.token = r.json().get('access_token')
            print("✅ 토큰 OK")
            return True
        return False
    
    async def buy(self, symbol="005930", qty=1):
        if not self.token:
            await self.get_token()
        
        data = {
            "CANO": self.account[:8],
            "ACNT_PRDT_CD": self.account[8:],
            "PDNO": symbol,
            "ORD_DVSN": "01",
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0"
        }
        
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.key,
            "appsecret": self.secret,
            "tr_id": "VTTC0802U"
        }
        
        print(f"🛒 {symbol} {qty}주 매수...")
        r = requests.post(f"{self.url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, json=data)
        
        if r.status_code == 200 and r.json().get('rt_cd') == '0':
            print("✅ 매수 성공!")
            return True
        else:
            print(f"❌ 실패: {r.text}")
            return False

async def main():
    t = Trader()
    await t.buy()

if __name__ == "__main__":
    asyncio.run(main()) 