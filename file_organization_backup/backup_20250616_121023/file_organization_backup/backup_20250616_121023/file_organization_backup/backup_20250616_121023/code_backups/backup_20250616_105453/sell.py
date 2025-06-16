"""
💰 매도 테스트
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
    
    async def check(self):
        if not self.token:
            await self.get_token()
        
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.key,
            "appsecret": self.secret,
            "tr_id": "VTTC8434R"
        }
        
        params = {
            "CANO": self.account[:8],
            "ACNT_PRDT_CD": self.account[8:],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        r = requests.get(f"{self.url}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
        
        if r.status_code == 200:
            stocks = r.json().get('output1', [])
            print("📊 보유 종목:")
            for s in stocks:
                if int(s.get('HLDG_QTY', 0)) > 0:
                    print(f"  {s.get('PDNO')} ({s.get('PRDT_NAME')}): {s.get('HLDG_QTY')}주")
            return stocks
        return []
    
    async def sell(self, symbol, qty):
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
            "tr_id": "VTTC0801U"
        }
        
        print(f"💰 {symbol} {qty}주 매도...")
        r = requests.post(f"{self.url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, json=data)
        
        if r.status_code == 200 and r.json().get('rt_cd') == '0':
            print("✅ 매도 성공!")
            return True
        else:
            print(f"❌ 실패: {r.text}")
            return False

async def main():
    t = Trader()
    stocks = await t.check()
    
    # 삼성전자 있으면 매도
    for s in stocks:
        if s.get('PDNO') == '005930' and int(s.get('HLDG_QTY', 0)) > 0:
            await t.sell('005930', int(s.get('HLDG_QTY')))
            break

if __name__ == "__main__":
    asyncio.run(main()) 