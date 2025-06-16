"""
🔄 매매 사이클
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
    
    async def token(self):
        data = {"grant_type": "client_credentials", "appkey": self.key, "appsecret": self.secret}
        r = requests.post(f"{self.url}/oauth2/tokenP", json=data)
        if r.status_code == 200:
            self.token = r.json().get('access_token')
            return True
        return False
    
    async def buy(self, code="005930"):
        if not self.token: await self.token()
        
        data = {"CANO": self.account[:8], "ACNT_PRDT_CD": self.account[8:], "PDNO": code, "ORD_DVSN": "01", "ORD_QTY": "1", "ORD_UNPR": "0"}
        headers = {"Content-Type": "application/json", "authorization": f"Bearer {self.token}", "appkey": self.key, "appsecret": self.secret, "tr_id": "VTTC0802U"}
        
        print(f"🛒 {code} 매수...")
        r = requests.post(f"{self.url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, json=data)
        return r.status_code == 200 and r.json().get('rt_cd') == '0'
    
    async def sell(self, code, qty):
        if not self.token: await self.token()
        
        data = {"CANO": self.account[:8], "ACNT_PRDT_CD": self.account[8:], "PDNO": code, "ORD_DVSN": "01", "ORD_QTY": str(qty), "ORD_UNPR": "0"}
        headers = {"Content-Type": "application/json", "authorization": f"Bearer {self.token}", "appkey": self.key, "appsecret": self.secret, "tr_id": "VTTC0801U"}
        
        print(f"💰 {code} 매도...")
        r = requests.post(f"{self.url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, json=data)
        return r.status_code == 200 and r.json().get('rt_cd') == '0'
    
    async def check(self):
        if not self.token: await self.token()
        
        headers = {"Content-Type": "application/json", "authorization": f"Bearer {self.token}", "appkey": self.key, "appsecret": self.secret, "tr_id": "VTTC8434R"}
        params = {"CANO": self.account[:8], "ACNT_PRDT_CD": self.account[8:], "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        
        r = requests.get(f"{self.url}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
        
        if r.status_code == 200:
            for s in r.json().get('output1', []):
                if int(s.get('HLDG_QTY', 0)) > 0:
                    print(f"📊 {s.get('PDNO')}: {s.get('HLDG_QTY')}주")
                    return s.get('PDNO'), int(s.get('HLDG_QTY'))
        return None, 0

async def main():
    t = Trader()
    
    print("1. 매수")
    if await t.buy():
        print("✅ 매수 OK")
        
        print("2. 잔고 확인")
        await asyncio.sleep(3)
        code, qty = await t.check()
        
        if qty > 0:
            print(f"3. {code} {qty}주 보유")
            if input("매도? (y/n): ") == 'y':
                if await t.sell(code, qty):
                    print("✅ 매도 OK")

if __name__ == "__main__":
    asyncio.run(main()) 