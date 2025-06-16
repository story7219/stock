"""
🧪 매도 테스트
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class SimpleTrader:
    def __init__(self):
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.app_key = os.getenv('MOCK_KIS_APP_KEY')
        self.app_secret = os.getenv('MOCK_KIS_APP_SECRET')
        self.account_number = os.getenv('MOCK_KIS_ACCOUNT_NUMBER')
        self.access_token = None
    
    async def get_token(self):
        """토큰 발급"""
        url = f"{self.base_url}/oauth2/tokenP"
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            self.access_token = result.get('access_token')
            print("✅ 토큰 발급 성공!")
            return True
        return False
    
    async def check_balance(self):
        """잔고 확인"""
        if not self.access_token:
            await self.get_token()
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "VTTC8434R"
        }
        
        params = {
            "CANO": self.account_number[:8],
            "ACNT_PRDT_CD": self.account_number[8:],
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
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('rt_cd') == '0':
                output1 = result.get('output1', [])
                print("📊 보유 종목:")
                for stock in output1:
                    symbol = stock.get('PDNO', '')
                    name = stock.get('PRDT_NAME', '')
                    quantity = int(stock.get('HLDG_QTY', 0))
                    if quantity > 0:
                        print(f"  {symbol} ({name}): {quantity}주")
                return output1
        return []
    
    async def sell_stock(self, symbol, quantity):
        """매도"""
        if not self.access_token:
            await self.get_token()
        
        order_data = {
            "CANO": self.account_number[:8],
            "ACNT_PRDT_CD": self.account_number[8:],
            "PDNO": symbol,
            "ORD_DVSN": "01",  # 시장가
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0"
        }
        
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "VTTC0801U"  # 매도
        }
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        
        print(f"💰 {symbol} {quantity}주 매도 시도...")
        response = requests.post(url, headers=headers, json=order_data)
        
        print(f"📡 응답: {response.status_code}")
        print(f"📡 내용: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('rt_cd') == '0':
                print("✅ 매도 성공!")
                return True
        
        print("❌ 매도 실패!")
        return False

async def main():
    trader = SimpleTrader()
    
    print("📊 현재 잔고 확인...")
    stocks = await trader.check_balance()
    
    # 삼성전자가 있으면 매도
    for stock in stocks:
        if stock.get('PDNO') == '005930':
            quantity = int(stock.get('HLDG_QTY', 0))
            if quantity > 0:
                print(f"💰 삼성전자 {quantity}주 매도 진행...")
                await trader.sell_stock('005930', quantity)
                break

if __name__ == "__main__":
    asyncio.run(main()) 