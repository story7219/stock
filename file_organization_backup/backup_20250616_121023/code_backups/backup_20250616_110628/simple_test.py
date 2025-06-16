"""
🧪 초간단 매수 테스트
"""

import asyncio
import sys
import os
import logging
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 직접 import (경로 문제 해결)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import requests

class SimpleTrader:
    """초간단 트레이더"""
    
    def __init__(self):
        self.base_url = "https://openapivts.koreainvestment.com:29443"  # 모의투자
        self.app_key = os.getenv('MOCK_KIS_APP_KEY')
        self.app_secret = os.getenv('MOCK_KIS_APP_SECRET')
        self.account_number = os.getenv('MOCK_KIS_ACCOUNT_NUMBER')
        self.access_token = None
        
        print(f"🔑 앱키: {self.app_key[:10]}...")
        print(f"🔐 시크릿: {self.app_secret[:10]}...")
        print(f"🏦 계좌: {self.account_number}")
    
    async def get_token(self):
        """토큰 발급"""
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            response = requests.post(url, json=data)
            print(f"📡 토큰 요청 응답: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                print("✅ 토큰 발급 성공!")
                return True
            else:
                print(f"❌ 토큰 발급 실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 토큰 발급 오류: {e}")
            return False
    
    async def buy_stock(self, symbol="005930", quantity=1):
        """주식 매수"""
        try:
            if not self.access_token:
                success = await self.get_token()
                if not success:
                    return False
            
            # 주문 데이터
            order_data = {
                "CANO": self.account_number[:8],
                "ACNT_PRDT_CD": self.account_number[8:],
                "PDNO": symbol,
                "ORD_DVSN": "01",  # 시장가
                "ORD_QTY": str(quantity),
                "ORD_UNPR": "0"
            }
            
            # 헤더
            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "VTTC0802U"  # 모의투자 매수
            }
            
            # API 호출
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            
            print(f"🛒 {symbol} {quantity}주 매수 시도...")
            print(f"📡 주문 데이터: {order_data}")
            
            response = requests.post(url, headers=headers, json=order_data)
            
            print(f"📡 응답 상태: {response.status_code}")
            print(f"📡 응답 내용: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    print("✅ 매수 주문 성공!")
                    return True
                else:
                    print(f"❌ 매수 실패: {result.get('msg1')}")
                    return False
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 매수 오류: {e}")
            return False

async def main():
    """메인 테스트"""
    print("🚀 초간단 매수 테스트 시작")
    
    trader = SimpleTrader()
    
    # 환경 변수 확인
    if not trader.app_key or not trader.app_secret or not trader.account_number:
        print("❌ 환경 변수가 설정되지 않았습니다!")
        print("📝 .env 파일을 확인하세요:")
        print("MOCK_KIS_APP_KEY=your_app_key")
        print("MOCK_KIS_APP_SECRET=your_app_secret")
        print("MOCK_KIS_ACCOUNT_NUMBER=your_account_number")
        return
    
    # 매수 테스트
    success = await trader.buy_stock("005930", 1)
    
    if success:
        print("🎉 테스트 성공!")
    else:
        print("😞 테스트 실패!")

if __name__ == "__main__":
    asyncio.run(main()) 