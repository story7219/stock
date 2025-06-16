"""
🔄 완전한 매매 사이클 테스트
매수 → 잔고확인 → 매도 → 최종확인
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class CompleteTrader:
    """완전한 매매 테스트용 트레이더"""
    
    def __init__(self):
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.app_key = os.getenv('MOCK_KIS_APP_KEY')
        self.app_secret = os.getenv('MOCK_KIS_APP_SECRET')
        self.account_number = os.getenv('MOCK_KIS_ACCOUNT_NUMBER')
        self.access_token = None
        
        print(f"🔑 앱키: {self.app_key[:10]}...")
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
                "tr_id": "VTTC0802U"  # 매수
            }
            
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            
            print(f"🛒 {symbol} {quantity}주 매수 시도...")
            response = requests.post(url, headers=headers, json=order_data)
            
            print(f"📡 응답: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    order_no = result.get('output', {}).get('ODNO', '')
                    print(f"✅ 매수 성공! 주문번호: {order_no}")
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
    
    async def check_balance(self):
        """잔고 확인"""
        try:
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
                    print("\n📊 현재 보유 종목:")
                    print("-" * 40)
                    
                    holdings = []
                    for stock in output1:
                        symbol = stock.get('PDNO', '')
                        name = stock.get('PRDT_NAME', '')
                        quantity = int(stock.get('HLDG_QTY', 0))
                        avg_price = float(stock.get('PCHS_AVG_PRIC', 0))
                        current_price = float(stock.get('PRPR', 0))
                        
                        if quantity > 0:
                            profit_loss = (current_price - avg_price) * quantity
                            profit_rate = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
                            
                            print(f"📈 {symbol} ({name})")
                            print(f"   보유: {quantity}주")
                            print(f"   평단: {avg_price:,.0f}원")
                            print(f"   현재: {current_price:,.0f}원")
                            print(f"   손익: {profit_loss:,.0f}원 ({profit_rate:+.2f}%)")
                            print()
                            
                            holdings.append({
                                'symbol': symbol,
                                'name': name,
                                'quantity': quantity,
                                'avg_price': avg_price,
                                'current_price': current_price,
                                'profit_loss': profit_loss,
                                'profit_rate': profit_rate
                            })
                    
                    if not holdings:
                        print("   보유 종목 없음")
                    
                    return holdings
            
            return []
            
        except Exception as e:
            print(f"❌ 잔고 확인 오류: {e}")
            return []
    
    async def sell_stock(self, symbol, quantity):
        """매도"""
        try:
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
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    order_no = result.get('output', {}).get('ODNO', '')
                    print(f"✅ 매도 성공! 주문번호: {order_no}")
                    return True
                else:
                    print(f"❌ 매도 실패: {result.get('msg1')}")
                    return False
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 매도 오류: {e}")
            return False

async def full_trading_cycle():
    """완전한 매매 사이클"""
    trader = CompleteTrader()
    
    print("🚀 완전한 매매 사이클 테스트 시작")
    print("=" * 50)
    
    # 환경 변수 확인
    if not trader.app_key or not trader.app_secret or not trader.account_number:
        print("❌ 환경 변수가 설정되지 않았습니다!")
        return
    
    try:
        # 1. 초기 잔고 확인
        print("\n1️⃣ 초기 잔고 확인")
        initial_holdings = await trader.check_balance()
        
        # 2. 매수 테스트
        print("\n2️⃣ 매수 테스트")
        symbol = "005930"  # 삼성전자
        quantity = 1
        
        buy_success = await trader.buy_stock(symbol, quantity)
        
        if not buy_success:
            print("❌ 매수 실패로 테스트 중단")
            return
        
        # 3. 매수 후 잔고 확인
        print("\n3️⃣ 매수 후 잔고 확인")
        print("⏳ 5초 대기 후 잔고 확인...")
        await asyncio.sleep(5)
        
        after_buy_holdings = await trader.check_balance()
        
        # 4. 매도 여부 확인
        print("\n4️⃣ 매도 테스트 여부 확인")
        
        # 삼성전자 보유 확인
        samsung_holding = None
        for holding in after_buy_holdings:
            if holding['symbol'] == symbol:
                samsung_holding = holding
                break
        
        if samsung_holding and samsung_holding['quantity'] > 0:
            print(f"📊 {symbol} {samsung_holding['quantity']}주 보유 확인!")
            print(f"💰 현재 손익: {samsung_holding['profit_loss']:,.0f}원 ({samsung_holding['profit_rate']:+.2f}%)")
            
            user_input = input("\n매도 테스트를 진행하시겠습니까? (y/n): ")
            
            if user_input.lower() == 'y':
                print("\n5️⃣ 매도 테스트 진행")
                
                sell_success = await trader.sell_stock(symbol, samsung_holding['quantity'])
                
                if sell_success:
                    print("✅ 매도 완료!")
                    
                    # 6. 최종 잔고 확인
                    print("\n6️⃣ 최종 잔고 확인")
                    print("⏳ 5초 대기 후 최종 잔고 확인...")
                    await asyncio.sleep(5)
                    
                    final_holdings = await trader.check_balance()
                    
                    print("\n🎉 완전한 매매 사이클 테스트 완료!")
                else:
                    print("❌ 매도 실패!")
            else:
                print("⏭️ 매도 테스트 건너뜀")
        else:
            print("⚠️ 매수 후에도 보유 종목이 확인되지 않음")
            print("   (체결 대기 중이거나 주문 실패 가능성)")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(full_trading_cycle()) 