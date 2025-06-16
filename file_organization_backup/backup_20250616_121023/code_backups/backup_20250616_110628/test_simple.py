# -*- coding: utf-8 -*-
# test_simple.py - 한국투자증권 API 테스트 (IP 등록 전 Mock 데이터 활용)
import requests
import random
from datetime import datetime

class MockKISAPI:
    """KIS API 모의 클래스 - IP 등록 전 로직 테스트용"""
    
    def __init__(self):
        self.access_token = "mock_token"
        self.balance = 1000000  # 모의 잔고 100만원
        self.holdings = {}  # 보유 주식
        
    def get_access_token(self):
        """가짜 토큰 발급"""
        print("[TOKEN] 모의 토큰 발급 완료")
        return "mock_access_token_12345"
    
    def get_stock_price(self, stock_code):
        """가짜 주식 가격 반환 (실제와 유사한 변동)"""
        mock_prices = {
            "005930": {"name": "삼성전자", "current_price": 71000 + random.randint(-2000, 2000)},
            "000660": {"name": "SK하이닉스", "current_price": 126000 + random.randint(-3000, 3000)},
            "035420": {"name": "NAVER", "current_price": 194000 + random.randint(-5000, 5000)},
            "207940": {"name": "삼성바이오로직스", "current_price": 780000 + random.randint(-20000, 20000)},
            "005380": {"name": "현대차", "current_price": 205000 + random.randint(-5000, 5000)}
        }
        
        if stock_code in mock_prices:
            stock = mock_prices[stock_code]
            change = random.randint(-3, 3)  # -3% ~ +3% 변동
            stock["change_rate"] = change
            stock["change_amount"] = int(stock["current_price"] * change / 100)
            return stock
        else:
            return {
                "name": "알 수 없는 종목",
                "current_price": 50000 + random.randint(-5000, 5000),
                "change_rate": random.randint(-2, 2),
                "change_amount": random.randint(-1000, 1000)
            }
    
    def buy_stock(self, stock_code, quantity):
        """가짜 매수 주문"""
        stock_info = self.get_stock_price(stock_code)
        total_cost = stock_info["current_price"] * quantity
        
        if self.balance >= total_cost:
            self.balance -= total_cost
            if stock_code in self.holdings:
                self.holdings[stock_code] += quantity
            else:
                self.holdings[stock_code] = quantity
                
            print(f"[BUY] 매수 성공: {stock_info['name']} {quantity}주 ({stock_info['current_price']:,}원)")
            print(f"[BALANCE] 잔고: {self.balance:,}원")
            return {"order_id": f"buy_{random.randint(1000, 9999)}", "status": "success"}
        else:
            print(f"[ERROR] 매수 실패: 잔고 부족 (필요: {total_cost:,}원, 보유: {self.balance:,}원)")
            return {"order_id": None, "status": "failed", "reason": "insufficient_balance"}
    
    def sell_stock(self, stock_code, quantity):
        """가짜 매도 주문"""
        if stock_code in self.holdings and self.holdings[stock_code] >= quantity:
            stock_info = self.get_stock_price(stock_code)
            total_amount = stock_info["current_price"] * quantity
            
            self.balance += total_amount
            self.holdings[stock_code] -= quantity
            
            if self.holdings[stock_code] == 0:
                del self.holdings[stock_code]
            
            print(f"[SELL] 매도 성공: {stock_info['name']} {quantity}주 ({stock_info['current_price']:,}원)")
            print(f"[BALANCE] 잔고: {self.balance:,}원")
            return {"order_id": f"sell_{random.randint(1000, 9999)}", "status": "success"}
        else:
            print(f"[ERROR] 매도 실패: 보유 수량 부족")
            return {"order_id": None, "status": "failed", "reason": "insufficient_stock"}
    
    def get_portfolio(self):
        """포트폴리오 조회"""
        print("\n[PORTFOLIO] 현재 포트폴리오:")
        print(f"[CASH] 현금 잔고: {self.balance:,}원")
        
        if not self.holdings:
            print("[STOCKS] 보유 주식: 없음")
            return
            
        total_value = self.balance
        print("[STOCKS] 보유 주식:")
        for stock_code, quantity in self.holdings.items():
            stock_info = self.get_stock_price(stock_code)
            value = stock_info["current_price"] * quantity
            total_value += value
            print(f"   - {stock_info['name']} ({stock_code}): {quantity}주 = {value:,}원")
        
        print(f"[TOTAL] 총 자산: {total_value:,}원")
        profit = total_value - 1000000
        profit_rate = (profit / 1000000) * 100
        print(f"[PROFIT] 수익: {profit:,}원 ({profit_rate:.2f}%)")


def simple_trading_strategy(api):
    """간단한 트레이딩 전략 테스트"""
    print("\n[STRATEGY] 간단한 트레이딩 봇 전략 테스트:")
    
    # 관심 종목
    watchlist = ["005930", "000660", "035420"]
    
    for stock_code in watchlist:
        stock_info = api.get_stock_price(stock_code)
        print(f"\n[ANALYSIS] {stock_info['name']} 분석:")
        print(f"   현재가: {stock_info['current_price']:,}원")
        print(f"   변동률: {stock_info['change_rate']:.2f}%")
        
        # 간단한 전략: 상승률이 2% 이상이면 매수, -2% 이하면 매도
        if stock_info['change_rate'] >= 2.0:
            print(f"   [SIGNAL] 매수 신호 발생!")
            api.buy_stock(stock_code, 10)
        elif stock_info['change_rate'] <= -2.0:
            print(f"   [SIGNAL] 매도 신호 발생!")
            if stock_code in api.holdings:
                api.sell_stock(stock_code, min(5, api.holdings.get(stock_code, 0)))
        else:
            print(f"   [WAIT] 관망")


def test_kis_connection_attempt():
    """실제 KIS API 연결 시도 (IP 등록 확인용)"""
    print("\n[CONNECTION] 실제 KIS API 연결 테스트:")
    
    try:
        # 모의투자 URL로 토큰 요청
        url = "https://openapivts.koreainvestment.com/oauth2/tokenP"
        
        # 더미 데이터로 테스트 (실제 키가 없어도 IP 등록 여부 확인 가능)
        data = {
            "grant_type": "client_credentials",
            "appkey": "dummy_key",
            "appsecret": "dummy_secret"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=5)
        
        print(f"응답 코드: {response.status_code}")
        response_text = response.text.lower()
        
        if "ip" in response_text and ("등록" in response_text or "register" in response_text):
            print("[ERROR] IP 등록이 필요합니다.")
        elif response.status_code == 401:
            print("[OK] IP는 등록되어 있지만 API 키가 잘못되었습니다.")
        else:
            print(f"응답 내용: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 연결 오류: {e}")


def main():
    """메인 테스트 함수"""
    print("한국투자증권 API 테스트 시작")
    print("=" * 50)
    
    # Mock API 테스트
    api = MockKISAPI()
    
    # 토큰 발급 테스트
    token = api.get_access_token()
    
    # 초기 포트폴리오
    api.get_portfolio()
    
    # 트레이딩 전략 테스트
    simple_trading_strategy(api)
    
    # 최종 포트폴리오
    api.get_portfolio()
    
    # 실제 KIS API 연결 시도
    test_kis_connection_attempt()
    
    print("\n[COMPLETE] 테스트 완료!")
    print("[INFO] IP 등록 후 실제 API 키를 .env 파일에 설정하여 실제 테스트를 진행하세요.")


if __name__ == "__main__":
    main()