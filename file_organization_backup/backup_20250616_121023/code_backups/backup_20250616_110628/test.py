# mock_data.py
class MockKISAPI:
    """KIS API 모의 클래스 - IP 등록 전 로직 테스트용"""
    
    def __init__(self):
        self.access_token = "mock_token"
        
    def get_access_token(self):
        """가짜 토큰 반환"""
        print("📝 모의 토큰 발급 완료")
        return "mock_access_token_12345"
    
    def get_stock_price(self, stock_code):
        """가짜 주식 가격 반환"""
        mock_prices = {
            "005930": {"current_price": 71000, "change": 1000},  # 삼성전자
            "000660": {"current_price": 556000, "change": -5000}, # SK하이닉스
            "035420": {"current_price": 264000, "change": 2000}   # NAVER
        }
        
        if stock_code in mock_prices:
            return mock_prices[stock_code]
        else:
            return {"current_price": 50000, "change": 0}
    
    def buy_stock(self, stock_code, quantity):
        """가짜 매수 주문"""
        print(f"📈 모의 매수: {stock_code} {quantity}주")
        return {"order_id": "mock_order_123", "status": "success"}
    
    def sell_stock(self, stock_code, quantity):
        """가짜 매도 주문"""
        print(f"📉 모의 매도: {stock_code} {quantity}주")
        return {"order_id": "mock_order_456", "status": "success"}

# 테스트 봇 로직
def test_trading_bot():
    """트레이딩 봇 로직 테스트"""
    
    # Mock API 사용
    api = MockKISAPI()
    
    # 토큰 발급 테스트
    token = api.get_access_token()
    
    # 주식 가격 조회 테스트
    samsung_price = api.get_stock_price("005930")
    print(f"삼성전자 현재가: {samsung_price['current_price']}")
    
    # 간단한 매매 로직 테스트
    if samsung_price['change'] > 0:
        result = api.buy_stock("005930", 10)
        print(f"매수 결과: {result}")
    else:
        result = api.sell_stock("005930", 5)
        print(f"매도 결과: {result}")

if __name__ == "__main__":
    test_trading_bot()