import requests
import json
from typing import Dict

class KoreaInvestmentBroker:
    """
    한국투자증권의 REST/WebSocket API를 사용하여 거래를 실행하는 브로커 클래스입니다.
    실제 구현을 위해서는 API 키 발급 및 상세 명세 확인이 필요합니다.
    """
    def __init__(self, api_key, api_secret):
        """
        API 키와 시크릿을 사용하여 브로커를 초기화합니다.
        
        Args:
            api_key (str): 한국투자증권에서 발급받은 API 키
            api_secret (str): 한국투자증권에서 발급받은 API 시크릿
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://openapi.koreainvestment.com:9443"  # 실거래 URL
        # self.base_url = "https://openapivts.koreainvestment.com:29443" # 모의투자 URL
        self.access_token = self._get_access_token()

    def _get_access_token(self):
        """
        접근 토큰을 발급받습니다.
        """
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.api_key,
            "appsecret": self.api_secret
        }
        path = "/oauth2/tokenP"
        url = f"{self.base_url}{path}"
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            response.raise_for_status()
            return response.json().get("access_token")
        except requests.exceptions.RequestException as e:
            print(f"한국투자증권 API 접근 토큰 발급 실패: {e}")
            return None

    def get_account_balance(self):
        """
        계좌 잔고를 조회합니다.
        실제 구현이 필요합니다.
        """
        print("한국투자증권 계좌 잔고를 조회합니다.")
        # TODO: 한국투자증권 API를 사용하여 잔고 조회 로직 구현
        # 예: path = '/api/v1/accounts/balance'
        #     headers = {'Authorization': f'Bearer {self.access_token}', ...}
        #     response = requests.get(f"{self.base_url}{path}", headers=headers)
        return {"cash": 10000000, "stocks": []} # 예시 데이터

    def place_order(self, stock_code, quantity, price, order_type='buy'):
        """
        주문을 실행합니다.
        
        Args:
            stock_code (str): 종목 코드
            quantity (int): 수량
            price (int): 가격 (지정가)
            order_type (str): 'buy' 또는 'sell'
        """
        print(f"한국투자증권 API를 통해 주문을 실행합니다: {stock_code}, {quantity}주, {price}원, {order_type}")
        # TODO: 한국투자증권 API를 사용하여 주문 실행 로직 구현
        # 예: path = '/api/v1/orders'
        #     headers = {'Authorization': f'Bearer {self.access_token}', ...}
        #     body = {'stock_code': stock_code, ...}
        #     response = requests.post(f"{self.base_url}{path}", headers=headers, data=json.dumps(body))
        
        return {"order_id": "mock_order_12345", "status": "success"} # 예시 데이터

    def get_order_status(self, order_id):
        """
        주문 상태를 조회합니다.
        """
        print(f"주문 상태를 조회합니다: {order_id}")
        # TODO: 한국투자증권 API를 사용하여 주문 상태 조회 로직 구현
        return {"status": "filled"} # 예시 데이터


class ExecutionManager:
    """
    전략에 따라 실제 주문을 실행하고 포트폴리오를 관리합니다.
    """
    def __init__(self, broker: KoreaInvestmentBroker):
        """
        브로커 인스턴스를 사용하여 실행 관리자를 초기화합니다.

        Args:
            broker (KoreaInvestmentBroker): 주문 실행에 사용할 브로커의 인스턴스
        """
        if not isinstance(broker, KoreaInvestmentBroker):
            raise TypeError("broker는 KoreaInvestmentBroker의 인스턴스여야 합니다.")
        self.broker = broker

    def get_current_portfolio(self):
        """
        브로커를 통해 현재 계좌의 포트폴리오 정보를 가져옵니다.
        """
        print("현재 포트폴리오 정보를 조회합니다...")
        balance = self.broker.get_account_balance()
        # 실제로는 보유 주식 목록과 평가 금액 등을 포함해야 합니다.
        return balance

    def rebalance_portfolio(self, target_portfolio: dict):
        """
        현재 포트폴리오를 목표 포트폴리오에 맞게 리밸런싱합니다.

        Args:
            target_portfolio (dict): 목표 포트폴리오. 
                                     예: {'005930': 0.2, '000660': 0.15, ...}
        """
        print("\n포트폴리오 리밸런싱을 시작합니다...")
        print(f"목표 포트폴리오: {target_portfolio}")

        # 1. 현재 계좌 정보 가져오기
        current_portfolio = self.get_current_portfolio()
        # 편의상 총 자산을 현금으로 가정합니다. 실제로는 주식 평가액을 포함해야 합니다.
        total_asset_value = current_portfolio.get('cash', 0) 
        print(f"현재 총 자산 (가정): {total_asset_value:,.0f}원")

        # 2. 목표 포트폴리오에 따른 각 종목별 목표 금액 계산
        target_allocations = {
            stock: total_asset_value * weight 
            for stock, weight in target_portfolio.items()
        }
        print(f"종목별 목표 금액: {target_allocations}")

        # 3. 현재 보유량과 목표 보유량 비교하여 주문 생성 (실제 구현 필요)
        # 이 예제에서는 간단히 매수 주문만 생성합니다.
        # 실제 로직은 현재 보유량을 확인하고, 초과분은 매도, 부족분은 매수해야 합니다.
        
        print("\n필요 주문을 생성 및 실행합니다 (예시: 매수만 진행)...")
        for stock_code, target_value in target_allocations.items():
            # TODO: 현재 주가를 가져오는 로직 필요 (데이터 레이어 연동)
            current_price = 50000  # 예시 현재가
            
            quantity_to_buy = int(target_value // current_price)
            
            if quantity_to_buy > 0:
                print(f"  - {stock_code}: {quantity_to_buy} 주 매수 주문 실행")
                self.broker.place_order(
                    stock_code=stock_code,
                    quantity=quantity_to_buy,
                    price=current_price, # 실제로는 시장가/지정가 등 옵션 필요
                    order_type='buy'
                )
            else:
                print(f"  - {stock_code}: 매수 수량 없음")

        print("\n리밸런싱 완료.") 