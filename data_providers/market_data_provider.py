# -*- coding: utf-8 -*-
"""
시장 데이터를 제공하는 다양한 소스에 대한 인터페이스와 구현을 포함하는 모듈입니다.
"""

from abc import ABC, abstractmethod
import pandas as pd
# from config import KIS_APP_KEY, KIS_APP_SECRET, KIS_BASE_URL
# 위와 같이 직접 참조하는 대신, 초기화 시점에 주입받도록 변경 (의존성 분리)

class MarketDataProvider(ABC):
    """
    시장 데이터 제공자에 대한 추상 베이스 클래스(ABC)입니다.
    모든 데이터 제공자는 이 클래스를 상속받아 필요한 메서드를 구현해야 합니다.
    """

    @abstractmethod
    def get_price(self, stock_code: str) -> dict:
        """
        특정 종목의 현재 가격 정보를 가져옵니다.

        :param stock_code: 종목 코드
        :return: 가격 정보 (예: {'current_price': 10000, 'volume': 12345})
        """
        pass

    @abstractmethod
    def get_historical_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        특정 종목의 과거 시계열 데이터를 가져옵니다.

        :param stock_code: 종목 코드
        :param start_date: 조회 시작일 (YYYY-MM-DD)
        :param end_date: 조회 종료일 (YYYY-MM-DD)
        :return: 시계열 데이터 (Pandas DataFrame)
        """
        pass

class KISMarketDataProvider(MarketDataProvider):
    """
    한국투자증권(KIS) API를 사용하여 시장 데이터를 제공하는 클래스입니다.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """
        KISMarketDataProvider를 초기화합니다.

        :param api_key: KIS API 키
        :param api_secret: KIS API 시크릿
        :param base_url: KIS API 기본 URL (실전/모의투자)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.access_token = self._get_access_token()
        print("KIS API Access Token 발급 완료")


    def _get_access_token(self) -> str:
        """
        KIS API 접근 토큰을 발급받습니다.
        실제 구현에서는 requests 라이브러리를 사용하여 API를 호출해야 합니다.
        """
        # 여기에 토큰 발급 로직 구현 (requests.post 사용)
        # 예시:
        # headers = {"content-type": "application/json"}
        # body = {"grant_type": "client_credentials", "appkey": self.api_key, "appsecret": self.api_secret}
        # PATH = "oauth2/tokenP"
        # URL = f"{self.base_url}/{PATH}"
        # res = requests.post(URL, headers=headers, json=body)
        # access_token = res.json()["access_token"]
        # return access_token
        print("실제 토큰 발급 로직이 필요합니다.")
        return "dummy_access_token" # 임시 토큰

    def get_price(self, stock_code: str) -> dict:
        """
        KIS API를 사용하여 특정 종목의 현재가를 조회합니다.
        실제 구현에서는 requests 라이브러리를 사용하여 API를 호출해야 합니다.
        """
        # 여기에 현재가 조회 API 호출 로직 구현
        print(f"{stock_code}의 현재가를 KIS API로 조회합니다 (구현 필요).")
        return {'current_price': 50000, 'volume': 1000} # 예시 데이터

    def get_historical_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        KIS API를 사용하여 특정 종목의 일봉/주봉/월봉 데이터를 조회합니다.
        실제 구현에서는 requests 라이브러리를 사용하여 API를 호출해야 합니다.
        """
        # 여기에 과거 데이터 조회 API 호출 로직 구현
        print(f"{stock_code}의 {start_date}부터 {end_date}까지의 과거 데이터를 KIS API로 조회합니다 (구현 필요).")
        # 예시 데이터프레임 생성
        data = {
            'date': pd.to_datetime(['2023-01-02', '2023-01-03']),
            'open': [10000, 10200],
            'high': [10300, 10400],
            'low': [9900, 10100],
            'close': [10200, 10300],
            'volume': [100000, 120000]
        }
        return pd.DataFrame(data)

# 필요에 따라 다른 데이터 제공자 구현
# class EbestMarketDataProvider(MarketDataProvider):
#     ...

def get_data_provider(provider_name: str, config: dict) -> MarketDataProvider:
    """
    설정에 맞는 데이터 제공자 인스턴스를 생성하여 반환하는 팩토리 함수입니다.

    :param provider_name: 사용할 제공자 이름 (예: 'kis', 'ebest')
    :param config: 설정 값들이 담긴 딕셔너리
    :return: MarketDataProvider 인스턴스
    """
    if provider_name.lower() == 'kis':
        return KISMarketDataProvider(
            api_key=config['KIS_APP_KEY'],
            api_secret=config['KIS_APP_SECRET'],
            base_url=config['KIS_BASE_URL']
        )
    # elif provider_name.lower() == 'ebest':
    #     return EbestMarketDataProvider(...)
    else:
        raise ValueError(f"지원하지 않는 데이터 제공자입니다: {provider_name}")

if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트 코드
    # 실제 사용 시에는 config 모듈에서 직접 값을 가져와야 합니다.
    # from config import IS_MOCK, KIS_APP_KEY, KIS_APP_SECRET, KIS_BASE_URL
    
    # 예시 설정 (실제로는 config 모듈에서 가져와야 함)
    mock_config = {
        "KIS_APP_KEY": "your_mock_key",
        "KIS_APP_SECRET": "your_mock_secret",
        "KIS_BASE_URL": "https://openapivts.koreainvestment.com:29443"
    }

    try:
        # 'kis' 제공자 인스턴스 생성
        data_provider = get_data_provider('kis', mock_config)

        # 현재가 조회 테스트
        samsung_price = data_provider.get_price("005930") # 삼성전자
        print(f"삼성전자 현재가 정보: {samsung_price}")

        # 과거 데이터 조회 테스트
        samsung_history = data_provider.get_historical_data("005930", "2023-01-01", "2023-01-31")
        print("삼성전자 과거 데이터:")
        print(samsung_history.head())

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"데이터 제공자 테스트 중 오류 발생: {e}") 