"""
핵심 트레이딩 엔진
- 한국투자증권 API 연동
- 잔고 조회, 주문 실행 등 핵심 거래 기능 담당
"""
import requests
import json
import logging
import config

logger = logging.getLogger(__name__)

class KisTrader:
    """한국투자증권 API를 사용하는 트레이딩 클래스"""
    def __init__(self):
        self.is_mock = config.IS_MOCK_TRADING
        self.base_url = "https://openapivts.koreainvestment.com:29443" if self.is_mock else "https://openapi.koreainvestment.com:9443"
        self.app_key = config.KIS_APP_KEY
        self.app_secret = config.KIS_APP_SECRET
        self.account_no = config.KIS_ACCOUNT_NO
        self.access_token = self._get_access_token()
        
        if self.access_token:
            logger.info(f"✅ KIS API 접속 성공 ({'모의투자' if self.is_mock else '실전투자'})")
        else:
            logger.error("❌ KIS API 접속 실패. config.py의 API 키와 계정 정보를 확인하세요.")
    
    def _get_access_token(self):
        """접근 토큰 발급"""
        path = "/oauth2/tokenP"
        url = f"{self.base_url}{path}"
        headers = {"content-type": "application/json"}
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            res.raise_for_status()
            return res.json()["access_token"]
        except requests.exceptions.RequestException as e:
            logger.error(f"토큰 발급 실패: {e}, 응답: {res.text if 'res' in locals() else 'N/A'}")
            return None

    def _get_common_headers(self):
        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "custtype": "P"
        }

    def get_balance(self):
        """잔고 조회"""
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{self.base_url}{path}"
        headers = self._get_common_headers()
        headers["tr_id"] = "VTTC8434R" if self.is_mock else "TTTC8434R"
        params = {
            "CANO": self.account_no.split('-')[0], "ACNT_PRDT_CD": self.account_no.split('-')[1],
            "AFHR_FLASS_YN": "N", "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""
        }
        try:
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"잔고 조회 실패: {e}")
            return None

    def place_order(self, stock_code: str, order_type: str, quantity: int, price: int = 0):
        """주문 실행 (order_type: 'buy' 또는 'sell')"""
        path = "/uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.base_url}{path}"
        
        order_code = "01" if price > 0 else "02" # 01: 지정가, 02: 시장가
        tr_id_suffix = "0801U" if order_type == 'sell' else "0802U"
        
        headers = self._get_common_headers()
        headers["tr_id"] = ("VTTC" if self.is_mock else "TTTC") + tr_id_suffix
        
        body = {
            "CANO": self.account_no.split('-')[0], "ACNT_PRDT_CD": self.account_no.split('-')[1],
            "PDNO": stock_code, "ORD_DVSN": order_code, "ORD_QTY": str(quantity), "ORD_UNPR": str(price),
        }
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            res.raise_for_status()
            result = res.json()
            if result.get('rt_cd') == '0':
                logger.info(f"✅ 주문 성공: {stock_code} {quantity}주 {'매도' if order_type=='sell' else '매수'}")
                return result
            else:
                logger.error(f"❌ 주문 실패: {result.get('msg1', '알 수 없는 오류')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"주문 API 요청 실패: {e}, 응답: {res.text if 'res' in locals() else 'N/A'}")
            return None 