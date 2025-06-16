import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class TokenInfo:
    access_token: str
    token_type: str
    expires_at: datetime

@dataclass
class StockPrice:
    symbol: str
    name: str
    current_price: int
    change_rate: float
    volume: int
    trading_value: int

@dataclass
class OrderRequest:
    symbol: str
    order_type: str  # "01": 시장가, "00": 지정가
    quantity: int
    action: str  # "buy" or "sell"
    price: int = 0

class KISAPIClient:
    def __init__(self):
        self.app_key = os.getenv('KIS_APP_KEY')
        self.app_secret = os.getenv('KIS_APP_SECRET')
        self.account_number = os.getenv('KIS_ACCOUNT_NUMBER')
        
        # 운영/모의투자 선택 (모의투자로 시작)
        self.base_url = "https://openapi.koreainvestment.com:9000"  # 모의투자
        # self.base_url = "https://openapi.koreainvestment.com:9443"  # 실투자
        
        self.token_info: Optional[TokenInfo] = None
        
    def get_access_token(self) -> str:
        """접근토큰 발급 및 캐싱"""
        if self.token_info and self.token_info.expires_at > datetime.now():
            return self.token_info.access_token
            
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        if response.status_code == 200:
            self.token_info = TokenInfo(
                access_token=result['access_token'],
                token_type=result['token_type'],
                expires_at=datetime.now() + timedelta(hours=6)
            )
            return self.token_info.access_token
        else:
            raise Exception(f"토큰 발급 실패: {result}")
    
    def get_headers(self, tr_id: str) -> Dict[str, str]:
        """API 호출용 공통 헤더"""
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
    
    def get_current_price(self, symbol: str) -> StockPrice:
        """현재가 조회"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self.get_headers("FHKST01010100")
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol
        }
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if response.status_code == 200 and data['rt_cd'] == '0':
            output = data['output']
            return StockPrice(
                symbol=symbol,
                name=output['hts_kor_isnm'],
                current_price=int(output['stck_prpr']),
                change_rate=float(output['prdy_ctrt']),
                volume=int(output['acml_vol']),
                trading_value=int(output['acml_tr_pbmn'])
            )
        else:
            raise Exception(f"현재가 조회 실패: {data}")
    
    def get_trading_volume_ranking(self, limit: int = 20) -> List[StockPrice]:
        """거래량 순위 조회"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/volume-rank"
        headers = self.get_headers("FHPST01710000")
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code": "20171",
            "fid_input_iscd": "0000",
            "fid_div_cls_code": "0",
            "fid_blng_cls_code": "0",
            "fid_trgt_cls_code": "111111111",
            "fid_trgt_exls_cls_code": "0000000000",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": str(limit)
        }
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        stocks = []
        if response.status_code == 200 and data['rt_cd'] == '0':
            for item in data['output']:
                stocks.append(StockPrice(
                    symbol=item['mksc_shrn_iscd'],
                    name=item['hts_kor_isnm'],
                    current_price=int(item['stck_prpr']),
                    change_rate=float(item['prdy_ctrt']),
                    volume=int(item['acml_vol']),
                    trading_value=int(item['acml_tr_pbmn'])
                ))
        
        return stocks
    
    def place_order(self, order: OrderRequest) -> Dict:
        """주문 실행"""
        # 매수/매도에 따른 URL과 TR_ID 설정
        if order.action == "buy":
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            tr_id = "VTTC0802U"  # 모의투자 매수
        else:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            tr_id = "VTTC0801U"  # 모의투자 매도
        
        headers = self.get_headers(tr_id)
        
        data = {
            "CANO": self.account_number.split('-')[0],
            "ACNT_PRDT_CD": self.account_number.split('-')[1],
            "PDNO": order.symbol,
            "ORD_DVSN": order.order_type,
            "ORD_QTY": str(order.quantity),
            "ORD_UNPR": str(order.price) if order.order_type == "00" else "0"
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    
    def get_account_balance(self) -> Dict:
        """계좌잔고 조회"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = self.get_headers("VTTC8434R")
        
        params = {
            "CANO": self.account_number.split('-')[0],
            "ACNT_PRDT_CD": self.account_number.split('-')[1],
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
        return response.json() 