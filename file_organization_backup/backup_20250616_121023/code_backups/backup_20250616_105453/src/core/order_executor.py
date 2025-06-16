"""
📈 주문 실행기 - 실제 매수 강화
"""

import asyncio
import logging
import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime

from config import config

class OrderExecutor:
    """실제 주문 실행 강화"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://openapi.koreainvestment.com:9443" if not config.is_mock else "https://openapivts.koreainvestment.com:29443"
        self.access_token = None
        self.headers = {}
        
    async def initialize(self):
        """초기화 - 토큰 발급"""
        try:
            await self._get_access_token()
            self.logger.info("✅ 주문 실행기 초기화 완료")
        except Exception as e:
            self.logger.error(f"❌ 주문 실행기 초기화 실패: {e}")
            raise
    
    async def _get_access_token(self):
        """액세스 토큰 발급"""
        try:
            api_config = config.current_api_config
            
            url = f"{self.base_url}/oauth2/tokenP"
            data = {
                "grant_type": "client_credentials",
                "appkey": api_config['app_key'],
                "appsecret": api_config['app_secret']
            }
            
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                
                # 헤더 설정
                self.headers = {
                    "Content-Type": "application/json",
                    "authorization": f"Bearer {self.access_token}",
                    "appkey": api_config['app_key'],
                    "appsecret": api_config['app_secret'],
                    "tr_id": "VTTC0802U" if config.is_mock else "TTTC0802U"  # 모의/실거래 구분
                }
                
                self.logger.info("✅ 액세스 토큰 발급 성공")
            else:
                raise Exception(f"토큰 발급 실패: {response.text}")
                
        except Exception as e:
            self.logger.error(f"❌ 토큰 발급 오류: {e}")
            raise
    
    async def buy_market_order(self, symbol: str, quantity: int) -> bool:
        """시장가 매수 주문 - 검증된 로직 적용"""
        try:
            self.logger.info(f"🛒 시장가 매수 시도: {symbol} {quantity}주")
            
            # 1. 토큰 발급 (simple_test.py에서 검증된 방식)
            await self._get_access_token_simple()
            
            # 2. 주문 데이터 (검증된 형식)
            order_data = {
                "CANO": config.current_api_config['account_number'][:8],
                "ACNT_PRDT_CD": config.current_api_config['account_number'][8:],
                "PDNO": symbol,
                "ORD_DVSN": "01",  # 시장가
                "ORD_QTY": str(quantity),
                "ORD_UNPR": "0"
            }
            
            # 3. 헤더 (검증된 형식)
            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {self.access_token}",
                "appkey": config.current_api_config['app_key'],
                "appsecret": config.current_api_config['app_secret'],
                "tr_id": "VTTC0802U" if config.is_mock else "TTTC0802U"
            }
            
            # 4. API 호출
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            
            self.logger.info(f"📡 주문 데이터: {order_data}")
            
            response = requests.post(url, headers=headers, json=order_data, timeout=30)
            
            self.logger.info(f"📡 API 응답 상태: {response.status_code}")
            self.logger.info(f"📡 API 응답 내용: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                rt_cd = result.get('rt_cd', '')
                msg1 = result.get('msg1', '')
                
                if rt_cd == '0':  # 성공
                    output = result.get('output', {})
                    order_no = output.get('ODNO', '')
                    
                    self.logger.info(f"✅ 매수 주문 성공: {symbol} {quantity}주 (주문번호: {order_no})")
                    self.logger.info(f"🎯 KRX NXT로 주문 전송 완료")
                    
                    return True
                else:
                    self.logger.error(f"❌ 매수 주문 실패: {symbol} - {msg1}")
                    return False
            else:
                self.logger.error(f"❌ HTTP 오류: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 매수 주문 오류 ({symbol}): {e}")
            return False
    
    async def _get_access_token_simple(self):
        """간단한 토큰 발급 (검증된 방식)"""
        try:
            api_config = config.current_api_config
            
            url = f"{self.base_url}/oauth2/tokenP"
            data = {
                "grant_type": "client_credentials",
                "appkey": api_config['app_key'],
                "appsecret": api_config['app_secret']
            }
            
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                self.logger.info("✅ 액세스 토큰 발급 성공")
            else:
                raise Exception(f"토큰 발급 실패: {response.text}")
                
        except Exception as e:
            self.logger.error(f"❌ 토큰 발급 오류: {e}")
            raise
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            
            headers = self.headers.copy()
            headers["tr_id"] = "FHKST01010100"
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    output = result.get('output', {})
                    current_price = float(output.get('STCK_PRPR', 0))  # 현재가
                    return current_price
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 현재가 조회 오류 ({symbol}): {e}")
            return None
    
    async def _check_order_status(self, order_no: str) -> bool:
        """주문 체결 상태 확인"""
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-order"
            
            headers = self.headers.copy()
            headers["tr_id"] = "VTTC8001R" if config.is_mock else "TTTC8001R"
            
            params = {
                "CANO": config.current_api_config['account_number'][:8],
                "ACNT_PRDT_CD": config.current_api_config['account_number'][8:],
                "ODNO": order_no,
                "ORD_GNO_BRNO": "",
                "ODNO_NM": "",
                "INQR_DVSN": "00"
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    output = result.get('output', [])
                    if output:
                        order_status = output[0].get('ORD_STAT_CD', '')
                        return order_status == '02'  # 02: 체결완료
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 주문 상태 확인 오류: {e}")
            return False
    
    async def sell_market_order(self, symbol: str, quantity: int) -> bool:
        """시장가 매도 주문"""
        try:
            self.logger.info(f"💰 시장가 매도 시도: {symbol} {quantity}주")
            
            if not self.access_token:
                await self._get_access_token()
            
            order_data = {
                "CANO": config.current_api_config['account_number'][:8],
                "ACNT_PRDT_CD": config.current_api_config['account_number'][8:],
                "PDNO": symbol,
                "ORD_DVSN": "01",  # 시장가
                "ORD_QTY": str(quantity),
                "ORD_UNPR": "0"
            }
            
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            
            # 매도용 헤더 설정
            headers = self.headers.copy()
            headers["tr_id"] = "VTTC0801U" if config.is_mock else "TTTC0801U"  # 매도 TR_ID
            
            response = requests.post(url, headers=headers, json=order_data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    self.logger.info(f"✅ 매도 주문 성공: {symbol} {quantity}주")
                    return True
                else:
                    self.logger.error(f"❌ 매도 주문 실패: {result.get('msg1', '')}")
                    return False
            else:
                self.logger.error(f"❌ 매도 API 호출 실패: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 매도 주문 오류 ({symbol}): {e}")
            return False
    
    async def cleanup(self):
        """정리"""
        self.logger.info("🧹 주문 실행기 정리 완료")
    
    async def _check_balance(self, symbol: str) -> int:
        """잔고 확인"""
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
            
            headers = self.headers.copy()
            headers["tr_id"] = "VTTC8434R" if config.is_mock else "TTTC8434R"
            
            params = {
                "CANO": config.current_api_config['account_number'][:8],
                "ACNT_PRDT_CD": config.current_api_config['account_number'][8:],
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
                    for stock in output1:
                        if stock.get('PDNO') == symbol:
                            return int(stock.get('HLDG_QTY', 0))
            
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ 잔고 확인 오류 ({symbol}): {e}")
            return 0
    
    async def get_account_balance(self) -> dict:
        """계좌 잔고 전체 조회"""
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
            
            headers = self.headers.copy()
            headers["tr_id"] = "VTTC8908R" if config.is_mock else "TTTC8908R"
            
            params = {
                "CANO": config.current_api_config['account_number'][:8],
                "ACNT_PRDT_CD": config.current_api_config['account_number'][8:],
                "PDNO": "005930",  # 임시 종목코드
                "ORD_UNPR": "0",
                "ORD_DVSN": "01",
                "CMA_EVLU_AMT_ICLD_YN": "Y",
                "OVRS_ICLD_YN": "N"
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('rt_cd') == '0':
                    output = result.get('output', {})
                    cash_balance = int(output.get('ORD_PSBL_CASH', 0))
                    self.logger.info(f"💰 현금 잔고: {cash_balance:,}원")
                    return {'cash': cash_balance}
            
            return {'cash': 0}
            
        except Exception as e:
            self.logger.error(f"❌ 계좌 잔고 조회 오류: {e}")
            return {'cash': 0} 