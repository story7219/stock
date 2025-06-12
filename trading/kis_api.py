import requests
import json
import time
from datetime import datetime, timedelta
from utils.logger import log_event
from typing import Literal, Dict, Any, List, Optional, Tuple
import asyncio
import aiohttp

# --- API 상수 정의 ---
# URL 경로
PATH_OAUTH2_TOKEN = "oauth2/tokenP"
PATH_ORDER_CASH = "uapi/domestic-stock/v1/trading/order-cash"
PATH_INQUIRE_PRICE = "uapi/domestic-stock/v1/quotations/inquire-price"
PATH_INQUIRE_BALANCE = "uapi/domestic-stock/v1/trading/inquire-balance"
PATH_RANKING = "uapi/stock/v1/ranking/volume-rank"

# Transaction ID (tr_id)
TR_ID_INQUIRE_PRICE = "FHKST01010100"
TR_ID_ORDER_CASH_MOCK = "VTTC0802U"
TR_ID_ORDER_CASH_LIVE = "TTTC0802U"
TR_ID_INQUIRE_BALANCE_MOCK = "VTTC8434R"
TR_ID_INQUIRE_BALANCE_LIVE = "TTTC8434R" # 실전/모의 동일 (문서 확인 필요)
TR_ID_RANKING = "FHPST01710000"

# 주문 관련 상수
ORDER_SIDE_BUY = "02"
ORDER_SIDE_SELL = "01"
ORDER_TYPE_MARKET = "01" # 시장가

class KIS_API:
    """한국투자증권 REST API 핸들러 (리팩토링)"""
    def __init__(self, app_key: str, app_secret: str, account_number: str, mock: bool = True):
        self.mock = mock
        self.base_url = "https://openapivts.koreainvestment.com:29443" if mock else "https://openapi.koreainvestment.com:9443"
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_number = account_number
        
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        self.cano, self.acnt_prdt_cd = account_number.split('-')

    def _log(self, level: str, message: str):
        """클래스용 로깅 래퍼 함수"""
        log_event(level, f"[KIS_API] {message}")

    def _is_response_ok(self, response: Optional[Dict[str, Any]]) -> bool:
        """API 응답이 성공적인지 확인하는 헬퍼 함수"""
        if not response or response.get("rt_cd") != "0":
            self._log("ERROR", f"API 응답 실패: {response}")
            return False
        return True

    def _send_request(self, method: Literal['GET', 'POST'], path: str, headers: Dict[str, str] = None, 
                      params: Dict[str, Any] = None, body: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """API 요청 전송 내부 함수 (개선)"""
        url = f"{self.base_url}/{path}"
        
        final_headers = {
            "Content-Type": "application/json",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        if self.access_token:
            final_headers["authorization"] = f"Bearer {self.access_token}"
        if headers:
            final_headers.update(headers)

        try:
            if method == 'GET':
                response = requests.get(url, headers=final_headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=final_headers, data=json.dumps(body) if body else None)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            error_text = e.response.text if e.response else "N/A"
            self._log("ERROR", f"요청 실패: {e} - URL: {url}, 응답: {error_text}")
            return None
        except json.JSONDecodeError:
            self._log("ERROR", f"JSON 파싱 오류: {response.text}")
            return None

    def _ensure_token(self) -> bool:
        """토큰 유효성 검사 및 재발급 (개선)"""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return True
        return self._issue_token()

    def _issue_token(self) -> bool:
        """접근 토큰 발급"""
        body = {"grant_type": "client_credentials"}
        res = self._send_request('POST', PATH_OAUTH2_TOKEN, body=body)

        if res and 'access_token' in res:
            self.access_token = res['access_token']
            expires_in = res.get('expires_in', 86400)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300) # 5분 여유
            self._log("INFO", "접근 토큰 발급/갱신 성공.")
            return True
        
        error_details = (
            f"토큰 발급에 실패했습니다 (응답: {res}).\n\n"
            "**[조치 방법]**\n"
            "1. `.env` 파일에 설정된 'LIVE_KIS_APP_KEY'와 'LIVE_KIS_APP_SECRET' 값이 정확한지 다시 한번 확인하세요.\n"
            "2. 프로그램 시작 시 안내된 '공인 IP'가 한국투자증권 개발자 포털에 올바르게 등록되었는지 확인하세요."
        )
        self._log("CRITICAL", error_details)
        return False

    def get_current_price(self, ticker: str) -> Optional[float]:
        """주식 현재가 시세 조회"""
        if not self._ensure_token(): return None

        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
        headers = {"tr_id": TR_ID_INQUIRE_PRICE}
        
        res = self._send_request('GET', PATH_INQUIRE_PRICE, params=params, headers=headers)
        
        if self._is_response_ok(res):
            try:
                price_str = res.get('output', {}).get('stck_prpr')
                return float(price_str)
            except (ValueError, TypeError) as e:
                self._log("ERROR", f"현재가 파싱 오류 ({ticker}): {e}")
        return None

    def create_order(self, side: Literal["buy", "sell"], ticker: str, quantity: int) -> Optional[Dict[str, str]]:
        """주식 시장가 주문 실행 (현금)"""
        if not self._ensure_token(): return None

        side_code = ORDER_SIDE_BUY if side == "buy" else ORDER_SIDE_SELL
        tr_id = TR_ID_ORDER_CASH_MOCK if self.mock else TR_ID_ORDER_CASH_LIVE
        
        headers = {"tr_id": tr_id}
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": ticker,
            "ORD_DVSN": ORDER_TYPE_MARKET,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
        }
        
        self._log("INFO", f"[주문] 계좌:{self.account_number}, 종목:{ticker}, 구분:{side}, 수량:{quantity}")
        res = self._send_request('POST', PATH_ORDER_CASH, headers=headers, body=body)
        
        if self._is_response_ok(res):
            output = res.get('output', {})
            order_id = output.get('KRX_FWDG_ORD_ORGNO', '') + output.get('ODNO', '')
            self._log("INFO", f"주문 접수 성공: {order_id}")
            return {"status": "success", "order_id": order_id}
        return None

    def fetch_ranking_data(self, ranking_type: str) -> List[Dict[str, Any]]:
        """다양한 유형의 시장 순위 정보 조회"""
        if not self._ensure_token(): return []

        fid_map = {
            'volume': '0150', 'gainer': '0152', 'value': '0151',
            'foreign_buy': '0165', 'institution_buy': '0168',
        }
        if ranking_type not in fid_map:
            self._log("ERROR", f"지원하지 않는 랭킹 유형: {ranking_type}")
            return []
            
        headers = {"tr_id": TR_ID_RANKING}
        params = {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": fid_map[ranking_type], "FID_DIV_CLS_CODE": "0",
            "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000", "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "", "FID_VOL_CNT": "", "FID_PB_CLS_CODE": "0"
        }
        
        res = self._send_request('GET', PATH_RANKING, headers=headers, params=params)
        
        if self._is_response_ok(res):
            return res.get('output', [])
        return []

    def fetch_current_balance(self) -> Optional[Dict[str, Any]]:
        """현재 계좌의 잔고 정보 조회"""
        if not self._ensure_token(): return None

        tr_id = TR_ID_INQUIRE_BALANCE_MOCK if self.mock else TR_ID_INQUIRE_BALANCE_LIVE
        headers = {"tr_id": tr_id}
        params = {
            "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
            "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""
        }
        
        res = self._send_request('GET', PATH_INQUIRE_BALANCE, params=params, headers=headers)
        
        if self._is_response_ok(res):
            return res.get('output1', []) # 잔고 정보는 output1에 있음
        return None

    async def async_get_current_price(self, ticker: str, session: aiohttp.ClientSession) -> Tuple[str, Optional[float]]:
        """(비동기) aiohttp를 사용하여 단일 종목의 현재가를 조회합니다."""
        if not self._ensure_token():
            return ticker, None

        url = f"{self.base_url}/{PATH_INQUIRE_PRICE}"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
        headers = {
            "tr_id": TR_ID_INQUIRE_PRICE,
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                res = await response.json()
                if self._is_response_ok(res):
                    price_str = res.get('output', {}).get('stck_prpr')
                    return ticker, float(price_str)
        except (aiohttp.ClientError, ValueError, TypeError) as e:
            self._log("ERROR", f"[비동기 가격조회] {ticker} 처리 중 오류: {e}")
        
        return ticker, None

    async def fetch_prices_in_parallel(self, tickers: List[str]) -> Dict[str, float]:
        """(비동기) 여러 종목의 현재가를 병렬로 조회합니다."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.async_get_current_price(ticker, session) for ticker in tickers]
            results = await asyncio.gather(*tasks)
        
        # None 값을 제외하고 성공한 결과만 딕셔너리로 변환
        price_dict = {ticker: price for ticker, price in results if price is not None}
        
        if len(price_dict) != len(tickers):
            failed_tickers = [ticker for ticker in tickers if ticker not in price_dict]
            self._log("WARNING", f"[병렬 가격조회] 일부 종목 조회 실패: {failed_tickers}")

        return price_dict