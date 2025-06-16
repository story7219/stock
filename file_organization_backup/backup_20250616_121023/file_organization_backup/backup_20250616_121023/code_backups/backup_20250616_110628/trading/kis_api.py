import requests
import json
import time
from datetime import datetime, timedelta
from utils.logger import log_event
from typing import Literal, Dict, Any, List, Optional, Tuple
import asyncio
import aiohttp
import os
from utils.throttle import RateLimiter

class KIS_API:
    """한국투자증권 REST API 통합 핸들러 (동기/비동기 지원)"""
    
    # --- API 상수 정의 ---
    PATH_OAUTH2_TOKEN = "oauth2/tokenP"
    PATH_ORDER_CASH = "uapi/domestic-stock/v1/trading/order-cash"
    PATH_INQUIRE_PRICE = "uapi/domestic-stock/v1/quotations/inquire-price"
    PATH_INQUIRE_BALANCE = "uapi/domestic-stock/v1/trading/inquire-balance"
    PATH_RANKING_FLUCTUATION = "uapi/domestic-stock/v1/quotations/fluctuation-rank"
    PATH_RANKING_INVESTOR = "uapi/domestic-stock/v1/quotations/investor-rank"
    PATH_RANKING_SECTOR = "uapi/domestic-stock/v1/quotations/sector-rank"
    PATH_RANKING_VOLUME = "uapi/domestic-stock/v1/quotations/volume-rank"

    def __init__(self, app_key: str, app_secret: str, account_number: str, mock: bool = True, telegram_bot=None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_number = account_number
        self.mock = mock
        self.base_url = "https://openapi.koreainvestment.com:9000" if mock else "https://openapi.koreainvestment.com:9443"
        
        self.token_info: Optional[Dict[str, Any]] = None
        
        if account_number:
            self.cano, self.acnt_prdt_cd = account_number.split('-')
        else:
            self.cano, self.acnt_prdt_cd = None, None
        
        self.rate_limiter = RateLimiter(10)  # 1초 10건 제한 (개인 기준)
        self.telegram_bot = telegram_bot  # 텔레그램 봇 인스턴스 저장
        
    def _get_headers(self, tr_id: str) -> Dict[str, str]:
        """API 호출용 공통 헤더 생성"""
        self._ensure_token()
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token_info['access_token']}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }

    def _ensure_token(self):
        """토큰 유효성 검사 및 필요 시 자동 발급"""
        if self.token_info and datetime.now() < self.token_info['expires_at']:
            return
        
        url = f"{self.base_url}/{self.PATH_OAUTH2_TOKEN}"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            res = requests.post(url, headers=headers, json=body)
            res.raise_for_status()
            data = res.json()
            
            expires_in = int(data.get('expires_in', 86400))
            self.token_info = {
                'access_token': data['access_token'],
                'expires_at': datetime.now() + timedelta(seconds=expires_in - 300) # 5분 여유
            }
            log_event("INFO", "✅ [KIS_API] 접근 토큰 발급/갱신 성공.")
            if self.telegram_bot:
                self.telegram_bot.send_message("✅ [KIS_API] 접근 토큰 발급/갱신 성공")
        except requests.exceptions.RequestException as e:
            log_event("CRITICAL", f"🔥 [KIS_API] 토큰 발급 실패: {e.response.text if e.response else e}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"🔥 [KIS_API] 토큰 발급 실패: {e.response.text if e.response else e}")
            raise

    def get_current_price(self, ticker: str) -> Optional[float]:
        """주식 현재가 조회"""
        tr_id = "FHKST01010100"
        url = f"{self.base_url}/{self.PATH_INQUIRE_PRICE}"
        headers = self._get_headers(tr_id)
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": ticker}
        
        try:
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()
            data = res.json()
            if data.get('rt_cd') == '0':
                return float(data['output']['stck_prpr'])
            log_event("WARNING", f"⚠️ [KIS_API] 현재가 조회 실패({ticker}): {data.get('msg1')}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"⚠️ [KIS_API] 현재가 조회 실패({ticker}): {data.get('msg1')}")
            return None
        except Exception as e:
            log_event("ERROR", f"🔥 [KIS_API] 현재가 조회 중 오류({ticker}): {e}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"🔥 [KIS_API] 현재가 조회 중 오류({ticker}): {e}")
            return None
            
    # --- 비동기 병렬 조회 ---
    async def _fetch_price_async(self, ticker: str, session: aiohttp.ClientSession) -> Tuple[str, Optional[float]]:
        """단일 종목 현재가 비동기 조회"""
        await self.rate_limiter.acquire()  # Throttle 적용
        tr_id = "FHKST01010100"
        url = f"{self.base_url}/{self.PATH_INQUIRE_PRICE}"
        headers = self._get_headers(tr_id)
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": ticker}
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get('rt_cd') == '0':
                    return ticker, float(data['output']['stck_prpr'])
                return ticker, None
        except Exception:
            return ticker, None

    async def fetch_prices_in_parallel(self, tickers: List[str]) -> Dict[str, float]:
        """여러 종목의 현재가를 병렬로 빠르게 조회"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_price_async(ticker, session) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            
        price_dict = {ticker: price for ticker, price in results if price is not None}
        if len(price_dict) != len(tickers):
            failed = len(tickers) - len(price_dict)
            log_event("WARNING", f"⚠️ [KIS_API] {failed}개 종목 병렬 조회 실패.")
            
        return price_dict

    def inquire_balance(self):
        """잔고조회 예시 (실패시 알림)"""
        tr_id = "VTTC8434R" if self.mock else "TTTC8434R"
        url = f"{self.base_url}/{self.PATH_INQUIRE_BALANCE}"
        headers = self._get_headers(tr_id)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "UNPR_DVSN_CD": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN_CD": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        try:
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()
            data = res.json()
            if data.get('rt_cd') != '0':
                log_event("ERROR", f"🔥 [KIS_API] 잔고조회 실패: {data.get('msg1')}")
                if self.telegram_bot:
                    self.telegram_bot.send_message(f"🔥 [KIS_API] 잔고조회 실패: {data.get('msg1')}")
            return data
        except Exception as e:
            log_event("CRITICAL", f"🔥 [KIS_API] 잔고조회 오류: {e}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"🔥 [KIS_API] 잔고조회 오류: {e}")
            return None

    def create_order(self, side: Literal["buy", "sell"], ticker: str, quantity: int) -> Optional[Dict[str, str]]:
        """주식 시장가 주문 실행 (현금)"""
        if side == "buy":
            tr_id = "VTTC0802U" if self.mock else "TTTC0802U" # 매수
        else: # sell
            tr_id = "VTTC0801U" if self.mock else "TTTC0801U" # 매도

        url = f"{self.base_url}/{self.PATH_ORDER_CASH}"
        headers = self._get_headers(tr_id)
        
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": ticker,
            "ORD_DVSN": "01",  # 시장가
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
        }
        
        log_event("INFO", f"📦 [주문] {side.upper()} | {ticker} | {quantity}주")
        if self.telegram_bot:
            self.telegram_bot.send_message(f"📦 [주문] {side.upper()} | {ticker} | {quantity}주")
        
        try:
            res = requests.post(url, headers=headers, json=body)
            res.raise_for_status()
            data = res.json()
            
            if data.get('rt_cd') == '0':
                output = data.get('output', {})
                order_id = output.get('KRX_FWDG_ORD_ORGNO', '') + output.get('ODNO', '')
                log_event("SUCCESS", f"✅ [주문 성공] ID: {order_id}")
                if self.telegram_bot:
                    self.telegram_bot.send_message(f"✅ [주문 성공] {side.upper()} | {ticker} | {quantity}주 | 주문ID: {order_id}")
                return {"status": "success", "order_id": order_id}
            
            log_event("ERROR", f"🔥 [주문 실패] ({ticker}): {data.get('msg1')}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"🔥 [주문 실패] {side.upper()} | {ticker} | {quantity}주 | 사유: {data.get('msg1')}")
            return {"status": "fail", "message": data.get('msg1')}
        except Exception as e:
            log_event("CRITICAL", f"🔥 [주문 오류] ({ticker}): {e}")
            if self.telegram_bot:
                self.telegram_bot.send_message(f"🔥 [주문 오류] {side.upper()} | {ticker} | {quantity}주 | 오류: {e}")
            return None

    async def throttled_order(self, side, ticker, quantity):
        await self.rate_limiter.acquire()
        return self.create_order(side, ticker, quantity)

    # 여기에 다른 API 호출 함수(잔고조회 등)들을 추가할 수 있습니다.