"""
💎 핵심 트레이딩 엔진 (v7.0, 완전 비동기)
- 모든 API 통신을 httpx 기반의 비동기 방식으로 전환하여 성능 극대화
- 비동기 초기화 로직 도입으로 안정적인 시스템 시작 보장
- 각 API 종류별 비동기 레이트 리미터 적용
- 실시간 WebSocket 승인키 발급 또한 비동기로 전환
"""
import asyncio
import json
import logging
import time
import threading
import base64
from datetime import datetime, timedelta, time as time_obj
from collections import deque
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import os

import gspread
import websocket
from google.oauth2.service_account import Credentials
import httpx
import pandas_ta as ta # pandas-ta 임포트

from telegram_wrapper import TelegramNotifierWrapper
import config
from google_sheet_logger import GoogleSheetLogger
from abc import ABC, abstractmethod
import pandas as pd

logger = logging.getLogger(__name__)

# 데이터 모델 정의
class OrderSide(Enum): ...
@dataclass
class PriceData: ...
@dataclass
class BalanceInfo: ...

# === 유틸리티 클래스 (비동기) ===
class AsyncRateLimiter:
    __slots__ = ('calls', 'period', 'call_times', '_lock')
    def __init__(self, calls: int, period: int):
        self.calls, self.period, self.call_times, self._lock = calls, period, deque(), asyncio.Lock()
    async def __aenter__(self):
        async with self._lock:
            now = time.monotonic()
            while self.call_times and self.call_times[0] <= now - self.period: self.call_times.popleft()
            if len(self.call_times) >= self.calls:
                if (sleep_time := self.call_times[0] - (now - self.period)) > 0:
                    await asyncio.sleep(sleep_time)
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock: self.call_times.append(time.monotonic())

class OptimizedDailyApiCounter:
    __slots__ = ('daily_limit', 'counter_file', 'today_count', 'last_reset_date', '_lock')
    def __init__(self, daily_limit: Optional[int]):
        self.daily_limit, self.counter_file = daily_limit, "daily_api_count.json"
        self.today_count, self.last_reset_date, self._lock = 0, None, asyncio.Lock()
        self._load_counter()
    def _load_counter(self):
        try:
            if not os.path.exists(self.counter_file): self._reset_counter(); return
            with open(self.counter_file, 'r', encoding='utf-8') as f: data = json.load(f)
            self.today_count, self.last_reset_date = data.get('count', 0), data.get('date')
            if self.last_reset_date != datetime.now().strftime('%Y-%m-%d'): self._reset_counter()
        except Exception: self._reset_counter()
    def _reset_counter(self):
        self.today_count, self.last_reset_date = 0, datetime.now().strftime('%Y-%m-%d')
        self._save_counter()
    def _save_counter(self):
        try:
            with open(self.counter_file, 'w', encoding='utf-8') as f: json.dump({'count': self.today_count, 'date': self.last_reset_date}, f)
        except Exception as e: logger.error(f"❌ API 카운터 저장 실패: {e}")
    async def can_make_request(self) -> bool:
        if self.daily_limit is None: return True
        async with self._lock: return self.today_count < self.daily_limit
    async def increment(self):
        async with self._lock:
            self.today_count += 1
            if self.today_count % 10 == 0: await asyncio.to_thread(self._save_counter)
    async def get_remaining_calls(self) -> Union[int, float]:
        if self.daily_limit is None: return float('inf')
        async with self._lock: return max(0, self.daily_limit - self.today_count)

class OptimizedTokenManager:
    __slots__ = ('base_url', 'app_key', 'app_secret', 'limiter', 'token_file', '_token_cache', '_lock')
    def __init__(self, base_url, app_key, app_secret, limiter):
        self.base_url, self.app_key, self.app_secret, self.limiter = base_url, app_key, app_secret, limiter
        self.token_file, self._token_cache, self._lock = "kis_token.json", None, asyncio.Lock()
    def _save_token(self, token_data):
        try:
            token_data['expires_at'] = (datetime.now() + timedelta(seconds=token_data['expires_in'] - 600)).isoformat()
            with open(self.token_file, 'w', encoding='utf-8') as f: json.dump(token_data, f)
            self._token_cache = token_data
        except Exception as e: logger.error(f"❌ 토큰 저장 실패: {e}")
    def _load_token(self):
        try:
            if not os.path.exists(self.token_file): return None
            with open(self.token_file, 'r', encoding='utf-8') as f: token_data = json.load(f)
            if 'expires_at' in token_data and datetime.now() < datetime.fromisoformat(token_data['expires_at']): return token_data
            return None
        except Exception: return None
    async def _issue_new_token(self, client: httpx.AsyncClient):
        url, data = f"{self.base_url}/oauth2/tokenP", {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        try:
            async with self.limiter:
                response = await client.post(url, json=data); response.raise_for_status()
            new_token_data = response.json(); self._save_token(new_token_data)
            return new_token_data
        except Exception as e: logger.error(f"❌ 토큰 발급 중 예외: {e}", exc_info=True); return None
    async def get_valid_token(self, client: httpx.AsyncClient) -> Optional[str]:
        async with self._lock:
            now = datetime.now()
            if self._token_cache and now < datetime.fromisoformat(self._token_cache['expires_at']): return self._token_cache.get('access_token')
            loaded_token = self._load_token()
            if loaded_token and now < datetime.fromisoformat(loaded_token['expires_at']):
                self._token_cache = loaded_token; return loaded_token.get('access_token')
            new_token = await self._issue_new_token(client)
            return new_token.get('access_token') if new_token else None

# === 🏦 완전 비동기 핵심 거래 클래스 ===
class CoreTrader:
    def __init__(self, sheet_logger: Optional[GoogleSheetLogger] = None):
        self._load_configuration(); self.notifier = TelegramNotifierWrapper()
        self.worksheet = None; self.order_limiter = None; self.market_data_limiter = None; self.account_limiter = None
        self.global_limiter = None; self.daily_counter = None; self.token_manager = None; self.http_client = None
        self.stock_info_cache = {}
        self.sheet_logger = sheet_logger
        self._initialize_websocket_components()

    async def async_initialize(self) -> bool:
        logger.info("🔧 CoreTrader v7.0 비동기 초기화를 시작합니다...")
        self.http_client = httpx.AsyncClient(timeout=10)
        self.worksheet = await asyncio.to_thread(self._initialize_gspread)
        self._initialize_rate_limiters_async()
        self.token_manager = OptimizedTokenManager(self.base_url, self.app_key, self.app_secret, self.global_limiter)
        self.daily_counter = OptimizedDailyApiCounter(config.DAILY_API_LIMIT)
        token = await self.token_manager.get_valid_token(self.http_client)
        if not token:
            logger.error("❌ 유효한 API 토큰 획득 실패."); await self.close(); return False
        await self._log_initialization_status(); return True

    async def close(self):
        if self.http_client: await self.http_client.aclose(); logger.info("🔌 HTTP 클라이언트 종료됨.")

    def _load_configuration(self):
        self.app_key, self.app_secret = config.KIS_APP_KEY, config.KIS_APP_SECRET
        account_no_raw, self.base_url, self.is_mock = config.KIS_ACCOUNT_NO, config.KIS_BASE_URL, config.IS_MOCK
        if not all([self.app_key, self.app_secret, account_no_raw, self.base_url]): raise ValueError("❌ KIS 설정값 누락.")
        
        # 계좌번호 포맷팅 로직 추가
        try:
            parts = account_no_raw.split('-')
            if len(parts) == 2:
                self.account_no = f"{parts[0]}-{parts[1].zfill(2)}"
                logger.info(f"계좌번호 포맷팅: {account_no_raw} -> {self.account_no}")
            else:
                raise ValueError("계좌번호 형식이 올바르지 않습니다 (예: 12345678-01).")
        except Exception as e:
            raise ValueError(f"❌ 계좌번호 처리 중 오류 발생: {e}")

    def _initialize_rate_limiters_async(self):
        self.order_limiter = AsyncRateLimiter(config.ORDER_API_CALLS_PER_SEC, 1)
        self.market_data_limiter = AsyncRateLimiter(config.MARKET_DATA_API_CALLS_PER_SEC, 1)
        self.account_limiter = AsyncRateLimiter(config.ACCOUNT_API_CALLS_PER_SEC, 1)
        self.global_limiter = AsyncRateLimiter(config.TOTAL_API_CALLS_PER_SEC, 1)

    async def _log_initialization_status(self):
        logger.info(f"✅ CoreTrader 초기화 완료 ({'모의투자' if self.is_mock else '실전투자'})")
        logger.info(f"📈 일일 한도: {self.daily_counter.today_count}회 사용, {await self.daily_counter.get_remaining_calls()}회 남음")

    def _initialize_gspread(self):
        # ... (기존 gspread 초기화 로직과 동일) ...
        return None # 임시

    async def _send_request_async(self, method, path, limiter, headers=None, params=None, json_data=None):
        if not await self.daily_counter.can_make_request(): logger.error("❌ 일일 API 한도 초과"); return None
        token = await self.token_manager.get_valid_token(self.http_client)
        if not token: logger.error("❌ 유효한 토큰 없음"); return None
        req_headers = {"Content-Type":"application/json", "authorization":f"Bearer {token}", "appkey":self.app_key, "appsecret":self.app_secret}
        if headers: req_headers.update(headers)
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        async with self.global_limiter, limiter:
            for attempt in range(config.MAX_RETRY_ATTEMPTS):
                try:
                    await self.daily_counter.increment()
                    response = await self.http_client.request(method, url, headers=req_headers, params=params, json=json_data)
                    response.raise_for_status(); result = response.json()
                    if isinstance(result, dict) and result.get('rt_cd', '1') != '0':
                        logger.warning(f"⚠️ API 응답 오류 [{path}]: {result.get('rt_cd')} - {result.get('msg1', 'Unknown')}"); return None
                    return result
                except (httpx.HTTPStatusError, httpx.RequestError) as e: logger.warning(f"⚠️ API 요청 실패 [{path}]: {e} (시도 {attempt+1})")
                except json.JSONDecodeError: logger.error(f"❌ JSON 파싱 실패 [{path}]"); return None
                if attempt < config.MAX_RETRY_ATTEMPTS - 1: await asyncio.sleep(config.RETRY_DELAY_SECONDS)
        logger.error(f"❌ API 요청 최종 실패: {path}"); return None

    async def get_current_price(self, symbol: str) -> Optional[Dict]:
        res = await self._send_request_async("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", self.market_data_limiter,
            headers={"tr_id": "FHKST01010100"}, params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol})
        if not res or 'output' not in res: return None
        out = res['output']; return {'price':int(out.get('stck_prpr',0)), 'symbol':symbol, 'change_rate':float(out.get('prdy_ctrt',0.0)), 'volume':int(out.get('acml_vol',0))}

    async def fetch_ranking_data(self, ranking_type: str, limit: int = 50) -> Optional[List[Dict]]:
        """
        다양한 주식 순위 정보를 비동기적으로 조회합니다.
        ranking_type: 'rise', 'volume', 'value', 'institution_net_buy', 'foreign_net_buy'
        """
        # API 경로, tr_id, 기본 파라미터를 랭킹 타입별로 매핑
        configs = {
            "rise": {
                "path": "/uapi/domestic-stock/v1/quotations/volume-rank",
                "tr_id": "FHPST01710000",
                "params": {"FID_INPUT_ISCD": "0002"}, # 상승률
                "output_key": "output1"
            },
            "volume": {
                "path": "/uapi/domestic-stock/v1/quotations/volume-rank",
                "tr_id": "FHPST01710000",
                "params": {"FID_INPUT_ISCD": "0001"}, # 거래량
                "output_key": "output1"
            },
            "value": {
                "path": "/uapi/domestic-stock/v1/quotations/volume-rank",
                "tr_id": "FHPST01710000",
                "params": {"FID_INPUT_ISCD": "0004"}, # 거래대금
                "output_key": "output1"
            },
            "institution_net_buy": {
                "path": "/uapi/domestic-stock/v1/quotations/inquire-investor-rank",
                "tr_id": "FHPST01720000",
                "params": {"FID_INPUT_ISCD1": "001", "FID_RANK_SORT_CLS_CODE": "0"}, # 기관 순매수
                "output_key": "output"
            },
            "foreign_net_buy": {
                "path": "/uapi/domestic-stock/v1/quotations/inquire-investor-rank",
                "tr_id": "FHPST01720000",
                "params": {"FID_INPUT_ISCD1": "002", "FID_RANK_SORT_CLS_CODE": "0"}, # 외국인 순매수
                "output_key": "output"
            }
        }

        if ranking_type not in configs:
            logger.error(f"❌ 지원되지 않는 순위 타입: {ranking_type}")
            return None

        config = configs[ranking_type]
        
        base_params = {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "0", "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "000000000",
            "FID_INPUT_PRICE_1": "", "FID_INPUT_PRICE_2": "", "FID_VOL_CNT": str(limit)
        }
        
        # 각 타입에 맞는 파라미터로 업데이트 (누락되었던 부분 수정)
        if "params" in config:
            base_params.update(config["params"])
        
        # 'volume-rank' API는 FID_INPUT_ISCD 값을 파라미터에서 직접 받아서 설정해야 함
        # 이 부분이 없으면 항상 기본값("0000")으로 요청되어 오류 발생
        if config["path"].endswith("volume-rank"):
            if ranking_type == 'rise':
                base_params["FID_INPUT_ISCD"] = "0002"
            elif ranking_type == 'volume':
                base_params["FID_INPUT_ISCD"] = "0001"
            elif ranking_type == 'value':
                base_params["FID_INPUT_ISCD"] = "0004"

        res = await self._send_request_async(
            "GET", 
            config["path"],
            self.market_data_limiter,
            headers={"tr_id": config["tr_id"]},
            params=base_params
        )

        output_key = config["output_key"]
        if not res or output_key not in res:
            logger.warning(f"⚠️ {ranking_type} 순위 데이터 조회 실패")
            return None
            
        return res[output_key]

    async def get_balance(self) -> Optional[Dict]:
        tr_id, parts = ("VTTC8434R" if self.is_mock else "TTTC8434R"), self.account_no.split('-')
        params = {
            "CANO": parts[0],
            "ACNT_PRDT_CD": parts[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00"
        }
        # 모의투자 계좌는 상품코드를 보내면 오류를 반환하는 경우가 있음
        if self.is_mock:
            del params["ACNT_PRDT_CD"]
            
        return await self._send_request_async("GET", "/uapi/domestic-stock/v1/trading/inquire-balance", self.account_limiter, headers={"tr_id": tr_id}, params=params)

    async def execute_order(self, symbol: str, order_type: str, quantity: int, price: int = 0, order_condition: str = "0", log_payload: Optional[Dict] = None) -> bool:
        """지정된 조건으로 주문을 실행하고, 성공 시 구글 시트에 로그를 남깁니다."""
        if not self.is_real_trading:
            logger.info(f"🧪 [모의 거래] {order_type.upper()}: {symbol}, {quantity}주 at {price if price > 0 else '시장가'}")
            if self.sheet_logger:
                await self._log_trade_to_sheet(symbol, order_type, quantity, price, log_payload)
            return True

        if not self.daily_counter.can_make_request():
            logger.error("❌ 일일 API 한도 초과"); return False

        req = self._build_order_request(symbol, order_type.upper(), quantity, price)
        if not req: return False
        res = await self._send_request_async("POST", "/uapi/domestic-stock/v1/trading/order-cash", self.order_limiter, headers=req['headers'], json_data=req['json_data'])
        result = self._process_order_response(res, symbol, order_type, quantity, price)
        success = result and result.get('success', False)
        if success:
            await self._log_trade_to_sheet(symbol, order_type, quantity, price, log_payload)
            await self._send_order_notification(result)
        else:
            self._log_failed_order(symbol, order_type, quantity, result)
        return success

    def _build_order_request(self, symbol, side, quantity, price):
        parts = self.account_no.split('-'); side_map = {'BUY':'VTTC0802U', 'SELL':'VTTC0801U'} if self.is_mock else {'BUY':'TTTC0802U', 'SELL':'TTTC0801U'}
        return {"headers":{"tr_id":side_map[side]}, "json_data":{"CANO":parts[0],"ACNT_PRDT_CD":parts[1],"PDNO":symbol,"ORD_DVSN":"01" if price==0 else "00","ORD_QTY":str(quantity),"ORD_UNPR":str(int(price))}}

    def _process_order_response(self, response, symbol, side, quantity, price):
        if not response or 'output' not in response: return {'success': False, 'message': 'API 응답 없음'}
        success = response.get('rt_cd') == '0'
        return {'success':success, 'symbol':symbol, 'side':side, 'quantity':quantity, 'price':price, 'message':response.get('msg1'), 'timestamp':datetime.now().isoformat()}

    async def _log_trade_to_sheet(self, symbol: str, order_type: str, quantity: int, price: int, log_payload: Optional[Dict]):
        """매매 상세 정보를 구글 시트에 로깅하는 내부 헬퍼 메서드"""
        if not self.sheet_logger:
            return
        try:
            stock_name = await self.get_stock_name(symbol)
            order_price = price if price > 0 else (await self.get_current_price(symbol)).get('price', 0)

            trade_details = {
                'symbol': symbol,
                'name': stock_name,
                'order_type': order_type,
                'quantity': quantity,
                'price': order_price,
                'total_amount': order_price * quantity,
                'reason': log_payload.get('reason', '') if log_payload else '',
                'ai_comment': log_payload.get('status', '') if log_payload else '',
                'pnl_percent': log_payload.get('pnl_percent', ''),
                'realized_pnl': log_payload.get('realized_pnl', '')
            }
            # 구글 시트 I/O는 블로킹 작업이므로, 비동기 이벤트 루프를 막지 않도록 to_thread 사용
            await asyncio.to_thread(self.sheet_logger.log_trade, trade_details)
        except Exception as e:
            logger.error(f"⚠️ 구글 시트 로깅 중 예외 발생: {e}", exc_info=True)

    async def _send_order_notification(self, result: Dict):
        await self.notifier.send_message(f"📈 주문 완료: {result['side']} {result['symbol']} {result['quantity']}주")

    def _log_failed_order(self, symbol, side, quantity, result):
        logger.error(f"❌ [{symbol}] 주문 실패: {result.get('msg1', '알 수 없는 오류')}")
        message = (f"🚨 [주문 실패] 🚨\n"
                   f"종목: {symbol}\n"
                   f"수량: {quantity}주\n"
                   f"사유: {result.get('msg1', 'API 오류')}")
        self.notifier.send_message(message)

    # Websocket 및 기타 헬퍼들은 기존 로직 유지
    def _initialize_websocket_components(self): ...
    async def _get_ws_approval_key(self) -> Optional[str]: ...
    # ... 이하 생략 ...

    async def fetch_stocks_by_sector(self, sector_code: str) -> Optional[List[Dict]]:
        """
        특정 업종 코드에 속한 모든 종목의 리스트를 비동기적으로 조회합니다.
        """
        params = {
            "fid_input_iscd": sector_code,
        }
        res = await self._send_request_async(
            "GET", 
            "/uapi/domestic-stock/v1/quotations/inquire-asct-rec-item",
            self.market_data_limiter,
            headers={"tr_id": "FHKST03010200"}, # 업종별 구성종목
            params=params
        )
        if not res or 'output1' not in res:
            logger.warning(f"⚠️ 업종({sector_code})별 종목 조회 실패")
            return None
        return res['output1']

    async def fetch_news_headlines(self, symbol: str) -> Optional[List[Dict]]:
        """
        특정 종목에 대한 최신 뉴스/공시 헤드라인을 조회합니다.
        """
        params = {
            "fid_input_iscd": symbol,
            "fid_input_date_1": (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
            "fid_input_date_2": datetime.now().strftime('%Y%m%d'),
            "fid_div_code": "0", # 0: 전체
        }
        res = await self._send_request_async(
            "GET", 
            "/uapi/domestic-stock/v1/quotations/news-headline",
            self.market_data_limiter,
            headers={"tr_id": "FHPST04010000"},
            params=params
        )
        if not res or 'output' not in res:
            logger.warning(f"⚠️ 종목({symbol}) 뉴스 헤드라인 조회 실패")
            return None
        return res['output']

    async def fetch_daily_price_history(self, symbol: str, days_to_fetch: int = 100) -> Optional[List[Dict]]:
        """
        특정 종목의 일봉 데이터를 지정된 기간만큼 조회합니다.
        """
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
            "fid_org_adj_prc": "1", # 1: 수정주가
            "fid_period_div_code": "D", # D: 일봉
        }
        res = await self._send_request_async(
            "GET", 
            "/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            self.market_data_limiter,
            headers={"tr_id": "FHKST01010400"},
            params=params
        )
        if not res or 'output' not in res:
            logger.warning(f"⚠️ 종목({symbol}) 일봉 데이터 조회 실패")
            return None
        
        # API는 오래된 데이터부터 반환하므로, 최신 데이터 순으로 뒤집어줍니다.
        price_history = reversed(res['output']) 
        
        # 필요한 만큼만 잘라서 반환
        return list(price_history)[:days_to_fetch]

    async def fetch_minute_price_history(self, symbol: str) -> Optional[List[Dict]]:
        """
        특정 종목의 당일 분봉 데이터를 조회합니다. (1분 단위)
        """
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
            "fid_org_adj_prc": "1", # 1: 수정주가
            "fid_etc_cls_code": "1", # 1: 시간외포함
        }
        res = await self._send_request_async(
            "GET", 
            "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            self.market_data_limiter,
            headers={"tr_id": "FHKST01010600"},
            params=params
        )
        if not res or 'output1' not in res:
            logger.warning(f"⚠️ 종목({symbol}) 분봉 데이터 조회 실패")
            return None
        
        # API는 오래된 데이터부터 반환하므로, 최신 데이터 순으로 뒤집어줍니다.
        return list(reversed(res['output1']))

    async def fetch_investor_trading_trends(self, market: str = "KOSPI") -> Optional[List[Dict]]:
        """
        시장별(코스피/코스닥) 투자자 매매 동향을 조회합니다.
        """
        params = {
            "fid_input_iscd_1": "001" if market.upper() == "KOSPI" else "101", # 001: KOSPI, 101: KOSDAQ
            "fid_input_iscd_2": "0000", # 0000: 전체 투자자
            "fid_input_date_1": datetime.now().strftime('%Y%m%d'),
            "fid_input_date_2": datetime.now().strftime('%Y%m%d'),
        }
        res = await self._send_request_async(
            "GET", 
            "/uapi/domestic-stock/v1/quotations/inquire-investor-trend",
            self.market_data_limiter,
            headers={"tr_id": "FHPST01730000"},
            params=params
        )
        if not res or 'output' not in res:
            logger.warning(f"⚠️ {market} 투자자 매매 동향 조회 실패")
            return None
        return res['output']

    async def get_stock_name(self, symbol: str) -> str:
        """종목 코드로 종목명을 조회하고 캐시합니다."""
        if symbol in self.stock_info_cache:
            return self.stock_info_cache[symbol]
        
        url = "/uapi/domestic-stock/v1/quotations/search-info"
        params = {"BNSN_DATE": datetime.now().strftime('%Y%m%d'), "INPT_KEYB": symbol}
        
        try:
            # _send_request_async는 성공 시 dict, 실패 시 None을 반환
            data = await self._send_request_async("GET", url, self.market_data_limiter, params=params)
            
            # 응답이 유효한지 확인
            if data and isinstance(data.get('output1'), list) and data['output1']:
                stock_name = data['output1'][0].get('shot_iss_name') or data['output1'][0].get('prdt_name')
                if stock_name:
                    self.stock_info_cache[symbol] = stock_name
                    return stock_name
            else:
                logger.warning(f"종목명 조회 응답에 유효한 데이터가 없습니다 ({symbol}). Response: {data}")

        except Exception as e:
            logger.error(f"종목명 조회 중 예외 발생 ({symbol}): {e}", exc_info=True)
            
        return "N/A"

    async def fetch_historical_data(self, symbol: str, days: int = 100) -> List[Dict]:
        """
        지정된 기간 동안의 일봉 데이터를 조회합니다. (기본 100일)
        
        :param symbol: 종목 코드
        :param days: 조회할 거래일 수
        :return: 일봉 데이터 리스트
        """
        logger.info(f"🔍 [{symbol}] 최근 {days}일 일봉 데이터 조회...")
        res = await self._send_request_async(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            self.market_data_limiter,
            headers={"tr_id": "FHKST01010400"},
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol,
                "fid_period_div_code": "D",
                "fid_org_adj_prc": "1", # 수정주가 반영
            },
        )
        if res and res.get('output'):
            # API는 최대 30개만 반환하므로, 필요 시 여러 번 호출해야 하지만
            # 현재 구조에서는 기술적 분석을 위해 한 번의 호출로 충분한 데이터를 얻도록 시도
            return res['output'][:days]
        return []

    async def fetch_index_price_history(self, symbol: str, days_to_fetch: int = 100) -> Optional[List[Dict]]:
        """
        특정 지수(업종)의 일봉 데이터를 지정된 기간만큼 조회합니다. (KIS API 기반)
        :param symbol: 업종 코드 (코스피: 0001, 코스닥: 1001)
        :param days_to_fetch: 조회할 거래일 수
        """
        params = {
            "fid_cond_mrkt_div_code": "U",  # U: 업종
            "fid_input_iscd": symbol,
            "fid_period_dvsn_code": "D",    # D: 일
            "fid_orgn_adj_prc": "1",
        }
        res = await self._send_request_async(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-daily-index",
            self.market_data_limiter,
            headers={"tr_id": "FHKUP03010100"},
            params=params
        )

        if not res or 'output1' not in res:
            logger.warning(f"⚠️ 지수({symbol}) 일봉 데이터 조회 실패")
            return None

        price_history = reversed(res['output1'])
        return list(price_history)[:days_to_fetch]

    async def fetch_sector_ranking(self, market_type: str = "prd_thema") -> Optional[List[Dict]]:
        """
        업종/테마별 순위 정보를 비동기적으로 조회합니다.
        :param market_type: "prd_upjong" (업종) 또는 "prd_thema" (테마)
        """
        res = await self._send_request_async(
            "GET",
            "/uapi/domestic-stock/v1/quotations/psearch-result",
            self.market_data_limiter,
            headers={"tr_id": "HHKST03010300"},
            params={"USER_ID": "HHKST03010300", "PROD_ID": market_type}
        )
        if not res or 'output' not in res:
            logger.warning(f"⚠️ 테마/업종 순위({market_type}) 데이터 조회 실패")
            return None
        return res['output']

    async def fetch_stocks_by_theme(self, theme_id: str) -> Optional[List[Dict]]:
        """
        특정 테마에 속한 모든 종목의 리스트를 비동기적으로 조회합니다.
        :param theme_id: `fetch_sector_ranking`에서 얻은 테마 ID
        """
        res = await self._send_request_async(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-search-thema-stock",
            self.market_data_limiter,
            headers={"tr_id": "HHKST03010400"},
            params={"USER_ID": "HHKST03010400", "THEME_ID": theme_id}
        )
        if not res or 'output' not in res:
            logger.warning(f"⚠️ 테마({theme_id})별 종목 조회 실패")
            return None
        return res['output']

    async def fetch_detailed_investor_trends(self, symbol: str) -> Optional[Dict[str, int]]:
        """
        특정 종목의 당일 상세 투자자별 순매수 수량(연기금, 사모펀드 등)을 조회합니다.
        
        :param symbol: 종목 코드
        :return: 투자자별 순매수 수량 딕셔너리 또는 실패 시 None
        """
        logger.info(f"🔍 [{symbol}] 상세 투자자별 매매 동향 조회...")
        
        res = await self._send_request_async(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            self.market_data_limiter,
            headers={"tr_id": "FHKST01010900"}, # 주식현재가 투자자
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol
            }
        )
        
        if not res or 'output' not in res or not res['output']:
            logger.warning(f"⚠️ [{symbol}] 상세 투자자 동향 데이터가 없습니다.")
            return None
            
        trends_data = res['output']
        
        # 주요 투자 주체별 순매수량(수량)을 추출하여 구조화합니다.
        try:
            trends = {
                "pension_fund": int(trends_data.get('pns_ntby_qty', 0)),         # 연기금
                "private_equity": int(trends_data.get('pbid_ntby_qty', 0)),       # 사모펀드
                "insurance": int(trends_data.get('insu_ntby_qty', 0)),            # 보험
                "investment_trust": int(trends_data.get('ivst_ntby_qty', 0)),     # 투신
                "bank": int(trends_data.get('bnk_ntby_qty', 0)),                   # 은행
                "other_financial": int(trends_data.get('fina_ntby_qty', 0)),      # 기타금융
                "other_corporations": int(trends_data.get('corp_ntby_qty', 0)),   # 기타법인
                "program": int(trends_data.get('prgm_net_buy_qty', 0)),         # 프로그램
                "foreign": int(trends_data.get('frgn_ntby_qty', 0)),              # 외국인
                "institution": int(trends_data.get('inst_ntby_qty', 0)),          # 기관계
                "individual": int(trends_data.get('ant_ntby_qty', 0))             # 개인
            }
            logger.info(f"✅ [{symbol}] 상세 투자자 동향 수집 완료.")
            return trends
        except Exception as e:
            logger.error(f"💥 [{symbol}] 상세 투자자별 동향 조회 중 예기치 않은 오류: {e}", exc_info=True)
            return None

    async def get_current_price_info(self, symbol: str) -> Optional[Dict]:
        """
        종목의 현재 가격 정보를 조회합니다.
        
        :param symbol: 종목 코드
        :return: 종목의 현재 가격 정보 또는 None
        """
        res = await self._send_request_async("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", self.market_data_limiter,
            headers={"tr_id": "FHKST01010100"}, params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol})
        if not res or 'output' not in res:
            logger.warning(f"⚠️ 종목({symbol}) 현재가 조회 실패")
            return None
        out = res['output']; return {'price':int(out.get('stck_prpr',0)), 'symbol':symbol, 'change_rate':float(out.get('prdy_ctrt',0.0)), 'volume':int(out.get('acml_vol',0))}

    async def get_technical_indicators(self, symbol: str, days: int = 200) -> Optional[pd.DataFrame]:
        """
        일봉 데이터를 기반으로 주요 기술적 지표를 계산합니다.
        (현재 numpy 호환성 문제로 임시 비활성화)
        """
        logger.warning(f"[{symbol}] 기술적 지표 계산 기능이 임시 비활성화되었습니다.")
        return None
        
        # 원본 코드 (임시 주석 처리)
        # try:
        #     logger.info(f"[{symbol}] KIS API로 일봉 데이터 조회 (기간: {days}일)...")
        #     daily_chart = await self.fetch_daily_price_history(symbol, days_to_fetch=days)
        #     if daily_chart is None or daily_chart.empty:
        #         logger.warning(f"[{symbol}] 기술적 지표 계산을 위한 일봉 데이터가 없습니다.")
        #         return None
        #
        #     # pandas-ta를 사용하여 기술적 지표 추가
        #     daily_chart.ta.ema(length=5, append=True)
        #     daily_chart.ta.ema(length=20, append=True)
        #     daily_chart.ta.ema(length=60, append=True)
        #     daily_chart.ta.ichimoku(append=True)
        #
        #     # 불필요한 컬럼 정리
        #     daily_chart.drop(['STCK_CLPR', 'STCK_OPN', 'STCK_HGPR', 'STCK_LWPR', 'ACML_VOL', 'ACML_TR_PBMN'], axis=1, inplace=True, errors='ignore')
        #     logger.info(f"✅ [{symbol}] 기술적 지표 계산 완료.")
        #     return daily_chart
        #
        # except Exception as e:
        #     logger.error(f"❌ [{symbol}] pandas-ta 지표 계산 중 오류: {e}", exc_info=True)
        #     return None

# --- 전략 추상 클래스 ---
class Strategy(ABC):
    """
    매매 전략에 대한 추상 베이스 클래스(ABC)입니다.
    모든 매매 전략은 이 클래스를 상속받아 `generate_signals` 메서드를 구현해야 합니다.
    """
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        주어진 데이터를 바탕으로 매매 신호(매수/매도/보유)를 생성합니다.

        :param data: 분석할 시계열 데이터 (Pandas DataFrame)
        :return: 매매 신호가 포함된 DataFrame. 'signal' 컬럼에 1(매수), -1(매도), 0(보유)을 표시합니다.
        """
        pass

# --- 예시 전략: 간단한 이동 평균 교차 전략 ---
class MovingAverageCrossStrategy(Strategy):
    """
    간단한 이동 평균 교차 전략을 구현한 클래스입니다.
    단기 이동 평균선이 장기 이동 평균선을 상향 돌파하면 매수, 하향 돌파하면 매도합니다.
    """
    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        이동 평균 교차 전략에 따라 매매 신호를 생성합니다.

        :param data: 'close' 컬럼(종가)을 포함하는 시계열 데이터
        :return: 'signal' 컬럼이 추가된 DataFrame
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # 이동 평균 계산
        signals['short_mavg'] = data['close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        # 매수 신호 (단기 > 장기)
        signals['signal'][self.long_window:] = \
            (signals['short_mavg'][self.long_window:] > signals['long_mavg'][self.long_window:]).astype(float)

        # 포지션 변경 (신호가 변경되는 시점)
        signals['positions'] = signals['signal'].diff()
        
        # 'positions' 컬럼을 사용하여 최종 신호를 결정: 1.0 (매수), -1.0 (매도)
        # 실제로는 positions가 1.0일때 매수, -1.0일때 매도
        print("이동 평균 교차 전략 신호 생성 완료")
        return signals

# if __name__ == '__main__': 블록을 비동기 함수로 변경
async def main_test():
    """
    CoreTrader 모듈의 주요 기능들을 테스트하기 위한 비동기 함수.
    """
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    trader = CoreTrader(sheet_logger=None)
    initialized = await trader.async_initialize()
    if not initialized:
        print("❌ CoreTrader 초기화 실패")
        return

    symbol = "005930" # 삼성전자
    
    print(f"--- [{symbol}] 데이터 조회 테스트 ---")
    price = await trader.get_current_price(symbol)
    print(f"현재가 정보: {price}")
    
    balance = await trader.get_balance()
    # print(f"계좌 잔고: {balance}") # 응답이 길어 주석 처리
    
    # 기술적 지표 테스트
    indicators = await trader.get_technical_indicators(symbol)
    if indicators is not None and not indicators.empty:
        print(f"\n--- [{symbol}] 기술적 지표 (최근 5일) ---")
        print(indicators.tail())
    else:
        print(f"[{symbol}] 기술적 지표를 가져오지 못했습니다.")

    await trader.close()

if __name__ == "__main__":
    # 비동기 테스트 함수 실행
    asyncio.run(main_test())
