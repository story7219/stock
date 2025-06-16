"""
핵심 트레이딩 엔진 (v5.0, GitHub Actions 최적화)
- 환경 변수에서 모든 설정을 로드하여 보안 및 안정성 강화.
- 주문, 로깅, 알림을 하나로 통합한 강력한 주문 함수 제공.
"""
import requests
import json
import logging
import time
from collections import deque
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta, time as time_obj
import os
import sys
from utils.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

# --- ⚙️ API 호출 조절기 ---
class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls; self.period = period; self.call_times = deque()
    def __enter__(self):
        now = time.monotonic()
        while self.call_times and self.call_times[0] <= now - self.period: self.call_times.popleft()
        if len(self.call_times) >= self.calls:
            sleep_time = self.call_times[0] - (now - self.period)
            logger.warning(f"API 호출 제한(초당 {self.calls}회) 도달. {sleep_time:.2f}초 대기.")
            time.sleep(sleep_time)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.call_times.append(time.monotonic())

# --- 💎 자동 토큰 관리 시스템 ---
class TokenManager:
    def __init__(self, base_url, app_key, app_secret, limiter):
        self.base_url, self.app_key, self.app_secret, self.limiter = base_url, app_key, app_secret, limiter
        self.token_file = "kis_token.json"
        self.access_token = None
        self.renewal_time = time_obj(hour=7, minute=0) # KST 07:00

    def _save_token(self, token_data):
        token_data['expires_at'] = (datetime.now() + timedelta(seconds=token_data['expires_in'] - 600)).isoformat()
        with open(self.token_file, 'w') as f: json.dump(token_data, f)
        logger.info("새 API 토큰을 파일에 저장했습니다.")

    def _load_token(self):
        if not os.path.exists(self.token_file): return None
        with open(self.token_file, 'r') as f:
            try: return json.load(f)
            except json.JSONDecodeError: return None

    def _issue_new_token(self):
        with self.limiter:
            url = f"{self.base_url}/oauth2/tokenP"
            body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
            try:
                res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(body))
                res.raise_for_status()
                token_data = res.json()
                self._save_token(token_data)
                return token_data
            except Exception as e:
                logger.error(f"토큰 발급 실패: {e}"); return None

    def get_valid_token(self):
        token_data = self._load_token()
        if not token_data or datetime.fromisoformat(token_data['expires_at']) <= datetime.now():
            logger.warning("유효한 토큰이 없어 새 토큰을 발급합니다.")
            new_token_data = self._issue_new_token()
            return new_token_data['access_token'] if new_token_data else None
        
        if not self.access_token: logger.info("파일에서 유효한 토큰을 로드했습니다.")
        self.access_token = token_data['access_token']
        return self.access_token

# --- 🏦 핵심 거래 클래스 ---
class CoreTrader:
    def __init__(self):
        self.is_mock = os.environ.get('IS_MOCK', 'true').lower() == 'true'
        self.base_url = "https://openapivts.koreainvestment.com:29443" if self.is_mock else "https://openapi.koreainvestment.com:9443"
        self.app_key = os.environ.get('KIS_APP_KEY')
        self.app_secret = os.environ.get('KIS_APP_SECRET')
        self.account_no = os.environ.get('KIS_ACCOUNT_NO')

        if not all([self.app_key, self.app_secret, self.account_no]):
            sys.exit("❌ 필수 환경변수(KIS_*)가 설정되지 않았습니다. 워크플로우의 `env` 설정을 확인하세요.")

        self.order_limiter = RateLimiter(calls=2, period=1)
        self.market_data_limiter = RateLimiter(calls=5, period=1)
        self.token_manager = TokenManager(self.base_url, self.app_key, self.app_secret, self.order_limiter)
        self.notifier = TelegramNotifier()
        self.worksheet = self._initialize_gspread()

    def _initialize_gspread(self):
        try:
            gcp_sa_key = os.environ.get('GCP_SA_KEY')
            spreadsheet_id = os.environ.get('GOOGLE_SPREADSHEET_ID')
            if not gcp_sa_key or not spreadsheet_id:
                logger.warning("⚠️ Google Sheets 환경변수(GCP_SA_KEY, GOOGLE_SPREADSHEET_ID)가 없어 로깅이 비활성화됩니다.")
                return None
            
            creds = Credentials.from_service_account_info(json.loads(gcp_sa_key))
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet('trade_log')
            logger.info("✅ Google Sheets 연동 성공 ('trade_log' 워크시트)")
            return worksheet
        except gspread.exceptions.WorksheetNotFound:
            logger.info("'trade_log' 워크시트가 없어 새로 생성합니다.")
            return spreadsheet.add_worksheet(title='trade_log', rows="1000", cols="10")
        except Exception as e:
            logger.error(f"❌ Google Sheets 초기화 실패: {e}"); return None

    def _log_trade_to_sheet(self, log_data):
        if not self.worksheet: return
        try:
            headers = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'status', 'pnl_percent']
            if self.worksheet.row_count == 0: self.worksheet.append_row(headers)
            row = [log_data.get(h, '') for h in headers]
            self.worksheet.append_row(row)
            logger.info(f"📝 Google Sheet에 거래 기록 완료: {log_data.get('symbol')} {log_data.get('side')}")
        except Exception as e: logger.error(f"❌ Google Sheet 기록 실패: {e}")

    def _send_request(self, method, path, headers=None, params=None, json_data=None):
        token = self.token_manager.get_valid_token()
        if not token: raise Exception("유효한 API 토큰이 없습니다.")
        common_headers = {
            "Content-Type": "application/json", "authorization": f"Bearer {token}",
            "appKey": self.app_key, "appSecret": self.app_secret, "custtype": "P"
        }
        if headers: common_headers.update(headers)
        
        try:
            res = requests.request(method, f"{self.base_url}{path}", headers=common_headers, params=params, data=json.dumps(json_data) if json_data else None)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logger.error(f"API 요청 실패 ({path}): {e}"); return None

    def get_current_price(self, symbol):
        with self.market_data_limiter:
            res = self._send_request("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", 
                                     headers={"tr_id": "FHKST01010100"}, 
                                     params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol})
            return {'price': int(res['output']['stck_prpr']), 'name': res['output']['hts_kor_isnm']} if res and res.get('rt_cd') == '0' else None

    def get_balance(self, part='all'):
        with self.order_limiter:
            tr_id = "VTTC8434R" if self.is_mock else "TTTC8434R"
            params = {"CANO": self.account_no.split('-')[0], "ACNT_PRDT_CD": self.account_no.split('-')[1], "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "01", "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "Y", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00"}
            res = self._send_request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance", headers={"tr_id": tr_id}, params=params)

            if not res or res.get('rt_cd') != '0':
                logger.error(f"잔고 조회 실패: {res.get('msg1') if res else '응답 없음'}"); return None
            
            if part == 'all': return res
            if part == 'cash': return int(res.get('output2', [{}])[0].get('dnca_tot_amt', 0))
            if part == 'stocks': return res.get('output1', [])
            if part == symbol: # 특정 종목 조회
                return next((s for s in res.get('output1', []) if s['pdno'] == symbol), {})
            return None

    def get_top_ranking_stocks(self, top_n=10):
        with self.market_data_limiter:
            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_NO": "0171", "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", "FID_BLNG_CLS_CODE": "1", "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "000000", "FID_INPUT_PRICE_1": "1000", "FID_VOL_CNT": "100000"}
            res = self._send_request("GET", "/uapi/domestic-stock/v1/quotations/inquire-ranking", headers={"tr_id": "FHPST01710000"}, params=params)
            
            if not res or res.get('rt_cd') != '0':
                logger.error(f"상승률 상위 종목 조회 실패: {res.get('msg1') if res else '응답 없음'}"); return []
            
            return [{'symbol': i['iscd_symb'], 'name': i['hts_kor_isnm'], 'price': int(i.get('stck_prpr', 0))} for i in res.get('output', [])[:top_n]]

    def execute_order(self, symbol, side, quantity, price=0, log_payload=None):
        with self.order_limiter:
            tr_id = ("VTTC0802U" if side == 'buy' else "VTTC0801U") if self.is_mock else ("TTTC0802U" if side == 'buy' else "TTTC0801U")
            body = {"CANO": self.account_no.split('-')[0], "ACNT_PRDT_CD": self.account_no.split('-')[1], "PDNO": symbol, "ORD_DVSN": "01" if price == 0 else "00", "ORD_QTY": str(quantity), "ORD_UNPR": str(price)}
            res = self._send_request("POST", "/uapi/domestic-stock/v1/trading/order-cash", headers={"tr_id": tr_id}, json_data=body)
            
            if res and res.get('rt_cd') == '0':
                current_price = self.get_current_price(symbol)['price'] if price == 0 else price
                log_data = {'symbol': symbol, 'side': side, 'quantity': quantity, 'price': current_price}
                if log_payload: log_data.update(log_payload)
                self._log_trade_to_sheet(log_data)
                self.notifier.send_message(f"✅ [{log_data.get('status', side)}] {symbol} {quantity}주 주문 성공")
                return True
            else:
                msg = res.get('msg1') if res else '주문 응답 없음'
                logger.error(f"❌ [{side}] {symbol} 주문 실패: {msg}")
                self.notifier.send_message(f"❌ [{side}] {symbol} 주문 실패: {msg}")
                return False

    def get_market_summary(self):
        try:
            kospi = self._send_request("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", headers={"tr_id": "FHPUP01010100"}, params={"FID_INPUT_ISCD": "U001"})['output']
            kosdaq = self._send_request("GET", "/uapi/domestic-stock/v1/quotations/inquire-price", headers={"tr_id": "FHPUP01010100"}, params={"FID_INPUT_ISCD": "U201"})['output']
            summary = (f"- **코스피**: {kospi['prpr']} ({kospi['prdy_vrss_sign']}{kospi['prdy_vrss']}, {kospi['prdy_ctrt']}%)\n"
                       f"- **코스닥**: {kosdaq['prpr']} ({kosdaq['prdy_vrss_sign']}{kosdaq['prdy_vrss']}, {kosdaq['prdy_ctrt']}%)\n")
            return summary
        except Exception as e: return f"시장 지수 조회 실패: {e}"

    def get_todays_trades_from_sheet(self):
        if not self.worksheet: return "거래 내역 시트 미설정"
        try:
            records = self.worksheet.get_all_records()
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_records = [r for r in records if str(r.get('timestamp', '')).startswith(today_str)]
            if not today_records: return "오늘 발생한 거래가 없습니다."
            return "\n".join([f"- {r.get('timestamp')} | {r.get('symbol')} | {r.get('side')} | {r.get('status', '')} | {r.get('pnl_percent', '')}" for r in today_records])
        except Exception as e: return f"거래 내역 조회 오류: {e}" 