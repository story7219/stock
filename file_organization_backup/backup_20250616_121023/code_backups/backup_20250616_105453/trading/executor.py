# executor.py
# 한국투자증권 실전/모의투자 API 연동 및 주문 실행 함수 정의

import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
from utils.telegram_bot import send_message
from utils.logger import log_event

# .env에서 환경변수 불러오기
load_dotenv()
KIS_APP_KEY = os.getenv("KIS_APP_KEY")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET")
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")

# 실전은 "https://openapi.koreainvestment.com:9443"
# 모의투자는 "https://openapivts.koreainvestment.com:29443"
BASE_URL = "https://openapivts.koreainvestment.com:29443"

_ACCESS_TOKEN = ""
_TOKEN_EXPIRATION = None

def _get_access_token():
    """인증 토큰을 발급받는 내부 함수"""
    global _ACCESS_TOKEN, _TOKEN_EXPIRATION
    
    # 토큰이 유효하면 재사용
    if _ACCESS_TOKEN and _TOKEN_EXPIRATION and datetime.now() < _TOKEN_EXPIRATION:
        return _ACCESS_TOKEN

    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET
    }
    path = "/oauth2/tokenP"
    url = f"{BASE_URL}{path}"
    
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        data = res.json()
        
        _ACCESS_TOKEN = data['access_token']
        # 토큰 만료 시간 저장 (약간의 여유시간을 둠)
        _TOKEN_EXPIRATION = datetime.now() + pd.Timedelta(seconds=data['expires_in'] - 60)
        
        log_event("INFO", "한투 API 신규 토큰 발급 성공")
        return _ACCESS_TOKEN
    except requests.exceptions.RequestException as e:
        log_event("ERROR", f"토큰 발급 실패: {e}")
        send_message(f"🔥[API 오류] 인증 토큰 발급에 실패했습니다: {e}")
        return None

def execute_order_kis(ticker, signal, quantity):
    """
    한국투자증권 API를 활용하여 실제 주문을 실행합니다.
    """
    token = _get_access_token()
    if not token:
        return "토큰 발급 실패"

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": KIS_APP_KEY,
        "appSecret": KIS_APP_SECRET,
        "tr_id": "VTTC0802U" if signal == 'buy' else "VTTC0801U", # 모의투자 매수/매도
    }
    
    body = {
        "CANO": KIS_ACCOUNT_NO.split('-')[0],
        "ACNT_PRDT_CD": KIS_ACCOUNT_NO.split('-')[1],
        "PDNO": ticker.replace(".KS", ""), # .KS 제거
        "ORD_DVSN": "01", # 시장가
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0", # 시장가 주문이므로 0
    }
    
    path = "/uapi/domestic-stock/v1/trading/order-cash"
    url = f"{BASE_URL}{path}"

    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        data = res.json()
        
        if data['rt_cd'] == '0':
            log_message = f"✅ [주문 성공] {ticker}: {signal.upper()} {quantity}주. (응답: {data['msg1']})"
            log_event("ORDER_SUCCESS", log_message)
            send_message(log_message)
            return "주문 성공"
        else:
            log_message = f"❌ [주문 실패] {ticker}: {signal.upper()} {quantity}주. (응답: {data['msg1']})"
            log_event("ORDER_FAIL", log_message)
            send_message(log_message)
            return f"주문 실패: {data['msg1']}"

    except requests.exceptions.RequestException as e:
        log_event("ERROR", f"주문 API 요청 실패: {e}")
        send_message(f"🔥[API 오류] 주문 실행 중 오류가 발생했습니다: {e}")
        return f"주문 API 요청 실패: {e}"

# 기존 validate_order 함수는 main.py로 이동하거나 삭제 (PortfolioManager가 관리)
# def validate_order(signal, account_info, risk_limit): ... 