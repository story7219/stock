import requests
import json
import time
import os
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict
import logging
import threading
import schedule

# --- 한국어 주석 ---

# ==============================================================================
# 1. API 속도 제한 관리 클래스 (RateLimiter)
# ==============================================================================
class RateLimiter:
    """
    한국투자증권 API의 요청 속도 제한(1초당 10회)을 관리합니다.
    - 모든 요청은 이 클래스를 통해 '승인' 받은 후 전송됩니다.
    - 지정된 횟수를 초과할 경우, 다음 요청까지 자동으로 대기 시간을 부여합니다.
    """
    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period_sec = period_sec
        # 요청 기록을 저장할 덱(deque) 생성. 오래된 기록부터 자동 삭제에 용이.
        self.request_timestamps = deque()

    def wait(self):
        """
        요청을 보내기 전 호출해야 하는 함수.
        필요 시 다음 요청이 가능해질 때까지 실행을 잠시 멈춥니다(sleep).
        """
        while True:
            now = time.monotonic()
            
            # 오래된 타임스탬프(기간을 벗어난)를 덱에서 제거
            while self.request_timestamps and self.request_timestamps[0] <= now - self.period_sec:
                self.request_timestamps.popleft()

            if len(self.request_timestamps) < self.max_calls:
                # 요청 횟수 제한에 여유가 있으면 루프 탈출
                break
            
            # 가장 오래된 요청 시간 기준으로 대기 시간 계산
            sleep_time = self.period_sec - (now - self.request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 현재 요청의 타임스탬프 기록
        self.request_timestamps.append(time.monotonic())


# ==============================================================================
# 2. 한국투자증권 API 통신 메인 클래스
# ==============================================================================
class KIS_API:
    """
    한국투자증권 REST API와의 모든 통신을 담당하는 클래스.
    - 실전/모의투자 서버 자동 전환
    - 접근 토큰 자동 발급 및 갱신
    - 모든 API 요청에 RateLimiter 자동 적용
    """
    def __init__(self, app_key: str, app_secret: str, account_no: str, is_mock_env: bool = True):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.is_mock_env = is_mock_env

        # 계좌번호를 '-' 기준으로 분리하여 CANO와 ACNT_PRDT_CD를 설정합니다.
        # 주문 API 호출 시 반드시 필요합니다.
        try:
            self.cano = self.account_no.split('-')[0]
            self.acnt_prdt_cd = self.account_no.split('-')[1]
        except IndexError:
            # 사용자가 계좌번호를 잘못 입력했을 경우를 대비한 방어 코드
            logging.error("계좌번호 형식이 올바르지 않습니다. 'XXXXXXXX-XX' 형식을 사용해주세요.")
            raise ValueError("계좌번호 형식이 올바르지 않습니다. 'XXXXXXXX-XX' 형식을 사용해주세요.")

        # MOCK(모의투자)/LIVE(실전투자) 환경에 따른 URL 및 Rate Limit 자동 설정
        if is_mock_env:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.env_type = "MOCK"
            self.token_file = "mock_token.json"  # 모의투자 토큰 파일
            logging.info("MOCK(모의투자) 환경으로 API를 초기화합니다.")
            calls = 2
            period = 1.0
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.env_type = "LIVE"
            self.token_file = "live_token.json"  # 실전투자 토큰 파일
            logging.info("LIVE(실전투자) 환경으로 API를 초기화합니다.")
            calls = 10
            period = 1.0

        # RateLimiter 인스턴스 생성
        self.rate_limiter = RateLimiter(max_calls=calls, period_sec=period)
        logging.info(f"RateLimiter가 {period}초당 {calls}회 호출로 설정되었습니다.")
        
        # 접근 토큰 및 만료 시간 초기화
        self.access_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        
        # 텔레그램 설정 (환경변수에서 로드)
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if self.telegram_bot_token and self.telegram_chat_id:
            logging.info("📱 텔레그램 알림 기능이 활성화되었습니다.")
        else:
            logging.warning("📱 텔레그램 설정이 없습니다. 알림 기능이 비활성화됩니다.")
        
        # 저장된 토큰 로드 시도 후, 없거나 만료되면 새로 발급
        self._load_token_from_file()
        if not self._is_token_valid():
            self._issue_token()

        # 자동 토큰 발급 스케줄링 설정
        self.auto_token_enabled = True
        self.token_schedule_thread = None
        
        # 환경변수에서 토큰 발급 시간 설정 (기본값: 06:00)
        self.token_issue_hour = int(os.getenv('TOKEN_ISSUE_HOUR', '6'))
        self.token_issue_minute = int(os.getenv('TOKEN_ISSUE_MINUTE', '0'))
        
        # 스케줄링 시작
        self._start_token_scheduler()
        
        logging.info("📅 자동 토큰 발급 스케줄러가 시작되었습니다.")
        logging.info(f"⏰ 매일 {self.token_issue_hour:02d}:{self.token_issue_minute:02d}에 새로운 토큰을 발급합니다.")

    def _load_token_from_file(self):
        """파일에서 저장된 토큰 로드"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r', encoding='utf-8') as f:
                    token_data = json.load(f)
                
                self.access_token = token_data.get('access_token')
                expiry_str = token_data.get('token_expiry')
                
                if expiry_str:
                    self.token_expiry = datetime.fromisoformat(expiry_str)
                
                if self._is_token_valid():
                    remaining_time = self.token_expiry - datetime.now()
                    logging.info(f"저장된 토큰을 로드했습니다. (남은 시간: {remaining_time})")
                else:
                    logging.info("저장된 토큰이 만료되었습니다.")
            else:
                logging.info("저장된 토큰 파일이 없습니다.")
        except Exception as e:
            logging.warning(f"토큰 파일 로드 중 오류: {e}")
            self.access_token = None
            self.token_expiry = None

    def _save_token_to_file(self):
        """토큰을 파일에 저장"""
        try:
            token_data = {
                'access_token': self.access_token,
                'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
                'issued_at': datetime.now().isoformat()
            }
            
            with open(self.token_file, 'w', encoding='utf-8') as f:
                json.dump(token_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"토큰이 {self.token_file}에 저장되었습니다.")
        except Exception as e:
            logging.error(f"토큰 파일 저장 중 오류: {e}")

    def _is_token_valid(self) -> bool:
        """토큰이 유효한지 확인"""
        if not self.access_token:
            return False
        
        if not self.token_expiry:
            return False
        
        # 만료 30분 전까지를 유효한 것으로 간주 (여유시간 확보)
        # 24시간 토큰이므로 30분 여유를 두고 미리 재발급 준비
        return datetime.now() < (self.token_expiry - timedelta(minutes=30))

    def _send_telegram_message(self, message: str):
        """텔레그램으로 메시지 전송"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logging.info("📱 텔레그램 알림 전송 완료")
            else:
                logging.warning(f"📱 텔레그램 알림 전송 실패: {response.status_code}")
                
        except Exception as e:
            logging.error(f"📱 텔레그램 알림 전송 중 오류: {e}")

    def _issue_token(self):
        """접근 토큰 발급 (24시간 제한 체크 + 텔레그램 알림)"""
        try:
            # 발급 가능 여부 확인
            if not self._can_issue_new_token():
                error_msg = "⚠️ 토큰 발급 실패: 24시간 제한"
                logging.error(error_msg)
                
                # 텔레그램 알림
                telegram_msg = f"""
🚫 <b>KIS API 토큰 발급 실패</b>

📊 환경: {self.env_type}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
❌ 사유: 24시간 제한 (하루 1회만 발급 가능)

💡 다음 발급 가능 시간을 확인하세요.
"""
                self._send_telegram_message(telegram_msg)
                raise Exception("토큰 발급 24시간 제한에 걸렸습니다.")
            
            logging.info(f"{self.env_type} 환경에서 새로운 토큰 발급을 시작합니다.")
            
            # 텔레그램 발급 시작 알림
            start_msg = f"""
🔄 <b>KIS API 토큰 발급 시작</b>

📊 환경: {self.env_type}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔑 계정: {self.account_no}

토큰 발급을 시도합니다...
"""
            self._send_telegram_message(start_msg)
            
            url = f"{self.base_url}/oauth2/tokenP"
            
            headers = {
                "content-type": "application/json; charset=utf-8"
            }
            
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            logging.info("토큰 발급을 요청합니다.")
            response = requests.post(url, headers=headers, json=data)
            
            # 상세한 오류 정보 출력
            if response.status_code != 200:
                error_detail = f"토큰 발급 실패 - 상태 코드: {response.status_code}"
                logging.error(error_detail)
                logging.error(f"응답 내용: {response.text}")
                
                # 텔레그램 오류 알림
                error_msg = f"""
❌ <b>KIS API 토큰 발급 실패</b>

📊 환경: {self.env_type}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🚫 상태 코드: {response.status_code}
📝 오류 내용: {response.text[:200]}...

💡 해결 방법을 확인하세요.
"""
                self._send_telegram_message(error_msg)
                
                if response.status_code == 403:
                    try:
                        error_response = response.json()
                        if 'error_code' in error_response and error_response['error_code'] == 'EGW00133':
                            logging.error("🚫 토큰 발급 빈도 제한 오류!")
                    except:
                        pass
            
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            
            if not self.access_token:
                raise Exception("응답에서 access_token을 찾을 수 없습니다.")
            
            # 토큰 만료 시간 설정 (정확히 24시간 후)
            self.token_expiry = datetime.now() + timedelta(hours=24)
            
            logging.info("✅ 접근 토큰 발급이 완료되었습니다.")
            
            # 토큰을 파일에 저장
            self._save_token_to_file()
            
            # 텔레그램 성공 알림
            success_msg = f"""
✅ <b>KIS API 토큰 발급 성공!</b>

📊 환경: {self.env_type}
⏰ 발급 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔑 계정: {self.account_no}
⏳ 만료 시간: {self.token_expiry.strftime('%Y-%m-%d %H:%M:%S')}
🕐 유효 기간: 24시간

🎯 다음 발급 가능: {(datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')}

🚀 자동매매 시스템이 정상 작동합니다!
"""
            self._send_telegram_message(success_msg)
            
        except Exception as e:
            error_msg = f"접근 토큰 발급에 실패했습니다 - {e}"
            logging.error(error_msg)
            
            # 텔레그램 예외 오류 알림
            exception_msg = f"""
💥 <b>KIS API 토큰 발급 예외 오류</b>

📊 환경: {self.env_type}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🚫 오류: {str(e)[:200]}

🔧 시스템 점검이 필요합니다.
"""
            self._send_telegram_message(exception_msg)
            raise e

    def _can_issue_new_token(self) -> bool:
        """새로운 토큰 발급이 가능한지 확인 (24시간 제한 체크 + 텔레그램 알림)"""
        try:
            if not os.path.exists(self.token_file):
                logging.info("🆕 첫 토큰 발급이므로 발급 가능합니다.")
                return True
            
            with open(self.token_file, 'r', encoding='utf-8') as f:
                token_data = json.load(f)
            
            issued_at_str = token_data.get('issued_at')
            if not issued_at_str:
                logging.info("🆕 발급 시간 정보가 없으므로 발급 가능합니다.")
                return True
            
            issued_at = datetime.fromisoformat(issued_at_str)
            now = datetime.now()
            elapsed = now - issued_at
            
            # 24시간(1440분) 경과 확인
            if elapsed >= timedelta(hours=24):
                logging.info(f"✅ 마지막 발급으로부터 24시간이 경과했습니다. (경과: {elapsed})")
                
                # 텔레그램 발급 가능 알림
                available_msg = f"""
⏰ <b>KIS API 토큰 발급 가능</b>

📊 환경: {self.env_type}
🕐 마지막 발급: {issued_at.strftime('%Y-%m-%d %H:%M:%S')}
⏳ 경과 시간: {elapsed}
✅ 상태: 24시간 경과, 발급 가능

🔄 새로운 토큰 발급을 시작합니다.
"""
                self._send_telegram_message(available_msg)
                return True
            else:
                remaining = timedelta(hours=24) - elapsed
                next_available = issued_at + timedelta(hours=24)
                
                logging.warning(f"⏳ 아직 24시간이 경과하지 않았습니다.")
                logging.warning(f"   남은 대기 시간: {remaining}")
                logging.warning(f"   다음 발급 가능: {next_available}")
                
                # 텔레그램 대기 알림
                waiting_msg = f"""
⏳ <b>KIS API 토큰 발급 대기 중</b>

📊 환경: {self.env_type}
🕐 마지막 발급: {issued_at.strftime('%Y-%m-%d %H:%M:%S')}
⏰ 경과 시간: {elapsed}
⏳ 남은 대기 시간: {remaining}
🎯 다음 발급 가능: {next_available.strftime('%Y-%m-%d %H:%M:%S')}

💡 24시간 제한으로 인해 대기 중입니다.
"""
                self._send_telegram_message(waiting_msg)
                return False
                
        except Exception as e:
            logging.error(f"토큰 발급 가능 여부 확인 중 오류: {e}")
            return True  # 오류 시 발급 시도

    def check_token_renewal_needed(self) -> bool:
        """토큰 갱신이 필요한지 확인"""
        status = self.get_token_status()
        return status["status"] in ["없음", "만료", "만료임박"]

    def get_token_status(self) -> Dict:
        """토큰 상태 정보 반환 (24시간 토큰 관리)"""
        if not self.access_token:
            return {"status": "없음", "message": "토큰이 발급되지 않았습니다."}
        
        if not self.token_expiry:
            return {"status": "불명", "message": "토큰 만료 시간을 알 수 없습니다."}
        
        now = datetime.now()
        remaining = self.token_expiry - now
        
        if now >= self.token_expiry:
            status_info = {
                "status": "만료", 
                "message": f"토큰이 만료되었습니다. 새로운 토큰 발급이 필요합니다.",
                "expired_at": self.token_expiry,
                "action": "새 토큰 발급 필요"
            }
            
            # 만료 알림
            expire_msg = f"""
⚠️ <b>KIS API 토큰 만료</b>

📊 환경: {self.env_type}
⏰ 만료 시간: {self.token_expiry.strftime('%Y-%m-%d %H:%M:%S')}
🚫 상태: 만료됨

🔄 새로운 토큰 발급이 필요합니다.
"""
            self._send_telegram_message(expire_msg)
            return status_info
            
        elif remaining <= timedelta(minutes=30):
            return {
                "status": "만료임박", 
                "message": f"토큰이 곧 만료됩니다. (남은 시간: {remaining})",
                "expires_at": self.token_expiry,
                "remaining_time": str(remaining),
                "action": "새 토큰 발급 준비 필요"
            }
        else:
            return {
                "status": "유효", 
                "message": f"토큰이 유효합니다. (남은 시간: {remaining})",
                "expires_at": self.token_expiry,
                "remaining_time": str(remaining),
                "action": "정상 사용 가능"
            }

    def _get_valid_token(self) -> str:
        """유효한 접근 토큰을 반환합니다."""
        if not self._is_token_valid():
            logging.error("🚫 유효한 토큰이 없습니다!")
            logging.error("💡 해결 방법:")
            logging.error("1. 프로그램을 다시 시작하여 토큰 재발급 시도")
            logging.error("2. 토큰은 하루에 1회만 발급 가능하므로 내일 다시 시도")
            logging.error(f"3. 토큰 파일({self.token_file}) 확인")
            raise Exception("유효한 토큰이 없습니다. 토큰 재발급이 필요합니다.")
        
        return self.access_token

    def _send_request(self, method: str, path: str, headers: Optional[Dict] = None, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Dict:
        """
        API 서버에 요청을 보내는 범용 함수.
        - RateLimiter 적용
        - 토큰 관리 자동화
        """
        # 1. API 요청 전 속도 제한 체크 (가장 중요)
        self.rate_limiter.wait()

        # 2. 유효한 토큰 가져오기
        token = self._get_valid_token()

        # 3. 공통 헤더 설정
        common_headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        if headers:
            common_headers.update(headers)

        # 4. 요청 실행
        url = f"{self.base_url}{path}"
        try:
            # 요청 상세 로깅 (디버깅용)
            logging.info(f"🌐 API 요청: {method} {url}")
            if body:
                logging.info(f"📤 요청 데이터: {json.dumps(body, ensure_ascii=False)}")
            
            response = requests.request(method, url, headers=common_headers, params=params, data=json.dumps(body) if body else None)
            
            # 응답 상세 로깅
            logging.info(f"📥 응답 상태: {response.status_code}")
            logging.info(f"📥 응답 데이터: {response.text}")
            
            response.raise_for_status()
            
            # 응답 데이터 처리
            if response.text:
                return response.json()
            return {}

        except requests.exceptions.HTTPError as e:
            # API 서버에서 보낸 에러 메시지(rt_cd, msg1 등)를 포함하여 출력
            try:
                error_data = e.response.json()
                logging.error(f"API 오류: {e.response.status_code} - {error_data}")
            except json.JSONDecodeError:
                logging.error(f"API 오류: {e.response.status_code} - {e.response.text}")
            raise e
        except Exception as e:
            logging.error(f"네트워크/기타 오류: {e}")
            raise e

    # --- 공개적으로 사용할 API 함수들 ---
    
    def get_balance(self, account_no: str = None) -> Dict:
        """
        주식 잔고 현황을 조회합니다.
        """
        if account_no is None:
            account_no = self.account_no
            
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {"tr_id": "TTTC8434R" if not self.is_mock_env else "VTTC8434R"}
        params = {
            "CANO": account_no.split('-')[0],
            "ACNT_PRDT_CD": account_no.split('-')[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        return self._send_request('GET', path, headers=headers, params=params)

    def get_current_price(self, symbol: str) -> Dict:
        """
        지정된 종목의 현재가를 조회합니다.
        (API 명세: /uapi/domestic-stock/v1/quotations/inquire-price)
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {"tr_id": "FHKST01010100"}
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol}
        
        return self._send_request('GET', path, headers=headers, params=params)

    def _place_order(self, side, code, quantity):
        """
        내부 매수/매도 주문 처리 함수 (시장가 주문).
        trader.py의 명령을 실제 주문으로 변환하는 집행부 역할을 합니다.

        Args:
            side (str): 'BUY' 또는 'SELL'
            code (str): 종목 코드 (예: '005930')
            quantity (int): 주문 수량
        """
        path = "/uapi/domestic-stock/v1/trading/order-cash"
        
        # KIS API 명세에 따라 매수와 매도의 tr_id가 다르며, 실전/모의투자도 다릅니다.
        if self.is_mock_env:
            tr_id = "VTTC0802U" if side == 'BUY' else "VTTC0801U"  # 모의투자: 매수/매도
        else:
            tr_id = "TTTC0802U" if side == 'BUY' else "TTTC0801U"  # 실전투자: 매수/매도

        headers = {"tr_id": tr_id}
        
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": code,
            "ORD_DVSN": "01",  # 01: 지정가, 02: 시장가 (모의투자에서는 지정가 권장)
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",   # 시장가 주문 시 가격은 0으로 설정
        }
        
        # 모의투자에서는 지정가 주문이 더 안정적일 수 있음
        if self.is_mock_env:
            try:
                # 현재가 조회하여 지정가로 주문
                price_info = self.get_current_price(code)
                if price_info and price_info.get('rt_cd') == '0':
                    current_price = price_info['output']['stck_prpr']
                    # 매수는 현재가보다 약간 높게, 매도는 현재가로 설정
                    if side == 'BUY':
                        order_price = str(int(int(current_price) * 1.01))  # 1% 높게
                    else:
                        order_price = current_price
                    
                    body["ORD_DVSN"] = "00"  # 지정가
                    body["ORD_UNPR"] = order_price
                    logging.info(f"📊 모의투자 지정가 주문: {code} {side} {quantity}주 @ {order_price}원")
            except:
                logging.warning("현재가 조회 실패, 시장가로 주문합니다.")
        
        # _send_request를 통해 KIS 서버에 최종 주문을 전송합니다.
        return self._send_request(
            method="POST", 
            path=path, 
            headers=headers,
            body=body
        )

    def buy_order(self, code, quantity):
        """시장가 매수 주문을 실행합니다."""
        logging.info(f"[주문 요청] 시장가 매수: {code}, 수량: {quantity}")
        return self._place_order('BUY', code, quantity)

    def sell_order(self, code, quantity):
        """시장가 매도 주문을 실행합니다."""
        logging.info(f"[주문 요청] 시장가 매도: {code}, 수량: {quantity}")
        return self._place_order('SELL', code, quantity)

    def _start_token_scheduler(self):
        """토큰 자동 발급 스케줄러 시작"""
        try:
            # 기존 스케줄 모두 삭제
            schedule.clear()
            
            # 환경변수에서 설정한 시간에 토큰 발급
            schedule_time = f"{self.token_issue_hour:02d}:{self.token_issue_minute:02d}"
            schedule.every().day.at(schedule_time).do(self._scheduled_token_renewal)
            
            # 스케줄러 스레드 시작
            if not hasattr(self, '_scheduler_thread') or not self._scheduler_thread.is_alive():
                self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
                self._scheduler_thread.start()
                
            logging.info("📅 자동 토큰 발급 스케줄러가 시작되었습니다.")
            logging.info(f"⏰ 매일 {schedule_time}에 새로운 토큰을 발급합니다.")
            
        except Exception as e:
            logging.error(f"토큰 스케줄러 시작 중 오류: {e}")

    def _run_scheduler(self):
        """스케줄러 실행 루프"""
        while self.auto_token_enabled:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 스케줄 확인
            except Exception as e:
                logging.error(f"스케줄러 실행 중 오류: {e}")
                time.sleep(60)

    def _scheduled_token_renewal(self):
        """스케줄된 토큰 발급"""
        try:
            schedule_time = f"{self.token_issue_hour:02d}:{self.token_issue_minute:02d}"
            logging.info(f"📅 {schedule_time} - 토큰 발급을 시작합니다...")
            
            # 텔레그램 시작 알림
            morning_start_msg = f"""
🌅 <b>{schedule_time} - 거래 준비 시작</b>

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 목적: 토큰 발급 + AI 반성
🔄 환경: {self.env_type}

📋 <b>진행 순서:</b>
1️⃣ 토큰 발급
2️⃣ 전날 거래 분석
3️⃣ 제미나이 반성 시간
4️⃣ 오늘 전략 수립

시작합니다...
"""
            self._send_telegram_message(morning_start_msg)
            
            # 1단계: 토큰 발급
            logging.info("1️⃣ 토큰 발급을 시작합니다...")
            self._issue_token()
            
            # 2단계: 전날 거래 리뷰 (trader에서 처리하도록 신호 전송)
            logging.info("2️⃣ 전날 거래 리뷰를 시작합니다...")
            self._trigger_daily_review()
            
        except Exception as e:
            # 예외 오류 알림
            error_msg = f"""
💥 <b>자동 준비 과정 오류</b>

⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🚫 오류: {str(e)[:200]}

🔧 시스템 점검이 필요합니다.
"""
            self._send_telegram_message(error_msg)
            logging.error(f"자동 준비 과정 중 예외 오류: {e}")

    def _trigger_daily_review(self):
        """일일 리뷰 트리거 (trader에서 처리하도록 신호 생성)"""
        try:
            # 리뷰 신호 파일 생성
            review_signal = {
                'trigger_time': datetime.now().isoformat(),
                'review_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'status': 'pending'
            }
            
            with open('daily_review_signal.json', 'w', encoding='utf-8') as f:
                json.dump(review_signal, f, ensure_ascii=False, indent=2)
            
            logging.info("📊 일일 리뷰 신호가 생성되었습니다.")
            
        except Exception as e:
            logging.error(f"일일 리뷰 신호 생성 실패: {e}")

    def get_next_token_schedule(self) -> str:
        """다음 토큰 발급 예정 시간 반환"""
        try:
            now = datetime.now()
            # 설정된 시간
            today_schedule = now.replace(
                hour=self.token_issue_hour, 
                minute=self.token_issue_minute, 
                second=0, 
                microsecond=0
            )
            
            if now < today_schedule:
                # 아직 오늘 설정 시간이 지나지 않았으면 오늘
                next_schedule = today_schedule
            else:
                # 이미 지났으면 내일
                next_schedule = today_schedule + timedelta(days=1)
            
            return next_schedule.strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logging.error(f"다음 스케줄 계산 중 오류: {e}")
            return "계산 실패"

    def stop_token_scheduler(self):
        """토큰 스케줄러 중지"""
        self.auto_token_enabled = False
        schedule.clear()
        logging.info("📅 토큰 스케줄러가 중지되었습니다.")

# ==============================================================================
# 3. 사용 예시 (이 파일을 직접 실행할 경우)
# ==============================================================================
if __name__ == '__main__':
    # .env 파일 등에서 실제 키와 계좌번호를 불러와야 합니다.
    # 아래는 예시 값이며, 실제 값으로 변경해야 동작합니다.
    MY_APP_KEY = "PSJHToqNQYzVvVH1DfkndIodXaCsEgAHBHPr"
    MY_APP_SECRET = "W5ts9iDYGxjNGaPdKqDcjAQz2FdLwakr/2sC3K44zs9dtljT2P8UbB/zOo2hsWZpkP/kraOmF9P1vqqcHxbz/YiVwKcR6FCmj/WZdoAdnCfQi/KMntP9V1b6dn7RLoOiTZtgwLaoVfWKJPP+hcmxNI/st+oCp3iDv/ZdKoQg4Hu9OG4myW0="
    MY_ACCOUNT_NO = "50128558-01" # ex) 12345678-01

    # KIS_API 클래스 인스턴스화 (모의투자 환경)
    api = KIS_API(app_key=MY_APP_KEY, app_secret=MY_APP_SECRET, account_no=MY_ACCOUNT_NO, is_mock_env=True)

    # --- 사용 예시 ---
    try:
        # 1. 삼성전자 현재가 조회
        price_data = api.get_current_price("005930")
        print("\n[삼성전자 현재가 조회 결과]")
        print(price_data)

        # 2. 내 주식 잔고 조회
        balance_data = api.get_balance(MY_ACCOUNT_NO)
        print("\n[주식 잔고 조회 결과]")
        print(balance_data)

        # 3. (참고) 짧은 시간 연속 호출 테스트
        print("\n[Rate Limiter 연속 호출 테스트 시작]")
        start_time = time.time()
        for i in range(15):
            print(f"{i+1}번째 호출 중...")
            api.get_current_price("000660") # SK하이닉스
        end_time = time.time()
        print(f"15회 호출에 걸린 시간: {end_time - start_time:.2f}초")
        print("RateLimiter에 의해 호출 속도가 자동으로 조절되었음을 확인할 수 있습니다.")

    except Exception as e:
        print(f"\n메인 실행 중 오류 발생: {e}") 