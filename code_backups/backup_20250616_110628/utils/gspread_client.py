import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from config import SPREADSHEET_ID
from utils.logger import log_event
import os

# 구글 API 접근 범위 설정
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# 인증 정보 파일 경로
CREDS_FILE = 'credentials.json'

class MockGspreadClient:
    """구글 시트 클라이언트 모의 객체"""
    def __init__(self):
        self.client = None
        
    def update_sheet(self, data):
        """시트 업데이트 (모의)"""
        print(f"[Google Sheets] 데이터 기록: {data}")

class GSpreadClient:
    """
    구글 스프레드시트와의 연동을 관리하는 클라이언트 클래스
    """
    def __init__(self, spreadsheet_id):
        self.spreadsheet_id = spreadsheet_id
        self.client = self._authenticate()
        self.spreadsheet = None # 먼저 None으로 초기화
        if self.client:
            try:
                self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            except Exception as e:
                log_event("ERROR", f"스프레드시트를 여는 데 실패했습니다: {e}. 구글 시트 기능이 비활성화됩니다.")
                self.client = None # 클라이언트를 다시 None으로 설정하여 이후 작업 방지

    def _authenticate(self):
        """서비스 계정을 사용하여 구글 API에 인증합니다."""
        try:
            creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPES)
            client = gspread.authorize(creds)
            log_event("INFO", "Google Sheets 인증에 성공했습니다.")
            return client
        except FileNotFoundError:
            log_event("WARNING", f"인증 파일({CREDS_FILE})을 찾을 수 없습니다. 구글 시트 기능이 비활성화됩니다.")
            return None
        except Exception as e:
            log_event("ERROR", f"Google Sheets 인증 중 오류 발생: {e}. 구글 시트 기능이 비활성화됩니다.")
            return None

    def get_worksheet(self, worksheet_name="trade_log"):
        """지정된 이름의 워크시트를 가져오거나 없으면 새로 생성합니다."""
        if not self.client:
            return None
        try:
            worksheet = self.spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = self.spreadsheet.add_worksheet(title=worksheet_name, rows="1000", cols="20")
        return worksheet

    def append_log(self, data, worksheet_name="trade_log"):
        """
        주어진 데이터를 워크시트의 마지막 행에 추가합니다.
        워크시트가 비어있는 경우, 데이터의 key를 헤더로 추가합니다.

        :param data: dict. 기록할 데이터. 예: {'timestamp': '...', 'strategy': '...', ...}
        :param worksheet_name: str. 데이터를 기록할 워크시트 이름
        """
        worksheet = self.get_worksheet(worksheet_name)
        if not worksheet:
            log_event("WARNING", "워크시트를 가져올 수 없어 로그 기록에 실패했습니다.")
            return

        try:
            # 워크시트가 비어있는지 확인 (헤더만 있거나 아예 없는 경우)
            if len(worksheet.get_all_records()) == 0:
                headers = list(data.keys())
                worksheet.append_row(headers)

            # 데이터 추가
            values = list(data.values())
            worksheet.append_row(values)
            log_event("INFO", f"'{worksheet_name}' 시트에 로그를 추가했습니다: {values}")

        except Exception as e:
            log_event("ERROR", f"스프레드시트 로그 추가 중 오류 발생: {e}")


# 싱글톤처럼 사용할 클라이언트 인스턴스
gspread_client = MockGspreadClient()

if __name__ == '__main__':
    # 테스트용 코드
    if gspread_client.client:
        test_log_data = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': 'test',
            'symbol': 'TEST.KS',
            'action': 'BUY',
            'price': 10000,
            'quantity': 10,
            'total_amount': 100000,
            'reason': '모듈 테스트'
        }
        gspread_client.append_log(test_log_data, worksheet_name="test_log") 