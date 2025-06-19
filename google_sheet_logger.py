import gspread
from google.oauth2.service_account import Credentials
import logging
from datetime import datetime
from typing import Optional
import asyncio
import os

logger = logging.getLogger(__name__)

class GoogleSheetLogger:
    def __init__(self, credentials_path: str, spreadsheet_key: Optional[str]):
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/drive'
        ]
        self.credentials_path = credentials_path
        self.spreadsheet_key = spreadsheet_key
        self.spreadsheet = None
        self.initialized = False

        if not self.spreadsheet_key:
            logger.info("구글 시트 키가 없어 비활성화합니다.")
            return
        
        if not os.path.exists(self.credentials_path):
            logger.warning(f"'{self.credentials_path}' 파일을 찾을 수 없습니다. 구글 시트 로깅이 비활성화됩니다.")
            self.credentials_path = None

    async def async_initialize(self):
        """비동기적으로 구글 시트에 연결합니다."""
        if not self.spreadsheet_key or not self.credentials_path:
            return

        try:
            await self._connect()
            self.initialized = True
        except gspread.exceptions.SpreadsheetNotFound:
            logger.warning(f"스프레드시트를 찾을 수 없습니다. 키가 정확한지, 시트가 공유 상태인지 확인하세요.")
        except Exception as e:
            logger.error(f"❌ 구글 시트 초기화 중 예상치 못한 오류 발생: {e}", exc_info=True)

    async def _connect(self):
        """구글 시트에 연결하고 스프레드시트를 엽니다."""
        logger.info("🔑 구글 시트 인증을 시작합니다...")
        
        def blocking_io():
            creds = Credentials.from_service_account_file(self.credentials_path, scopes=self.scope)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(self.spreadsheet_key)
            return spreadsheet

        self.spreadsheet = await asyncio.to_thread(blocking_io)
        logger.info(f"✅ 구글 시트 '{self.spreadsheet.title}'에 성공적으로 연결되었습니다.")

    async def _setup_header(self):
        """워크시트의 헤더가 비어있으면 설정합니다."""
        if not self.initialized or not self.spreadsheet:
            return
        
        try:
            worksheet = self.spreadsheet.sheet1
            
            def blocking_io():
                if worksheet.acell('A1').value is None:
                    header = [
                        "매매일시", "종목코드", "종목명", "매매구분", "수량", 
                        "가격", "주문사유", "총 매매금액", "수익률(%)", "실현손익", 
                        "AI 코멘트"
                    ]
                    worksheet.insert_row(header, 1)
                    return True
                return False

            if await asyncio.to_thread(blocking_io):
                logger.info("📄 구글 시트에 헤더를 설정했습니다.")
        except Exception as e:
            logger.error(f"⚠️ 구글 시트 헤더 설정 실패: {e}", exc_info=True)

    async def log_trade(self, trade_details: dict):
        """매매 기록을 구글 시트에 한 행으로 추가합니다."""
        if not self.initialized or not self.spreadsheet:
            logger.warning("구글 시트가 초기화되지 않아 매매 기록을 로깅할 수 없습니다.")
            return

        try:
            log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 필수 필드 확인
            required_fields = ['symbol', 'order_type', 'quantity', 'price', 'reason']
            if not all(field in trade_details for field in required_fields):
                logger.error(f"매매 기록에 필수 필드가 누락되었습니다: {trade_details}")
                return

            row_data = [
                log_time,
                trade_details.get('symbol', ''),
                trade_details.get('name', ''), # 종목명은 외부에서 추가 필요
                trade_details.get('order_type', '').upper(),
                trade_details.get('quantity', 0),
                trade_details.get('price', 0),
                trade_details.get('reason', ''),
                trade_details.get('total_amount', 0),
                trade_details.get('pnl_percent', ''),
                trade_details.get('realized_pnl', ''),
                trade_details.get('ai_comment', '')
            ]

            def blocking_io():
                worksheet = self.spreadsheet.sheet1
                worksheet.insert_row(row_data, 2)
            
            await asyncio.to_thread(blocking_io)
            logger.info(f"📈 구글 시트에 매매 기록 추가: {trade_details.get('symbol')} {trade_details.get('order_type')}")
        except Exception as e:
            logger.error(f"⚠️ 구글 시트 로깅 실패: {e}", exc_info=True)

    @classmethod
    def load_spreadsheet_key(cls, path: str) -> Optional[str]:
        """파일에서 스프레드시트 키를 읽어옵니다."""
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"스프레드시트 키 파일 '{path}'을(를) 찾을 수 없습니다.")
            return None

    async def get_or_create_worksheet(self, title: str):
        """기존 워크시트를 가져오거나 없으면 새로 생성합니다."""
        if not self.initialized:
            logger.warning("Logger가 초기화되지 않아 워크시트를 가져올 수 없습니다.")
            return None
        
        def blocking_io():
            try:
                return self.spreadsheet.worksheet(title)
            except gspread.exceptions.WorksheetNotFound:
                return None

        worksheet = await asyncio.to_thread(blocking_io)

        if worksheet:
            logger.info(f"기존 워크시트 '{title}'를 찾았습니다.")
            return worksheet
        else:
            logger.info(f"워크시트 '{title}'를 찾을 수 없어 새로 생성합니다.")
            
            async def create_worksheet_async():
                return self.spreadsheet.add_worksheet(title=title, rows="1", cols="1")

            try:
                worksheet = await asyncio.to_thread(create_worksheet_async)
                return worksheet
            except Exception as e:
                logger.error(f"⚠️ 구글 시트 워크시트 생성 실패: {e}", exc_info=True)
                return None

if __name__ == '__main__':
    # 테스트용 코드
    # 구글 서비스 계정 키 파일 경로 (service_account.json)
    # 이 파일은 구글 클라우드에서 다운로드해야 합니다.
    key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')
    # 데이터를 기록할 구글 시트 이름
    sheet_name = os.getenv('GOOGLE_SHEET_NAME', '자동매매일지')

    if not os.path.exists(key_path):
        print(f"'{key_path}' 파일을 찾을 수 없습니다. 구글 서비스 계정 키 파일을 다운로드하고 경로를 설정하세요.")
    else:
        logger.info(f"테스트를 위해 '{sheet_name}' 시트에 연결합니다.")
        sheet_logger = GoogleSheetLogger(credentials_path=key_path, spreadsheet_key=sheet_name)
        
        if sheet_logger.spreadsheet:
            # 테스트 데이터
            test_trade = {
                'symbol': '005930',
                'name': '삼성전자',
                'order_type': 'buy',
                'quantity': 1,
                'price': 75000,
                'total_amount': 75000,
                'reason': 'AI 추천 (거래대금 상위)',
                'pnl_percent': '',
                'realized_pnl': '',
                'ai_comment': '척후병 투입'
            }
            sheet_logger.log_trade(test_trade)
            print("테스트 매매 기록을 구글 시트에 추가했습니다.") 