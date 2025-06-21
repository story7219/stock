"""
구글 시트 로거 모듈 (Google Sheet Logger)
GitHub 참고: Google Sheets API v4 기반 로깅 시스템
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import json
import os
from pathlib import Path

# Google Sheets API 관련 임포트 (선택적)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread 라이브러리가 설치되지 않았습니다. pip install gspread google-auth 명령으로 설치하세요.")

logger = logging.getLogger(__name__)

class GoogleSheetLogger:
    """구글 시트 로거 클래스"""
    
    def __init__(self, credentials_file: Optional[str] = None, 
                 spreadsheet_key: Optional[str] = None,
                 fallback_to_csv: bool = True):
        """
        구글 시트 로거 초기화
        
        Args:
            credentials_file: 구글 서비스 계정 인증 파일 경로
            spreadsheet_key: 구글 스프레드시트 키
            fallback_to_csv: 구글 시트 연결 실패 시 CSV 파일로 대체 여부
        """
        self.credentials_file = credentials_file or os.getenv('GOOGLE_CREDENTIALS_FILE')
        self.spreadsheet_key = spreadsheet_key or os.getenv('GOOGLE_SPREADSHEET_KEY')
        self.fallback_to_csv = fallback_to_csv
        
        # 로컬 백업 디렉토리
        self.backup_dir = Path("data/logs")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 구글 시트 클라이언트
        self.client = None
        self.spreadsheet = None
        self.is_connected = False
        
        # 초기화 시도
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """비동기 초기화"""
        await self._connect_to_google_sheets()
    
    async def _connect_to_google_sheets(self):
        """구글 시트 연결"""
        try:
            if not GSPREAD_AVAILABLE:
                logger.warning("gspread 라이브러리가 없어 CSV 모드로 작동합니다.")
                return
            
            if not self.credentials_file or not self.spreadsheet_key:
                logger.warning("구글 시트 인증 정보가 없어 CSV 모드로 작동합니다.")
                return
            
            if not os.path.exists(self.credentials_file):
                logger.warning(f"인증 파일을 찾을 수 없습니다: {self.credentials_file}")
                return
            
            # 인증 스코프 설정
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # 서비스 계정 인증
            credentials = Credentials.from_service_account_file(
                self.credentials_file, scopes=scopes
            )
            
            # gspread 클라이언트 생성
            self.client = gspread.authorize(credentials)
            
            # 스프레드시트 열기
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_key)
            
            self.is_connected = True
            logger.info("✅ 구글 시트 연결 성공")
            
        except Exception as e:
            logger.error(f"❌ 구글 시트 연결 실패: {e}")
            if self.fallback_to_csv:
                logger.info("📄 CSV 파일 모드로 전환합니다.")
            self.is_connected = False
    
    async def log_trading_data(self, data: Dict[str, Any], sheet_name: str = "거래기록"):
        """거래 데이터 로깅"""
        try:
            # 기본 필드 추가
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **data
            }
            
            if self.is_connected:
                await self._log_to_google_sheet(log_data, sheet_name)
            else:
                await self._log_to_csv(log_data, f"{sheet_name}.csv")
            
            logger.info(f"📝 거래 데이터 로깅 완료: {data.get('stock_code', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ 거래 데이터 로깅 실패: {e}")
            # 백업으로 로컬 파일에 저장
            await self._emergency_backup(log_data, sheet_name)
    
    async def log_analysis_result(self, result: Dict[str, Any], sheet_name: str = "분석결과"):
        """분석 결과 로깅"""
        try:
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stock_code': result.get('stock_code', ''),
                'strategy': result.get('strategy', ''),
                'recommendation': result.get('recommendation', ''),
                'confidence': result.get('confidence', 0),
                'price': result.get('current_price', 0),
                'reason': result.get('reason', ''),
                'additional_data': json.dumps(result.get('additional_data', {}), ensure_ascii=False)
            }
            
            if self.is_connected:
                await self._log_to_google_sheet(log_data, sheet_name)
            else:
                await self._log_to_csv(log_data, f"{sheet_name}.csv")
            
            logger.info(f"📊 분석 결과 로깅 완료: {result.get('stock_code', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ 분석 결과 로깅 실패: {e}")
            await self._emergency_backup(log_data, sheet_name)
    
    async def log_portfolio_status(self, portfolio: Dict[str, Any], sheet_name: str = "포트폴리오"):
        """포트폴리오 상태 로깅"""
        try:
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_value': portfolio.get('total_value', 0),
                'cash': portfolio.get('cash', 0),
                'stock_value': portfolio.get('stock_value', 0),
                'profit_loss': portfolio.get('profit_loss', 0),
                'return_rate': portfolio.get('return_rate', 0),
                'holdings_count': len(portfolio.get('holdings', [])),
                'holdings_detail': json.dumps(portfolio.get('holdings', []), ensure_ascii=False)
            }
            
            if self.is_connected:
                await self._log_to_google_sheet(log_data, sheet_name)
            else:
                await self._log_to_csv(log_data, f"{sheet_name}.csv")
            
            logger.info("💼 포트폴리오 상태 로깅 완료")
            
        except Exception as e:
            logger.error(f"❌ 포트폴리오 로깅 실패: {e}")
            await self._emergency_backup(log_data, sheet_name)
    
    async def log_performance_metrics(self, metrics: Dict[str, Any], sheet_name: str = "성과지표"):
        """성과 지표 로깅"""
        try:
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'period': metrics.get('period', ''),
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'volatility': metrics.get('volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'trade_count': metrics.get('trade_count', 0)
            }
            
            if self.is_connected:
                await self._log_to_google_sheet(log_data, sheet_name)
            else:
                await self._log_to_csv(log_data, f"{sheet_name}.csv")
            
            logger.info("📈 성과 지표 로깅 완료")
            
        except Exception as e:
            logger.error(f"❌ 성과 지표 로깅 실패: {e}")
            await self._emergency_backup(log_data, sheet_name)
    
    async def log_error(self, error_data: Dict[str, Any], sheet_name: str = "오류로그"):
        """오류 로깅"""
        try:
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': error_data.get('error_type', ''),
                'error_message': str(error_data.get('error_message', '')),
                'function_name': error_data.get('function_name', ''),
                'stock_code': error_data.get('stock_code', ''),
                'severity': error_data.get('severity', 'ERROR'),
                'stack_trace': error_data.get('stack_trace', '')
            }
            
            if self.is_connected:
                await self._log_to_google_sheet(log_data, sheet_name)
            else:
                await self._log_to_csv(log_data, f"{sheet_name}.csv")
            
            logger.info("🚨 오류 로깅 완료")
            
        except Exception as e:
            logger.error(f"❌ 오류 로깅 실패: {e}")
            await self._emergency_backup(log_data, sheet_name)
    
    async def _log_to_google_sheet(self, data: Dict[str, Any], sheet_name: str):
        """구글 시트에 데이터 로깅"""
        try:
            # 워크시트 가져오기 또는 생성
            try:
                worksheet = self.spreadsheet.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
                # 헤더 추가
                headers = list(data.keys())
                worksheet.append_row(headers)
            
            # 데이터 행 추가
            values = [str(v) for v in data.values()]
            worksheet.append_row(values)
            
        except Exception as e:
            logger.error(f"❌ 구글 시트 쓰기 실패: {e}")
            # 연결 재시도
            await self._connect_to_google_sheets()
            raise
    
    async def _log_to_csv(self, data: Dict[str, Any], filename: str):
        """CSV 파일에 데이터 로깅"""
        try:
            file_path = self.backup_dir / filename
            
            # DataFrame 생성
            df = pd.DataFrame([data])
            
            # 파일이 존재하면 추가, 없으면 새로 생성
            if file_path.exists():
                df.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            else:
                df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            logger.error(f"❌ CSV 파일 쓰기 실패: {e}")
            raise
    
    async def _emergency_backup(self, data: Dict[str, Any], sheet_name: str):
        """긴급 백업 (JSON 파일)"""
        try:
            backup_file = self.backup_dir / f"emergency_backup_{sheet_name}.json"
            
            # 기존 데이터 로드
            backup_data = []
            if backup_file.exists():
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
            
            # 새 데이터 추가
            backup_data.append(data)
            
            # 파일 저장
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🆘 긴급 백업 완료: {backup_file}")
            
        except Exception as e:
            logger.error(f"❌ 긴급 백업 실패: {e}")
    
    async def get_recent_logs(self, sheet_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """최근 로그 조회"""
        try:
            if self.is_connected:
                return await self._get_logs_from_google_sheet(sheet_name, days)
            else:
                return await self._get_logs_from_csv(f"{sheet_name}.csv", days)
                
        except Exception as e:
            logger.error(f"❌ 로그 조회 실패: {e}")
            return []
    
    async def _get_logs_from_google_sheet(self, sheet_name: str, days: int) -> List[Dict[str, Any]]:
        """구글 시트에서 로그 조회"""
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
            records = worksheet.get_all_records()
            
            # 최근 n일 데이터 필터링
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_records = []
            
            for record in records:
                try:
                    timestamp = datetime.strptime(record.get('timestamp', ''), '%Y-%m-%d %H:%M:%S')
                    if timestamp >= cutoff_date:
                        recent_records.append(record)
                except:
                    continue
            
            return recent_records
            
        except Exception as e:
            logger.error(f"❌ 구글 시트 조회 실패: {e}")
            return []
    
    async def _get_logs_from_csv(self, filename: str, days: int) -> List[Dict[str, Any]]:
        """CSV 파일에서 로그 조회"""
        try:
            file_path = self.backup_dir / filename
            
            if not file_path.exists():
                return []
            
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 최근 n일 데이터 필터링
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[df['timestamp'] >= cutoff_date]
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ CSV 파일 조회 실패: {e}")
            return []
    
    async def create_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """일일 요약 리포트 생성"""
        try:
            target_date = date or datetime.now().strftime('%Y-%m-%d')
            
            # 각 시트에서 해당 날짜 데이터 조회
            trading_logs = await self._get_logs_by_date("거래기록", target_date)
            analysis_logs = await self._get_logs_by_date("분석결과", target_date)
            error_logs = await self._get_logs_by_date("오류로그", target_date)
            
            summary = {
                'date': target_date,
                'trading_count': len(trading_logs),
                'analysis_count': len(analysis_logs),
                'error_count': len(error_logs),
                'most_analyzed_stocks': self._get_most_frequent_stocks(analysis_logs),
                'error_summary': self._summarize_errors(error_logs)
            }
            
            # 요약 로깅
            await self.log_daily_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 일일 요약 생성 실패: {e}")
            return {}
    
    async def _get_logs_by_date(self, sheet_name: str, date: str) -> List[Dict[str, Any]]:
        """특정 날짜의 로그 조회"""
        try:
            all_logs = await self.get_recent_logs(sheet_name, 30)  # 최근 30일
            date_logs = []
            
            for log in all_logs:
                log_date = log.get('timestamp', '')[:10]  # YYYY-MM-DD 부분만 추출
                if log_date == date:
                    date_logs.append(log)
            
            return date_logs
            
        except Exception as e:
            logger.error(f"❌ 날짜별 로그 조회 실패: {e}")
            return []
    
    def _get_most_frequent_stocks(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """가장 많이 분석된 종목 조회"""
        try:
            stock_counts = {}
            for log in logs:
                stock_code = log.get('stock_code', '')
                if stock_code:
                    stock_counts[stock_code] = stock_counts.get(stock_code, 0) + 1
            
            # 상위 5개 종목
            sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
            return [{'stock_code': code, 'count': count} for code, count in sorted_stocks[:5]]
            
        except Exception as e:
            logger.error(f"❌ 빈도 분석 실패: {e}")
            return []
    
    def _summarize_errors(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """오류 요약"""
        try:
            if not error_logs:
                return {'total': 0, 'by_type': {}}
            
            error_types = {}
            for log in error_logs:
                error_type = log.get('error_type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            return {
                'total': len(error_logs),
                'by_type': error_types
            }
            
        except Exception as e:
            logger.error(f"❌ 오류 요약 실패: {e}")
            return {'total': 0, 'by_type': {}}
    
    async def log_daily_summary(self, summary: Dict[str, Any], sheet_name: str = "일일요약"):
        """일일 요약 로깅"""
        try:
            log_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': summary.get('date', ''),
                'trading_count': summary.get('trading_count', 0),
                'analysis_count': summary.get('analysis_count', 0),
                'error_count': summary.get('error_count', 0),
                'top_stocks': json.dumps(summary.get('most_analyzed_stocks', []), ensure_ascii=False),
                'error_summary': json.dumps(summary.get('error_summary', {}), ensure_ascii=False)
            }
            
            if self.is_connected:
                await self._log_to_google_sheet(log_data, sheet_name)
            else:
                await self._log_to_csv(log_data, f"{sheet_name}.csv")
            
            logger.info("📋 일일 요약 로깅 완료")
            
        except Exception as e:
            logger.error(f"❌ 일일 요약 로깅 실패: {e}")
    
    async def cleanup_old_logs(self, days_to_keep: int = 90):
        """오래된 로그 정리"""
        try:
            if not self.is_connected:
                # CSV 파일 정리
                await self._cleanup_csv_files(days_to_keep)
                return
            
            # 구글 시트는 수동으로 정리 (데이터 손실 방지)
            logger.info(f"구글 시트의 {days_to_keep}일 이전 데이터는 수동으로 정리해주세요.")
            
        except Exception as e:
            logger.error(f"❌ 로그 정리 실패: {e}")
    
    async def _cleanup_csv_files(self, days_to_keep: int):
        """CSV 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for csv_file in self.backup_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df_filtered = df[df['timestamp'] >= cutoff_date]
                        
                        if len(df_filtered) < len(df):
                            df_filtered.to_csv(csv_file, index=False, encoding='utf-8-sig')
                            logger.info(f"📁 {csv_file.name} 정리 완료: {len(df) - len(df_filtered)}개 행 삭제")
                
                except Exception as e:
                    logger.error(f"❌ {csv_file.name} 정리 실패: {e}")
            
        except Exception as e:
            logger.error(f"❌ CSV 파일 정리 실패: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 조회"""
        return {
            'is_connected': self.is_connected,
            'has_credentials': bool(self.credentials_file and os.path.exists(self.credentials_file)),
            'has_spreadsheet_key': bool(self.spreadsheet_key),
            'gspread_available': GSPREAD_AVAILABLE,
            'fallback_mode': not self.is_connected and self.fallback_to_csv,
            'backup_dir': str(self.backup_dir)
        }
    
    async def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            if self.is_connected:
                # 간단한 읽기 테스트
                worksheets = self.spreadsheet.worksheets()
                logger.info(f"✅ 구글 시트 연결 테스트 성공: {len(worksheets)}개 워크시트")
                return True
            else:
                # CSV 모드 테스트
                test_data = {'test': 'connection_test', 'timestamp': datetime.now().isoformat()}
                await self._log_to_csv(test_data, "connection_test.csv")
                logger.info("✅ CSV 모드 연결 테스트 성공")
                return True
                
        except Exception as e:
            logger.error(f"❌ 연결 테스트 실패: {e}")
            return False
    
    async def close(self):
        """리소스 정리"""
        try:
            if self.client:
                # gspread 클라이언트는 별도 종료 메서드가 없음
                pass
            
            logger.info("✅ 구글 시트 로거 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 구글 시트 로거 정리 실패: {e}")

# 전역 인스턴스 (선택적 사용)
_global_logger = None

def get_global_logger() -> GoogleSheetLogger:
    """전역 로거 인스턴스 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = GoogleSheetLogger()
    return _global_logger

async def log_trading_data(data: Dict[str, Any], sheet_name: str = "거래기록"):
    """전역 함수: 거래 데이터 로깅"""
    logger_instance = get_global_logger()
    await logger_instance.log_trading_data(data, sheet_name)

async def log_analysis_result(result: Dict[str, Any], sheet_name: str = "분석결과"):
    """전역 함수: 분석 결과 로깅"""
    logger_instance = get_global_logger()
    await logger_instance.log_analysis_result(result, sheet_name) 