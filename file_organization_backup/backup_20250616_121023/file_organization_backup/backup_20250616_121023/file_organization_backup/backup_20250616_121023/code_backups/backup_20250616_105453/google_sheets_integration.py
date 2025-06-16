import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import json
import logging
from dataclasses import asdict
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

from auto_refactoring_system import RefactoringSession, RefactoringProposal
from quality_analyzer import QualityReport

logger = logging.getLogger(__name__)

class GoogleSheetsManager:
    """구글 시트 관리 클래스"""
    
    def __init__(self):
        self.setup_google_sheets()
        self.spreadsheet_id = None
        self.worksheet = None
        
    def setup_google_sheets(self):
        """구글 시트 API 설정"""
        try:
            load_dotenv()
            
            # 서비스 계정 키 파일 경로
            credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'google_credentials.json')
            
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"구글 인증 파일을 찾을 수 없습니다: {credentials_path}")
            
            # 구글 시트 API 스코프 설정
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # 인증 정보 로드
            credentials = Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
            
            # gspread 클라이언트 생성
            self.gc = gspread.authorize(credentials)
            
            logger.info("구글 시트 API 설정 완료")
            
        except Exception as e:
            logger.error(f"구글 시트 API 설정 실패: {e}")
            raise
    
    def create_or_get_spreadsheet(self, spreadsheet_name: str = "코드품질_리팩토링_로그"):
        """스프레드시트 생성 또는 기존 시트 가져오기"""
        try:
            # 기존 스프레드시트 찾기
            try:
                spreadsheet = self.gc.open(spreadsheet_name)
                logger.info(f"기존 스프레드시트 발견: {spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                # 새 스프레드시트 생성
                spreadsheet = self.gc.create(spreadsheet_name)
                logger.info(f"새 스프레드시트 생성: {spreadsheet_name}")
                
                # 공유 설정 (선택사항)
                # spreadsheet.share('your-email@gmail.com', perm_type='user', role='writer')
            
            self.spreadsheet = spreadsheet
            self.setup_worksheets()
            
            return spreadsheet
            
        except Exception as e:
            logger.error(f"스프레드시트 생성/접근 실패: {e}")
            raise
    
    def setup_worksheets(self):
        """워크시트 설정 및 헤더 생성"""
        try:
            # 1. 품질 분석 결과 시트
            try:
                self.quality_sheet = self.spreadsheet.worksheet("품질분석결과")
            except gspread.WorksheetNotFound:
                self.quality_sheet = self.spreadsheet.add_worksheet(
                    title="품질분석결과", rows=1000, cols=20
                )
                
                # 헤더 설정
                quality_headers = [
                    "날짜", "시간", "전체점수", "분석파일수", "총코드라인", 
                    "평균복잡도", "코드스멜수", "보안이슈수", "유지보수성점수",
                    "트렌드분석", "주요권장사항", "Gemini분석요약"
                ]
                self.quality_sheet.append_row(quality_headers)
            
            # 2. 리팩토링 제안 시트
            try:
                self.refactoring_sheet = self.spreadsheet.worksheet("리팩토링제안")
            except gspread.WorksheetNotFound:
                self.refactoring_sheet = self.spreadsheet.add_worksheet(
                    title="리팩토링제안", rows=1000, cols=15
                )
                
                # 헤더 설정
                refactoring_headers = [
                    "날짜", "시간", "세션ID", "파일경로", "이슈유형", 
                    "설명", "위험도", "신뢰도", "상태", "승인여부", 
                    "적용여부", "개선효과", "원본코드미리보기", "제안코드미리보기"
                ]
                self.refactoring_sheet.append_row(refactoring_headers)
            
            # 3. 일일 요약 시트
            try:
                self.summary_sheet = self.spreadsheet.worksheet("일일요약")
            except gspread.WorksheetNotFound:
                self.summary_sheet = self.spreadsheet.add_worksheet(
                    title="일일요약", rows=1000, cols=12
                )
                
                # 헤더 설정
                summary_headers = [
                    "날짜", "품질점수", "품질변화", "총제안수", "승인수", 
                    "거부수", "적용수", "주요개선사항", "다음액션", 
                    "코드품질등급", "보안위험도", "전체평가"
                ]
                self.summary_sheet.append_row(summary_headers)
            
            logger.info("워크시트 설정 완료")
            
        except Exception as e:
            logger.error(f"워크시트 설정 실패: {e}")
            raise
    
    def save_quality_analysis(self, report: QualityReport):
        """품질 분석 결과를 구글 시트에 저장"""
        try:
            # 데이터 준비
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # 메트릭 계산
            total_files = len(report.file_metrics)
            total_loc = sum(m.lines_of_code for m in report.file_metrics)
            avg_complexity = sum(m.complexity for m in report.file_metrics) / total_files if total_files > 0 else 0
            total_smells = sum(len(m.code_smells) for m in report.file_metrics)
            total_security = sum(len(m.security_issues) for m in report.file_metrics)
            avg_maintainability = sum(m.maintainability_index for m in report.file_metrics) / total_files if total_files > 0 else 0
            
            # 주요 권장사항 요약
            recommendations_summary = " | ".join(report.recommendations[:3])  # 상위 3개만
            
            # Gemini 분석 요약 (처음 200자만)
            gemini_summary = report.gemini_analysis[:200] + "..." if len(report.gemini_analysis) > 200 else report.gemini_analysis
            
            # 행 데이터 생성
            row_data = [
                date_str,
                time_str,
                round(report.overall_score, 2),
                total_files,
                total_loc,
                round(avg_complexity, 2),
                total_smells,
                total_security,
                round(avg_maintainability, 2),
                report.trend_analysis,
                recommendations_summary,
                gemini_summary
            ]
            
            # 구글 시트에 추가
            self.quality_sheet.append_row(row_data)
            
            logger.info(f"품질 분석 결과 저장 완료: {date_str} {time_str}")
            
        except Exception as e:
            logger.error(f"품질 분석 결과 저장 실패: {e}")
            raise
    
    def save_refactoring_proposals(self, session: RefactoringSession):
        """리팩토링 제안을 구글 시트에 저장"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            for proposal in session.proposals:
                # 코드 미리보기 (처음 100자만)
                original_preview = proposal.original_code[:100] + "..." if len(proposal.original_code) > 100 else proposal.original_code
                proposed_preview = proposal.proposed_code[:100] + "..." if len(proposal.proposed_code) > 100 else proposal.proposed_code
                
                # 행 데이터 생성
                row_data = [
                    date_str,
                    time_str,
                    session.session_id,
                    proposal.file_path,
                    proposal.issue_type,
                    proposal.description,
                    proposal.risk_level,
                    f"{proposal.confidence:.0%}",
                    "대기중",  # 초기 상태
                    "미정",    # 승인여부
                    "미적용",  # 적용여부
                    proposal.explanation,
                    original_preview,
                    proposed_preview
                ]
                
                # 구글 시트에 추가
                self.refactoring_sheet.append_row(row_data)
            
            logger.info(f"리팩토링 제안 저장 완료: {len(session.proposals)}개 제안")
            
        except Exception as e:
            logger.error(f"리팩토링 제안 저장 실패: {e}")
            raise
    
    def save_daily_summary(self, report: QualityReport, session: RefactoringSession):
        """일일 요약을 구글 시트에 저장"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            
            # 품질 등급 결정
            score = report.overall_score
            if score >= 80:
                quality_grade = "우수"
            elif score >= 60:
                quality_grade = "양호"
            elif score >= 40:
                quality_grade = "보통"
            else:
                quality_grade = "개선필요"
            
            # 보안 위험도 계산
            total_security_issues = sum(len(m.security_issues) for m in report.file_metrics)
            if total_security_issues == 0:
                security_risk = "낮음"
            elif total_security_issues <= 3:
                security_risk = "보통"
            else:
                security_risk = "높음"
            
            # 품질 변화 (이전 데이터와 비교 - 간단한 예시)
            quality_change = "분석중"  # 실제로는 이전 데이터와 비교
            
            # 주요 개선사항 요약
            major_improvements = []
            low_risk_count = len([p for p in session.proposals if p.risk_level == "LOW"])
            medium_risk_count = len([p for p in session.proposals if p.risk_level == "MEDIUM"])
            high_risk_count = len([p for p in session.proposals if p.risk_level == "HIGH"])
            
            if low_risk_count > 0:
                major_improvements.append(f"낮은위험도 {low_risk_count}건")
            if medium_risk_count > 0:
                major_improvements.append(f"중간위험도 {medium_risk_count}건")
            if high_risk_count > 0:
                major_improvements.append(f"높은위험도 {high_risk_count}건")
            
            improvements_text = ", ".join(major_improvements)
            
            # 다음 액션 제안
            next_actions = []
            if high_risk_count > 0:
                next_actions.append("보안이슈 우선해결")
            if medium_risk_count > 0:
                next_actions.append("구조개선 검토")
            if low_risk_count > 0:
                next_actions.append("코드정리 진행")
            
            next_action_text = ", ".join(next_actions) if next_actions else "현재상태 유지"
            
            # 전체 평가
            if score >= 70 and total_security_issues == 0:
                overall_evaluation = "매우양호"
            elif score >= 50 and total_security_issues <= 2:
                overall_evaluation = "양호"
            elif score >= 30:
                overall_evaluation = "개선필요"
            else:
                overall_evaluation = "즉시개선필요"
            
            # 행 데이터 생성
            row_data = [
                date_str,
                round(report.overall_score, 2),
                quality_change,
                len(session.proposals),
                session.approved_count,
                session.rejected_count,
                session.applied_count,
                improvements_text,
                next_action_text,
                quality_grade,
                security_risk,
                overall_evaluation
            ]
            
            # 구글 시트에 추가
            self.summary_sheet.append_row(row_data)
            
            logger.info(f"일일 요약 저장 완료: {date_str}")
            
        except Exception as e:
            logger.error(f"일일 요약 저장 실패: {e}")
            raise
    
    def update_proposal_status(self, session_id: str, proposal_index: int, status: str, approved: bool = False, applied: bool = False):
        """리팩토링 제안 상태 업데이트"""
        try:
            # 해당 세션의 제안들 찾기
            all_values = self.refactoring_sheet.get_all_values()
            
            for i, row in enumerate(all_values[1:], start=2):  # 헤더 제외
                if len(row) >= 3 and row[2] == session_id:  # 세션 ID 확인
                    # 상태 업데이트
                    self.refactoring_sheet.update_cell(i, 9, status)  # 상태 컬럼
                    self.refactoring_sheet.update_cell(i, 10, "승인" if approved else "거부" if status == "거부됨" else "미정")  # 승인여부
                    self.refactoring_sheet.update_cell(i, 11, "적용완료" if applied else "미적용")  # 적용여부
                    break
            
            logger.info(f"제안 상태 업데이트: {session_id} - {status}")
            
        except Exception as e:
            logger.error(f"제안 상태 업데이트 실패: {e}")
    
    def get_quality_trend(self, days: int = 7) -> Dict[str, Any]:
        """최근 품질 트렌드 분석"""
        try:
            # 품질 분석 데이터 가져오기
            all_values = self.quality_sheet.get_all_values()
            
            if len(all_values) <= 1:  # 헤더만 있는 경우
                return {"message": "분석할 데이터가 부족합니다"}
            
            # 최근 데이터 추출
            recent_data = all_values[-days:] if len(all_values) > days else all_values[1:]
            
            scores = []
            dates = []
            
            for row in recent_data:
                if len(row) >= 3:
                    try:
                        score = float(row[2])  # 전체점수 컬럼
                        scores.append(score)
                        dates.append(row[0])  # 날짜 컬럼
                    except ValueError:
                        continue
            
            if not scores:
                return {"message": "유효한 점수 데이터가 없습니다"}
            
            # 트렌드 분석
            avg_score = sum(scores) / len(scores)
            trend = "상승" if scores[-1] > scores[0] else "하락" if scores[-1] < scores[0] else "안정"
            
            return {
                "average_score": round(avg_score, 2),
                "latest_score": scores[-1],
                "trend": trend,
                "data_points": len(scores),
                "date_range": f"{dates[0]} ~ {dates[-1]}" if len(dates) > 1 else dates[0]
            }
            
        except Exception as e:
            logger.error(f"품질 트렌드 분석 실패: {e}")
            return {"error": str(e)}

class AutomatedGoogleSheetsLogger:
    """자동화된 구글 시트 로깅 시스템"""
    
    def __init__(self):
        self.sheets_manager = GoogleSheetsManager()
        self.sheets_manager.create_or_get_spreadsheet()
    
    async def log_daily_analysis(self, report: QualityReport, session: RefactoringSession):
        """일일 분석 결과를 구글 시트에 로깅"""
        try:
            logger.info("=== 구글 시트 로깅 시작 ===")
            
            # 1. 품질 분석 결과 저장
            self.sheets_manager.save_quality_analysis(report)
            
            # 2. 리팩토링 제안 저장
            self.sheets_manager.save_refactoring_proposals(session)
            
            # 3. 일일 요약 저장
            self.sheets_manager.save_daily_summary(report, session)
            
            # 4. 트렌드 분석
            trend_data = self.sheets_manager.get_quality_trend()
            
            logger.info("=== 구글 시트 로깅 완료 ===")
            logger.info(f"📊 품질 점수: {report.overall_score}")
            logger.info(f"📈 트렌드: {trend_data.get('trend', '분석중')}")
            logger.info(f"💡 제안 수: {len(session.proposals)}")
            
            return {
                "success": True,
                "quality_score": report.overall_score,
                "proposals_count": len(session.proposals),
                "trend": trend_data
            }
            
        except Exception as e:
            logger.error(f"구글 시트 로깅 실패: {e}")
            return {"success": False, "error": str(e)}

# 기존 시스템에 통합
async def run_daily_analysis_with_sheets():
    """구글 시트 로깅이 포함된 일일 분석"""
    try:
        # 1. 품질 분석 실행
        from quality_analyzer import CodeQualityAnalyzer
        from auto_refactoring_system import AutoRefactoringSystem
        
        analyzer = CodeQualityAnalyzer()
        report = await analyzer.run_quality_analysis()
        
        # 2. 리팩토링 제안 생성
        refactoring_system = AutoRefactoringSystem()
        session = await refactoring_system.run_semi_automatic_refactoring(report)
        
        # 3. 구글 시트에 로깅
        sheets_logger = AutomatedGoogleSheetsLogger()
        result = await sheets_logger.log_daily_analysis(report, session)
        
        print(f"\n🎯 일일 분석 및 구글 시트 로깅 완료!")
        print(f"📊 품질 점수: {report.overall_score}/100")
        print(f"💡 리팩토링 제안: {len(session.proposals)}개")
        print(f"📈 구글 시트 저장: {'성공' if result['success'] else '실패'}")
        
        if result['success']:
            trend = result.get('trend', {})
            if 'trend' in trend:
                print(f"📈 품질 트렌드: {trend['trend']}")
        
        return result
        
    except Exception as e:
        logger.error(f"일일 분석 실행 실패: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "daily":
        # 일일 분석 + 구글 시트 로깅 실행
        asyncio.run(run_daily_analysis_with_sheets())
    else:
        print("사용법: python google_sheets_integration.py daily") 