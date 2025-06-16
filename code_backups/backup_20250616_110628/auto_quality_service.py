"""
완전 자동화된 코드 품질 검사 서비스
Windows 서비스로 실행되어 매일 오전 7시에 자동으로 품질 검사를 수행합니다.
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
import asyncio
from typing import Optional

# 프로젝트 루트 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quality_analyzer import CodeQualityAnalyzer

# 로깅 설정
log_file = project_root / "auto_quality_service.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoQualityService:
    """자동 품질 검사 서비스"""
    
    def __init__(self):
        self.analyzer = CodeQualityAnalyzer()
        self.running = False
        self.target_time = dt_time(7, 0)  # 오전 7시
        self.last_run_date = None
        self.missed_run_file = Path.cwd() / ".last_quality_check"
        
    def should_run_analysis(self) -> bool:
        """분석을 실행해야 하는지 확인 (놓친 실행 포함)"""
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()
        
        # 마지막 실행 날짜 확인
        last_run = self.get_last_run_date()
        
        # 어제 실행하지 못했으면 지금 실행
        yesterday = current_date - timedelta(days=1)
        if last_run < yesterday:
            logger.info("🔄 놓친 품질 검사를 실행합니다")
            return True
        
        # 오늘 이미 실행했으면 실행하지 않음
        if last_run == current_date:
            return False
        
        # 현재 시간이 목표 시간(오전 7시)을 지났는지 확인
        if current_time >= self.target_time:
            return True
            
        return False
    
    def get_last_run_date(self) -> datetime.date:
        """마지막 실행 날짜 가져오기"""
        try:
            if self.missed_run_file.exists():
                with open(self.missed_run_file, 'r') as f:
                    date_str = f.read().strip()
                    return datetime.fromisoformat(date_str).date()
        except Exception:
            pass
        
        # 파일이 없으면 어제 날짜 반환 (첫 실행을 위해)
        return datetime.now().date() - timedelta(days=1)
    
    def save_last_run_date(self, date: datetime.date):
        """마지막 실행 날짜 저장"""
        try:
            with open(self.missed_run_file, 'w') as f:
                f.write(date.isoformat())
        except Exception as e:
            logger.warning(f"실행 날짜 저장 실패: {e}")
    
    async def run_daily_analysis(self):
        """일일 품질 분석 실행"""
        try:
            logger.info("🚀 자동 품질 분석 시작")
            
            # 분석 실행
            report = await self.analyzer.run_quality_analysis()
            
            # 실행 날짜 저장
            current_date = datetime.now().date()
            self.save_last_run_date(current_date)
            
            # 결과 요약 로그
            logger.info(f"✅ 품질 분석 완료!")
            logger.info(f"📊 전체 점수: {report.overall_score}/100")
            logger.info(f"📁 분석된 파일 수: {len(report.file_metrics)}")
            logger.info(f"💡 권장사항 수: {len(report.recommendations)}")
            
            # 중요한 이슈가 있으면 특별 알림
            if report.overall_score < 50:
                logger.warning(f"⚠️ 품질 점수가 낮습니다: {report.overall_score}/100")
                self.send_alert(f"코드 품질 점수가 {report.overall_score}점으로 낮습니다!")
            
            total_issues = sum(len(m.code_smells) + len(m.security_issues) for m in report.file_metrics)
            if total_issues > 10:
                logger.warning(f"⚠️ 총 {total_issues}개의 이슈가 발견되었습니다")
                self.send_alert(f"총 {total_issues}개의 코드 이슈가 발견되었습니다!")
            
            logger.info("🎉 일일 품질 분석 완료")
            
        except Exception as e:
            logger.error(f"❌ 품질 분석 실패: {e}")
            self.send_alert(f"품질 분석 중 오류 발생: {str(e)}")
    
    def send_alert(self, message: str):
        """알림 전송 (Windows 알림)"""
        try:
            import win10toast
            toaster = win10toast.ToastNotifier()
            toaster.show_toast(
                "코드 품질 검사",
                message,
                duration=10,
                icon_path=None
            )
        except ImportError:
            # win10toast가 없으면 로그로만 기록
            logger.warning(f"알림: {message}")
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")
    
    def start_service(self):
        """서비스 시작"""
        self.running = True
        logger.info("🔄 자동 품질 검사 서비스 시작")
        logger.info(f"⏰ 매일 {self.target_time.strftime('%H:%M')}에 자동 실행됩니다")
        
        while self.running:
            try:
                if self.should_run_analysis():
                    # 비동기 분석 실행
                    asyncio.run(self.run_daily_analysis())
                
                # 1분마다 체크
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 사용자에 의해 서비스 중단")
                break
            except Exception as e:
                logger.error(f"❌ 서비스 실행 중 오류: {e}")
                time.sleep(300)  # 5분 후 재시도
    
    def stop_service(self):
        """서비스 중단"""
        self.running = False
        logger.info("🛑 자동 품질 검사 서비스 중단")

class QualityScheduler:
    """품질 검사 스케줄러 (구글 시트 로깅 포함)"""
    
    def __init__(self):
        self.analyzer = CodeQualityAnalyzer()
        
    async def run_scheduled_analysis(self):
        """스케줄된 분석 실행 (구글 시트 로깅 포함)"""
        try:
            logger.info("=== 자동 품질 검사 + 구글 시트 로깅 시작 ===")
            
            # 1. 품질 분석 실행
            report = await self.analyzer.run_quality_analysis()
            
            # 2. 리팩토링 제안 생성
            from auto_refactoring_system import AutoRefactoringSystem
            refactoring_system = AutoRefactoringSystem()
            proposals = await refactoring_system.generate_refactoring_proposals(report)
            
            # 세션 생성 (자동 실행용)
            from auto_refactoring_system import RefactoringSession
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session = RefactoringSession(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                proposals=proposals
            )
            
            # 3. 구글 시트에 로깅
            from google_sheets_integration import AutomatedGoogleSheetsLogger
            sheets_logger = AutomatedGoogleSheetsLogger()
            result = await sheets_logger.log_daily_analysis(report, session)
            
            # 4. 결과 로깅
            if result['success']:
                logger.info("✅ 구글 시트 로깅 성공")
                trend = result.get('trend', {})
                if 'trend' in trend:
                    logger.info(f"📈 품질 트렌드: {trend['trend']}")
            else:
                logger.error(f"❌ 구글 시트 로깅 실패: {result.get('error')}")
            
            # 5. 중요한 이슈가 있으면 알림
            if report.overall_score < 50:
                logger.warning(f"⚠️ 품질 점수가 낮습니다: {report.overall_score}")
            
            total_issues = sum(len(m.code_smells) + len(m.security_issues) for m in report.file_metrics)
            if total_issues > 10:
                logger.warning(f"⚠️ 총 {total_issues}개의 이슈가 발견되었습니다")
            
            logger.info("=== 자동 품질 검사 + 구글 시트 로깅 완료 ===")
            
        except Exception as e:
            logger.error(f"스케줄된 분석 실패: {e}")

def create_windows_task():
    """Windows 작업 스케줄러에 작업 등록 (컴퓨터 꺼져있어도 실행 가능)"""
    try:
        import subprocess
        
        # 현재 스크립트 경로
        script_path = Path(__file__).absolute()
        python_path = sys.executable
        
        # 작업 스케줄러 명령어 생성
        task_name = "AutoCodeQualityCheck"
        
        # 기존 작업 삭제 (있다면)
        try:
            subprocess.run([
                "schtasks", "/delete", "/tn", task_name, "/f"
            ], capture_output=True, check=False)
        except:
            pass
        
        # 새 작업 생성 - 스마트 실행 옵션 추가
        cmd = [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", f'"{python_path}" "{script_path}" service',
            "/sc", "daily",
            "/st", "07:00",
            "/ru", "SYSTEM",  # 시스템 권한으로 실행
            "/rl", "HIGHEST",  # 최고 권한
            "/f"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 추가 설정: 놓친 실행 시 다음 부팅 시 실행
            configure_advanced_settings(task_name)
            
            logger.info("✅ Windows 작업 스케줄러에 등록 완료")
            logger.info("📅 매일 오전 7시에 자동으로 실행됩니다")
            logger.info("🔄 컴퓨터가 꺼져있었다면 다음 부팅 시 실행됩니다")
            return True
        else:
            logger.error(f"❌ 작업 스케줄러 등록 실패: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Windows 작업 생성 실패: {e}")
        return False

def configure_advanced_settings(task_name: str):
    """고급 설정 적용 (놓친 실행 처리)"""
    try:
        import subprocess
        
        # XML 설정 파일 생성
        xml_config = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>자동 코드 품질 검사 서비스</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2024-01-01T07:00:00</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
    <BootTrigger>
      <Delay>PT5M</Delay>
    </BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>{sys.executable}</Command>
      <Arguments>"{Path(__file__).absolute()}" service</Arguments>
    </Exec>
  </Actions>
</Task>"""
        
        # 임시 XML 파일 저장
        xml_file = Path.cwd() / "temp_task.xml"
        with open(xml_file, 'w', encoding='utf-16') as f:
            f.write(xml_config)
        
        # XML로 작업 업데이트
        subprocess.run([
            "schtasks", "/create", "/xml", str(xml_file), "/tn", task_name, "/f"
        ], capture_output=True)
        
        # 임시 파일 삭제
        xml_file.unlink(missing_ok=True)
        
        logger.info("🔧 고급 설정 적용 완료")
        
    except Exception as e:
        logger.warning(f"고급 설정 적용 실패: {e}")

def remove_windows_task():
    """Windows 작업 스케줄러에서 작업 제거"""
    try:
        import subprocess
        
        task_name = "AutoCodeQualityCheck"
        
        result = subprocess.run([
            "schtasks", "/delete", "/tn", task_name, "/f"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Windows 작업 스케줄러에서 제거 완료")
            return True
        else:
            logger.error(f"❌ 작업 제거 실패: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Windows 작업 제거 실패: {e}")
        return False

def install_service():
    """서비스 설치"""
    print("🔧 자동 품질 검사 서비스 설치 중...")
    
    # 필요한 패키지 설치
    try:
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "win10toast", "google-generativeai", "schedule", "python-dotenv"
        ], check=True)
        print("✅ 필요한 패키지 설치 완료")
    except Exception as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False
    
    # Windows 작업 스케줄러에 등록
    if create_windows_task():
        print("🎉 자동 품질 검사 서비스 설치 완료!")
        print("📅 매일 오전 7시에 자동으로 코드 품질 검사가 실행됩니다.")
        print("📁 결과는 quality_reports 폴더에 저장됩니다.")
        return True
    else:
        print("❌ 서비스 설치 실패")
        return False

def uninstall_service():
    """서비스 제거"""
    print("🗑️ 자동 품질 검사 서비스 제거 중...")
    
    if remove_windows_task():
        print("✅ 자동 품질 검사 서비스 제거 완료")
        return True
    else:
        print("❌ 서비스 제거 실패")
        return False

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("""
🔍 자동 코드 품질 검사 서비스

사용법:
    python auto_quality_service.py [명령어]

명령어:
    install     - 서비스 설치 (Windows 작업 스케줄러에 등록)
    uninstall   - 서비스 제거
    service     - 서비스 실행 (내부 사용)
    now         - 즉시 분석 실행
    status      - 서비스 상태 확인

설치 후에는 매일 오전 7시에 자동으로 코드 품질 검사가 실행됩니다.
""")
        return
    
    command = sys.argv[1].lower()
    
    if command == "install":
        install_service()
        
    elif command == "uninstall":
        uninstall_service()
        
    elif command == "service":
        # 실제 서비스 실행
        service = AutoQualityService()
        service.start_service()
        
    elif command == "now":
        # 즉시 분석 실행 - 이 부분이 수동 실행을 처리
        print("📊 즉시 분석을 시작합니다...")
        service = AutoQualityService()
        asyncio.run(service.run_daily_analysis())
        
    elif command == "status":
        # 서비스 상태 확인
        try:
            import subprocess
            result = subprocess.run([
                "schtasks", "/query", "/tn", "AutoCodeQualityCheck"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 자동 품질 검사 서비스가 설치되어 있습니다")
                print("📅 매일 오전 7시에 자동 실행됩니다")
            else:
                print("❌ 자동 품질 검사 서비스가 설치되어 있지 않습니다")
                print("💡 'python auto_quality_service.py install' 명령으로 설치하세요")
        except Exception as e:
            print(f"❌ 상태 확인 실패: {e}")
    
    else:
        print(f"❌ 알 수 없는 명령어: {command}")

if __name__ == "__main__":
    main() 