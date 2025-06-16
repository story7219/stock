"""
구글 시트 연결 및 기본 기능 테스트
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_google_sheets_connection():
    """구글 시트 연결 테스트"""
    try:
        print("🔗 구글 시트 연결 테스트 시작...")
        
        from google_sheets_integration import GoogleSheetsManager
        
        # 1. 구글 시트 매니저 초기화
        sheets_manager = GoogleSheetsManager()
        print("✅ 구글 시트 API 연결 성공")
        
        # 2. 테스트용 스프레드시트 생성
        spreadsheet = sheets_manager.create_or_get_spreadsheet("테스트_코드품질_로그")
        print(f"✅ 스프레드시트 생성/접근 성공: {spreadsheet.title}")
        print(f"📄 스프레드시트 URL: {spreadsheet.url}")
        
        # 3. 워크시트 확인
        worksheets = spreadsheet.worksheets()
        print(f"✅ 워크시트 수: {len(worksheets)}")
        for ws in worksheets:
            print(f"   📋 {ws.title}")
        
        # 4. 테스트 데이터 추가
        test_data = [
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            85.5,  # 테스트 점수
            10,    # 파일 수
            1500,  # 코드 라인
            5.2,   # 평균 복잡도
            3,     # 코드 스멜
            1,     # 보안 이슈
            78.3,  # 유지보수성
            "테스트 트렌드",
            "테스트 권장사항",
            "테스트 Gemini 분석"
        ]
        
        sheets_manager.quality_sheet.append_row(test_data)
        print("✅ 테스트 데이터 추가 성공")
        
        # 5. 데이터 읽기 테스트
        all_values = sheets_manager.quality_sheet.get_all_values()
        print(f"✅ 데이터 읽기 성공: {len(all_values)}행")
        
        return True
        
    except Exception as e:
        print(f"❌ 구글 시트 연결 테스트 실패: {e}")
        logger.error(f"상세 오류: {e}", exc_info=True)
        return False

async def test_mock_analysis():
    """모의 분석 데이터로 전체 시스템 테스트"""
    try:
        print("\n🧪 모의 분석 데이터 테스트 시작...")
        
        # 모의 데이터 생성
        from quality_analyzer import QualityReport, CodeMetrics
        from auto_refactoring_system import RefactoringSession, RefactoringProposal
        
        # 모의 코드 메트릭
        mock_metrics = [
            CodeMetrics(
                file_path="test_file1.py",
                lines_of_code=150,
                complexity=8,
                maintainability_index=75.5,
                test_coverage=0.0,
                code_smells=["매직 넘버 과다 사용"],
                security_issues=[],
                performance_issues=[]
            ),
            CodeMetrics(
                file_path="test_file2.py", 
                lines_of_code=200,
                complexity=12,
                maintainability_index=65.2,
                test_coverage=0.0,
                code_smells=["긴 함수 발견"],
                security_issues=["하드코딩된 API 키 의심"],
                performance_issues=[]
            )
        ]
        
        # 모의 품질 보고서
        mock_report = QualityReport(
            timestamp=datetime.now().isoformat(),
            overall_score=70.3,
            file_metrics=mock_metrics,
            gemini_analysis="테스트용 Gemini 분석 결과입니다. 전반적으로 양호한 상태이나 몇 가지 개선점이 있습니다.",
            recommendations=[
                "매직 넘버를 상수로 추출하세요",
                "긴 함수를 작은 단위로 분할하세요", 
                "보안 이슈를 해결하세요"
            ],
            trend_analysis="품질이 지속적으로 개선되고 있습니다"
        )
        
        # 모의 리팩토링 제안
        mock_proposals = [
            RefactoringProposal(
                file_path="test_file1.py",
                issue_type="매직 넘버",
                description="3개의 매직 넘버를 상수로 추출",
                original_code="if count > 100: timeout = 5000",
                proposed_code="MAX_COUNT = 100\nDEFAULT_TIMEOUT = 5000\nif count > MAX_COUNT: timeout = DEFAULT_TIMEOUT",
                confidence=0.9,
                risk_level="LOW",
                explanation="매직 넘버를 의미있는 상수로 변경하여 가독성 향상"
            )
        ]
        
        mock_session = RefactoringSession(
            session_id="test_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.now().isoformat(),
            proposals=mock_proposals
        )
        
        # 구글 시트에 저장 테스트
        from google_sheets_integration import AutomatedGoogleSheetsLogger
        
        sheets_logger = AutomatedGoogleSheetsLogger()
        result = await sheets_logger.log_daily_analysis(mock_report, mock_session)
        
        if result['success']:
            print("✅ 모의 데이터 구글 시트 저장 성공")
            print(f"📊 품질 점수: {result['quality_score']}")
            print(f"💡 제안 수: {result['proposals_count']}")
            
            trend = result.get('trend', {})
            if 'trend' in trend:
                print(f"📈 트렌드: {trend['trend']}")
            
            return True
        else:
            print(f"❌ 모의 데이터 저장 실패: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ 모의 분석 테스트 실패: {e}")
        logger.error(f"상세 오류: {e}", exc_info=True)
        return False

async def test_real_analysis():
    """실제 코드 분석 테스트"""
    try:
        print("\n🔍 실제 코드 분석 테스트 시작...")
        
        from google_sheets_integration import run_daily_analysis_with_sheets
        
        result = await run_daily_analysis_with_sheets()
        
        if result['success']:
            print("✅ 실제 분석 및 구글 시트 저장 성공!")
            return True
        else:
            print(f"❌ 실제 분석 실패: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ 실제 분석 테스트 실패: {e}")
        logger.error(f"상세 오류: {e}", exc_info=True)
        return False

async def run_all_tests():
    """모든 테스트 실행"""
    print("🧪 구글 시트 통합 시스템 전체 테스트 시작\n")
    
    tests = [
        ("구글 시트 연결", test_google_sheets_connection),
        ("모의 데이터 테스트", test_mock_analysis),
        ("실제 분석 테스트", test_real_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name} 테스트")
        print('='*50)
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} 테스트 성공")
            else:
                print(f"❌ {test_name} 테스트 실패")
                
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("📊 테스트 결과 요약")
    print('='*50)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n🎯 전체 결과: {success_count}/{len(results)} 성공")
    
    if success_count == len(results):
        print("🎉 모든 테스트가 성공했습니다!")
        print("📈 구글 시트 통합 시스템이 정상적으로 작동합니다")
    else:
        print("⚠️ 일부 테스트가 실패했습니다")
        print("💡 오류 메시지를 확인하고 설정을 점검하세요")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 