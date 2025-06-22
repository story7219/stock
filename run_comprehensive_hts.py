"""
종합 HTS 시스템 실행기
- 데이터 업데이트 후 GUI 실행
- 모든 대가들의 분석 방법론 포함
- 재무제표, 차트, 기업정보 통합 표시
"""

import os
import sys
from datetime import datetime

def check_requirements():
    """필수 요구사항 확인"""
    required_files = [
        'data/stock_data.csv',
        'comprehensive_hts_gui.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 필수 파일이 없습니다:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def update_sample_data():
    """샘플 데이터 업데이트"""
    try:
        print("🔄 샘플 데이터 업데이트 중...")
        import subprocess
        result = subprocess.run([sys.executable, 'create_sample_data.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 샘플 데이터 업데이트 완료")
            return True
        else:
            print(f"⚠️ 데이터 업데이트 경고: {result.stderr}")
            return True  # 경고는 있지만 계속 진행
    except Exception as e:
        print(f"❌ 데이터 업데이트 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🚀 종합 HTS 투자 분석 시스템 시작")
    print("=" * 50)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 1. 요구사항 확인
    if not check_requirements():
        print("\n❌ 시스템 요구사항을 만족하지 않습니다.")
        print("먼저 create_sample_data.py를 실행하여 데이터를 생성하세요.")
        input("Press Enter to exit...")
        return
    
    # 2. 데이터 업데이트
    if not update_sample_data():
        print("\n⚠️ 데이터 업데이트에 문제가 있지만 계속 진행합니다.")
    
    # 3. GUI 시스템 실행
    try:
        print("\n🖥️ GUI 시스템 실행 중...")
        print("=" * 50)
        print("📊 기능 안내:")
        print("• 좌측: 투자 대가 선택 + 거래대금 순위")
        print("• 중앙: 종목차트 + 재무제표 + 현금흐름표 + 손익계산서")
        print("• 우측: 기업정보 + 분석결과")
        print("=" * 50)
        
        from comprehensive_hts_gui import ComprehensiveHTS
        
        # GUI 실행
        app = ComprehensiveHTS()
        app.run()
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print("필요한 라이브러리를 설치하세요:")
        print("pip install pandas numpy matplotlib tkinter")
        
    except Exception as e:
        print(f"❌ 시스템 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n📅 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🙏 종합 HTS 시스템을 이용해 주셔서 감사합니다!")

if __name__ == "__main__":
    main() 