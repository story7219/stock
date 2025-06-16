"""
구글 시트 통합 테스트를 위한 설정 및 패키지 설치
"""

import subprocess
import sys
import os
from pathlib import Path

def install_required_packages():
    """필요한 패키지 설치"""
    packages = [
        "gspread",
        "google-auth", 
        "google-auth-oauthlib",
        "google-auth-httplib2",
        "python-dotenv",
        "google-generativeai"
    ]
    
    print("📦 필요한 패키지 설치 중...")
    
    for package in packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 설치 실패: {e}")
            return False
    
    print("🎉 모든 패키지 설치 완료!")
    return True

def check_environment():
    """환경 설정 확인"""
    print("\n🔍 환경 설정 확인 중...")
    
    # .env 파일 확인
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env 파일이 없습니다")
        create_sample_env()
        return False
    
    # 환경 변수 확인
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    google_creds = os.getenv('GOOGLE_CREDENTIALS_PATH', 'google_credentials.json')
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY가 설정되지 않았습니다")
        return False
    else:
        print("✅ GEMINI_API_KEY 설정됨")
    
    if not Path(google_creds).exists():
        print(f"❌ 구글 인증 파일이 없습니다: {google_creds}")
        print("💡 구글 서비스 계정 키 파일을 다운로드하여 저장하세요")
        return False
    else:
        print("✅ 구글 인증 파일 확인됨")
    
    return True

def create_sample_env():
    """샘플 .env 파일 생성"""
    sample_content = """# Gemini API 키
GEMINI_API_KEY=your_gemini_api_key_here

# 구글 서비스 계정 키 파일 경로
GOOGLE_CREDENTIALS_PATH=google_credentials.json
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print("📝 샘플 .env 파일을 생성했습니다")
    print("💡 실제 API 키를 입력하고 다시 실행하세요")

if __name__ == "__main__":
    print("🚀 구글 시트 통합 테스트 준비")
    
    if install_required_packages():
        if check_environment():
            print("\n✅ 모든 준비가 완료되었습니다!")
            print("🧪 이제 테스트를 실행할 수 있습니다")
        else:
            print("\n❌ 환경 설정을 완료하고 다시 실행하세요")
    else:
        print("\n❌ 패키지 설치에 실패했습니다") 