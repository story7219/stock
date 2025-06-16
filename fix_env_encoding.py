#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.env 파일 한글 인코딩 문제 해결 스크립트
Windows에서 한글이 깨지는 문제를 근본적으로 해결합니다.
"""

import os
import shutil
from pathlib import Path

def create_utf8_env_file():
    """UTF-8 인코딩으로 .env 파일을 생성합니다."""
    
    # 백업 생성
    if os.path.exists('.env'):
        shutil.copy('.env', '.env.backup.before_fix')
        print("📋 기존 .env 파일을 .env.backup.before_fix로 백업했습니다.")
    
    # UTF-8 인코딩으로 새 .env 파일 생성
    env_content = """# 환경 설정(true: 모의투자, false: 실전투자)
IS_MOCK=true

# MOCK (모의 투자) 환경설정
MOCK_KIS_APP_KEY=PSJHToqNQYzVvVH1DfkndIodXaCsEgAHBHPr
MOCK_KIS_APP_SECRET=W5ts9iDYGxjNGaPdKqDcjAQz2FdLwakr/2sC3K44zs9dtljT2P8UbB/zOo2hsWZpkP/kraOmF9P1vqqcHxbz/YiVwKcR6FCmj/WZdoAdnCfQi/KMntP9V1b6dn7RLoOiTZtgwLaoVfWKJPP+hcmxNI/st+oCp3iDv/ZdKoQg4Hu9OG4myW0=
MOCK_KIS_ACCOUNT_NUMBER=50128558-01

# LIVE (실전 투자) 환경설정 (사용시에만 활용)
LIVE_KIS_APP_KEY=PSGofcwBn3wobX5ha24kZhTIFd1gwCEpnp4X
LIVE_KIS_APP_SECRET=VDDe6yqRvHEKSdAUhji0Ba5aFvbrPP6b3b/3J5YfF7igu83eH72HRB2PLmXiB4zSwHVgA0U/3vjVm6VjDKBN22zUlx6bYzw3v/y77u8/UGOdwylXt1jzUT9KDuUhHuHcibVSQMupMs9C4Asbn6HrJ61EbVv9143LakeSzcWsCbf48SPlNPk=
LIVE_KIS_ACCOUNT_NUMBER=64841154-01

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN=7877945865:AAFng10_N3aJPaoutEo5UBs3T-CpFD-fWxg
TELEGRAM_CHAT_ID=796943082

GEMINI_API_KEY=AIzaSyCOBC-vf_gyrzlWCOoY32OuF5w45eGKIBQ
GOOGLE_SERVICE_ACCOUNT_FILE=SERVICE_ACCOUNT_JSON
GOOGLE_SPREADSHEET_ID=1apZe5eUizqa8gdlge0MUZNkLvEv7mkuHLd6XPqnTFsc
GOOGLE_WORKSHEET_NAME=매매기록

# ZAPIER API 키
ZAPIER_NLA_API_KEY=d11636695312aad0e8862b265a2a28ef

# 토큰 발급 시간 설정 (24시간 형식)
TOKEN_ISSUE_HOUR=7
TOKEN_ISSUE_MINUTE=0

ORDER_API_CALLS_PER_SEC=1
TOTAL_API_CALLS_PER_SEC=5

# 다른 시간 예시:
# TOKEN_ISSUE_HOUR=8    # 실전 8시
# TOKEN_ISSUE_MINUTE=30 # 30분
# 결과: 08:30에 토큰 발급"""

    # UTF-8 BOM 없이 저장
    with open('.env', 'w', encoding='utf-8', newline='\n') as f:
        f.write(env_content)
    
    print("✅ .env 파일이 UTF-8 인코딩으로 생성되었습니다!")
    return True

def verify_encoding():
    """파일 인코딩을 확인합니다."""
    try:
        # UTF-8로 읽기 시도
        with open('.env', 'r', encoding='utf-8') as f:
            content = f.read()
            if '환경 설정' in content and '모의투자' in content:
                print("✅ UTF-8 인코딩으로 한글이 정상적으로 읽힙니다!")
                return True
            else:
                print("❌ 한글이 여전히 깨져 있습니다.")
                return False
    except UnicodeDecodeError:
        print("❌ UTF-8로 읽을 수 없습니다. 인코딩 문제가 있습니다.")
        return False

def update_config_for_encoding():
    """config.py를 수정하여 강제로 UTF-8로 읽도록 합니다."""
    
    print("🔧 config.py에 인코딩 강제 설정을 추가합니다...")
    
    # config.py 읽기
    with open('config.py', 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # dotenv 로드 부분 수정
    if 'load_dotenv()' in config_content:
        # UTF-8 강제 로드를 위한 커스텀 함수로 대체
        updated_content = config_content.replace(
            '# .env 파일 로드\nload_dotenv()',
            '''# .env 파일 로드 (UTF-8 강제)
def load_dotenv_utf8():
    """UTF-8 인코딩으로 .env 파일을 강제 로드"""
    import os
    from pathlib import Path
    
    env_file = Path('.env')
    if not env_file.exists():
        return
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"⚠️ .env 파일 로드 실패: {e}")
        # 기본 dotenv 사용
        try:
            load_dotenv()
        except:
            pass

load_dotenv_utf8()'''
        )
        
        # 수정된 내용 저장
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("✅ config.py가 UTF-8 강제 로드로 수정되었습니다!")
        return True
    else:
        print("⚠️ config.py에서 load_dotenv() 호출을 찾을 수 없습니다.")
        return False

def main():
    """메인 실행 함수"""
    print("🔧 .env 파일 한글 인코딩 문제 해결 시작...")
    print("=" * 60)
    
    # 1단계: UTF-8 .env 파일 생성
    if create_utf8_env_file():
        # 2단계: 인코딩 확인
        if verify_encoding():
            # 3단계: config.py 수정
            update_config_for_encoding()
            
            print("\n" + "=" * 60)
            print("🎉 한글 인코딩 문제 해결 완료!")
            print("💡 앞으로는 다음 방법으로 .env 파일을 편집하세요:")
            print("   1. VS Code에서 편집 시: 우하단에서 인코딩을 'UTF-8'로 설정")
            print("   2. 메모장 사용 시: '다른 이름으로 저장' → 인코딩을 'UTF-8'로 선택")
            print("   3. 이 스크립트 재실행: python fix_env_encoding.py")
            print("=" * 60)
        else:
            print("❌ 인코딩 확인 실패")
    else:
        print("❌ .env 파일 생성 실패")

if __name__ == "__main__":
    main() 