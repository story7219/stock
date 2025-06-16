#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.env 파일 읽기 테스트
"""

import os
from dotenv import load_dotenv

def test_env_reading():
    """환경변수 읽기 테스트"""
    print("🧪 .env 파일 읽기 테스트")
    print("=" * 40)
    
    # .env 파일 로드
    load_dotenv()
    
    # 주요 환경변수 확인
    print(f"IS_MOCK: {os.getenv('IS_MOCK')}")
    print(f"MOCK_KIS_APP_KEY: ...{os.getenv('MOCK_KIS_APP_KEY', '')[-4:]}")
    print(f"ORDER_API_CALLS_PER_SEC: {os.getenv('ORDER_API_CALLS_PER_SEC')}")
    print(f"TOTAL_API_CALLS_PER_SEC: {os.getenv('TOTAL_API_CALLS_PER_SEC')}")
    
    # UTF-8로 직접 파일 읽기 테스트
    print("\n📄 파일 직접 읽기 테스트:")
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"첫 번째 줄: {first_line}")
            if '환경 설정' in first_line:
                print("✅ UTF-8로 한글이 정상 읽힙니다!")
            else:
                print("❌ 한글이 깨져 있습니다.")
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
    
    print("=" * 40)

if __name__ == "__main__":
    test_env_reading() 