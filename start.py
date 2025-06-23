#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra Stock Analysis System - 안전한 시작 스크립트
"""

import sys
import os
import logging
from pathlib import Path

def setup_environment():
    """환경 설정"""
    # 프로젝트 루트 경로를 Python 경로에 추가
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # 필요한 디렉토리 생성
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

def check_dependencies():
    """필수 패키지 확인"""
    required_packages = [
        'tkinter', 'pandas', 'numpy', 'matplotlib', 
        'requests', 'beautifulsoup4', 'yfinance',
        'google.generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'google.generativeai':
                import google.generativeai
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지가 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 다음 명령으로 설치하세요:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """메인 실행 함수"""
    print("🚀 Ultra Stock Analysis System v5.0")
    print("=" * 50)
    
    # 환경 설정
    setup_environment()
    
    # 의존성 확인
    if not check_dependencies():
        input("\n⚠️ Enter 키를 눌러 종료하세요...")
        return
    
    print("✅ 모든 패키지가 정상적으로 설치되어 있습니다.")
    print("🎨 GUI 애플리케이션을 시작합니다...\n")
    
    try:
        from main import main as run_main
        run_main()
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        logging.error(f"애플리케이션 실행 실패: {e}")
        input("\n⚠️ Enter 키를 눌러 종료하세요...")

if __name__ == "__main__":
    main() 