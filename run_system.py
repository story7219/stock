#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis System Launcher
시스템 실행을 위한 간단한 런처 스크립트

Usage:
    python run_system.py
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_python_version():
    """Python 버전 확인"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        sys.exit(1)
    print(f"✅ Python 버전: {sys.version}")

def check_dependencies():
    """필수 패키지 설치 확인"""
    required_packages = [
        'pandas', 'numpy', 'aiohttp', 'beautifulsoup4',
        'yfinance', 'google-generativeai', 'python-telegram-bot',
        'gspread', 'schedule', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 누락된 패키지: {missing_packages}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 필수 패키지가 설치되어 있습니다.")
    return True

def check_env_file():
    """환경 변수 파일 확인"""
    env_file = PROJECT_ROOT / '.env'
    
    if not env_file.exists():
        print("❌ .env 파일이 없습니다.")
        print("다음 내용으로 .env 파일을 생성하세요:")
        print("""
# Gemini AI 설정
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash-8b
GEMINI_TEMPERATURE=0.05
GEMINI_MAX_TOKENS=16384

# 텔레그램 설정
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 구글 시트 설정
GOOGLE_SHEETS_CREDENTIALS=credentials.json
GOOGLE_SHEETS_ID=your_google_sheets_id

# 기타 설정
LOG_LEVEL=INFO
MAX_WORKERS=10
BATCH_SIZE=50
        """)
        return False
    
    print("✅ .env 파일이 존재합니다.")
    return True

def create_directories():
    """필요한 디렉토리 생성"""
    directories = ['logs', 'data', 'reports', 'backups']
    
    for directory in directories:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 디렉토리 생성: {directory}/")

def show_menu():
    """메뉴 표시"""
    print("\n" + "="*60)
    print("🚀 Stock Analysis System v5.0 런처")
    print("🤖 Powered by Gemini 1.5 Flash")
    print("="*60)
    print("1. 전체 분석 실행 (기본)")
    print("2. 빠른 상태 점검")
    print("3. 자동 스케줄러 모드")
    print("4. 시스템 정보 확인")
    print("5. 패키지 설치/업데이트")
    print("6. 로그 파일 보기")
    print("0. 종료")
    print("="*60)

def install_packages():
    """패키지 설치/업데이트"""
    print("📦 패키지 설치/업데이트 중...")
    
    try:
        # setup.py를 통한 설치
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                      cwd=PROJECT_ROOT, check=True)
        print("✅ 패키지 설치 완료")
        
        # 개발용 패키지 설치 여부 확인
        dev_install = input("개발용 패키지도 설치하시겠습니까? (y/n): ").lower()
        if dev_install == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], 
                          cwd=PROJECT_ROOT, check=True)
            print("✅ 개발용 패키지 설치 완료")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")

def show_logs():
    """로그 파일 보기"""
    logs_dir = PROJECT_ROOT / 'logs'
    
    if not logs_dir.exists():
        print("❌ 로그 디렉토리가 없습니다.")
        return
    
    log_files = list(logs_dir.glob('*.log'))
    
    if not log_files:
        print("❌ 로그 파일이 없습니다.")
        return
    
    print("\n📋 로그 파일 목록:")
    for i, log_file in enumerate(log_files, 1):
        print(f"{i}. {log_file.name}")
    
    try:
        choice = int(input("보실 로그 파일 번호를 선택하세요 (0: 취소): "))
        if choice == 0:
            return
        
        if 1 <= choice <= len(log_files):
            log_file = log_files[choice - 1]
            print(f"\n📄 {log_file.name} (마지막 50줄):")
            print("-" * 60)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        else:
            print("❌ 잘못된 선택입니다.")
            
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ 로그 파일 읽기 실패: {e}")

def show_system_info():
    """시스템 정보 표시"""
    print("\n🔧 시스템 정보:")
    print(f"Python 버전: {sys.version}")
    print(f"프로젝트 경로: {PROJECT_ROOT}")
    print(f"운영체제: {os.name}")
    
    # 디스크 사용량 확인
    try:
        import shutil
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        print(f"디스크 용량: {free // (2**30)} GB 사용 가능")
    except:
        pass
    
    # 메모리 사용량 확인
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"메모리 사용률: {memory.percent}%")
    except ImportError:
        print("메모리 정보 확인을 위해 psutil 패키지를 설치하세요.")

async def run_main_system(command: str = "analyze"):
    """메인 시스템 실행"""
    try:
        from main import main as main_function
        
        # 명령어 인자 설정
        original_argv = sys.argv.copy()
        sys.argv = ['main.py', command]
        
        try:
            await main_function()
        finally:
            # 원래 argv 복원
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ 시스템 실행 오류: {e}")

def main():
    """메인 함수"""
    # 초기 검사
    check_python_version()
    
    if not check_dependencies():
        install_choice = input("필수 패키지를 설치하시겠습니까? (y/n): ").lower()
        if install_choice == 'y':
            install_packages()
        else:
            print("❌ 필수 패키지가 없어 실행할 수 없습니다.")
            sys.exit(1)
    
    if not check_env_file():
        print("❌ 환경 설정이 필요합니다.")
        sys.exit(1)
    
    create_directories()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n선택하세요 (0-6): ").strip()
            
            if choice == '0':
                print("👋 시스템을 종료합니다.")
                break
            elif choice == '1':
                print("🚀 전체 분석을 시작합니다...")
                asyncio.run(run_main_system("analyze"))
            elif choice == '2':
                print("⚡ 빠른 상태 점검을 시작합니다...")
                asyncio.run(run_main_system("check"))
            elif choice == '3':
                print("⏰ 자동 스케줄러 모드를 시작합니다...")
                print("중지하려면 Ctrl+C를 누르세요.")
                asyncio.run(run_main_system("schedule"))
            elif choice == '4':
                show_system_info()
            elif choice == '5':
                install_packages()
            elif choice == '6':
                show_logs()
            else:
                print("❌ 잘못된 선택입니다. 0-6 사이의 숫자를 입력하세요.")
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")

if __name__ == "__main__":
    main() 