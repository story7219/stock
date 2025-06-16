#!/usr/bin/env python3
"""
코드 품질 검사 실행 스크립트
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quality_analyzer import QualityScheduler, run_immediate_analysis

def main():
    """메인 함수"""
    print("🔍 코드 품질 분석 시스템")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "now":
            print("📊 즉시 분석을 시작합니다...")
            asyncio.run(run_immediate_analysis())
            
        elif command == "schedule":
            print("⏰ 스케줄러를 시작합니다 (매일 오전 7시 실행)")
            scheduler = QualityScheduler()
            scheduler.start_scheduler()
            
        elif command == "help":
            print_help()
            
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print_help()
    else:
        print_help()

def print_help():
    """도움말 출력"""
    print("""
사용법:
    python run_quality_check.py [명령어]

명령어:
    now       - 즉시 품질 분석 실행
    schedule  - 스케줄러 시작 (매일 오전 7시 자동 실행)
    help      - 이 도움말 표시

예시:
    python run_quality_check.py now        # 지금 바로 분석
    python run_quality_check.py schedule   # 스케줄러 시작
""")

if __name__ == "__main__":
    main() 