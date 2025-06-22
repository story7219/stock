#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Premium HTS - 메인 실행 파일
세계 최고 수준의 HTS 시스템 런처
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hts_system.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """필수 종속성 확인"""
    required_packages = [
        'tkinter',
        'asyncio', 
        'pandas',
        'numpy',
        'matplotlib',
        'sqlalchemy',
        'aiofiles',
        'structlog',
        'pydantic',
        'yfinance'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        logger.error(f"누락된 패키지: {missing_packages}")
        print(f"다음 패키지를 설치해주세요: {', '.join(missing_packages)}")
        print("설치 명령어: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """메인 함수 - 프리미엄 HTS 시스템 시작"""
    try:
        # 시스템 시작 메시지
        print("=" * 60)
        print("🚀 고성능 HTS (Home Trading System) v5.0")
        print("비동기 처리 | 멀티레벨 캐싱 | 성능 최적화")
        print("=" * 60)
        
        logger.info("HTS 애플리케이션 시작")
        
        # 시스템 정보 출력
        print(f"📂 작업 디렉토리: {os.getcwd()}")
        print(f"🐍 Python 버전: {sys.version}")
        print(f"📁 프로젝트 루트: {project_root}")
        print()
        
        # 종속성 확인
        print("🔍 필수 패키지 확인 중...")
        if not check_dependencies():
            print("❌ 패키지 확인 실패")
            sys.exit(1)
        print("✅ 모든 필수 패키지 확인 완료")
        print()
        
        # 시스템 구성 요소 확인
        print("🔧 시스템 구성 요소 확인 중...")
        
        # 디렉토리 구조 확인
        required_dirs = ['config', 'core', 'ui_interfaces', 'data']
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"✅ {dir_name}/ 디렉토리 존재")
            else:
                print(f"⚠️ {dir_name}/ 디렉토리 없음")
        
        # 주요 파일 확인
        required_files = [
            'config/settings.py',
            'core/cache_manager.py', 
            'core/database_manager.py',
            'core/performance_monitor.py',
            'ui_interfaces/optimized_hts_gui.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} 파일 존재")
            else:
                print(f"❌ {file_path} 파일 없음")
        
        print()
        
        # GUI 모듈 임포트
        print("📦 GUI 모듈 로딩 중...")
        try:
            from ui_interfaces.optimized_hts_gui import OptimizedHTS
            print("✅ GUI 모듈 로딩 성공")
        except Exception as e:
            print(f"❌ GUI 모듈 로딩 실패: {e}")
            raise
        
        # 애플리케이션 생성
        print("🚀 HTS 애플리케이션 생성 중...")
        try:
            app = OptimizedHTS()
            print("✅ HTS 애플리케이션 생성 성공")
            logger.info("프리미엄 HTS GUI 초기화 완료")
        except Exception as e:
            print(f"❌ HTS 애플리케이션 생성 실패: {e}")
            raise
        
        # GUI 실행
        print("🖥️ GUI 창 실행 중...")
        print("=" * 60)
        print("GUI 창이 열립니다. 창을 닫으면 프로그램이 종료됩니다.")
        print("=" * 60)
        
        # 메인 루프 실행
        app.run()
        
    except ImportError as e:
        logger.error(f"모듈 임포트 실패: {e}")
        print(f"❌ 모듈 임포트 실패: {e}")
        print("필요한 패키지가 설치되어 있는지 확인해주세요.")
        print("설치 명령어: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류: {e}")
        print(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        print("자세한 오류 정보는 hts_system.log 파일을 확인해주세요.")
        sys.exit(1)
        
    finally:
        print("\n" + "=" * 60)
        print("🏁 Ultra Premium HTS 시스템 종료")
        print("=" * 60)
        logger.info("Ultra Premium HTS 시스템 종료")

if __name__ == "__main__":
    main() 