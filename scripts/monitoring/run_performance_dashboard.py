#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_performance_dashboard.py
모듈: 실시간 성능 모니터링 대시보드 실행 스크립트
목적: Streamlit 대시보드 실행 및 성능 모니터링

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Usage:
    python run_performance_dashboard.py
    또는
    streamlit run run_performance_dashboard.py

Dependencies:
    - Python 3.11+
    - streamlit>=1.28.0
    - plotly>=5.17.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - psutil>=5.9.0

Performance:
    - 대시보드 시작: < 5초
    - 실시간 업데이트: < 1초
    - 메모리 사용량: < 500MB

License: MIT
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """의존성 체크"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 설치됨")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} 설치되지 않음")
    
    if missing_packages:
        logger.error(f"누락된 패키지: {missing_packages}")
        logger.info("다음 명령어로 설치하세요:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def run_dashboard():
    """대시보드 실행"""
    try:
        logger.info("🚀 실시간 성능 모니터링 대시보드 시작...")
        
        # 의존성 체크
        if not check_dependencies():
            logger.error("의존성 체크 실패. 대시보드를 실행할 수 없습니다.")
            return False
        
        # 대시보드 모듈 임포트
        try:
            from src.realtime_monitoring_dashboard import PerformanceDashboard
            logger.info("✅ 대시보드 모듈 로드 완료")
        except ImportError as e:
            logger.error(f"❌ 대시보드 모듈 로드 실패: {e}")
            return False
        
        # 대시보드 실행
        dashboard = PerformanceDashboard()
        dashboard.run_dashboard()
        
        return True
        
    except Exception as e:
        logger.error(f"대시보드 실행 실패: {e}")
        return False


def run_with_streamlit():
    """Streamlit으로 대시보드 실행"""
    try:
        logger.info("📊 Streamlit으로 대시보드 실행...")
        
        # Streamlit 명령어 실행
        dashboard_file = project_root / "src" / "realtime_monitoring_dashboard.py"
        
        if not dashboard_file.exists():
            logger.error(f"대시보드 파일을 찾을 수 없습니다: {dashboard_file}")
            return False
        
        # Streamlit 실행
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info(f"실행 명령어: {' '.join(cmd)}")
        
        # 서브프로세스로 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("✅ Streamlit 대시보드가 시작되었습니다.")
        logger.info("🌐 브라우저에서 http://localhost:8501 을 열어주세요.")
        
        # 프로세스 모니터링
        try:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Streamlit 실행 실패: {stderr}")
                return False
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단되었습니다.")
            process.terminate()
            return True
        
        return True
        
    except Exception as e:
        logger.error(f"Streamlit 실행 실패: {e}")
        return False


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="실시간 성능 모니터링 대시보드")
    parser.add_argument(
        "--mode",
        choices=["direct", "streamlit"],
        default="streamlit",
        help="실행 모드 (direct: 직접 실행, streamlit: Streamlit 서버)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit 서버 포트 (기본값: 8501)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Streamlit 서버 호스트 (기본값: localhost)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 실시간 성능 모니터링 대시보드")
    logger.info("=" * 60)
    
    if args.mode == "direct":
        logger.info("📊 직접 실행 모드")
        success = run_dashboard()
    else:
        logger.info("🌐 Streamlit 서버 모드")
        success = run_with_streamlit()
    
    if success:
        logger.info("✅ 대시보드가 성공적으로 실행되었습니다.")
    else:
        logger.error("❌ 대시보드 실행에 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main() 