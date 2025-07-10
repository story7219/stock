#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_monitoring_dashboard.py
모듈: 실시간 모니터링 대시보드 실행 스크립트
목적: 실시간 데이터 시스템 모니터링 대시보드 실행

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Usage:
    python run_monitoring_dashboard.py
    또는
    streamlit run run_monitoring_dashboard.py

Dependencies:
    - Python 3.11+
    - streamlit, plotly, dash
    - psutil, prometheus_client
    - requests, aiohttp

Features:
    - 실시간 시스템 상태 모니터링
    - 데이터 품질 대시보드
    - 성능 메트릭 추적
    - 실시간 알림 시스템
    - 모바일 반응형 디자인

Performance:
    - 실시간 업데이트: < 1초
    - 대시보드 로딩: < 3초
    - 알림 전송: < 5초

License: MIT
"""

import sys
import os
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """의존성 체크"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'psutil',
        'requests',
        'aiohttp'
    ]
    
    optional_packages = [
        'prometheus_client',
        'websockets',
        'asyncio_mqtt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 설치됨")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} 설치되지 않음")
    
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 설치됨 (선택사항)")
        except ImportError:
            logger.warning(f"⚠️ {package} 설치되지 않음 (선택사항)")
    
    if missing_packages:
        logger.error(f"누락된 필수 패키지: {missing_packages}")
        logger.info("다음 명령어로 설치하세요:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def load_monitoring_config(config_path: str = None) -> Dict[str, Any]:
    """모니터링 설정 파일 로드"""
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 기본 모니터링 설정
            config = {
                "dashboard": {
                    "port": 8501,
                    "update_interval_seconds": 1.0,
                    "max_data_points": 1000
                },
                "alerts": {
                    "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
                    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
                    "email_smtp_server": os.getenv("EMAIL_SMTP_SERVER", ""),
                    "email_username": os.getenv("EMAIL_USERNAME", ""),
                    "email_password": os.getenv("EMAIL_PASSWORD", "")
                },
                "thresholds": {
                    "cpu_threshold_percent": 80.0,
                    "memory_threshold_percent": 85.0,
                    "disk_threshold_percent": 90.0,
                    "latency_threshold_ms": 100.0,
                    "error_rate_threshold_percent": 5.0
                },
                "data_quality": {
                    "data_coverage_threshold_percent": 95.0,
                    "data_delay_threshold_seconds": 60.0,
                    "anomaly_detection_enabled": True
                },
                "performance": {
                    "websocket_enabled": True,
                    "websocket_port": 8765,
                    "prometheus_enabled": True,
                    "prometheus_port": 8000
                }
            }
        
        logger.info("모니터링 설정 파일 로드 완료")
        return config
        
    except Exception as e:
        logger.error(f"모니터링 설정 파일 로드 실패: {e}")
        raise


def validate_monitoring_config(config: Dict[str, Any]) -> bool:
    """모니터링 설정 유효성 검증"""
    try:
        # 대시보드 설정 검증
        dashboard_config = config.get("dashboard", {})
        if not dashboard_config.get("port"):
            logger.error("대시보드 포트가 설정되지 않았습니다.")
            return False
        
        # 임계값 설정 검증
        thresholds_config = config.get("thresholds", {})
        required_thresholds = [
            "cpu_threshold_percent",
            "memory_threshold_percent", 
            "disk_threshold_percent",
            "latency_threshold_ms",
            "error_rate_threshold_percent"
        ]
        
        for threshold in required_thresholds:
            if threshold not in thresholds_config:
                logger.error(f"임계값 설정 누락: {threshold}")
                return False
        
        logger.info("모니터링 설정 유효성 검증 완료")
        return True
        
    except Exception as e:
        logger.error(f"모니터링 설정 유효성 검증 실패: {e}")
        return False


class MonitoringDashboardRunner:
    """모니터링 대시보드 실행기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = None
        self.alert_manager = None
        self.notification_system = None
        self.is_running = False
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            from src.realtime_monitoring_dashboard import (
                MonitoringConfig, RealTimeMonitor, AlertManager, NotificationSystem
            )
            
            # MonitoringConfig 생성
            monitoring_config = MonitoringConfig(
                dashboard_port=self.config["dashboard"]["port"],
                update_interval_seconds=self.config["dashboard"]["update_interval_seconds"],
                max_data_points=self.config["dashboard"]["max_data_points"],
                slack_webhook_url=self.config["alerts"]["slack_webhook_url"],
                telegram_bot_token=self.config["alerts"]["telegram_bot_token"],
                telegram_chat_id=self.config["alerts"]["telegram_chat_id"],
                email_smtp_server=self.config["alerts"]["email_smtp_server"],
                email_username=self.config["alerts"]["email_username"],
                email_password=self.config["alerts"]["email_password"],
                cpu_threshold_percent=self.config["thresholds"]["cpu_threshold_percent"],
                memory_threshold_percent=self.config["thresholds"]["memory_threshold_percent"],
                disk_threshold_percent=self.config["thresholds"]["disk_threshold_percent"],
                latency_threshold_ms=self.config["thresholds"]["latency_threshold_ms"],
                error_rate_threshold_percent=self.config["thresholds"]["error_rate_threshold_percent"],
                data_coverage_threshold_percent=self.config["data_quality"]["data_coverage_threshold_percent"],
                data_delay_threshold_seconds=self.config["data_quality"]["data_delay_threshold_seconds"],
                anomaly_detection_enabled=self.config["data_quality"]["anomaly_detection_enabled"],
                websocket_enabled=self.config["performance"]["websocket_enabled"],
                websocket_port=self.config["performance"]["websocket_port"],
                prometheus_enabled=self.config["performance"]["prometheus_enabled"],
                prometheus_port=self.config["performance"]["prometheus_port"]
            )
            
            # 컴포넌트 초기화
            self.monitor = RealTimeMonitor(monitoring_config)
            self.alert_manager = AlertManager(monitoring_config)
            self.notification_system = NotificationSystem(monitoring_config)
            
            logger.info("모니터링 대시보드 초기화 완료")
            
        except Exception as e:
            logger.error(f"모니터링 대시보드 초기화 실패: {e}")
            raise
    
    async def start(self):
        """시스템 시작"""
        try:
            self.is_running = True
            
            # 모니터링 시작
            self.monitor.start_monitoring()
            self.alert_manager.start_alert_manager()
            self.notification_system.start_notification_system()
            
            logger.info("모니터링 대시보드 시작")
            
        except Exception as e:
            logger.error(f"모니터링 대시보드 시작 실패: {e}")
            raise
        finally:
            self.is_running = False
    
    async def test_system(self):
        """시스템 테스트"""
        try:
            # 모니터링 테스트
            latest_metrics = self.monitor.get_latest_metrics()
            logger.info(f"최신 메트릭: {len(latest_metrics)} 개")
            
            # 알림 시스템 테스트
            test_alert = {
                'level': 'info',
                'message': '시스템 테스트 알림',
                'metric': 'test',
                'value': 0,
                'threshold': 0
            }
            self.alert_manager.process_alert(test_alert)
            logger.info("알림 시스템 테스트 완료")
            
            # 알림 히스토리 조회
            alert_history = self.alert_manager.get_alert_history(1)
            logger.info(f"알림 히스토리: {len(alert_history)} 개")
            
        except Exception as e:
            logger.error(f"시스템 테스트 실패: {e}")
    
    async def stop(self):
        """시스템 중지"""
        self.is_running = False
        if self.monitor:
            self.monitor.stop_monitoring()
        if self.alert_manager:
            self.alert_manager.stop_alert_manager()
        if self.notification_system:
            self.notification_system.stop_notification_system()
        logger.info("모니터링 대시보드 중지")


async def run_monitoring_dashboard(config: Dict[str, Any]):
    """모니터링 대시보드 실행"""
    try:
        runner = MonitoringDashboardRunner(config)
        await runner.initialize()
        
        # 시스템 테스트
        await runner.test_system()
        
        # 시스템 시작
        await runner.start()
        
    except Exception as e:
        logger.error(f"모니터링 대시보드 실행 실패: {e}")
        raise


def print_banner():
    """배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                실시간 모니터링 대시보드                      ║
    ║                                                              ║
    ║  📊 실시간 시스템 상태 모니터링                              ║
    ║  📈 데이터 품질 대시보드                                    ║
    ║  ⚡ 성능 메트릭 추적                                        ║
    ║  🔔 실시간 알림 시스템                                      ║
    ║  📱 모바일 반응형 디자인                                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def create_streamlit_app():
    """Streamlit 앱 생성"""
    try:
        import streamlit as st
        from src.realtime_monitoring_dashboard import create_streamlit_dashboard
        
        # Streamlit 앱 실행
        create_streamlit_dashboard()
        
    except Exception as e:
        logger.error(f"Streamlit 앱 생성 실패: {e}")
        raise


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="실시간 모니터링 대시보드")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="모니터링 설정 파일 경로"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="의존성만 체크"
    )
    parser.add_argument(
        "--validate-config",
        action="store_true", 
        help="설정만 검증"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="시스템 테스트만 실행"
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Streamlit 대시보드 실행"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        # 의존성 체크
        if not check_dependencies():
            logger.error("의존성 체크 실패")
            sys.exit(1)
        
        if args.check_deps:
            logger.info("의존성 체크 완료")
            return
        
        # 설정 로드
        config = load_monitoring_config(args.config)
        
        # 설정 검증
        if not validate_monitoring_config(config):
            logger.error("설정 검증 실패")
            sys.exit(1)
        
        if args.validate_config:
            logger.info("설정 검증 완료")
            return
        
        # Streamlit 대시보드 실행
        if args.streamlit:
            logger.info("Streamlit 대시보드 시작...")
            create_streamlit_app()
        else:
            # 모니터링 대시보드 실행
            if args.test:
                logger.info("시스템 테스트 실행...")
                asyncio.run(run_monitoring_dashboard(config))
            else:
                logger.info("실시간 모니터링 대시보드 시작...")
                asyncio.run(run_monitoring_dashboard(config))
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 