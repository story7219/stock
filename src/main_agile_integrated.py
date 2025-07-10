            import psutil
from __future__ import annotations
from agile_dashboard import AgileDashboard
from agile_trading_strategy import AgileTradingStrategy
from datetime import datetime
from main_integrated import IntegratedTradingSystem
from typing import Dict, Any
import asyncio
import json
import logging
import signal
import sys
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: main_agile_integrated.py
모듈: 소액 투자 민첩성 전략 통합 메인
목적: 기존 시스템 + 민첩성 전략 통합 실행

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, logging
    - main_integrated, agile_trading_strategy, agile_dashboard
"""




# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agile_trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AgileIntegratedTradingSystem:
    """소액 투자 민첩성 통합 트레이딩 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False

        # 기존 시스템
        self.main_system = None
        self.agile_strategy = None
        self.agile_dashboard = None

        # 종료 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 등)"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def start_system(self):
        """시스템 시작"""
        logger.info("Starting Agile Integrated Trading System...")

        try:
            # 설정 로드
            await self._load_configuration()

            # 서비스 초기화
            await self._initialize_services()

            # 시스템 시작
            self.running = True
            await self._run_system()

        except Exception as e:
            logger.error(f"System startup error: {e}")
            await self._shutdown_system()

    async def _load_configuration(self):
        """설정 로드"""
        logger.info("Loading agile configuration...")

        # 기본 설정
        default_config = {
            'main_system_config': {
                'kis_config': {
                    'symbols_to_track': ['005930', '000660', '035420'],
                    'update_interval': 1.0
                },
                'push_config': {
                    'enabled': True,
                    'firebase_service_account_path': 'config/firebase_service_account.json',
                    'device_tokens': [],
                    'signal_threshold': 0.8,
                    'risk_threshold': 0.7,
                    'max_notifications_per_hour': 10
                },
                'pipeline_config': {
                    'symbols_to_track': ['005930', '000660', '035420'],
                    'update_interval': 1.0,
                    'news_api_key': '',
                    'dart_api_key': '',
                    'push_config': {
                        'enabled': True,
                        'device_tokens': [],
                        'signal_threshold': 0.8,
                        'risk_threshold': 0.7
                    }
                },
                'system_config': {
                    'dashboard_port': 8000,
                    'log_level': 'INFO',
                    'auto_restart': True,
                    'health_check_interval': 30
                }
            },
            'agile_config': {
                'max_position_size': 1000000,  # 100만원
                'min_market_cap': 100000000000,  # 1000억원
                'max_market_cap': 5000000000000,  # 5조원
                'volume_surge_threshold': 2.0,
                'news_impact_threshold': 0.7,
                'theme_strength_threshold': 0.6,
                'instant_entry_timeout': 30,
                'fast_exit_timeout': 30,
                'max_hold_time': 7,
                'push_config': {
                    'enabled': True,
                    'device_tokens': [],
                    'signal_threshold': 0.7,
                    'risk_threshold': 0.6
                }
            },
            'agile_dashboard_config': {
                'host': '0.0.0.0',
                'port': 8001
            }
        }

        # 사용자 설정과 병합
        self.config = self._merge_configs(default_config, self.config)

        logger.info("Agile configuration loaded successfully")

    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """설정 병합"""
        merged = default.copy()

        def deep_merge(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value

        deep_merge(merged, user)
        return merged

    async def _initialize_services(self):
        """서비스 초기화"""
        logger.info("Initializing agile services...")

        try:
            # 기존 시스템 초기화
            self.main_system = IntegratedTradingSystem(self.config['main_system_config'])
            logger.info("Main system initialized")

            # 민첩성 전략 초기화
            self.agile_strategy = AgileTradingStrategy(self.config['agile_config'])
            logger.info("Agile strategy initialized")

            # 민첩성 대시보드 초기화
            self.agile_dashboard = AgileDashboard(self.agile_strategy)
            logger.info("Agile dashboard initialized")

            logger.info("All agile services initialized successfully")

        except Exception as e:
            logger.error(f"Service initialization error: {e}")
            raise

    async def _run_system(self):
        """시스템 실행"""
        logger.info("Running agile integrated trading system...")

        try:
            # 모든 서비스 태스크 시작
            tasks = [
                asyncio.create_task(self._run_main_system()),
                asyncio.create_task(self._run_agile_strategy()),
                asyncio.create_task(self._run_agile_dashboard()),
                asyncio.create_task(self._run_agile_health_monitor()),
                asyncio.create_task(self._run_agile_system_monitor())
            ]

            # 태스크 실행
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"System runtime error: {e}")
        finally:
            await self._shutdown_system()

    async def _run_main_system(self):
        """기존 시스템 실행"""
        try:
            await self.main_system.start_system()
        except Exception as e:
            logger.error(f"Main system error: {e}")
            if self.config['main_system_config']['system_config']['auto_restart']:
                logger.info("Restarting main system...")
                await asyncio.sleep(5)
                await self._run_main_system()

    async def _run_agile_strategy(self):
        """민첩성 전략 실행"""
        try:
            await self.agile_strategy.start_strategy()
        except Exception as e:
            logger.error(f"Agile strategy error: {e}")
            if self.config['main_system_config']['system_config']['auto_restart']:
                logger.info("Restarting agile strategy...")
                await asyncio.sleep(5)
                await self._run_agile_strategy()

    async def _run_agile_dashboard(self):
        """민첩성 대시보드 실행"""
        try:
            # 대시보드 업데이트 태스크
            while self.running:
                await self.agile_dashboard.update_dashboard()
                await asyncio.sleep(5)  # 5초마다 업데이트
        except Exception as e:
            logger.error(f"Agile dashboard error: {e}")
            if self.config['main_system_config']['system_config']['auto_restart']:
                logger.info("Restarting agile dashboard...")
                await asyncio.sleep(5)
                await self._run_agile_dashboard()

    async def _run_agile_health_monitor(self):
        """민첩성 헬스 모니터링"""
        while self.running:
            try:
                await self._check_agile_system_health()
                await asyncio.sleep(self.config['main_system_config']['system_config']['health_check_interval'])
            except Exception as e:
                logger.error(f"Agile health monitor error: {e}")
                await asyncio.sleep(10)

    async def _run_agile_system_monitor(self):
        """민첩성 시스템 모니터링"""
        while self.running:
            try:
                await self._monitor_agile_system_status()
                await asyncio.sleep(60)  # 1분마다 상태 체크
            except Exception as e:
                logger.error(f"Agile system monitor error: {e}")
                await asyncio.sleep(60)

    async def _check_agile_system_health(self):
        """민첩성 시스템 헬스 체크"""
        try:
            # 각 서비스 상태 확인
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'main_system': self.main_system is not None,
                'agile_strategy': self.agile_strategy is not None,
                'agile_dashboard': self.agile_dashboard is not None,
                'system_running': self.running
            }

            # 민첩성 전략 상태 확인
            if self.agile_strategy:
                strategy_status = self.agile_strategy.get_strategy_status()
                health_status['agile_strategy_status'] = strategy_status

            # 기존 시스템 상태 확인
            if self.main_system:
                main_status = {
                    'kis_integration': self.main_system.kis_integration is not None,
                    'push_service': self.main_system.push_service is not None,
                    'data_pipeline': self.main_system.data_pipeline is not None
                }
                health_status['main_system_status'] = main_status

            logger.info(f"Agile system health check: {health_status}")

            # 상태 파일 저장
            with open('agile_system_health.json', 'w', encoding='utf-8') as f:
                json.dump(health_status, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Agile health check error: {e}")

    async def _monitor_agile_system_status(self):
        """민첩성 시스템 상태 모니터링"""
        try:
            # 메모리 사용량, CPU 사용량 등 시스템 리소스 모니터링

            system_status = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections())
            }

            logger.info(f"Agile system status: {system_status}")

            # 리소스 사용량이 높으면 경고
            if system_status['cpu_percent'] > 80:
                if self.main_system and self.main_system.push_service:
                    await self.main_system.push_service.send_emergency_notification(
                        title="민첩성 시스템 리소스 경고",
                        message=f"CPU 사용량이 {system_status['cpu_percent']:.1f}%입니다",
                        emergency_data=system_status
                    )

            if system_status['memory_percent'] > 80:
                if self.main_system and self.main_system.push_service:
                    await self.main_system.push_service.send_emergency_notification(
                        title="민첩성 시스템 리소스 경고",
                        message=f"메모리 사용량이 {system_status['memory_percent']:.1f}%입니다",
                        emergency_data=system_status
                    )

        except ImportError:
            logger.warning("psutil not available, skipping system monitoring")
        except Exception as e:
            logger.error(f"Agile system monitoring error: {e}")

    async def _shutdown_system(self):
        """시스템 종료"""
        logger.info("Shutting down agile integrated trading system...")

        self.running = False

        try:
            # 각 서비스 정리
            if self.main_system and self.main_system.data_pipeline and self.main_system.data_pipeline.session:
                await self.main_system.data_pipeline.session.close()
                logger.info("Main system session closed")

            if self.agile_strategy and self.agile_strategy.session:
                await self.agile_strategy.session.close()
                logger.info("Agile strategy session closed")

            logger.info("Agile system shutdown completed")

        except Exception as e:
            logger.error(f"Agile shutdown error: {e}")

async def main():
    """메인 실행 함수"""

    # 사용자 설정 (실제 환경에 맞게 수정)
    user_config = {
        'main_system_config': {
            'kis_config': {
                'symbols_to_track': ['005930', '000660', '035420', '051910', '006400']
            },
            'push_config': {
                'enabled': True,
                'device_tokens': [
                    'your_device_token_1',
                    'your_device_token_2'
                ]
            },
            'pipeline_config': {
                'news_api_key': 'your_news_api_key',
                'dart_api_key': 'your_dart_api_key'
            },
            'system_config': {
                'dashboard_port': 8000,
                'log_level': 'INFO',
                'auto_restart': True
            }
        },
        'agile_config': {
            'max_position_size': 1000000,  # 100만원
            'min_market_cap': 100000000000,  # 1000억원
            'max_market_cap': 5000000000000,  # 5조원
            'volume_surge_threshold': 2.0,
            'news_impact_threshold': 0.7,
            'theme_strength_threshold': 0.6,
            'instant_entry_timeout': 30,
            'fast_exit_timeout': 30,
            'max_hold_time': 7,
            'push_config': {
                'enabled': True,
                'device_tokens': [
                    'your_device_token_1',
                    'your_device_token_2'
                ],
                'signal_threshold': 0.7,
                'risk_threshold': 0.6
            }
        },
        'agile_dashboard_config': {
            'host': '0.0.0.0',
            'port': 8001
        }
    }

    # 시스템 시작
    system = AgileIntegratedTradingSystem(user_config)
    await system.start_system()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agile system interrupted by user")
    except Exception as e:
        logger.error(f"Agile system error: {e}")
        sys.exit(1)

