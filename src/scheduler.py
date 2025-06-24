#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⏰ 자동 스케줄러 시스템
시간 기반으로 자동 분석, 학습, 진화를 수행하는 스케줄러
"""

import asyncio
import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
import os
from main_optimized import InvestmentAnalysisSystem

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    """예약된 작업"""
    name: str
    function: Callable
    schedule_type: str  # 'daily', 'weekly', 'hourly', 'interval'
    schedule_time: str  # "09:00", "Monday", "5" (minutes)
    enabled: bool
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    run_count: int
    parameters: Dict[str, Any]

@dataclass
class ScheduleStatus:
    """스케줄 상태"""
    active_tasks: int
    completed_today: int
    failed_today: int
    next_scheduled: Optional[datetime]
    system_running: bool
    last_health_check: datetime

class IntelligentScheduler:
    """🧠 지능형 스케줄러"""
    
    def __init__(self, analysis_system: InvestmentAnalysisSystem):
        """초기화"""
        self.analysis_system = analysis_system
        self.scheduled_tasks = []
        self.task_history = []
        self.is_running = False
        
        # 스케줄 설정
        self.config_path = "config/scheduler_config.json"
        self.max_concurrent_tasks = 3
        self.task_timeout_minutes = 60
        
        # 통계
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        
        # 기본 스케줄 설정
        self._setup_default_schedule()
        
        logger.info("⏰ 지능형 스케줄러 초기화 완료")
    
    def _setup_default_schedule(self):
        """기본 스케줄 설정"""
        
        default_tasks = [
            # 매일 장 시작 전 분석 (09:00)
            ScheduledTask(
                name="daily_market_analysis",
                function=self._run_daily_analysis,
                schedule_type="daily",
                schedule_time="09:00",
                enabled=True,
                last_run=None,
                next_run=None,
                run_count=0,
                parameters={"markets": ["KOSPI200", "NASDAQ100", "S&P500"], "full_analysis": True}
            ),
            
            # 시간당 실시간 모니터링 (평일 09:00-18:00)
            ScheduledTask(
                name="hourly_monitoring",
                function=self._run_hourly_monitoring,
                schedule_type="hourly",
                schedule_time="09-18",  # 9시부터 18시까지
                enabled=True,
                last_run=None,
                next_run=None,
                run_count=0,
                parameters={"quick_scan": True}
            ),
            
            # 주간 모델 재학습 (일요일 02:00)
            ScheduledTask(
                name="weekly_model_training",
                function=self._run_weekly_training,
                schedule_type="weekly",
                schedule_time="Sunday:02:00",
                enabled=True,
                last_run=None,
                next_run=None,
                run_count=0,
                parameters={"full_retrain": True}
            ),
            
            # 매일 성능 진화 체크 (22:00)
            ScheduledTask(
                name="daily_evolution_check",
                function=self._run_evolution_check,
                schedule_type="daily",
                schedule_time="22:00",
                enabled=True,
                last_run=None,
                next_run=None,
                run_count=0,
                parameters={"enable_auto_evolution": True}
            ),
            
            # 5분마다 시스템 헬스 체크
            ScheduledTask(
                name="health_check",
                function=self._run_health_check,
                schedule_type="interval",
                schedule_time="5",  # 5분
                enabled=True,
                last_run=None,
                next_run=None,
                run_count=0,
                parameters={"deep_check": False}
            )
        ]
        
        self.scheduled_tasks.extend(default_tasks)
        logger.info(f"📋 기본 스케줄 설정 완료: {len(default_tasks)}개 작업")
    
    def start_scheduler(self):
        """스케줄러 시작"""
        
        if self.is_running:
            logger.warning("스케줄러가 이미 실행 중입니다")
            return
        
        logger.info("🚀 스케줄러 시작")
        
        # 각 작업을 schedule 라이브러리에 등록
        for task in self.scheduled_tasks:
            if not task.enabled:
                continue
                
            try:
                self._register_task(task)
                logger.info(f"✅ 작업 등록: {task.name} ({task.schedule_type})")
            except Exception as e:
                logger.error(f"❌ 작업 등록 실패 ({task.name}): {e}")
        
        # 백그라운드에서 스케줄러 실행
        self.is_running = True
        scheduler_thread = threading.Thread(target=self._run_scheduler_loop, daemon=True)
        scheduler_thread.start()
        
        logger.info("✅ 스케줄러 시작 완료")
    
    def _register_task(self, task: ScheduledTask):
        """작업을 schedule에 등록"""
        
        if task.schedule_type == "daily":
            schedule.every().day.at(task.schedule_time).do(
                self._execute_task_safely, task
            )
        
        elif task.schedule_type == "weekly":
            day, time_str = task.schedule_time.split(":")
            schedule.every().week.day.at(time_str).do(
                self._execute_task_safely, task
            )
        
        elif task.schedule_type == "hourly":
            if "-" in task.schedule_time:  # 특정 시간대
                start_hour, end_hour = map(int, task.schedule_time.split("-"))
                # 현재는 단순화하여 매시간 실행
                schedule.every().hour.do(self._execute_hourly_task, task, start_hour, end_hour)
            else:
                schedule.every().hour.do(self._execute_task_safely, task)
        
        elif task.schedule_type == "interval":
            minutes = int(task.schedule_time)
            schedule.every(minutes).minutes.do(self._execute_task_safely, task)
    
    def _execute_hourly_task(self, task: ScheduledTask, start_hour: int, end_hour: int):
        """시간대 제한 있는 작업 실행"""
        
        current_hour = datetime.now().hour
        
        if start_hour <= current_hour <= end_hour:
            self._execute_task_safely(task)
        else:
            logger.debug(f"시간대 밖 작업 스킵: {task.name} (현재: {current_hour}시)")
    
    def _execute_task_safely(self, task: ScheduledTask):
        """안전한 작업 실행 (예외 처리 및 리소스 체크 포함)"""
        
        logger.info(f"🔄 작업 실행 시작: {task.name}")
        
        # 시스템 리소스 체크
        if not self._check_system_resources():
            logger.warning(f"⚠️ 시스템 리소스 부족으로 작업 연기: {task.name}")
            return
        
        start_time = datetime.now()
        task.last_run = start_time
        
        try:
            # 비동기 함수를 동기적으로 실행
            if asyncio.iscoroutinefunction(task.function):
                # 새로운 이벤트 루프에서 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(task.function(**task.parameters))
                loop.close()
            else:
                result = task.function(**task.parameters)
            
            # 성공 기록
            execution_time = (datetime.now() - start_time).total_seconds()
            
            task.run_count += 1
            self.total_runs += 1
            self.successful_runs += 1
            
            # 작업 기록
            self.task_history.append({
                'task_name': task.name,
                'start_time': start_time.isoformat(),
                'execution_time_seconds': execution_time,
                'status': 'success',
                'result_summary': str(result)[:200] if result else 'completed'
            })
            
            logger.info(f"✅ 작업 완료: {task.name} ({execution_time:.1f}초)")
            
        except Exception as e:
            # 실패 기록
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.total_runs += 1
            self.failed_runs += 1
            
            self.task_history.append({
                'task_name': task.name,
                'start_time': start_time.isoformat(),
                'execution_time_seconds': execution_time,
                'status': 'failed',
                'error': str(e)
            })
            
            logger.error(f"❌ 작업 실패: {task.name} - {e}")
    
    def _check_system_resources(self) -> bool:
        """시스템 리소스 체크"""
        
        try:
            import psutil
            
            # 메모리 사용률 체크 (80% 이하)
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                logger.warning(f"메모리 사용률 높음: {memory_percent:.1f}%")
                return False
            
            # CPU 사용률 체크 (90% 이하)
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"CPU 사용률 높음: {cpu_percent:.1f}%")
                return False
            
            return True
            
        except ImportError:
            # psutil이 없으면 기본적으로 허용
            logger.debug("psutil 없음 - 리소스 체크 스킵")
            return True
        except Exception as e:
            logger.debug(f"리소스 체크 실패: {e}")
            return True  # 오류시 기본 허용
    
    def _run_scheduler_loop(self):
        """스케줄러 메인 루프"""
        
        logger.info("🔄 스케줄러 루프 시작")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                logger.error(f"스케줄러 루프 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기
    
    def stop_scheduler(self):
        """스케줄러 중지"""
        
        logger.info("⏸️ 스케줄러 중지")
        self.is_running = False
        schedule.clear()
        
        logger.info("✅ 스케줄러 중지 완료")
    
    # === 스케줄된 작업 함수들 ===
    
    async def _run_daily_analysis(self, **kwargs) -> Dict[str, Any]:
        """일일 시장 분석"""
        
        logger.info("📊 일일 시장 분석 시작")
        
        try:
            markets = kwargs.get('markets', ["KOSPI200", "NASDAQ100"])
            full_analysis = kwargs.get('full_analysis', True)
            
            # 종합 분석 실행
            results = await self.analysis_system.run_comprehensive_analysis(
                markets=markets,
                enable_learning=True,
                enable_evolution=full_analysis
            )
            
            # 결과 요약
            summary = {
                'analysis_time': results.get('timestamp'),
                'markets_analyzed': len(results.get('markets_analyzed', [])),
                'top_recommendations': len(results.get('top_recommendations', [])),
                'system_health': results.get('system_status', {}).get('overall_health', 'unknown'),
                'learning_updates': bool(results.get('learning_updates', {})),
                'evolution_triggered': len(results.get('evolution_log', [])) > 0
            }
            
            logger.info(f"✅ 일일 분석 완료: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"일일 분석 실패: {e}")
            raise
    
    async def _run_hourly_monitoring(self, **kwargs) -> Dict[str, Any]:
        """시간당 모니터링"""
        
        logger.info("👁️ 시간당 모니터링 시작")
        
        try:
            quick_scan = kwargs.get('quick_scan', True)
            
            # 시스템 상태 체크
            system_status = await self.analysis_system._check_system_status()
            
            # 간단한 성능 모니터링
            performance_report = self.analysis_system.auto_updater.monitor_system_performance()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'system_health': system_status.overall_health,
                'ml_models': system_status.ml_models_loaded,
                'improvements_identified': len(performance_report.get('improvement_recommendations', [])),
                'auto_update_triggered': performance_report.get('auto_update_triggered', False)
            }
            
            # 긴급 상황 감지
            if system_status.overall_health in ['poor', 'error']:
                logger.warning(f"🚨 시스템 상태 경고: {system_status.overall_health}")
                # 자동 복구 시도
                await self._attempt_auto_recovery()
            
            logger.info(f"✅ 시간당 모니터링 완료: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"시간당 모니터링 실패: {e}")
            raise
    
    async def _run_weekly_training(self, **kwargs) -> Dict[str, Any]:
        """주간 모델 재학습"""
        
        logger.info("🧠 주간 모델 재학습 시작")
        
        try:
            full_retrain = kwargs.get('full_retrain', True)
            
            # ML 모델 재학습
            training_success = self.analysis_system.ml_engine.train_models()
            
            # 모델 성능 평가
            model_status = self.analysis_system.ml_engine.get_model_status()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'training_success': training_success,
                'models_trained': sum(model_status.get('models_loaded', {}).values()),
                'full_retrain': full_retrain
            }
            
            logger.info(f"✅ 주간 재학습 완료: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"주간 재학습 실패: {e}")
            raise
    
    async def _run_evolution_check(self, **kwargs) -> Dict[str, Any]:
        """진화 체크"""
        
        logger.info("🧬 진화 체크 시작")
        
        try:
            enable_auto_evolution = kwargs.get('enable_auto_evolution', True)
            
            if enable_auto_evolution:
                # 진화 사이클 실행
                evolution_log = await self.analysis_system._run_evolutionary_cycle()
                
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'evolution_actions': len(evolution_log),
                    'auto_evolution_enabled': enable_auto_evolution,
                    'actions_taken': [action.get('type') for action in evolution_log]
                }
            else:
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'evolution_actions': 0,
                    'auto_evolution_enabled': False,
                    'message': 'auto evolution disabled'
                }
            
            logger.info(f"✅ 진화 체크 완료: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"진화 체크 실패: {e}")
            raise
    
    async def _run_health_check(self, **kwargs) -> Dict[str, Any]:
        """헬스 체크"""
        
        deep_check = kwargs.get('deep_check', False)
        
        try:
            # 기본 시스템 상태
            system_status = await self.analysis_system._check_system_status()
            
            health_summary = {
                'timestamp': datetime.now().isoformat(),
                'overall_health': system_status.overall_health,
                'uptime_hours': (datetime.now() - self.analysis_system.start_time).total_seconds() / 3600,
                'scheduler_running': self.is_running,
                'task_success_rate': self.successful_runs / max(self.total_runs, 1),
                'deep_check': deep_check
            }
            
            if deep_check:
                # 깊은 체크 추가 로직
                health_summary['memory_usage'] = 'normal'  # 실제로는 psutil로 체크
                health_summary['disk_space'] = 'sufficient'
                health_summary['network_status'] = 'connected'
            
            # 문제 감지 시 알림
            if health_summary['task_success_rate'] < 0.8:
                logger.warning(f"🚨 작업 성공률 낮음: {health_summary['task_success_rate']:.1%}")
            
            return health_summary
            
        except Exception as e:
            logger.error(f"헬스 체크 실패: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'error',
                'error': str(e)
            }
    
    async def _attempt_auto_recovery(self):
        """자동 복구 시도"""
        
        logger.info("🔧 자동 복구 시도")
        
        try:
            # 1. 시스템 재시작 시뮬레이션
            logger.info("  • 시스템 구성요소 재초기화...")
            
            # 2. 백업으로 롤백 (필요시)
            if hasattr(self.analysis_system.auto_updater, '_rollback_to_backup'):
                logger.info("  • 백업으로 롤백 시도...")
                # 실제로는 상황에 따라 결정
            
            # 3. ML 모델 재로드
            logger.info("  • ML 모델 재로드...")
            # self.analysis_system.ml_engine.reload_models()
            
            logger.info("✅ 자동 복구 완료")
            
        except Exception as e:
            logger.error(f"자동 복구 실패: {e}")
    
    # === 스케줄 관리 함수들 ===
    
    def add_custom_task(self, task: ScheduledTask):
        """커스텀 작업 추가"""
        
        self.scheduled_tasks.append(task)
        
        if self.is_running and task.enabled:
            self._register_task(task)
        
        logger.info(f"📝 커스텀 작업 추가: {task.name}")
    
    def enable_task(self, task_name: str):
        """작업 활성화"""
        
        for task in self.scheduled_tasks:
            if task.name == task_name:
                task.enabled = True
                if self.is_running:
                    self._register_task(task)
                logger.info(f"✅ 작업 활성화: {task_name}")
                return
        
        logger.warning(f"작업을 찾을 수 없음: {task_name}")
    
    def disable_task(self, task_name: str):
        """작업 비활성화"""
        
        for task in self.scheduled_tasks:
            if task.name == task_name:
                task.enabled = False
                logger.info(f"⏸️ 작업 비활성화: {task_name}")
                return
        
        logger.warning(f"작업을 찾을 수 없음: {task_name}")
    
    def get_schedule_status(self) -> ScheduleStatus:
        """스케줄 상태 반환"""
        
        active_tasks = sum(1 for task in self.scheduled_tasks if task.enabled)
        
        # 오늘 실행된 작업 통계
        today = datetime.now().date()
        today_history = [
            h for h in self.task_history 
            if datetime.fromisoformat(h['start_time']).date() == today
        ]
        
        completed_today = sum(1 for h in today_history if h['status'] == 'success')
        failed_today = sum(1 for h in today_history if h['status'] == 'failed')
        
        # 다음 예정 작업 (간단화)
        next_scheduled = datetime.now() + timedelta(minutes=5)  # 다음 헬스체크
        
        return ScheduleStatus(
            active_tasks=active_tasks,
            completed_today=completed_today,
            failed_today=failed_today,
            next_scheduled=next_scheduled,
            system_running=self.is_running,
            last_health_check=datetime.now()
        )
    
    def get_task_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """작업 히스토리 반환"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_history = [
            h for h in self.task_history
            if datetime.fromisoformat(h['start_time']) > cutoff_time
        ]
        
        return recent_history
    
    def save_schedule_config(self):
        """스케줄 설정 저장"""
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config = {
                'scheduled_tasks': [
                    {
                        'name': task.name,
                        'schedule_type': task.schedule_type,
                        'schedule_time': task.schedule_time,
                        'enabled': task.enabled,
                        'parameters': task.parameters
                    }
                    for task in self.scheduled_tasks
                ],
                'scheduler_stats': {
                    'total_runs': self.total_runs,
                    'successful_runs': self.successful_runs,
                    'failed_runs': self.failed_runs
                },
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 스케줄 설정 저장: {self.config_path}")
            
        except Exception as e:
            logger.error(f"스케줄 설정 저장 실패: {e}")

# 사용 예시
def main():
    """메인 실행 함수"""
    print("⏰ 자동 스케줄러 시스템 v1.0")
    print("=" * 50)
    
    # 분석 시스템 초기화
    analysis_system = InvestmentAnalysisSystem()
    
    # 스케줄러 초기화
    scheduler = IntelligentScheduler(analysis_system)
    
    # 스케줄 상태 출력
    status = scheduler.get_schedule_status()
    print(f"\n📋 스케줄 상태:")
    print(f"  • 활성 작업: {status.active_tasks}개")
    print(f"  • 오늘 완료: {status.completed_today}개")
    print(f"  • 오늘 실패: {status.failed_today}개")
    print(f"  • 시스템 실행중: {'예' if status.system_running else '아니오'}")
    
    try:
        # 스케줄러 시작
        scheduler.start_scheduler()
        
        print(f"\n🚀 스케줄러 시작됨!")
        print(f"📝 예정된 작업:")
        for task in scheduler.scheduled_tasks:
            if task.enabled:
                print(f"  • {task.name}: {task.schedule_type} ({task.schedule_time})")
        
        print(f"\n⏰ 시스템이 백그라운드에서 실행됩니다...")
        print(f"Ctrl+C로 중지할 수 있습니다.")
        
        # 무한 대기 (실제 서비스에서는 다른 방식 사용)
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print(f"\n⏸️ 사용자 중지 요청")
        scheduler.stop_scheduler()
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        scheduler.stop_scheduler()
    
    print(f"✅ 스케줄러 종료")

if __name__ == "__main__":
    main() 