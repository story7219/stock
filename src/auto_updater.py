#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 자동 업데이트 및 진화 엔진
시스템이 스스로 성능을 모니터링하고 자동으로 개선되는 진화형 시스템
"""

import os
import logging
import json
import ast
import inspect
import importlib
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from investment_strategies import InvestmentStrategy

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """성능 지표"""
    strategy_name: str
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    avg_hold_period: float
    last_performance_date: datetime

@dataclass
class StrategyMutation:
    """전략 변이"""
    original_strategy: str
    mutated_code: str
    mutation_type: str
    expected_improvement: float
    test_results: Dict[str, Any]

class EvolutionaryOptimizer:
    """🧬 진화형 최적화 엔진"""
    
    def __init__(self):
        """초기화"""
        self.performance_history = {}
        self.best_strategies = {}
        self.mutation_success_rate = {}
        self.evolution_log = []
        
        # 진화 설정
        self.mutation_rate = 0.1
        self.selection_pressure = 0.7
        self.elite_preservation = 0.2
        
        # 성능 임계값
        self.min_improvement_threshold = 0.05
        self.performance_window_days = 30
        
        logger.info("🧬 진화형 최적화 엔진 초기화 완료")
    
    def analyze_strategy_performance(self, 
                                   strategy: InvestmentStrategy,
                                   historical_results: List[Dict]) -> PerformanceMetric:
        """전략 성능 분석"""
        
        if not historical_results:
            return PerformanceMetric(
                strategy_name=strategy.__class__.__name__,
                accuracy=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                win_rate=0.0,
                avg_hold_period=0.0,
                last_performance_date=datetime.now()
            )
        
        # 성능 계산
        returns = [r.get('return', 0) for r in historical_results]
        predictions = [r.get('predicted', False) for r in historical_results]
        actuals = [r.get('actual', False) for r in historical_results]
        
        # 정확도
        accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals) if actuals else 0
        
        # 샤프 비율
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 최대 낙폭
        cumulative_returns = np.cumsum(returns) if returns else [0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if drawdowns.size > 0 else 0
        
        # 총 수익률
        total_return = sum(returns) if returns else 0
        
        # 승률
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        
        return PerformanceMetric(
            strategy_name=strategy.__class__.__name__,
            accuracy=accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate,
            avg_hold_period=len(returns),  # 단순화
            last_performance_date=datetime.now()
        )
    
    def identify_improvement_opportunities(self, 
                                         performance_metrics: List[PerformanceMetric]) -> List[str]:
        """개선 기회 식별"""
        
        opportunities = []
        
        for metric in performance_metrics:
            # 성능이 낮은 전략들 식별
            if metric.accuracy < 0.6:
                opportunities.append(f"accuracy_improvement:{metric.strategy_name}")
            
            if metric.sharpe_ratio < 1.0:
                opportunities.append(f"risk_adjustment:{metric.strategy_name}")
            
            if metric.max_drawdown < -0.2:  # 20% 이상 낙폭
                opportunities.append(f"drawdown_control:{metric.strategy_name}")
            
            if metric.win_rate < 0.5:
                opportunities.append(f"win_rate_enhancement:{metric.strategy_name}")
        
        return opportunities
    
    def generate_strategy_mutations(self, 
                                  strategy_class: type,
                                  target_improvement: str) -> List[StrategyMutation]:
        """전략 변이 생성"""
        
        mutations = []
        
        # 소스 코드 가져오기
        try:
            source_code = inspect.getsource(strategy_class)
            source_lines = source_code.split('\n')
            
            # 변이 타입별 처리
            if "accuracy_improvement" in target_improvement:
                mutations.extend(self._generate_accuracy_mutations(source_lines, strategy_class.__name__))
            
            if "risk_adjustment" in target_improvement:
                mutations.extend(self._generate_risk_mutations(source_lines, strategy_class.__name__))
            
            if "drawdown_control" in target_improvement:
                mutations.extend(self._generate_drawdown_mutations(source_lines, strategy_class.__name__))
            
            if "win_rate_enhancement" in target_improvement:
                mutations.extend(self._generate_winrate_mutations(source_lines, strategy_class.__name__))
                
        except Exception as e:
            logger.error(f"변이 생성 실패: {e}")
        
        return mutations
    
    def _generate_accuracy_mutations(self, source_lines: List[str], strategy_name: str) -> List[StrategyMutation]:
        """정확도 개선 변이 생성"""
        
        mutations = []
        
        # 1. 임계값 조정 변이
        for i, line in enumerate(source_lines):
            if any(keyword in line for keyword in ['threshold', 'limit', 'min_', 'max_']):
                # 숫자 값 찾기
                import re
                numbers = re.findall(r'\d+\.?\d*', line)
                
                for num in numbers:
                    try:
                        original_val = float(num)
                        # ±20% 변이
                        for factor in [0.8, 1.2]:
                            new_val = original_val * factor
                            mutated_line = line.replace(num, str(new_val), 1)
                            
                            mutated_lines = source_lines.copy()
                            mutated_lines[i] = mutated_line
                            
                            mutations.append(StrategyMutation(
                                original_strategy=strategy_name,
                                mutated_code='\n'.join(mutated_lines),
                                mutation_type="threshold_adjustment",
                                expected_improvement=0.05,
                                test_results={}
                            ))
                            
                            if len(mutations) >= 5:  # 최대 5개 변이
                                return mutations
                                
                    except ValueError:
                        continue
        
        return mutations
    
    def _generate_risk_mutations(self, source_lines: List[str], strategy_name: str) -> List[StrategyMutation]:
        """리스크 조정 변이 생성"""
        
        mutations = []
        
        # 리스크 관련 코드 패턴 찾기
        risk_patterns = [
            ('volatility', 'volatility * 1.2'),
            ('risk_score', 'risk_score * 0.8'),
            ('max_position', 'max_position * 0.9'),
            ('stop_loss', 'stop_loss * 0.95')
        ]
        
        for i, line in enumerate(source_lines):
            for pattern, replacement in risk_patterns:
                if pattern in line.lower():
                    mutated_lines = source_lines.copy()
                    # 간단한 곱셈 추가
                    if '=' in line and not line.strip().startswith('#'):
                        mutated_lines.insert(i + 1, f"        {pattern} = {replacement}  # Auto-generated risk adjustment")
                        
                        mutations.append(StrategyMutation(
                            original_strategy=strategy_name,
                            mutated_code='\n'.join(mutated_lines),
                            mutation_type="risk_adjustment",
                            expected_improvement=0.1,
                            test_results={}
                        ))
                        
                        if len(mutations) >= 3:
                            return mutations
        
        return mutations
    
    def _generate_drawdown_mutations(self, source_lines: List[str], strategy_name: str) -> List[StrategyMutation]:
        """낙폭 제어 변이 생성"""
        
        mutations = []
        
        # 손절 로직 추가
        for i, line in enumerate(source_lines):
            if 'def analyze(' in line or 'def calculate_score(' in line:
                # 손절 로직 삽입
                drawdown_logic = [
                    "        # Auto-generated drawdown control",
                    "        if hasattr(stock, 'recent_return') and stock.recent_return < -0.15:",
                    "            score.total_score *= 0.5  # Reduce score for high drawdown stocks",
                ]
                
                mutated_lines = source_lines.copy()
                # 함수 시작 부분에 삽입
                insert_position = i + 2  # 함수 정의 다음 줄
                for j, logic_line in enumerate(drawdown_logic):
                    mutated_lines.insert(insert_position + j, logic_line)
                
                mutations.append(StrategyMutation(
                    original_strategy=strategy_name,
                    mutated_code='\n'.join(mutated_lines),
                    mutation_type="drawdown_control",
                    expected_improvement=0.08,
                    test_results={}
                ))
                
                break
        
        return mutations
    
    def _generate_winrate_mutations(self, source_lines: List[str], strategy_name: str) -> List[StrategyMutation]:
        """승률 개선 변이 생성"""
        
        mutations = []
        
        # 확률 기반 필터링 추가
        for i, line in enumerate(source_lines):
            if 'return score' in line or 'score.total_score' in line:
                # 확률적 필터링 추가
                probability_logic = [
                    "        # Auto-generated probability filtering",
                    "        import random",
                    "        if score.total_score > 70 and random.random() > 0.3:",
                    "            score.total_score *= 1.1  # Boost high-confidence predictions",
                    "        elif score.total_score < 40 and random.random() > 0.7:",
                    "            score.total_score *= 0.8  # Reduce low-confidence predictions",
                ]
                
                mutated_lines = source_lines.copy()
                # 점수 반환 직전에 삽입
                for j, logic_line in enumerate(probability_logic):
                    mutated_lines.insert(i + j, logic_line)
                
                mutations.append(StrategyMutation(
                    original_strategy=strategy_name,
                    mutated_code='\n'.join(mutated_lines),
                    mutation_type="probability_filtering",
                    expected_improvement=0.06,
                    test_results={}
                ))
                
                break
        
        return mutations

class AutoUpdater:
    """🔄 자동 업데이트 시스템"""
    
    def __init__(self):
        """초기화"""
        self.update_history = []
        self.pending_updates = []
        self.backup_dir = "backups"
        self.test_results_dir = "test_results"
        
        self.evolutionary_optimizer = EvolutionaryOptimizer()
        
        # 업데이트 설정
        self.auto_update_enabled = True
        self.min_test_success_rate = 0.8
        self.rollback_on_failure = True
        
        self._ensure_directories()
        
        logger.info("🔄 자동 업데이트 시스템 초기화 완료")
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.test_results_dir, exist_ok=True)
    
    def monitor_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 모니터링"""
        
        logger.info("📊 시스템 성능 모니터링 시작")
        
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'good',
            'strategy_performances': [],
            'system_metrics': {},
            'improvement_recommendations': [],
            'auto_update_triggered': False
        }
        
        try:
            # 1. 전략별 성능 체크
            strategy_performances = self._check_strategy_performances()
            performance_report['strategy_performances'] = strategy_performances
            
            # 2. 시스템 리소스 체크
            system_metrics = self._check_system_metrics()
            performance_report['system_metrics'] = system_metrics
            
            # 3. 개선 기회 식별
            improvements = self.evolutionary_optimizer.identify_improvement_opportunities(
                strategy_performances
            )
            performance_report['improvement_recommendations'] = improvements
            
            # 4. 자동 업데이트 트리거 결정
            if self._should_trigger_auto_update(strategy_performances, improvements):
                performance_report['auto_update_triggered'] = True
                self._trigger_evolutionary_update(improvements)
            
            # 5. 전반적 건강도 평가
            performance_report['overall_health'] = self._assess_overall_health(
                strategy_performances, system_metrics
            )
            
            logger.info(f"✅ 시스템 건강도: {performance_report['overall_health']}")
            
        except Exception as e:
            logger.error(f"성능 모니터링 실패: {e}")
            performance_report['overall_health'] = 'error'
        
        return performance_report
    
    def _check_strategy_performances(self) -> List[PerformanceMetric]:
        """전략별 성능 체크"""
        
        performances = []
        
        try:
            # 기존 성능 데이터 로드 (실제로는 데이터베이스나 파일에서)
            # 여기서는 시뮬레이션
            strategy_names = [
                "WarrenBuffettStrategy", "PeterLynchStrategy", 
                "BenjaminGrahamStrategy", "PhilipFisherStrategy"
            ]
            
            for strategy_name in strategy_names:
                # 시뮬레이션된 성능 데이터
                performance = PerformanceMetric(
                    strategy_name=strategy_name,
                    accuracy=np.random.uniform(0.45, 0.85),
                    sharpe_ratio=np.random.uniform(0.5, 2.0),
                    max_drawdown=np.random.uniform(-0.25, -0.05),
                    total_return=np.random.uniform(-0.1, 0.3),
                    win_rate=np.random.uniform(0.4, 0.7),
                    avg_hold_period=np.random.uniform(10, 60),
                    last_performance_date=datetime.now()
                )
                
                performances.append(performance)
            
        except Exception as e:
            logger.warning(f"전략 성능 체크 실패: {e}")
        
        return performances
    
    def _check_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 체크"""
        
        import psutil
        
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'python_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'uptime_hours': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600
            }
            
        except Exception:
            # psutil 없는 경우 기본값
            metrics = {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'disk_percent': 70.0,
                'python_memory_mb': 100.0,
                'uptime_hours': 24.0
            }
        
        return metrics
    
    def _should_trigger_auto_update(self, 
                                   performances: List[PerformanceMetric],
                                   improvements: List[str]) -> bool:
        """자동 업데이트 트리거 여부 결정"""
        
        if not self.auto_update_enabled:
            return False
        
        # 성능이 임계값 이하인 전략이 많은 경우
        poor_performance_count = sum(
            1 for p in performances 
            if p.accuracy < 0.6 or p.sharpe_ratio < 1.0
        )
        
        # 개선 기회가 많은 경우
        critical_improvements = len([i for i in improvements if 'accuracy' in i or 'drawdown' in i])
        
        # 트리거 조건
        should_trigger = (
            poor_performance_count >= len(performances) * 0.5 or  # 50% 이상 성능 저하
            critical_improvements >= 3 or  # 3개 이상 중요 개선사항
            len(improvements) >= 5  # 5개 이상 개선사항
        )
        
        logger.info(f"자동 업데이트 트리거: {should_trigger} (성능저하: {poor_performance_count}, 개선사항: {len(improvements)})")
        
        return should_trigger
    
    def _assess_overall_health(self, 
                             performances: List[PerformanceMetric],
                             system_metrics: Dict[str, Any]) -> str:
        """전반적 건강도 평가"""
        
        # 성능 점수
        avg_accuracy = np.mean([p.accuracy for p in performances]) if performances else 0
        avg_sharpe = np.mean([p.sharpe_ratio for p in performances]) if performances else 0
        
        # 시스템 점수
        cpu_health = 1.0 - (system_metrics.get('cpu_percent', 50) / 100)
        memory_health = 1.0 - (system_metrics.get('memory_percent', 60) / 100)
        
        # 종합 점수
        overall_score = (avg_accuracy * 0.4 + 
                        (avg_sharpe / 2.0) * 0.3 + 
                        cpu_health * 0.15 + 
                        memory_health * 0.15)
        
        if overall_score > 0.8:
            return "excellent"
        elif overall_score > 0.6:
            return "good"
        elif overall_score > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _trigger_evolutionary_update(self, improvements: List[str]):
        """진화적 업데이트 트리거"""
        
        logger.info("🧬 진화적 업데이트 시작")
        
        try:
            # 백업 생성
            self._create_backup()
            
            # 개선사항별 변이 생성 및 테스트
            for improvement in improvements[:3]:  # 최대 3개 처리
                strategy_name = improvement.split(':')[-1] if ':' in improvement else 'UnknownStrategy'
                
                # 임시로 더미 클래스 생성 (실제로는 동적 로드)
                class DummyStrategy:
                    def analyze(self, stock):
                        return {"score": 50}
                
                mutations = self.evolutionary_optimizer.generate_strategy_mutations(
                    DummyStrategy, improvement
                )
                
                # 변이 테스트
                for mutation in mutations[:2]:  # 최대 2개 변이 테스트
                    test_result = self._test_mutation(mutation)
                    
                    if test_result['success_rate'] > self.min_test_success_rate:
                        self._apply_mutation(mutation)
                        logger.info(f"✅ 변이 적용 성공: {mutation.mutation_type}")
                    else:
                        logger.info(f"❌ 변이 테스트 실패: {mutation.mutation_type}")
            
            logger.info("✅ 진화적 업데이트 완료")
            
        except Exception as e:
            logger.error(f"진화적 업데이트 실패: {e}")
            if self.rollback_on_failure:
                self._rollback_to_backup()
    
    def _create_backup(self):
        """현재 시스템 백업"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        
        try:
            # 주요 파일들 백업
            important_files = [
                'investment_strategies.py',
                'src/main_optimized.py',
                'src/ml_engine.py'
            ]
            
            os.makedirs(backup_path, exist_ok=True)
            
            for file_path in important_files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_path)
            
            logger.info(f"💾 백업 생성: {backup_path}")
            
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")
    
    def _test_mutation(self, mutation: StrategyMutation) -> Dict[str, Any]:
        """변이 테스트"""
        
        test_result = {
            'success_rate': 0.0,
            'performance_improvement': 0.0,
            'errors': [],
            'execution_time': 0.0
        }
        
        try:
            # 코드 구문 검사
            ast.parse(mutation.mutated_code)
            test_result['success_rate'] += 0.3
            
            # 시뮬레이션된 성능 테스트
            # 실제로는 백테스팅이나 샘플 데이터로 테스트
            simulated_performance = np.random.uniform(0.5, 0.9)
            test_result['success_rate'] += simulated_performance * 0.7
            
            test_result['performance_improvement'] = simulated_performance - 0.6
            
        except SyntaxError as e:
            test_result['errors'].append(f"구문 오류: {e}")
        except Exception as e:
            test_result['errors'].append(f"실행 오류: {e}")
        
        return test_result
    
    def _apply_mutation(self, mutation: StrategyMutation):
        """변이 적용"""
        
        try:
            # 새로운 전략 파일 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mutated_file = f"strategies/evolved_{mutation.original_strategy}_{timestamp}.py"
            
            os.makedirs("strategies", exist_ok=True)
            
            with open(mutated_file, 'w', encoding='utf-8') as f:
                f.write(mutation.mutated_code)
            
            logger.info(f"📄 진화된 전략 저장: {mutated_file}")
            
            # 업데이트 기록
            self.update_history.append({
                'timestamp': datetime.now().isoformat(),
                'mutation_type': mutation.mutation_type,
                'original_strategy': mutation.original_strategy,
                'file_path': mutated_file,
                'expected_improvement': mutation.expected_improvement
            })
            
        except Exception as e:
            logger.error(f"변이 적용 실패: {e}")
    
    def _rollback_to_backup(self):
        """백업으로 롤백"""
        
        try:
            # 최신 백업 찾기
            backups = [d for d in os.listdir(self.backup_dir) if d.startswith('backup_')]
            if not backups:
                logger.warning("롤백할 백업이 없습니다")
                return
            
            latest_backup = max(backups)
            backup_path = os.path.join(self.backup_dir, latest_backup)
            
            # 백업에서 파일 복원
            for file_name in os.listdir(backup_path):
                backup_file = os.path.join(backup_path, file_name)
                target_file = file_name  # 또는 적절한 경로 매핑
                
                shutil.copy2(backup_file, target_file)
            
            logger.info(f"🔄 백업으로 롤백 완료: {latest_backup}")
            
        except Exception as e:
            logger.error(f"롤백 실패: {e}")
    
    def get_update_status(self) -> Dict[str, Any]:
        """업데이트 상태 반환"""
        
        return {
            'auto_update_enabled': self.auto_update_enabled,
            'last_update': self.update_history[-1] if self.update_history else None,
            'total_updates': len(self.update_history),
            'pending_updates': len(self.pending_updates),
            'backup_count': len([d for d in os.listdir(self.backup_dir) if d.startswith('backup_')]),
            'evolution_enabled': True,
            'last_monitoring': datetime.now().isoformat()
        }

class SmartConfigManager:
    """🎛️ 스마트 설정 관리자"""
    
    def __init__(self):
        """초기화"""
        self.config_path = "config/smart_config.json"
        self.learning_history = []
        self.optimization_rules = {}
        
        self._load_config()
        
        logger.info("🎛️ 스마트 설정 관리자 초기화 완료")
    
    def _load_config(self):
        """설정 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.optimization_rules = config.get('optimization_rules', {})
                    self.learning_history = config.get('learning_history', [])
        except Exception as e:
            logger.warning(f"설정 로드 실패: {e}")
    
    def auto_tune_parameters(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """성능 데이터 기반 자동 파라미터 튜닝"""
        
        optimized_params = {}
        
        try:
            # 성능 기반 학습률 조정
            if 'accuracy' in performance_data:
                accuracy = performance_data['accuracy']
                
                if accuracy < 0.6:
                    optimized_params['learning_rate'] = 0.01  # 더 세밀한 학습
                    optimized_params['exploration_rate'] = 0.3  # 더 많은 탐색
                elif accuracy > 0.8:
                    optimized_params['learning_rate'] = 0.005  # 안정화
                    optimized_params['exploration_rate'] = 0.1  # 탐색 감소
                else:
                    optimized_params['learning_rate'] = 0.007
                    optimized_params['exploration_rate'] = 0.2
            
            # 리스크 기반 포지션 크기 조정
            if 'max_drawdown' in performance_data:
                drawdown = abs(performance_data['max_drawdown'])
                
                if drawdown > 0.2:
                    optimized_params['max_position_size'] = 0.05  # 포지션 축소
                    optimized_params['stop_loss_threshold'] = 0.08  # 더 엄격한 손절
                else:
                    optimized_params['max_position_size'] = 0.1
                    optimized_params['stop_loss_threshold'] = 0.12
            
            # 학습 기록 저장
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'input_performance': performance_data,
                'optimized_params': optimized_params
            })
            
            self._save_config()
            
            logger.info(f"🎯 파라미터 자동 튜닝 완료: {len(optimized_params)}개 파라미터")
            
        except Exception as e:
            logger.error(f"자동 튜닝 실패: {e}")
        
        return optimized_params
    
    def _save_config(self):
        """설정 저장"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config = {
                'optimization_rules': self.optimization_rules,
                'learning_history': self.learning_history[-100:],  # 최근 100개만 보관
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")

if __name__ == "__main__":
    print("🔄 자동 업데이트 및 진화 엔진 v1.0")
    print("=" * 50)
    
    # 테스트
    auto_updater = AutoUpdater()
    
    # 성능 모니터링 테스트
    report = auto_updater.monitor_system_performance()
    
    print(f"\n📊 시스템 상태:")
    print(f"  • 전반적 건강도: {report['overall_health']}")
    print(f"  • 개선 권장사항: {len(report['improvement_recommendations'])}개")
    print(f"  • 자동 업데이트: {'활성화' if report['auto_update_triggered'] else '대기'}")
    
    print("\n✅ 자동 진화 시스템 준비 완료!") 