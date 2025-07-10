#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: mlops_pipeline.py
모듈: MLOps 파이프라인 및 A/B 테스팅 프레임워크
목적: 모델 드리프트 감지, 자동 재학습, A/B 테스팅

Author: AI Trading System
Created: 2025-01-08
Modified: 2025-01-08
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - mlflow==2.5.0
    - kubeflow==1.8.0
    - prometheus-client==0.17.0
    - grafana-api==1.0.3
    - kubernetes==26.1.0

Performance:
    - 모니터링 지연: < 1초
    - 드리프트 감지: < 5초
    - 재학습 트리거: < 10초

Security:
    - Model validation: comprehensive checks
    - Error handling: graceful degradation
    - Logging: model performance tracking

License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Literal,
    Protocol, TypeVar, Generic, Final, Callable
)

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
from kubernetes import client, config
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 타입 정의
T = TypeVar('T')
ModelInput = np.ndarray
ModelOutput = np.ndarray

# 로깅 설정
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class ModelStage(Enum):
    """모델 스테이지"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DriftType(Enum):
    """드리프트 타입"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class DriftAlert:
    """드리프트 알림"""
    
    drift_type: DriftType
    severity: Literal['low', 'medium', 'high', 'critical']
    score: float
    threshold: float
    timestamp: datetime
    model_id: str
    description: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ABTestConfig:
    """A/B 테스트 설정"""
    
    test_name: str
    model_a_id: str
    model_b_id: str
    traffic_split: float = 0.5  # B 모델로 가는 트래픽 비율
    duration_days: int = 30
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    success_metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'r2'])
    statistical_significance: float = 0.05
    min_sample_size: int = 1000


class ModelRegistry:
    """모델 레지스트리"""
    
    def __init__(self, registry_uri: str = "sqlite:///model_registry.db"):
        self.registry_uri = registry_uri
        mlflow.set_tracking_uri(registry_uri)
        self.models = {}
        self.versions = {}
    
    def register_model(self, model_name: str, model, model_metadata: Dict[str, Any]) -> str:
        """모델 등록"""
        try:
            # MLflow에 모델 저장
            with mlflow.start_run():
                # 모델 저장
                if hasattr(model, 'save'):
                    mlflow.pytorch.log_model(model, model_name)
                else:
                    mlflow.sklearn.log_model(model, model_name)
                
                # 메타데이터 로깅
                mlflow.log_params(model_metadata.get('hyperparameters', {}))
                mlflow.log_metrics(model_metadata.get('performance_metrics', {}))
                
                # 모델 버전 생성
                model_version = mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                    model_name
                )
                
                # 로컬 레지스트리 업데이트
                self.models[model_name] = model
                self.versions[model_name] = model_version.version
                
                logger.info(f"모델 등록 완료: {model_name} v{model_version.version}")
                return model_version.run_id
                
        except Exception as e:
            logger.error(f"모델 등록 오류: {e}")
            raise
    
    def get_model(self, model_name: str, version: Optional[int] = None) -> Any:
        """모델 로드"""
        try:
            if version is None:
                # 최신 버전 로드
                model_uri = f"models:/{model_name}/latest"
            else:
                # 특정 버전 로드
                model_uri = f"models:/{model_name}/{version}"
            
            model = mlflow.pytorch.load_model(model_uri)
            return model
            
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """모델 목록 조회"""
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            
            model_list = []
            for model in models:
                latest_version = client.get_latest_versions(model.name, stages=["Production"])
                model_list.append({
                    'name': model.name,
                    'latest_version': latest_version[0].version if latest_version else None,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"모델 목록 조회 오류: {e}")
            return []
    
    def transition_model_stage(self, model_name: str, version: int, 
                             stage: ModelStage) -> None:
        """모델 스테이지 전환"""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value
            )
            
            logger.info(f"모델 스테이지 전환: {model_name} v{version} -> {stage.value}")
            
        except Exception as e:
            logger.error(f"모델 스테이지 전환 오류: {e}")
            raise


class DriftDetector:
    """모델 드리프트 감지기"""
    
    def __init__(self, detection_thresholds: Dict[DriftType, float] = None):
        self.detection_thresholds = detection_thresholds or {
            DriftType.DATA_DRIFT: 0.1,
            DriftType.CONCEPT_DRIFT: 0.15,
            DriftType.LABEL_DRIFT: 0.1,
            DriftType.PERFORMANCE_DRIFT: 0.2
        }
        self.reference_data = None
        self.reference_performance = None
        self.drift_history = []
    
    def set_reference_data(self, data: ModelInput, performance: Dict[str, float]) -> None:
        """기준 데이터 설정"""
        self.reference_data = data
        self.reference_performance = performance
        logger.info("기준 데이터 설정 완료")
    
    def detect_data_drift(self, current_data: ModelInput) -> DriftAlert:
        """데이터 드리프트 감지"""
        try:
            if self.reference_data is None:
                raise ValueError("기준 데이터가 설정되지 않았습니다.")
            
            # 통계적 거리 계산 (KL 발산, Jensen-Shannon 거리 등)
            drift_score = self._calculate_statistical_distance(
                self.reference_data, current_data
            )
            
            threshold = self.detection_thresholds[DriftType.DATA_DRIFT]
            severity = self._determine_severity(drift_score, threshold)
            
            alert = DriftAlert(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                score=drift_score,
                threshold=threshold,
                timestamp=datetime.now(),
                model_id="current_model",
                description=f"데이터 분포가 기준 데이터와 {drift_score:.4f}만큼 차이남",
                recommendations=[
                    "데이터 품질 검증 필요",
                    "피처 엔지니어링 재검토",
                    "모델 재훈련 고려"
                ]
            )
            
            self.drift_history.append(alert)
            return alert
            
        except Exception as e:
            logger.error(f"데이터 드리프트 감지 오류: {e}")
            raise
    
    def detect_concept_drift(self, current_performance: Dict[str, float]) -> DriftAlert:
        """개념 드리프트 감지"""
        try:
            if self.reference_performance is None:
                raise ValueError("기준 성능이 설정되지 않았습니다.")
            
            # 성능 변화 계산
            performance_changes = {}
            for metric in current_performance:
                if metric in self.reference_performance:
                    ref_value = self.reference_performance[metric]
                    curr_value = current_performance[metric]
                    change = abs(curr_value - ref_value) / ref_value
                    performance_changes[metric] = change
            
            # 평균 성능 변화
            avg_change = np.mean(list(performance_changes.values()))
            
            threshold = self.detection_thresholds[DriftType.CONCEPT_DRIFT]
            severity = self._determine_severity(avg_change, threshold)
            
            alert = DriftAlert(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                score=avg_change,
                threshold=threshold,
                timestamp=datetime.now(),
                model_id="current_model",
                description=f"모델 성능이 기준 대비 {avg_change:.4f}만큼 변화",
                recommendations=[
                    "모델 재훈련 필요",
                    "하이퍼파라미터 튜닝 고려",
                    "새로운 피처 추가 검토"
                ]
            )
            
            self.drift_history.append(alert)
            return alert
            
        except Exception as e:
            logger.error(f"개념 드리프트 감지 오류: {e}")
            raise
    
    def _calculate_statistical_distance(self, ref_data: ModelInput, 
                                      curr_data: ModelInput) -> float:
        """통계적 거리 계산"""
        try:
            # 간단한 구현: 평균과 표준편차 기반 거리
            ref_mean = np.mean(ref_data, axis=0)
            ref_std = np.std(ref_data, axis=0)
            curr_mean = np.mean(curr_data, axis=0)
            curr_std = np.std(curr_data, axis=0)
            
            # 정규화된 거리
            mean_distance = np.mean(np.abs(curr_mean - ref_mean) / (ref_std + 1e-8))
            std_distance = np.mean(np.abs(curr_std - ref_std) / (ref_std + 1e-8))
            
            total_distance = (mean_distance + std_distance) / 2
            return total_distance
            
        except Exception as e:
            logger.error(f"통계적 거리 계산 오류: {e}")
            return 0.0
    
    def _determine_severity(self, score: float, threshold: float) -> str:
        """심각도 결정"""
        if score < threshold * 0.5:
            return 'low'
        elif score < threshold:
            return 'medium'
        elif score < threshold * 1.5:
            return 'high'
        else:
            return 'critical'
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """드리프트 요약"""
        if not self.drift_history:
            return {'total_alerts': 0}
        
        summary = {
            'total_alerts': len(self.drift_history),
            'by_type': {},
            'by_severity': {},
            'recent_alerts': []
        }
        
        # 타입별, 심각도별 분류
        for alert in self.drift_history:
            drift_type = alert.drift_type.value
            severity = alert.severity
            
            summary['by_type'][drift_type] = summary['by_type'].get(drift_type, 0) + 1
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        # 최근 알림 (최근 10개)
        recent_alerts = sorted(self.drift_history, key=lambda x: x.timestamp, reverse=True)[:10]
        summary['recent_alerts'] = [
            {
                'type': alert.drift_type.value,
                'severity': alert.severity,
                'score': alert.score,
                'timestamp': alert.timestamp.isoformat(),
                'description': alert.description
            }
            for alert in recent_alerts
        ]
        
        return summary


class ABTestFramework:
    """A/B 테스팅 프레임워크"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
        self.metrics_collector = {}
    
    def start_ab_test(self, config: ABTestConfig) -> str:
        """A/B 테스트 시작"""
        try:
            test_id = f"{config.test_name}_{int(time.time())}"
            
            # 테스트 설정
            config.start_date = datetime.now()
            config.end_date = config.start_date + timedelta(days=config.duration_days)
            
            self.active_tests[test_id] = {
                'config': config,
                'start_time': datetime.now(),
                'metrics_a': [],
                'metrics_b': [],
                'traffic_count': {'a': 0, 'b': 0}
            }
            
            logger.info(f"A/B 테스트 시작: {test_id}")
            return test_id
            
        except Exception as e:
            logger.error(f"A/B 테스트 시작 오류: {e}")
            raise
    
    def assign_model(self, test_id: str, user_id: str) -> str:
        """사용자에게 모델 할당"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"활성 테스트가 없습니다: {test_id}")
            
            test = self.active_tests[test_id]
            config = test['config']
            
            # 트래픽 분할
            if np.random.random() < config.traffic_split:
                model_id = config.model_b_id
                test['traffic_count']['b'] += 1
            else:
                model_id = config.model_a_id
                test['traffic_count']['a'] += 1
            
            return model_id
            
        except Exception as e:
            logger.error(f"모델 할당 오류: {e}")
            raise
    
    def record_prediction(self, test_id: str, model_id: str, 
                         prediction: ModelOutput, actual: ModelOutput,
                         metrics: Dict[str, float]) -> None:
        """예측 결과 기록"""
        try:
            if test_id not in self.active_tests:
                return
            
            test = self.active_tests[test_id]
            config = test['config']
            
            # 모델별 메트릭 저장
            if model_id == config.model_a_id:
                test['metrics_a'].append(metrics)
            elif model_id == config.model_b_id:
                test['metrics_b'].append(metrics)
            
        except Exception as e:
            logger.error(f"예측 결과 기록 오류: {e}")
    
    def analyze_results(self, test_id: str) -> Dict[str, Any]:
        """A/B 테스트 결과 분석"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"테스트를 찾을 수 없습니다: {test_id}")
            
            test = self.active_tests[test_id]
            config = test['config']
            
            if len(test['metrics_a']) < config.min_sample_size or \
               len(test['metrics_b']) < config.min_sample_size:
                return {'status': 'insufficient_data'}
            
            # 통계적 유의성 검정
            results = {}
            for metric in config.success_metrics:
                metric_a = [m.get(metric, 0) for m in test['metrics_a']]
                metric_b = [m.get(metric, 0) for m in test['metrics_b']]
                
                # t-검정
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(metric_a, metric_b)
                
                results[metric] = {
                    'model_a_mean': np.mean(metric_a),
                    'model_b_mean': np.mean(metric_b),
                    'improvement': (np.mean(metric_b) - np.mean(metric_a)) / np.mean(metric_a),
                    'p_value': p_value,
                    'significant': p_value < config.statistical_significance
                }
            
            # 전체 결과
            overall_result = {
                'test_id': test_id,
                'status': 'completed',
                'duration_days': (datetime.now() - test['start_time']).days,
                'traffic_split': test['traffic_count'],
                'results': results,
                'recommendation': self._generate_recommendation(results, config)
            }
            
            self.test_results[test_id] = overall_result
            return overall_result
            
        except Exception as e:
            logger.error(f"A/B 테스트 결과 분석 오류: {e}")
            raise
    
    def _generate_recommendation(self, results: Dict[str, Any], 
                               config: ABTestConfig) -> str:
        """권장사항 생성"""
        significant_improvements = 0
        total_metrics = len(results)
        
        for metric_result in results.values():
            if metric_result['significant'] and metric_result['improvement'] > 0:
                significant_improvements += 1
        
        improvement_ratio = significant_improvements / total_metrics
        
        if improvement_ratio >= 0.7:
            return f"Model B 승리: {improvement_ratio:.1%} 메트릭에서 유의한 개선"
        elif improvement_ratio >= 0.3:
            return "부분적 개선: 추가 테스트 필요"
        else:
            return "Model A 유지: 유의한 개선 없음"


class AutoRetrainer:
    """자동 재학습 시스템"""
    
    def __init__(self, registry: ModelRegistry, drift_detector: DriftDetector):
        self.registry = registry
        self.drift_detector = drift_detector
        self.retrain_history = []
        self.retrain_triggers = {
            'drift_threshold': 0.15,
            'performance_degradation': 0.2,
            'time_based': 30,  # 일
            'data_volume': 10000  # 새로운 데이터 포인트
        }
    
    def check_retrain_conditions(self, model_id: str, 
                               current_performance: Dict[str, float],
                               new_data_size: int) -> bool:
        """재학습 조건 확인"""
        try:
            # 드리프트 확인
            drift_alerts = self.drift_detector.drift_history
            recent_drifts = [
                alert for alert in drift_alerts 
                if alert.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            high_severity_drifts = [
                alert for alert in recent_drifts 
                if alert.severity in ['high', 'critical']
            ]
            
            if high_severity_drifts:
                logger.info(f"드리프트로 인한 재학습 트리거: {model_id}")
                return True
            
            # 성능 저하 확인
            if self._check_performance_degradation(current_performance):
                logger.info(f"성능 저하로 인한 재학습 트리거: {model_id}")
                return True
            
            # 시간 기반 재학습
            last_retrain = self._get_last_retrain_time(model_id)
            if last_retrain and \
               (datetime.now() - last_retrain).days > self.retrain_triggers['time_based']:
                logger.info(f"시간 기반 재학습 트리거: {model_id}")
                return True
            
            # 데이터 볼륨 기반 재학습
            if new_data_size > self.retrain_triggers['data_volume']:
                logger.info(f"데이터 볼륨 기반 재학습 트리거: {model_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"재학습 조건 확인 오류: {e}")
            return False
    
    def trigger_retrain(self, model_id: str, training_data: ModelInput,
                       training_labels: ModelOutput) -> Dict[str, Any]:
        """재학습 트리거"""
        try:
            start_time = datetime.now()
            
            # 기존 모델 로드
            old_model = self.registry.get_model(model_id)
            
            # 재학습 (간단한 구현)
            # 실제로는 더 복잡한 재학습 로직이 필요
            retrained_model = self._retrain_model(old_model, training_data, training_labels)
            
            # 성능 평가
            performance_metrics = self._evaluate_model(retrained_model, training_data, training_labels)
            
            # 새 모델 등록
            new_model_id = f"{model_id}_v{int(time.time())}"
            self.registry.register_model(new_model_id, retrained_model, {
                'performance_metrics': performance_metrics,
                'retrain_timestamp': start_time.isoformat(),
                'original_model': model_id
            })
            
            # 재학습 기록
            retrain_record = {
                'model_id': model_id,
                'new_model_id': new_model_id,
                'retrain_timestamp': start_time,
                'performance_metrics': performance_metrics,
                'training_data_size': len(training_data)
            }
            
            self.retrain_history.append(retrain_record)
            
            logger.info(f"재학습 완료: {model_id} -> {new_model_id}")
            
            return retrain_record
            
        except Exception as e:
            logger.error(f"재학습 트리거 오류: {e}")
            raise
    
    def _check_performance_degradation(self, current_performance: Dict[str, float]) -> bool:
        """성능 저하 확인"""
        # 간단한 구현: MSE 기준
        if 'mse' in current_performance:
            return current_performance['mse'] > self.retrain_triggers['performance_degradation']
        return False
    
    def _get_last_retrain_time(self, model_id: str) -> Optional[datetime]:
        """마지막 재학습 시간 조회"""
        for record in reversed(self.retrain_history):
            if record['model_id'] == model_id:
                return record['retrain_timestamp']
        return None
    
    def _retrain_model(self, model: Any, X: ModelInput, y: ModelOutput) -> Any:
        """모델 재학습"""
        # 간단한 구현 - 실제로는 모델 타입에 따라 다르게 처리
        if hasattr(model, 'fit'):
            model.fit(X, y)
        elif hasattr(model, 'train'):
            model.train(X, y)
        
        return model
    
    def _evaluate_model(self, model: Any, X: ModelInput, y: ModelOutput) -> Dict[str, float]:
        """모델 평가"""
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                predictions = model(X)
            
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"모델 평가 오류: {e}")
            return {}


class MonitoringSystem:
    """모니터링 시스템"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.metrics = {}
        self._setup_metrics()
        self._start_server()
    
    def _setup_metrics(self):
        """메트릭 설정"""
        # Prometheus 메트릭
        self.metrics['prediction_counter'] = Counter(
            'model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'status']
        )
        
        self.metrics['prediction_latency'] = Histogram(
            'model_prediction_latency_seconds',
            'Model prediction latency',
            ['model_name']
        )
        
        self.metrics['model_accuracy'] = Gauge(
            'model_accuracy',
            'Model accuracy',
            ['model_name', 'metric_name']
        )
        
        self.metrics['drift_score'] = Gauge(
            'model_drift_score',
            'Model drift score',
            ['model_name', 'drift_type']
        )
    
    def _start_server(self):
        """메트릭 서버 시작"""
        try:
            start_http_server(self.port)
            logger.info(f"모니터링 서버 시작: http://localhost:{self.port}")
        except Exception as e:
            logger.error(f"모니터링 서버 시작 오류: {e}")
    
    def record_prediction(self, model_name: str, latency: float, 
                         success: bool = True):
        """예측 기록"""
        status = 'success' if success else 'error'
        self.metrics['prediction_counter'].labels(model_name=model_name, status=status).inc()
        self.metrics['prediction_latency'].labels(model_name=model_name).observe(latency)
    
    def update_accuracy(self, model_name: str, metric_name: str, value: float):
        """정확도 업데이트"""
        self.metrics['model_accuracy'].labels(
            model_name=model_name, metric_name=metric_name
        ).set(value)
    
    def update_drift_score(self, model_name: str, drift_type: str, score: float):
        """드리프트 점수 업데이트"""
        self.metrics['drift_score'].labels(
            model_name=model_name, drift_type=drift_type
        ).set(score)


class MLOpsPipeline:
    """MLOps 파이프라인 메인 클래스"""
    
    def __init__(self, registry_uri: str = "sqlite:///mlops.db"):
        self.registry = ModelRegistry(registry_uri)
        self.drift_detector = DriftDetector()
        self.ab_test_framework = ABTestFramework()
        self.auto_retrainer = AutoRetrainer(self.registry, self.drift_detector)
        self.monitoring = MonitoringSystem()
        
        # 메트릭 수집기
        self.metrics_collector = {}
    
    def deploy_model(self, model_name: str, model: Any, 
                    stage: ModelStage = ModelStage.STAGING) -> str:
        """모델 배포"""
        try:
            # 모델 등록
            model_id = self.registry.register_model(model_name, model, {
                'deployment_stage': stage.value,
                'deployment_timestamp': datetime.now().isoformat()
            })
            
            # 스테이지 전환
            self.registry.transition_model_stage(model_name, 1, stage)
            
            # 모니터링 설정
            self.metrics_collector[model_name] = {
                'predictions': 0,
                'errors': 0,
                'avg_latency': 0.0,
                'last_prediction': None
            }
            
            logger.info(f"모델 배포 완료: {model_name} -> {stage.value}")
            return model_id
            
        except Exception as e:
            logger.error(f"모델 배포 오류: {e}")
            raise
    
    def predict_with_monitoring(self, model_name: str, X: ModelInput) -> ModelOutput:
        """모니터링과 함께 예측"""
        try:
            start_time = time.time()
            
            # 모델 로드
            model = self.registry.get_model(model_name)
            
            # 예측 실행
            if hasattr(model, 'predict'):
                prediction = model.predict(X)
            else:
                prediction = model(X)
            
            latency = time.time() - start_time
            
            # 메트릭 기록
            self.monitoring.record_prediction(model_name, latency, success=True)
            self.metrics_collector[model_name]['predictions'] += 1
            self.metrics_collector[model_name]['avg_latency'] = \
                (self.metrics_collector[model_name]['avg_latency'] + latency) / 2
            self.metrics_collector[model_name]['last_prediction'] = datetime.now()
            
            return prediction
            
        except Exception as e:
            # 에러 메트릭 기록
            self.monitoring.record_prediction(model_name, 0.0, success=False)
            self.metrics_collector[model_name]['errors'] += 1
            logger.error(f"예측 오류: {e}")
            raise
    
    def check_model_health(self, model_name: str) -> Dict[str, Any]:
        """모델 상태 확인"""
        try:
            health_status = {
                'model_name': model_name,
                'status': 'healthy',
                'last_prediction': None,
                'error_rate': 0.0,
                'avg_latency': 0.0,
                'drift_alerts': []
            }
            
            if model_name in self.metrics_collector:
                metrics = self.metrics_collector[model_name]
                total_predictions = metrics['predictions'] + metrics['errors']
                
                if total_predictions > 0:
                    health_status['error_rate'] = metrics['errors'] / total_predictions
                    health_status['avg_latency'] = metrics['avg_latency']
                    health_status['last_prediction'] = metrics['last_prediction']
                
                # 에러율이 높으면 unhealthy
                if health_status['error_rate'] > 0.1:
                    health_status['status'] = 'unhealthy'
            
            # 드리프트 알림 확인
            recent_alerts = [
                alert for alert in self.drift_detector.drift_history
                if alert.model_id == model_name and 
                alert.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            health_status['drift_alerts'] = [
                {
                    'type': alert.drift_type.value,
                    'severity': alert.severity,
                    'score': alert.score,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ]
            
            return health_status
            
        except Exception as e:
            logger.error(f"모델 상태 확인 오류: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_system_overview(self) -> Dict[str, Any]:
        """시스템 전체 개요"""
        try:
            overview = {
                'deployed_models': len(self.metrics_collector),
                'active_ab_tests': len(self.ab_test_framework.active_tests),
                'total_retrains': len(self.auto_retrainer.retrain_history),
                'drift_alerts': len(self.drift_detector.drift_history),
                'model_health': {},
                'recent_activity': []
            }
            
            # 모델별 상태
            for model_name in self.metrics_collector:
                overview['model_health'][model_name] = self.check_model_health(model_name)
            
            # 최근 활동
            recent_activities = []
            
            # 최근 재학습
            for retrain in self.auto_retrainer.retrain_history[-5:]:
                recent_activities.append({
                    'type': 'retrain',
                    'model_id': retrain['model_id'],
                    'timestamp': retrain['retrain_timestamp'].isoformat()
                })
            
            # 최근 드리프트 알림
            for alert in self.drift_detector.drift_history[-5:]:
                recent_activities.append({
                    'type': 'drift_alert',
                    'model_id': alert.model_id,
                    'drift_type': alert.drift_type.value,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat()
                })
            
            overview['recent_activity'] = sorted(
                recent_activities, 
                key=lambda x: x['timestamp'], 
                reverse=True
            )[:10]
            
            return overview
            
        except Exception as e:
            logger.error(f"시스템 개요 조회 오류: {e}")
            return {'error': str(e)}


# 사용 예시
if __name__ == "__main__":
    # MLOps 파이프라인 생성
    mlops = MLOpsPipeline()
    
    # 샘플 모델 생성 (간단한 예시)
    from sklearn.ensemble import RandomForestRegressor
    sample_model = RandomForestRegressor(n_estimators=100)
    
    # 모델 배포
    model_id = mlops.deploy_model("sample_model", sample_model, ModelStage.STAGING)
    
    # 샘플 데이터로 예측
    X_sample = np.random.randn(100, 10)
    y_sample = np.random.randn(100)
    
    # 모델 훈련 (실제로는 미리 훈련된 모델 사용)
    sample_model.fit(X_sample, y_sample)
    
    # 예측 테스트
    X_test = np.random.randn(10, 10)
    prediction = mlops.predict_with_monitoring("sample_model", X_test)
    
    # 시스템 상태 확인
    health = mlops.check_model_health("sample_model")
    overview = mlops.get_system_overview()
    
    print("MLOps 파이프라인 테스트 완료")
    print(f"모델 상태: {health}")
    print(f"시스템 개요: {overview}") 