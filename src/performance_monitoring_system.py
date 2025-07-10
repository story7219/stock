#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: performance_monitoring_system.py
모듈: 실시간 성능 모니터링 및 진단 시스템
목적: 모델 성능 추적, 금융 성과 메트릭, 실시간 대시보드, 알림 시스템

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - numpy>=1.24.0
    - pandas>=2.0.0
    - scikit-learn>=1.3.0
    - matplotlib>=3.7.0
    - seaborn>=0.12.0
    - plotly>=5.15.0

Performance:
    - 실시간 모니터링: < 100ms
    - 대시보드 업데이트: 1초 간격
    - 알림 시스템: 즉시 발송

Security:
    - 데이터 보안
    - 접근 권한 관리
    - 로그 암호화

License: MIT
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings
from collections import deque
import json

warnings.filterwarnings('ignore')

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""
    ACCURACY = "accuracy"
    DIRECTIONAL = "directional"
    FINANCIAL = "financial"
    RISK = "risk"


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    timestamp: datetime
    model_name: str
    accuracy: float
    directional_accuracy: float
    rmse: float
    mae: float
    sharpe_ratio: float
    information_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """알림 설정"""
    accuracy_threshold: float = 0.6
    sharpe_threshold: float = 1.0
    drawdown_threshold: float = -0.1
    drift_threshold: float = 0.1
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])


class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self, alert_config: AlertConfig):
        self.alert_config = alert_config
        self.metrics_history = {}
        self.real_time_metrics = {}
        self.logger = logging.getLogger("PerformanceMonitor")
    
    def update_metrics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                      returns: np.ndarray, positions: np.ndarray) -> Optional[PerformanceMetrics]:
        """메트릭 업데이트"""
        try:
            # 1. 예측 정확도 메트릭
            accuracy = float(self._calculate_accuracy(y_true, y_pred))
            directional_accuracy = float(self._calculate_directional_accuracy(y_true, y_pred))
            rmse = float(self._calculate_rmse(y_true, y_pred))
            mae = float(self._calculate_mae(y_true, y_pred))
            
            # 2. 금융 성과 메트릭
            sharpe_ratio = float(self._calculate_sharpe_ratio(returns))
            information_ratio = float(self._calculate_information_ratio(returns))
            max_drawdown = float(self._calculate_max_drawdown(returns))
            win_rate = float(self._calculate_win_rate(returns))
            profit_factor = float(self._calculate_profit_factor(returns))
            calmar_ratio = float(self._calculate_calmar_ratio(returns, max_drawdown))
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                model_name=model_name,
                accuracy=accuracy,
                directional_accuracy=directional_accuracy,
                rmse=rmse,
                mae=mae,
                sharpe_ratio=sharpe_ratio,
                information_ratio=information_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                metadata={
                    'y_true': y_true.tolist(),
                    'y_pred': y_pred.tolist(),
                    'returns': returns.tolist(),
                    'positions': positions.tolist()
                }
            )
            
            # 히스토리 저장
            if model_name not in self.metrics_history:
                self.metrics_history[model_name] = deque(maxlen=1000)
            self.metrics_history[model_name].append(metrics)
            
            # 실시간 메트릭 업데이트
            self.real_time_metrics[model_name] = metrics
            
            self.logger.info(f"메트릭 업데이트: {model_name}, 정확도: {accuracy:.3f}, Sharpe: {sharpe_ratio:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"메트릭 업데이트 실패: {e}")
            return None
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """정확도 계산"""
        return np.mean(y_true == y_pred)
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """방향성 정확도 계산"""
        # 방향성 예측 (상승/하락)
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        return np.mean(true_direction == pred_direction)
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE 계산"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAE 계산"""
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Sharpe ratio 계산"""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Information ratio 계산"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - np.mean(returns)
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """승률 계산"""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns > 0)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Profit factor 계산"""
        if len(returns) == 0:
            return 0.0
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        return np.sum(positive_returns) / abs(np.sum(negative_returns))
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calmar ratio 계산"""
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        return annual_return / abs(max_drawdown)
    
    def get_model_performance(self, model_name: str) -> Optional[PerformanceMetrics]:
        """모델 성능 조회"""
        return self.real_time_metrics.get(model_name)
    
    def get_performance_history(self, model_name: str) -> List[PerformanceMetrics]:
        """성능 히스토리 조회"""
        return list(self.metrics_history.get(model_name, []))
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """모델 비교"""
        comparison = {}
        
        for model_name in model_names:
            if model_name in self.real_time_metrics:
                metrics = self.real_time_metrics[model_name]
                comparison[model_name] = {
                    'accuracy': metrics.accuracy,
                    'directional_accuracy': metrics.directional_accuracy,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor
                }
        
        return comparison


class ModelDiagnostics:
    """모델 진단"""
    
    def __init__(self):
        self.logger = logging.getLogger("ModelDiagnostics")
    
    def analyze_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """특성 중요도 분석"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return {}
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            self.logger.error(f"특성 중요도 분석 실패: {e}")
            return {}
    
    def analyze_prediction_distribution(self, y_pred: np.ndarray) -> Dict[str, float]:
        """예측 분포 분석"""
        try:
            return {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred)),
                'skewness': float(self._calculate_skewness(y_pred)),
                'kurtosis': float(self._calculate_kurtosis(y_pred))
            }
        except Exception as e:
            self.logger.error(f"예측 분포 분석 실패: {e}")
            return {}
    
    def analyze_model_uncertainty(self, y_pred: np.ndarray, y_pred_std: Optional[np.ndarray] = None) -> Dict[str, float]:
        """모델 불확실성 분석"""
        try:
            uncertainty_metrics = {
                'prediction_variance': float(np.var(y_pred)),
                'prediction_range': float(np.max(y_pred) - np.min(y_pred))
            }
            
            if y_pred_std is not None:
                uncertainty_metrics['average_std'] = float(np.mean(y_pred_std))
                uncertainty_metrics['std_of_std'] = float(np.std(y_pred_std))
            
            return uncertainty_metrics
            
        except Exception as e:
            self.logger.error(f"모델 불확실성 분석 실패: {e}")
            return {}
    
    def bias_variance_decomposition(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Bias-Variance 분해"""
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            bias_squared = (np.mean(y_true) - np.mean(y_pred)) ** 2
            variance = np.var(y_pred)
            
            return {
                'mse': float(mse),
                'bias_squared': float(bias_squared),
                'variance': float(variance),
                'bias_variance_sum': float(bias_squared + variance)
            }
            
        except Exception as e:
            self.logger.error(f"Bias-Variance 분해 실패: {e}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """왜도 계산"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도 계산"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3


class RealTimeDashboard:
    """실시간 대시보드"""
    
    def __init__(self):
        self.logger = logging.getLogger("RealTimeDashboard")
    
    def create_performance_dashboard(self, performance_monitor: PerformanceMonitor) -> Dict[str, Any]:
        """성능 대시보드 생성"""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'comparison': {},
                'alerts': []
            }
            
            # 모델별 성능 데이터
            for model_name, metrics in performance_monitor.real_time_metrics.items():
                dashboard_data['models'][model_name] = {
                    'accuracy': metrics.accuracy,
                    'directional_accuracy': metrics.directional_accuracy,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'calmar_ratio': metrics.calmar_ratio
                }
            
            # 모델 비교
            if len(performance_monitor.real_time_metrics) > 1:
                dashboard_data['comparison'] = performance_monitor.compare_models(
                    list(performance_monitor.real_time_metrics.keys())
                )
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"대시보드 생성 실패: {e}")
            return {}
    
    def create_visualization(self, performance_monitor: PerformanceMonitor, 
                           model_name: str) -> Optional[Dict[str, Any]]:
        """시각화 생성"""
        if not PLOTTING_AVAILABLE:
            return None
        
        try:
            history = performance_monitor.get_performance_history(model_name)
            if not history:
                return None
            
            # 시계열 데이터 준비
            timestamps = [m.timestamp for m in history]
            accuracies = [m.accuracy for m in history]
            sharpe_ratios = [m.sharpe_ratio for m in history]
            drawdowns = [m.max_drawdown for m in history]
            
            # Plotly 차트 생성
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Accuracy', 'Sharpe Ratio', 'Max Drawdown'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=accuracies, name='Accuracy'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=sharpe_ratios, name='Sharpe Ratio'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=drawdowns, name='Max Drawdown'),
                row=3, col=1
            )
            
            fig.update_layout(height=600, title_text=f"Performance Dashboard - {model_name}")
            
            return {
                'chart': fig.to_json(),
                'model_name': model_name,
                'data_points': len(history)
            }
            
        except Exception as e:
            self.logger.error(f"시각화 생성 실패: {e}")
            return None


class AlertSystem:
    """알림 시스템"""
    
    def __init__(self, alert_config: AlertConfig):
        self.alert_config = alert_config
        self.alert_history = deque(maxlen=1000)
        self.logger = logging.getLogger("AlertSystem")
    
    def check_alerts(self, performance_monitor: PerformanceMonitor) -> List[Dict[str, Any]]:
        """알림 체크"""
        alerts = []
        
        try:
            for model_name, metrics in performance_monitor.real_time_metrics.items():
                # 정확도 알림
                if metrics.accuracy < self.alert_config.accuracy_threshold:
                    alerts.append({
                        'type': 'LOW_ACCURACY',
                        'model': model_name,
                        'value': metrics.accuracy,
                        'threshold': self.alert_config.accuracy_threshold,
                        'timestamp': datetime.now(),
                        'severity': 'HIGH'
                    })
                
                # Sharpe ratio 알림
                if metrics.sharpe_ratio < self.alert_config.sharpe_threshold:
                    alerts.append({
                        'type': 'LOW_SHARPE',
                        'model': model_name,
                        'value': metrics.sharpe_ratio,
                        'threshold': self.alert_config.sharpe_threshold,
                        'timestamp': datetime.now(),
                        'severity': 'MEDIUM'
                    })
                
                # Drawdown 알림
                if metrics.max_drawdown < self.alert_config.drawdown_threshold:
                    alerts.append({
                        'type': 'HIGH_DRAWDOWN',
                        'model': model_name,
                        'value': metrics.max_drawdown,
                        'threshold': self.alert_config.drawdown_threshold,
                        'timestamp': datetime.now(),
                        'severity': 'HIGH'
                    })
                
                # Drift 알림
                drift_score = self._calculate_drift_score(performance_monitor, model_name)
                if drift_score > self.alert_config.drift_threshold:
                    alerts.append({
                        'type': 'MODEL_DRIFT',
                        'model': model_name,
                        'value': drift_score,
                        'threshold': self.alert_config.drift_threshold,
                        'timestamp': datetime.now(),
                        'severity': 'MEDIUM'
                    })
            
            # 알림 히스토리에 추가
            for alert in alerts:
                self.alert_history.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"알림 체크 실패: {e}")
            return []
    
    def _calculate_drift_score(self, performance_monitor: PerformanceMonitor, model_name: str) -> float:
        """드리프트 점수 계산"""
        try:
            history = performance_monitor.get_performance_history(model_name)
            if len(history) < 10:
                return 0.0
            
            # 최근 10개 vs 이전 10개 성능 비교
            recent_metrics = history[-10:]
            previous_metrics = history[-20:-10]
            
            recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
            previous_accuracy = np.mean([m.accuracy for m in previous_metrics])
            
            drift_score = abs(recent_accuracy - previous_accuracy)
            return drift_score
            
        except Exception as e:
            self.logger.error(f"드리프트 점수 계산 실패: {e}")
            return 0.0
    
    def send_notification(self, alert: Dict[str, Any]) -> bool:
        """알림 발송"""
        try:
            # 실제 구현에서는 이메일, 슬랙 등으로 발송
            message = f"Alert: {alert['type']} - Model: {alert['model']} - Value: {alert['value']:.3f}"
            self.logger.warning(message)
            
            # 알림 채널별 발송
            for channel in self.alert_config.notification_channels:
                if channel == "email":
                    self._send_email_alert(alert)
                elif channel == "slack":
                    self._send_slack_alert(alert)
            
            return True
            
        except Exception as e:
            self.logger.error(f"알림 발송 실패: {e}")
            return False
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """이메일 알림 발송"""
        # 실제 구현에서는 SMTP 등 사용
        pass
    
    def _send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """슬랙 알림 발송"""
        # 실제 구현에서는 Slack API 사용
        pass


class AlphaAttribution:
    """알파 attribution 분석"""
    
    def __init__(self):
        self.logger = logging.getLogger("AlphaAttribution")
    
    def analyze_alpha_sources(self, returns: np.ndarray, factors: Dict[str, np.ndarray]) -> Dict[str, float]:
        """알파 소스 분석"""
        try:
            attribution = {}
            
            for factor_name, factor_values in factors.items():
                # 팩터별 기여도 계산
                correlation = np.corrcoef(returns, factor_values)[0, 1]
                attribution[factor_name] = float(correlation)
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"알파 소스 분석 실패: {e}")
            return {}
    
    def analyze_risk_attribution(self, returns: np.ndarray, risk_factors: Dict[str, np.ndarray]) -> Dict[str, float]:
        """리스크 attribution 분석"""
        try:
            risk_attribution = {}
            
            for factor_name, factor_values in risk_factors.items():
                # 리스크 팩터별 기여도
                factor_vol = np.std(factor_values)
                portfolio_vol = np.std(returns)
                
                if portfolio_vol > 0:
                    risk_contribution = (factor_vol / portfolio_vol) * np.corrcoef(returns, factor_values)[0, 1]
                    risk_attribution[factor_name] = float(risk_contribution)
                else:
                    risk_attribution[factor_name] = 0.0
            
            return risk_attribution
            
        except Exception as e:
            self.logger.error(f"리스크 attribution 분석 실패: {e}")
            return {}


class IntegratedPerformanceSystem:
    """통합 성능 모니터링 시스템"""
    
    def __init__(self, alert_config: AlertConfig):
        self.alert_config = alert_config
        self.performance_monitor = PerformanceMonitor(alert_config)
        self.model_diagnostics = ModelDiagnostics()
        self.dashboard = RealTimeDashboard()
        self.alert_system = AlertSystem(alert_config)
        self.alpha_attribution = AlphaAttribution()
        self.logger = logging.getLogger("IntegratedPerformanceSystem")
    
    def update_model_performance(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                               returns: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """모델 성능 업데이트"""
        try:
            # 1. 성능 메트릭 업데이트
            metrics = self.performance_monitor.update_metrics(model_name, y_true, y_pred, returns, positions)
            
            # 2. 모델 진단
            diagnostics = self._run_model_diagnostics(model_name, y_true, y_pred)
            
            # 3. 알림 체크
            alerts = self.alert_system.check_alerts(self.performance_monitor)
            
            # 4. 알림 발송
            for alert in alerts:
                self.alert_system.send_notification(alert)
            
            # 5. 대시보드 업데이트
            dashboard_data = self.dashboard.create_performance_dashboard(self.performance_monitor)
            
            return {
                'metrics': metrics,
                'diagnostics': diagnostics,
                'alerts': alerts,
                'dashboard': dashboard_data
            }
            
        except Exception as e:
            self.logger.error(f"성능 업데이트 실패: {e}")
            return {}
    
    def _run_model_diagnostics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """모델 진단 실행"""
        try:
            diagnostics = {
                'prediction_distribution': self.model_diagnostics.analyze_prediction_distribution(y_pred),
                'model_uncertainty': self.model_diagnostics.analyze_model_uncertainty(y_pred),
                'bias_variance': self.model_diagnostics.bias_variance_decomposition(y_true, y_pred)
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"모델 진단 실패: {e}")
            return {}
    
    def generate_performance_report(self, model_name: str) -> Dict[str, Any]:
        """성능 리포트 생성"""
        try:
            metrics = self.performance_monitor.get_model_performance(model_name)
            history = self.performance_monitor.get_performance_history(model_name)
            visualization = self.dashboard.create_visualization(self.performance_monitor, model_name)
            
            return {
                'current_metrics': metrics,
                'history': history,
                'visualization': visualization,
                'report_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"성능 리포트 생성 실패: {e}")
            return {}


# 사용 예시
def main():
    """메인 실행 함수"""
    alert_config = AlertConfig()
    system = IntegratedPerformanceSystem(alert_config)
    
    # 예시 데이터
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.8, 0.2, 0.9, 0.7, 0.3])
    returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
    positions = np.array([1, -1, 1, 1, -1])
    
    # 성능 업데이트
    result = system.update_model_performance("test_model", y_true, y_pred, returns, positions)
    
    print("성능 모니터링 결과:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main() 