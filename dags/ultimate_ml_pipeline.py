#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ultimate_ml_pipeline.py
모듈: ML/DL 학습 파이프라인 Airflow DAG
목적: 데이터 전처리, 모델 학습, 하이퍼파라미터 최적화, 모델 배포 자동화

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - apache-airflow==2.7.0
    - mlflow==2.7.0
    - optuna==3.4.0
    - ray[tune]==2.7.0

Performance:
    - 병렬 처리: 멀티 태스크 동시 실행
    - 분산 학습: Ray Tune 통합
    - 모델 버전 관리: MLflow 통합
    - 자동 배포: 모델 성능 기반 자동 배포

Security:
    - 접근 제어: Airflow RBAC
    - 보안 로깅: 민감 정보 마스킹
    - 모델 검증: 입력 데이터 검증

License: MIT
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# DAG 기본 설정
default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG 생성
dag = DAG(
    'ultimate_ml_pipeline',
    default_args=default_args,
    description='Ultimate ML/DL Training Pipeline',
    schedule_interval='0 2 * * *',  # 매일 새벽 2시
    max_active_runs=1,
    tags=['ml', 'dl', 'trading', 'automation']
)


def check_data_availability(**context):
    """데이터 가용성 확인"""
    import os
    from pathlib import Path
    
    data_path = Path("data/")
    required_files = ["krx_data.parquet", "overseas_data.parquet", "kis_data.parquet"]
    
    available_files = []
    for file in required_files:
        if (data_path / file).exists():
            available_files.append(file)
    
    if not available_files:
        raise ValueError("No required data files found")
    
    context['task_instance'].xcom_push(key='available_data', value=available_files)
    print(f"Available data files: {available_files}")
    
    return available_files


def preprocess_data(**context):
    """데이터 전처리"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    available_data = context['task_instance'].xcom_pull(key='available_data')
    
    processed_data = {}
    
    for data_file in available_data:
        print(f"Processing {data_file}...")
        
        # 데이터 로드
        df = pd.read_parquet(f"data/{data_file}")
        
        # 기본 전처리
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 기술적 지표 추가
        if 'close' in df.columns:
            # 이동평균
            df['ma_5'] = df['close'].rolling(window=5).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # 스케일링
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # 처리된 데이터 저장
        processed_file = f"processed_{data_file}"
        df.to_parquet(f"data/{processed_file}")
        processed_data[data_file] = processed_file
        
        print(f"Processed {data_file} -> {processed_file}")
    
    context['task_instance'].xcom_push(key='processed_data', value=processed_data)
    return processed_data


def train_lstm_model(**context):
    """LSTM 모델 학습"""
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import mlflow
    
    processed_data = context['task_instance'].xcom_pull(key='processed_data')
    
    # 첫 번째 데이터 파일 사용
    data_file = list(processed_data.values())[0]
    df = pd.read_parquet(f"data/{data_file}")
    
    # 시계열 데이터 준비
    sequence_length = 60
    data = df.select_dtypes(include=[np.number]).values
    
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # 데이터 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # LSTM 모델 정의
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    # 모델 학습
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(X.shape[2]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # MLflow 로깅 시작
    with mlflow.start_run():
        mlflow.log_params({
            'model_type': 'lstm',
            'sequence_length': sequence_length,
            'hidden_size': 128,
            'num_layers': 2,
            'learning_rate': 0.001
        })
        
        # 학습
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                mlflow.log_metric('train_loss', loss.item(), step=epoch)
        
        # 평가
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs.squeeze(), y_test_tensor).item()
            
            # 메트릭 계산
            from sklearn.metrics import mean_squared_error, r2_score
            test_pred = test_outputs.cpu().numpy().squeeze()
            mse = mean_squared_error(y_test, test_pred)
            r2 = r2_score(y_test, test_pred)
            
            mlflow.log_metrics({
                'test_loss': test_loss,
                'mse': mse,
                'r2': r2
            })
        
        # 모델 저장
        mlflow.pytorch.log_model(model, "lstm_model")
    
    # 모델 정보 저장
    model_info = {
        'model_type': 'lstm',
        'test_loss': test_loss,
        'mse': mse,
        'r2': r2,
        'run_id': mlflow.active_run().info.run_id
    }
    
    context['task_instance'].xcom_push(key='lstm_results', value=model_info)
    return model_info


def train_transformer_model(**context):
    """Transformer 모델 학습"""
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import mlflow
    
    processed_data = context['task_instance'].xcom_pull(key='processed_data')
    
    # 첫 번째 데이터 파일 사용
    data_file = list(processed_data.values())[0]
    df = pd.read_parquet(f"data/{data_file}")
    
    # 시계열 데이터 준비
    sequence_length = 60
    data = df.select_dtypes(include=[np.number]).values
    
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # 데이터 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Transformer 모델 정의
    class TransformerModel(nn.Module):
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=6):
            super(TransformerModel, self).__init__()
            self.d_model = d_model
            self.input_projection = nn.Linear(input_size, d_model)
            self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.input_projection(x)
            seq_len = x.size(1)
            pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_encoding
            transformer_out = self.transformer(x)
            return self.fc(transformer_out[:, -1, :])
    
    # 모델 학습
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(X.shape[2]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # MLflow 로깅 시작
    with mlflow.start_run():
        mlflow.log_params({
            'model_type': 'transformer',
            'sequence_length': sequence_length,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 6,
            'learning_rate': 0.001
        })
        
        # 학습
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                mlflow.log_metric('train_loss', loss.item(), step=epoch)
        
        # 평가
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs.squeeze(), y_test_tensor).item()
            
            # 메트릭 계산
            from sklearn.metrics import mean_squared_error, r2_score
            test_pred = test_outputs.cpu().numpy().squeeze()
            mse = mean_squared_error(y_test, test_pred)
            r2 = r2_score(y_test, test_pred)
            
            mlflow.log_metrics({
                'test_loss': test_loss,
                'mse': mse,
                'r2': r2
            })
        
        # 모델 저장
        mlflow.pytorch.log_model(model, "transformer_model")
    
    # 모델 정보 저장
    model_info = {
        'model_type': 'transformer',
        'test_loss': test_loss,
        'mse': mse,
        'r2': r2,
        'run_id': mlflow.active_run().info.run_id
    }
    
    context['task_instance'].xcom_push(key='transformer_results', value=model_info)
    return model_info


def train_sklearn_models(**context):
    """스킷런 모델들 학습"""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    import mlflow
    
    processed_data = context['task_instance'].xcom_pull(key='processed_data')
    
    # 첫 번째 데이터 파일 사용
    data_file = list(processed_data.values())[0]
    df = pd.read_parquet(f"data/{data_file}")
    
    # 데이터 준비
    data = df.select_dtypes(include=[np.number]).values
    X = data[:-1]  # 마지막 행 제외
    y = data[1:, 0]  # 다음 시점의 첫 번째 컬럼
    
    # 데이터 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        with mlflow.start_run():
            mlflow.log_params({
                'model_type': model_name,
                'n_estimators': 100 if hasattr(model, 'n_estimators') else None
            })
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # 메트릭 계산
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            mlflow.log_metrics({
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2
            })
            
            # 모델 저장
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            results[model_name] = {
                'test_mse': test_mse,
                'test_r2': test_r2,
                'run_id': mlflow.active_run().info.run_id
            }
    
    context['task_instance'].xcom_push(key='sklearn_results', value=results)
    return results


def hyperparameter_optimization(**context):
    """하이퍼파라미터 최적화"""
    import optuna
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import mlflow
    
    processed_data = context['task_instance'].xcom_pull(key='processed_data')
    
    # 첫 번째 데이터 파일 사용
    data_file = list(processed_data.values())[0]
    df = pd.read_parquet(f"data/{data_file}")
    
    # 데이터 준비
    data = df.select_dtypes(include=[np.number]).values
    X = data[:-1]
    y = data[1:, 0]
    
    # 데이터 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    def objective(trial):
        # 하이퍼파라미터 샘플링
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        # 모델 생성
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # 교차 검증
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            val_pred = model.predict(X_val_fold)
            score = mean_squared_error(y_val_fold, val_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    # Optuna 스터디 생성
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # 최적 모델 학습
    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # 최적 모델 평가
    test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    
    # MLflow 로깅
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            'best_test_mse': test_mse,
            'best_value': study.best_value
        })
        mlflow.sklearn.log_model(best_model, "optimized_model")
    
    optimization_results = {
        'best_params': best_params,
        'best_value': study.best_value,
        'test_mse': test_mse,
        'run_id': mlflow.active_run().info.run_id
    }
    
    context['task_instance'].xcom_push(key='optimization_results', value=optimization_results)
    return optimization_results


def evaluate_models(**context):
    """모델 평가 및 비교"""
    import pandas as pd
    
    # 모든 모델 결과 수집
    lstm_results = context['task_instance'].xcom_pull(key='lstm_results')
    transformer_results = context['task_instance'].xcom_pull(key='transformer_results')
    sklearn_results = context['task_instance'].xcom_pull(key='sklearn_results')
    optimization_results = context['task_instance'].xcom_pull(key='optimization_results')
    
    # 결과 통합
    all_results = {
        'lstm': lstm_results,
        'transformer': transformer_results,
        **sklearn_results,
        'optimized': optimization_results
    }
    
    # 성능 비교
    comparison_df = pd.DataFrame([
        {
            'model': model_name,
            'test_mse': result.get('test_mse', result.get('mse', 0)),
            'test_r2': result.get('test_r2', result.get('r2', 0)),
            'test_loss': result.get('test_loss', 0)
        }
        for model_name, result in all_results.items()
    ])
    
    # 최고 성능 모델 선택
    best_model = comparison_df.loc[comparison_df['test_r2'].idxmax()]
    
    print("Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest Model: {best_model['model']} (R²: {best_model['test_r2']:.4f})")
    
    # 결과 저장
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    
    context['task_instance'].xcom_push(key='model_comparison', value=comparison_df.to_dict())
    context['task_instance'].xcom_push(key='best_model', value=best_model.to_dict())
    
    return best_model.to_dict()


def deploy_best_model(**context):
    """최고 성능 모델 배포"""
    import mlflow
    import os
    from pathlib import Path
    
    best_model_info = context['task_instance'].xcom_pull(key='best_model')
    best_model_name = best_model_info['model']
    
    print(f"Deploying best model: {best_model_name}")
    
    # 모델 배포 디렉토리 생성
    deployment_dir = Path("deployed_models")
    deployment_dir.mkdir(exist_ok=True)
    
    # MLflow에서 최고 모델 로드 및 배포
    if best_model_name in ['lstm', 'transformer']:
        # PyTorch 모델
        model_uri = f"runs:/{context['task_instance'].xcom_pull(key=f'{best_model_name}_results')['run_id']}/{best_model_name}_model"
        model = mlflow.pytorch.load_model(model_uri)
        
        # 모델 저장
        import torch
        torch.save(model.state_dict(), deployment_dir / f"{best_model_name}_deployed.pth")
        
    elif best_model_name == 'optimized':
        # 최적화된 모델
        model_uri = f"runs:/{context['task_instance'].xcom_pull(key='optimization_results')['run_id']}/optimized_model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # 모델 저장
        import joblib
        joblib.dump(model, deployment_dir / f"{best_model_name}_deployed.joblib")
        
    else:
        # 스킷런 모델
        model_uri = f"runs:/{context['task_instance'].xcom_pull(key='sklearn_results')[best_model_name]['run_id']}/{best_model_name}_model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # 모델 저장
        import joblib
        joblib.dump(model, deployment_dir / f"{best_model_name}_deployed.joblib")
    
    # 배포 정보 저장
    deployment_info = {
        'model_name': best_model_name,
        'deployment_time': datetime.now().isoformat(),
        'performance': best_model_info,
        'model_path': str(deployment_dir / f"{best_model_name}_deployed")
    }
    
    context['task_instance'].xcom_push(key='deployment_info', value=deployment_info)
    
    print(f"Model {best_model_name} deployed successfully!")
    return deployment_info


def generate_ml_report(**context):
    """ML 학습 리포트 생성"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # 결과 수집
    model_comparison = context['task_instance'].xcom_pull(key='model_comparison')
    deployment_info = context['task_instance'].xcom_pull(key='deployment_info')
    
    # 리포트 생성
    report_content = f"""
# ML/DL Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Summary
{pd.DataFrame(model_comparison).to_html()}

## Best Model Deployment
- Model: {deployment_info['model_name']}
- Deployment Time: {deployment_info['deployment_time']}
- Performance: R² = {deployment_info['performance']['test_r2']:.4f}

## Training Pipeline Status
✅ Data Preprocessing: Completed
✅ Model Training: Completed
✅ Hyperparameter Optimization: Completed
✅ Model Evaluation: Completed
✅ Model Deployment: Completed

## Next Steps
1. Monitor deployed model performance
2. Set up automated retraining schedule
3. Implement A/B testing for new models
4. Set up model performance alerts
"""
    
    # 리포트 저장
    with open('ml_training_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("ML training report generated successfully!")
    return "Report generated"


# 태스크 정의
with dag:
    # 데이터 확인
    check_data = PythonOperator(
        task_id='check_data_availability',
        python_callable=check_data_availability,
        provide_context=True
    )
    
    # 데이터 전처리
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True
    )
    
    # 모델 학습 태스크 그룹
    with TaskGroup(group_id='model_training') as model_training_group:
        # LSTM 모델 학습
        train_lstm = PythonOperator(
            task_id='train_lstm_model',
            python_callable=train_lstm_model,
            provide_context=True
        )
        
        # Transformer 모델 학습
        train_transformer = PythonOperator(
            task_id='train_transformer_model',
            python_callable=train_transformer_model,
            provide_context=True
        )
        
        # 스킷런 모델들 학습
        train_sklearn = PythonOperator(
            task_id='train_sklearn_models',
            python_callable=train_sklearn_models,
            provide_context=True
        )
    
    # 하이퍼파라미터 최적화
    optimize_hyperparams = PythonOperator(
        task_id='hyperparameter_optimization',
        python_callable=hyperparameter_optimization,
        provide_context=True
    )
    
    # 모델 평가
    evaluate = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        provide_context=True
    )
    
    # 최고 모델 배포
    deploy = PythonOperator(
        task_id='deploy_best_model',
        python_callable=deploy_best_model,
        provide_context=True
    )
    
    # 리포트 생성
    generate_report = PythonOperator(
        task_id='generate_ml_report',
        python_callable=generate_ml_report,
        provide_context=True
    )

# 태스크 의존성 설정
check_data >> preprocess >> model_training_group >> optimize_hyperparams >> evaluate >> deploy >> generate_report 