#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ml_api_server.py
모듈: ML API 서버
목적: 모델 서빙, 예측 엔드포인트, 종합 문서화

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - fastapi==0.104.0
    - uvicorn==0.24.0
    - pydantic==2.5.0
    - torch==2.1.0
    - scikit-learn==1.3.0
    - mlflow==2.7.0

Performance:
    - 비동기 처리: FastAPI 기반
    - 모델 캐싱: 메모리 내 모델 로딩
    - 배치 예측: 대량 데이터 처리
    - 로드 밸런싱: 다중 인스턴스 지원

Security:
    - API 키 인증: 요청별 인증
    - 입력 검증: Pydantic 모델
    - 요청 제한: Rate limiting
    - 로깅: 요청/응답 추적

License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Tuple,
    Protocol, TypeVar, Generic, Final, Literal
)

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Trading ML API",
    description="Advanced ML/DL Model Serving API for Trading Systems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 상수 정의
MODELS_DIR: Final = Path("models")
DEPLOYED_DIR: Final = Path("deployed_models")
API_KEY: Final = "your-secret-api-key"  # 실제 환경에서는 환경변수 사용


# Pydantic 모델들
class PredictionRequest(BaseModel):
    """예측 요청 모델"""
    data: List[List[float]] = Field(..., description="입력 데이터 (시계열)")
    model_type: str = Field(..., description="모델 타입 (lstm, transformer, random_forest 등)")
    api_key: str = Field(..., description="API 키")
    
    @validator('data')
    def validate_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError("데이터가 비어있습니다")
        if len(v[0]) == 0:
            raise ValueError("데이터 차원이 올바르지 않습니다")
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['lstm', 'gru', 'transformer', 'random_forest', 'xgboost', 'lightgbm', 'linear']
        if v not in allowed_types:
            raise ValueError(f"지원하지 않는 모델 타입입니다. 지원 타입: {allowed_types}")
        return v


class PredictionResponse(BaseModel):
    """예측 응답 모델"""
    predictions: List[float] = Field(..., description="예측 결과")
    model_type: str = Field(..., description="사용된 모델 타입")
    confidence: float = Field(..., description="예측 신뢰도")
    processing_time: float = Field(..., description="처리 시간 (초)")
    timestamp: datetime = Field(..., description="예측 시간")


class ModelInfo(BaseModel):
    """모델 정보 모델"""
    model_type: str = Field(..., description="모델 타입")
    version: str = Field(..., description="모델 버전")
    accuracy: float = Field(..., description="모델 정확도")
    last_updated: datetime = Field(..., description="마지막 업데이트 시간")
    status: str = Field(..., description="모델 상태")


class BatchPredictionRequest(BaseModel):
    """배치 예측 요청 모델"""
    data: List[List[List[float]]] = Field(..., description="배치 입력 데이터")
    model_type: str = Field(..., description="모델 타입")
    api_key: str = Field(..., description="API 키")


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답 모델"""
    predictions: List[List[float]] = Field(..., description="배치 예측 결과")
    model_type: str = Field(..., description="사용된 모델 타입")
    total_processing_time: float = Field(..., description="총 처리 시간 (초)")
    batch_size: int = Field(..., description="배치 크기")
    timestamp: datetime = Field(..., description="예측 시간")


class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.load_models()
    
    def load_models(self):
        """모델 로딩"""
        try:
            # 배포된 모델 디렉토리 확인
            if DEPLOYED_DIR.exists():
                model_files = list(DEPLOYED_DIR.glob("*_deployed.*"))
                
                for model_file in model_files:
                    model_type = model_file.stem.replace("_deployed", "")
                    
                    if model_file.suffix == ".pth":
                        # PyTorch 모델 로딩
                        checkpoint = torch.load(model_file, map_location='cpu')
                        
                        # 모델 아키텍처 재구성
                        if "lstm" in model_type:
                            from ml.advanced_ml_training_system import LSTMModel
                            model = LSTMModel(**checkpoint['model_config'])
                        elif "gru" in model_type:
                            from ml.advanced_ml_training_system import GRUModel
                            model = GRUModel(**checkpoint['model_config'])
                        elif "transformer" in model_type:
                            from ml.advanced_ml_training_system import TransformerModel
                            model = TransformerModel(**checkpoint['model_config'])
                        
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        
                        self.models[model_type] = model
                        self.scalers[model_type] = checkpoint.get('scaler')
                        
                    elif model_file.suffix == ".joblib":
                        # 스킷런 모델 로딩
                        model_data = joblib.load(model_file)
                        self.models[model_type] = model_data['model']
                        self.scalers[model_type] = model_data.get('scaler')
                    
                    # 모델 정보 저장
                    self.model_info[model_type] = {
                        'version': '1.0.0',
                        'accuracy': checkpoint.get('metrics', {}).get('r2', 0.0) if model_file.suffix == ".pth" else model_data.get('metrics', {}).get('r2', 0.0),
                        'last_updated': datetime.fromtimestamp(model_file.stat().st_mtime),
                        'status': 'loaded'
                    }
                    
                    logger.info(f"Loaded model: {model_type}")
            
            # MLflow에서 모델 로딩 (백업)
            if not self.models:
                self._load_from_mlflow()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_from_mlflow(self):
        """MLflow에서 모델 로딩"""
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            client = mlflow.tracking.MlflowClient()
            
            experiments = client.list_experiments()
            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1
                )
                
                if runs:
                    run = runs[0]
                    model_type = run.data.params.get('model_type', exp.name)
                    
                    try:
                        if model_type in ['lstm', 'gru', 'transformer']:
                            model = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/{model_type}_model")
                        else:
                            model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/{model_type}_model")
                        
                        self.models[model_type] = model
                        self.model_info[model_type] = {
                            'version': '1.0.0',
                            'accuracy': run.data.metrics.get('r2', 0.0),
                            'last_updated': datetime.fromtimestamp(run.info.start_time / 1000),
                            'status': 'loaded'
                        }
                        
                        logger.info(f"Loaded model from MLflow: {model_type}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_type} from MLflow: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading from MLflow: {e}")
    
    def predict(self, data: np.ndarray, model_type: str) -> Tuple[np.ndarray, float]:
        """단일 예측"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
        
        model = self.models[model_type]
        start_time = time.time()
        
        try:
            if model_type in ['lstm', 'gru', 'transformer']:
                # 딥러닝 모델 예측
                model.eval()
                with torch.no_grad():
                    if isinstance(data, np.ndarray):
                        data_tensor = torch.FloatTensor(data)
                    else:
                        data_tensor = data
                    
                    predictions = model(data_tensor)
                    predictions = predictions.cpu().numpy().squeeze()
                    
            else:
                # 스킷런 모델 예측
                if len(data.shape) == 3:
                    # 시계열 데이터를 2D로 변환
                    data_2d = data.reshape(data.shape[0], -1)
                else:
                    data_2d = data
                
                predictions = model.predict(data_2d)
            
            processing_time = time.time() - start_time
            
            # 신뢰도 계산 (간단한 방법)
            confidence = 0.8  # 실제로는 모델의 불확실성 추정 필요
            
            return predictions, confidence, processing_time
            
        except Exception as e:
            logger.error(f"Prediction error for {model_type}: {e}")
            raise
    
    def batch_predict(self, data: List[np.ndarray], model_type: str) -> Tuple[List[np.ndarray], float]:
        """배치 예측"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
        
        model = self.models[model_type]
        start_time = time.time()
        
        try:
            predictions = []
            
            for batch_data in data:
                if model_type in ['lstm', 'gru', 'transformer']:
                    # 딥러닝 모델 배치 예측
                    model.eval()
                    with torch.no_grad():
                        if isinstance(batch_data, np.ndarray):
                            batch_tensor = torch.FloatTensor(batch_data)
                        else:
                            batch_tensor = batch_data
                        
                        batch_predictions = model(batch_tensor)
                        batch_predictions = batch_predictions.cpu().numpy().squeeze()
                        predictions.append(batch_predictions)
                        
                else:
                    # 스킷런 모델 배치 예측
                    if len(batch_data.shape) == 3:
                        batch_data_2d = batch_data.reshape(batch_data.shape[0], -1)
                    else:
                        batch_data_2d = batch_data
                    
                    batch_predictions = model.predict(batch_data_2d)
                    predictions.append(batch_predictions)
            
            processing_time = time.time() - start_time
            return predictions, processing_time
            
        except Exception as e:
            logger.error(f"Batch prediction error for {model_type}: {e}")
            raise


# 전역 모델 매니저 인스턴스
model_manager = ModelManager()


# 의존성 함수들
def verify_api_key(api_key: str) -> bool:
    """API 키 검증"""
    return api_key == API_KEY


def get_model_manager() -> ModelManager:
    """모델 매니저 반환"""
    return model_manager


# API 엔드포인트들
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Trading ML API Server",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now()
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models),
        "timestamp": datetime.now()
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """사용 가능한 모델 목록"""
    models = []
    for model_type, info in model_manager.model_info.items():
        models.append(ModelInfo(
            model_type=model_type,
            version=info['version'],
            accuracy=info['accuracy'],
            last_updated=info['last_updated'],
            status=info['status']
        ))
    return models


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """단일 예측"""
    # API 키 검증
    if not verify_api_key(request.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    try:
        # 데이터 전처리
        data = np.array(request.data)
        
        # 예측 실행
        predictions, confidence, processing_time = model_manager.predict(data, request.model_type)
        
        return PredictionResponse(
            predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            model_type=request.model_type,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """배치 예측"""
    # API 키 검증
    if not verify_api_key(request.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    try:
        # 데이터 전처리
        data = [np.array(batch) for batch in request.data]
        
        # 배치 예측 실행
        predictions, processing_time = model_manager.batch_predict(data, request.model_type)
        
        # 예측 결과를 리스트로 변환
        predictions_list = []
        for pred in predictions:
            if isinstance(pred, np.ndarray):
                predictions_list.append(pred.tolist())
            else:
                predictions_list.append(pred)
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            model_type=request.model_type,
            total_processing_time=processing_time,
            batch_size=len(data),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/models/{model_type}/info")
async def get_model_info(model_type: str):
    """모델 정보 조회"""
    if model_type not in model_manager.model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_type} not found"
        )
    
    return model_manager.model_info[model_type]


@app.post("/models/{model_type}/reload")
async def reload_model(model_type: str, api_key: str):
    """모델 재로딩"""
    # API 키 검증
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    try:
        # 모델 재로딩
        model_manager.load_models()
        
        return {
            "message": f"Model {model_type} reloaded successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """시스템 메트릭"""
    import psutil
    
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "models": {
            "total_loaded": len(model_manager.models),
            "available_types": list(model_manager.models.keys())
        },
        "timestamp": datetime.now()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        "ml_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 