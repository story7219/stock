#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 로딩 테스트 스크립트
저장된 모델들이 정상적으로 로드되는지 확인
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tensorflow_import():
    """TensorFlow import 테스트"""
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow 버전: {tf.__version__}")
        return True
    except ImportError as e:
        logger.error(f"❌ TensorFlow import 실패: {e}")
        return False

def test_keras_import():
    """Keras import 테스트"""
    try:
        from tensorflow.keras.models import load_model, Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout
        logger.info("✅ Keras 모듈 import 성공")
        return True
    except ImportError as e:
        logger.error(f"❌ Keras import 실패: {e}")
        return False

def test_model_files():
    """모델 파일 존재 확인"""
    model_files = []
    
    # 모델 파일 경로들
    model_paths = [
        "backtest/models/best_keras_model.h5",
        "backtest/test_models/best_keras_model.h5",
        "backtest/test_models_basic/best_keras_model.h5",
        "backtest/test_models_seq_5/best_keras_model.h5",
        "backtest/test_models_seq_10/best_keras_model.h5",
        "backtest/test_models_seq_15/best_keras_model.h5"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            size = Path(path).stat().st_size
            model_files.append((path, size))
            logger.info(f"✅ 모델 파일 발견: {path} ({size:,} bytes)")
        else:
            logger.warning(f"⚠️ 모델 파일 없음: {path}")
    
    return model_files

def test_model_loading():
    """모델 로딩 테스트"""
    model_files = test_model_files()
    
    if not model_files:
        logger.error("❌ 로드할 모델 파일이 없습니다.")
        return False
    
    success_count = 0
    
    for model_path, size in model_files:
        try:
            from tensorflow.keras.models import load_model
            
            # 모델 로드
            model = load_model(model_path)
            logger.info(f"✅ 모델 로드 성공: {model_path}")
            
            # 모델 정보 출력
            logger.info(f"   - 모델 구조: {model.summary()}")
            logger.info(f"   - 입력 형태: {model.input_shape}")
            logger.info(f"   - 출력 형태: {model.output_shape}")
            
            # 간단한 예측 테스트
            if model.input_shape:
                input_shape = model.input_shape[0]
                if len(input_shape) > 1:
                    # 시계열 모델인 경우
                    test_input = np.random.random((1, input_shape[1], input_shape[2]))
                else:
                    # 일반 모델인 경우
                    test_input = np.random.random((1, input_shape[1]))
                
                prediction = model.predict(test_input, verbose=0)
                logger.info(f"   - 예측 테스트 성공: 출력 형태 {prediction.shape}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {model_path} - {e}")
    
    logger.info(f"📊 모델 로딩 결과: {success_count}/{len(model_files)} 성공")
    return success_count > 0

def test_market_predictor():
    """MarketPredictor 클래스 테스트"""
    try:
        from models.market_predictor import MarketPredictor
        
        predictor = MarketPredictor()
        logger.info("✅ MarketPredictor 클래스 생성 성공")
        
        # 데이터 로드 테스트
        data = predictor.load_data()
        logger.info(f"✅ 데이터 로드 성공: {len(data)} 개 데이터셋")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MarketPredictor 테스트 실패: {e}")
        return False

def test_sklearn_models():
    """Scikit-learn 모델 테스트"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # 간단한 모델 생성 테스트
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scaler = StandardScaler()
        
        # 가상 데이터로 테스트
        X = np.random.random((100, 5))
        y = np.random.random(100)
        
        # 스케일링
        X_scaled = scaler.fit_transform(X)
        
        # 모델 학습
        model.fit(X_scaled, y)
        
        # 예측 테스트
        prediction = model.predict(X_scaled[:5])
        
        logger.info("✅ Scikit-learn 모델 테스트 성공")
        logger.info(f"   - 예측 결과: {prediction.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scikit-learn 모델 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 모델 로딩 테스트 시작")
    
    results = {}
    
    # 1. TensorFlow import 테스트
    results['tensorflow'] = test_tensorflow_import()
    
    # 2. Keras import 테스트
    results['keras'] = test_keras_import()
    
    # 3. 모델 파일 확인
    model_files = test_model_files()
    results['model_files'] = len(model_files) > 0
    
    # 4. 모델 로딩 테스트
    results['model_loading'] = test_model_loading()
    
    # 5. MarketPredictor 테스트
    results['market_predictor'] = test_market_predictor()
    
    # 6. Scikit-learn 테스트
    results['sklearn'] = test_sklearn_models()
    
    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info("📊 테스트 결과 요약")
    logger.info("="*50)
    
    for test_name, result in results.items():
        status = "✅ 성공" if result else "❌ 실패"
        logger.info(f"{test_name:15}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\n전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        logger.info("🎉 모든 테스트 통과!")
    else:
        logger.warning("⚠️ 일부 테스트 실패")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 