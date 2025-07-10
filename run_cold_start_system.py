#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_cold_start_system.py
모듈: 콜드 스타트 및 데이터 동기화 시스템 실행 스크립트
목적: 통합 시스템 실행 및 테스트

Author: AI Trading System
Created: 2025-01-08
Modified: 2025-01-08
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - src/cold_start_system.py
    - src/data_synchronization_system.py

Performance:
    - 실행 시간: < 30초
    - 메모리 사용량: < 2GB
    - 처리 지연시간: < 100ms

Security:
    - 환경 변수: secure configuration
    - 에러 처리: comprehensive try-catch
    - 로깅: detailed audit trail

License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 모듈 임포트
from src.cold_start_system import (
    ColdStartSolver, TransferLearner, HybridPredictor, ConfidenceWeighter,
    ColdStartConfig
)
from src.data_synchronization_system import (
    DataSynchronizer, SchemaValidator, BackfillManager, ConsistencyChecker,
    SyncConfig, DataSchema
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cold_start_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ColdStartSystemRunner:
    """콜드 스타트 시스템 실행기"""
    
    def __init__(self):
        self.cold_start_config = ColdStartConfig()
        self.sync_config = SyncConfig()
        self.data_schema = DataSchema()
        
        # 시스템 컴포넌트들
        self.cold_start_solver: Optional[ColdStartSolver] = None
        self.transfer_learner: Optional[TransferLearner] = None
        self.hybrid_predictor: Optional[HybridPredictor] = None
        self.confidence_weighter: Optional[ConfidenceWeighter] = None
        self.data_synchronizer: Optional[DataSynchronizer] = None
        self.schema_validator: Optional[SchemaValidator] = None
        self.backfill_manager: Optional[BackfillManager] = None
        self.consistency_checker: Optional[ConsistencyChecker] = None
        
        logger.info("ColdStartSystemRunner initialized")
    
    async def initialize_systems(self) -> None:
        """모든 시스템 초기화"""
        try:
            logger.info("Initializing cold start and data synchronization systems...")
            
            # 콜드 스타트 시스템 초기화
            self.cold_start_solver = ColdStartSolver(self.cold_start_config)
            self.transfer_learner = TransferLearner(self.cold_start_config)
            self.hybrid_predictor = HybridPredictor(self.cold_start_config)
            self.confidence_weighter = ConfidenceWeighter(self.cold_start_config)
            
            # 데이터 동기화 시스템 초기화
            self.data_synchronizer = DataSynchronizer(self.sync_config, self.data_schema)
            self.schema_validator = SchemaValidator(self.data_schema)
            self.backfill_manager = BackfillManager(self.sync_config)
            self.consistency_checker = ConsistencyChecker(self.sync_config)
            
            # 연결 초기화
            await self.data_synchronizer.initialize_connections()
            
            # 사전 모델 로드
            await self.cold_start_solver.load_pre_trained_models()
            
            # 하이브리드 가중치 초기화
            await self.cold_start_solver.initialize_hybrid_weights()
            
            logger.info("All systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing systems: {e}")
            raise
    
    async def run_cold_start_demo(self) -> Dict[str, Any]:
        """콜드 스타트 데모 실행"""
        try:
            logger.info("Running cold start demo...")
            
            # 샘플 데이터 생성
            sample_data = self._create_sample_data()
            
            # 시장 상황 분석
            market_conditions = {
                'volatility': 0.3,
                'trend': 0.1,
                'volume': 0.5
            }
            
            # 최적 사전 모델 선택
            best_model_name = self.cold_start_solver.select_best_pre_trained_model(market_conditions)
            logger.info(f"Selected best pre-trained model: {best_model_name}")
            
            # 하이브리드 예측 생성
            if best_model_name in self.cold_start_solver.pre_trained_models:
                pre_trained_model = self.cold_start_solver.pre_trained_models[best_model_name]
                
                # 하이브리드 예측
                hybrid_result = await self.hybrid_predictor.generate_hybrid_prediction(
                    pre_trained_model=pre_trained_model,
                    realtime_model=pre_trained_model,  # 데모용으로 동일 모델 사용
                    input_data=sample_data,
                    model_name=best_model_name
                )
                
                logger.info(f"Hybrid prediction generated with confidence: {hybrid_result['confidence']:.3f}")
                
                return {
                    'selected_model': best_model_name,
                    'hybrid_result': hybrid_result,
                    'market_conditions': market_conditions
                }
            else:
                logger.warning(f"Pre-trained model {best_model_name} not found")
                return {'error': 'Model not found'}
            
        except Exception as e:
            logger.error(f"Error in cold start demo: {e}")
            raise
    
    async def run_data_sync_demo(self) -> Dict[str, Any]:
        """데이터 동기화 데모 실행"""
        try:
            logger.info("Running data synchronization demo...")
            
            # 샘플 과거 데이터 생성
            historical_data = self._create_historical_data()
            
            # 샘플 실시간 데이터 생성
            realtime_data = self._create_realtime_data()
            
            # 데이터 동기화
            synchronized_data = await self.data_synchronizer.synchronize_data(
                historical_data=historical_data,
                realtime_data=realtime_data
            )
            
            # 스키마 검증
            schema_valid = await self.schema_validator.validate_data_schema(synchronized_data)
            
            # 일관성 검사
            consistency_valid = await self.consistency_checker.check_data_consistency(synchronized_data)
            
            # 백필 통계
            backfill_stats = self.backfill_manager.get_backfill_statistics()
            
            # 일관성 리포트
            consistency_report = self.consistency_checker.get_consistency_report()
            
            logger.info(f"Data synchronization completed. Records: {len(synchronized_data)}")
            
            return {
                'synchronized_records': len(synchronized_data),
                'schema_valid': schema_valid,
                'consistency_valid': consistency_valid,
                'backfill_stats': backfill_stats,
                'consistency_report': consistency_report
            }
            
        except Exception as e:
            logger.error(f"Error in data sync demo: {e}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """샘플 데이터 생성"""
        try:
            import pandas as pd
            import numpy as np
            
            # 100개의 샘플 데이터 생성
            dates = pd.date_range(start='2025-01-01', periods=100, freq='1min')
            
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': ['AAPL'] * 100,
                'open': np.random.uniform(150, 160, 100),
                'high': np.random.uniform(155, 165, 100),
                'low': np.random.uniform(145, 155, 100),
                'close': np.random.uniform(150, 160, 100),
                'volume': np.random.randint(1000000, 5000000, 100)
            })
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            raise
    
    def _create_historical_data(self) -> pd.DataFrame:
        """과거 데이터 생성"""
        try:
            import pandas as pd
            import numpy as np
            
            # 과거 30일 데이터
            dates = pd.date_range(start='2024-12-01', end='2024-12-31', freq='1min')
            
            historical_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': ['AAPL'] * len(dates),
                'open': np.random.uniform(150, 160, len(dates)),
                'high': np.random.uniform(155, 165, len(dates)),
                'low': np.random.uniform(145, 155, len(dates)),
                'close': np.random.uniform(150, 160, len(dates)),
                'volume': np.random.randint(1000000, 5000000, len(dates))
            })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error creating historical data: {e}")
            raise
    
    def _create_realtime_data(self) -> pd.DataFrame:
        """실시간 데이터 생성"""
        try:
            import pandas as pd
            import numpy as np
            
            # 실시간 데이터 (2025년 1월 1일부터)
            dates = pd.date_range(start='2025-01-01', periods=1000, freq='1min')
            
            realtime_data = pd.DataFrame({
                'timestamp': dates,
                'symbol': ['AAPL'] * len(dates),
                'open': np.random.uniform(150, 160, len(dates)),
                'high': np.random.uniform(155, 165, len(dates)),
                'low': np.random.uniform(145, 155, len(dates)),
                'close': np.random.uniform(150, 160, len(dates)),
                'volume': np.random.randint(1000000, 5000000, len(dates))
            })
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"Error creating realtime data: {e}")
            raise
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """전체 시스템 데모 실행"""
        try:
            logger.info("Running full system demo...")
            
            start_time = time.time()
            
            # 1. 콜드 스타트 데모
            cold_start_result = await self.run_cold_start_demo()
            
            # 2. 데이터 동기화 데모
            data_sync_result = await self.run_data_sync_demo()
            
            # 3. 통합 결과
            total_time = time.time() - start_time
            
            result = {
                'cold_start_result': cold_start_result,
                'data_sync_result': data_sync_result,
                'total_execution_time': total_time,
                'status': 'success'
            }
            
            logger.info(f"Full demo completed in {total_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in full demo: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def print_demo_results(self, results: Dict[str, Any]) -> None:
        """데모 결과 출력"""
        try:
            print("\n" + "="*60)
            print("COLD START & DATA SYNCHRONIZATION SYSTEM DEMO RESULTS")
            print("="*60)
            
            if results.get('status') == 'success':
                # 콜드 스타트 결과
                cold_start = results.get('cold_start_result', {})
                if 'selected_model' in cold_start:
                    print(f"\n🔹 Cold Start System:")
                    print(f"   Selected Model: {cold_start['selected_model']}")
                    if 'hybrid_result' in cold_start:
                        hybrid = cold_start['hybrid_result']
                        print(f"   Confidence: {hybrid.get('confidence', 0):.3f}")
                        print(f"   Prediction Time: {hybrid.get('prediction_time', 0):.3f}s")
                
                # 데이터 동기화 결과
                data_sync = results.get('data_sync_result', {})
                print(f"\n🔹 Data Synchronization System:")
                print(f"   Synchronized Records: {data_sync.get('synchronized_records', 0):,}")
                print(f"   Schema Valid: {'✅' if data_sync.get('schema_valid') else '❌'}")
                print(f"   Consistency Valid: {'✅' if data_sync.get('consistency_valid') else '❌'}")
                
                # 백필 통계
                backfill_stats = data_sync.get('backfill_stats', {})
                if backfill_stats:
                    print(f"   Backfill Periods: {backfill_stats.get('total_missing_periods', 0)}")
                
                # 실행 시간
                print(f"\n⏱️  Total Execution Time: {results.get('total_execution_time', 0):.2f}s")
                
            else:
                print(f"\n❌ Demo failed: {results.get('error', 'Unknown error')}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing demo results: {e}")

async def main():
    """메인 실행 함수"""
    try:
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        # 시스템 실행기 초기화
        runner = ColdStartSystemRunner()
        
        # 시스템 초기화
        await runner.initialize_systems()
        
        # 전체 데모 실행
        results = await runner.run_full_demo()
        
        # 결과 출력
        runner.print_demo_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\n❌ System execution failed: {e}")
        return {'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main()) 