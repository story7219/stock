#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: qubole_setup.py
모듈: Qubole 클라우드 환경 설정
목적: Qubole Data Lake 초기 설정 및 테이블 생성

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - qds-sdk (Qubole Data Service SDK)
    - boto3 (AWS S3)
    - asyncio, logging, os, json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from qds_sdk.commands import HiveCommand
    from qds_sdk.qubole import Qubole
    QUBOLE_AVAILABLE = True
except ImportError:
    QUBOLE_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuboleSetup:
    """Qubole 설정 클래스"""

    def __init__(self) -> None:
        # Qubole API 설정
        self.api_token: Optional[str] = os.getenv('QUBOLE_API_TOKEN')
        self.api_url: str = os.getenv('QUBOLE_API_URL', 'https://api.qubole.com')

        # AWS S3 설정
        self.s3_bucket: str = os.getenv('S3_BUCKET', 'trading-data-lake')
        self.s3_region: str = os.getenv('S3_REGION', 'us-east-1')
        self.aws_access_key: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')

        # 데이터 레이크 경로
        self.data_lake_path: str = f"s3://{self.s3_bucket}"
        self.raw_data_path: str = f"s3://{self.s3_bucket}/raw"
        self.processed_data_path: str = f"s3://{self.s3_bucket}/processed"
        self.ml_data_path: str = f"s3://{self.s3_bucket}/ml"

    async def setup_qubole(self) -> None:
        """Qubole 초기 설정"""
        logger.info("Qubole 초기 설정 시작")

        try:
            # 1. Qubole 클라이언트 초기화
            await self._initialize_qubole()

            # 2. S3 버킷 설정
            await self._setup_s3_bucket()

            # 3. Hive 테이블 생성
            await self._create_hive_tables()

            # 4. 데이터 레이크 구조 생성
            await self._create_data_lake_structure()

            # 5. 연결 테스트
            await self._test_connections()

            logger.info("Qubole 초기 설정 완료")

        except Exception as e:
            logger.error(f"Qubole 초기 설정 실패: {e}")
            raise

    async def _initialize_qubole(self) -> None:
        """Qubole 클라이언트 초기화"""
        try:
            if not QUBOLE_AVAILABLE:
                raise ImportError("Qubole SDK가 설치되지 않았습니다.")

            if not self.api_token:
                raise ValueError("QUBOLE_API_TOKEN이 설정되지 않았습니다.")

            # Qubole SDK 초기화
            Qubole.configure(api_token=self.api_token)

            logger.info("Qubole 클라이언트 초기화 성공")

        except Exception as e:
            logger.error(f"Qubole 클라이언트 초기화 실패: {e}")
            raise

    async def _setup_s3_bucket(self) -> None:
        """S3 버킷 설정"""
        try:
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3가 설치되지 않았습니다.")

            if not self.aws_access_key or not self.aws_secret_key:
                raise ValueError("AWS 인증 정보가 설정되지 않았습니다.")

            # S3 클라이언트 생성
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )

            # 버킷 존재 확인 및 생성
            try:
                s3_client.head_bucket(Bucket=self.s3_bucket)
                logger.info(f"S3 버킷 '{self.s3_bucket}' 이미 존재")
            except:
                # 버킷 생성
                s3_client.create_bucket(
                    Bucket=self.s3_bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.s3_region}
                )
                logger.info(f"S3 버킷 '{self.s3_bucket}' 생성 완료")

            # 버킷 정책 설정 (선택사항)
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "QuboleAccess",
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:DeleteObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{self.s3_bucket}",
                            f"arn:aws:s3:::{self.s3_bucket}/*"
                        ]
                    }
                ]
            }

            s3_client.put_bucket_policy(
                Bucket=self.s3_bucket,
                Policy=json.dumps(bucket_policy)
            )

            logger.info("S3 버킷 설정 완료")

        except Exception as e:
            logger.error(f"S3 버킷 설정 실패: {e}")
            raise

    async def _create_hive_tables(self) -> None:
        """Hive 테이블 생성"""
        try:
            if not QUBOLE_AVAILABLE:
                raise ImportError("Qubole SDK가 설치되지 않았습니다.")

            # 원시 데이터 테이블
            raw_table_query = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS raw_stock_data (
                symbol STRING,
                timestamp TIMESTAMP,
                current_price DOUBLE,
                ohlcv MAP<STRING, STRING>,
                orderbook MAP<STRING, STRING>,
                category STRING,
                data_type STRING,
                collection_id STRING
            )
            PARTITIONED BY (date STRING)
            STORED AS PARQUET
            LOCATION '{self.raw_data_path}'
            TBLPROPERTIES ('parquet.compression'='SNAPPY')
            """

            # 처리된 OHLCV 테이블
            ohlcv_table_query = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS processed_ohlcv (
                symbol STRING,
                date STRING,
                low DOUBLE,
                high DOUBLE,
                open DOUBLE,
                close DOUBLE,
                volume BIGINT,
                data_points INT
            )
            PARTITIONED BY (date STRING)
            STORED AS PARQUET
            LOCATION '{self.processed_data_path}/ohlcv'
            TBLPROPERTIES ('parquet.compression'='SNAPPY')
            """

            # 기술적 지표 테이블
            technical_table_query = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS technical_indicators (
                symbol STRING,
                date STRING,
                ma_20 DOUBLE,
                ma_50 DOUBLE,
                rsi DOUBLE,
                macd DOUBLE,
                bollinger_upper DOUBLE,
                bollinger_lower DOUBLE,
                volume_ma DOUBLE
            )
            PARTITIONED BY (date STRING)
            STORED AS PARQUET
            LOCATION '{self.processed_data_path}/technical_indicators'
            TBLPROPERTIES ('parquet.compression'='SNAPPY')
            """

            # ML 특성 테이블
            features_table_query = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS ml_features (
                symbol STRING,
                date STRING,
                features ARRAY<DOUBLE>,
                scaled_features ARRAY<DOUBLE>,
                price_change DOUBLE,
                price_change_pct DOUBLE,
                volume_ma_ratio DOUBLE,
                volatility DOUBLE
            )
            PARTITIONED BY (date STRING)
            STORED AS PARQUET
            LOCATION '{self.ml_data_path}/features'
            TBLPROPERTIES ('parquet.compression'='SNAPPY')
            """

            # 예측 결과 테이블
            predictions_table_query = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS predictions (
                symbol STRING,
                date STRING,
                predicted_price DOUBLE,
                predicted_direction INT,
                confidence DOUBLE,
                model_version STRING
            )
            PARTITIONED BY (date STRING)
            STORED AS PARQUET
            LOCATION '{self.ml_data_path}/predictions'
            TBLPROPERTIES ('parquet.compression'='SNAPPY')
            """

            # 테이블 생성 실행
            tables = [
                ("raw_stock_data", raw_table_query),
                ("processed_ohlcv", ohlcv_table_query),
                ("technical_indicators", technical_table_query),
                ("ml_features", features_table_query),
                ("predictions", predictions_table_query)
            ]

            for table_name, query in tables:
                try:
                    hive_cmd = HiveCommand.run(query=query)
                    logger.info(f"Hive 테이블 '{table_name}' 생성 완료: {hive_cmd.id}")
                except Exception as e:
                    logger.error(f"Hive 테이블 '{table_name}' 생성 실패: {e}")

            logger.info("Hive 테이블 생성 완료")

        except Exception as e:
            logger.error(f"Hive 테이블 생성 실패: {e}")
            raise

    async def _create_data_lake_structure(self) -> None:
        """데이터 레이크 구조 생성"""
        try:
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3가 설치되지 않았습니다.")

            # S3 경로 구조 생성
            paths = [
                f"{self.raw_data_path}/",
                f"{self.processed_data_path}/",
                f"{self.processed_data_path}/ohlcv/",
                f"{self.processed_data_path}/technical_indicators/",
                f"{self.ml_data_path}/",
                f"{self.ml_data_path}/features/",
                f"{self.ml_data_path}/models/",
                f"{self.ml_data_path}/predictions/",
                f"{self.ml_data_path}/predictions/price/",
                f"{self.ml_data_path}/predictions/direction/",
                f"s3://{self.s3_bucket}/checkpoints/",
                f"s3://{self.s3_bucket}/logs/",
                f"s3://{self.s3_bucket}/backup/",
            ]

            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )

            for path in paths:
                # S3 경로는 실제로는 파일을 업로드할 때 자동으로 생성됨
                # 여기서는 더미 파일을 업로드하여 경로 생성
                bucket = path.replace(f"s3://{self.s3_bucket}/", "").rstrip("/")
                key = f"{bucket}/.keep"

                try:
                    s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=key,
                        Body=""
                    )
                    logger.info(f"데이터 레이크 경로 생성: {path}")
                except Exception as e:
                    logger.warning(f"경로 생성 실패 (무시됨): {path} - {e}")

            logger.info("데이터 레이크 구조 생성 완료")

        except Exception as e:
            logger.error(f"데이터 레이크 구조 생성 실패: {e}")
            raise

    async def _test_connections(self) -> bool:
        """연결 테스트"""
        try:
            if not QUBOLE_AVAILABLE:
                raise ImportError("Qubole SDK가 설치되지 않았습니다.")

            # Qubole 연결 테스트
            test_query = "SELECT 1 as test"
            hive_cmd = HiveCommand.run(query=test_query)
            logger.info(f"Qubole 연결 테스트 성공: {hive_cmd.id}")

            # S3 연결 테스트
            if BOTO3_AVAILABLE and self.aws_access_key and self.aws_secret_key:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.s3_region
                )

                response = s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    MaxKeys=1
                )
                logger.info("S3 연결 테스트 성공")

            logger.info("모든 연결 테스트 완료")
            return True

        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False

    async def create_sample_data(self) -> None:
        """샘플 데이터 생성"""
        try:
            if not QUBOLE_AVAILABLE:
                raise ImportError("Qubole SDK가 설치되지 않았습니다.")

            # 샘플 데이터 삽입 쿼리
            sample_data_query = f"""
            INSERT INTO raw_stock_data PARTITION (date='2025-01-27')
            VALUES
                ('005930', '2025-01-27 09:00:00', 75000.0,
                 map('open', '74800', 'high', '75500', 'low', '74500', 'close', '75000', 'volume', '1000000'),
                 map('bid1', '74900', 'ask1', '75100', 'bid_vol1', '100', 'ask_vol1', '150'),
                 'kospi', 'realtime', '20250127_090000_005930'),
                ('000660', '2025-01-27 09:00:00', 150000.0,
                 map('open', '149000', 'high', '152000', 'low', '148000', 'close', '150000', 'volume', '500000'),
                 map('bid1', '149500', 'ask1', '150500', 'bid_vol1', '200', 'ask_vol1', '180'),
                 'kospi', 'realtime', '20250127_090000_000660')
            """

            hive_cmd = HiveCommand.run(query=sample_data_query)
            logger.info(f"샘플 데이터 생성 완료: {hive_cmd.id}")

        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")

    async def show_tables(self) -> None:
        """테이블 목록 조회"""
        try:
            if not QUBOLE_AVAILABLE:
                raise ImportError("Qubole SDK가 설치되지 않았습니다.")

            show_tables_query = "SHOW TABLES"
            hive_cmd = HiveCommand.run(query=show_tables_query)
            logger.info(f"테이블 목록 조회 완료: {hive_cmd.id}")

        except Exception as e:
            logger.error(f"테이블 목록 조회 실패: {e}")


async def main() -> None:
    """메인 함수"""
    print("🔧 Qubole 클라우드 환경 설정 시작")
    print("=" * 60)

    setup = QuboleSetup()

    try:
        # 1. Qubole 초기 설정
        await setup.setup_qubole()

        # 2. 샘플 데이터 생성
        await setup.create_sample_data()

        # 3. 테이블 목록 조회
        await setup.show_tables()

        print("✅ Qubole 클라우드 환경 설정 완료")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(main())

