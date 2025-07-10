#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: database_setup.py
모듈: 데이터베이스 초기 설정 및 스키마 생성
목적: PostgreSQL, Redis, MongoDB 등 데이터베이스 초기화

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - sqlalchemy, asyncpg (PostgreSQL)
    - redis (Redis)
    - pymongo (MongoDB)
    - alembic (마이그레이션)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, JSON, Boolean, Text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.dialects.postgresql import UUID
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import alembic
    from alembic import command
    from alembic.config import Config
    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SQLAlchemy Base
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
else:
    Base = None


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    # PostgreSQL 설정
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_data"
    postgres_user: str = "postgres"
    postgres_password: str = "password"

    # Redis 설정
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # MongoDB 설정
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_db: str = "trading_data"
    mongodb_user: Optional[str] = None
    mongodb_password: Optional[str] = None

    # 설정
    create_tables: bool = True
    create_indexes: bool = True
    create_views: bool = True
    backup_existing: bool = True

    def __post_init__(self) -> None:
        # 환경변수에서 설정 로드
        self.postgres_host = os.getenv('POSTGRES_HOST', self.postgres_host)
        self.postgres_port = int(os.getenv('POSTGRES_PORT', str(self.postgres_port)))
        self.postgres_db = os.getenv('POSTGRES_DB', self.postgres_db)
        self.postgres_user = os.getenv('POSTGRES_USER', self.postgres_user)
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', self.postgres_password)

        self.redis_host = os.getenv('REDIS_HOST', self.redis_host)
        self.redis_port = int(os.getenv('REDIS_PORT', str(self.redis_port)))
        self.redis_db = int(os.getenv('REDIS_DB', str(self.redis_db)))
        self.redis_password = os.getenv('REDIS_PASSWORD', self.redis_password)

        self.mongodb_host = os.getenv('MONGODB_HOST', self.mongodb_host)
        self.mongodb_port = int(os.getenv('MONGODB_PORT', str(self.mongodb_port)))
        self.mongodb_db = os.getenv('MONGODB_DB', self.mongodb_db)
        self.mongodb_user = os.getenv('MONGODB_USER', self.mongodb_user)
        self.mongodb_password = os.getenv('MONGODB_PASSWORD', self.mongodb_password)


class DatabaseSetup:
    """데이터베이스 설정 클래스"""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self.postgres_engine = None
        self.async_postgres_engine = None
        self.redis_client: Optional[redis.Redis] = None
        self.mongo_client: Optional[MongoClient] = None

    async def setup_all(self) -> None:
        """모든 데이터베이스 설정"""
        logger.info("🚀 데이터베이스 설정 시작")

        try:
            # PostgreSQL 설정
            await self.setup_postgresql()

            # Redis 설정
            await self.setup_redis()

            # MongoDB 설정
            await self.setup_mongodb()

            # 테이블 생성
            if self.config.create_tables:
                await self.create_tables()

            # 인덱스 생성
            if self.config.create_indexes:
                await self.create_indexes()

            # 뷰 생성
            if self.config.create_views:
                await self.create_views()

            logger.info("✅ 모든 데이터베이스 설정 완료")

        except Exception as e:
            logger.error(f"데이터베이스 설정 실패: {e}")
            raise

    async def setup_postgresql(self) -> None:
        """PostgreSQL 설정"""
        try:
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError("SQLAlchemy가 설치되지 않았습니다.")

            logger.info("PostgreSQL 설정 시작")

            # 연결 URL
            sync_url = f"postgresql://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}"
            async_url = f"postgresql+asyncpg://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}"

            # 동기 엔진 생성
            self.postgres_engine = create_engine(
                sync_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            # 비동기 엔진 생성
            self.async_postgres_engine = create_async_engine(
                async_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            # 연결 테스트
            with self.postgres_engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"PostgreSQL 연결 성공: {version}")

            logger.info("PostgreSQL 설정 완료")

        except Exception as e:
            logger.error(f"PostgreSQL 설정 실패: {e}")
            raise

    async def setup_redis(self) -> None:
        """Redis 설정"""
        try:
            if not REDIS_AVAILABLE:
                raise ImportError("redis가 설치되지 않았습니다.")

            logger.info("Redis 설정 시작")

            # Redis 클라이언트 생성
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )

            # 연결 테스트
            await self.redis_client.ping()
            logger.info("Redis 연결 성공")

            # Redis 설정
            await self._configure_redis()

            logger.info("Redis 설정 완료")

        except Exception as e:
            logger.error(f"Redis 설정 실패: {e}")
            raise

    async def _configure_redis(self) -> None:
        """Redis 설정 구성"""
        try:
            # 메모리 정책 설정
            await self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')

            # 만료 시간 설정
            await self.redis_client.config_set('timeout', '300')

            # 로그 레벨 설정
            await self.redis_client.config_set('loglevel', 'notice')

            logger.info("Redis 설정 구성 완료")

        except Exception as e:
            logger.warning(f"Redis 설정 구성 실패: {e}")

    async def setup_mongodb(self) -> None:
        """MongoDB 설정"""
        try:
            if not PYMONGO_AVAILABLE:
                raise ImportError("pymongo가 설치되지 않았습니다.")

            logger.info("MongoDB 설정 시작")

            # MongoDB 연결 문자열
            if self.config.mongodb_user and self.config.mongodb_password:
                mongo_url = f"mongodb://{self.config.mongodb_user}:{self.config.mongodb_password}@{self.config.mongodb_host}:{self.config.mongodb_port}/{self.config.mongodb_db}"
            else:
                mongo_url = f"mongodb://{self.config.mongodb_host}:{self.config.mongodb_port}/{self.config.mongodb_db}"

            # MongoDB 클라이언트 생성
            self.mongo_client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )

            # 연결 테스트
            self.mongo_client.admin.command('ping')
            logger.info("MongoDB 연결 성공")

            # 데이터베이스 및 컬렉션 설정
            await self._setup_mongodb_collections()

            logger.info("MongoDB 설정 완료")

        except Exception as e:
            logger.error(f"MongoDB 설정 실패: {e}")
            raise

    async def _setup_mongodb_collections(self) -> None:
        """MongoDB 컬렉션 설정"""
        try:
            db = self.mongo_client[self.config.mongodb_db]

            # 컬렉션 생성
            collections = [
                'stock_prices',
                'orderbooks',
                'trades',
                'signals',
                'backtest_results',
                'performance_metrics'
            ]

            for collection_name in collections:
                if collection_name not in db.list_collection_names():
                    db.create_collection(collection_name)
                    logger.info(f"MongoDB 컬렉션 생성: {collection_name}")

            # 인덱스 생성
            await self._create_mongodb_indexes(db)

        except Exception as e:
            logger.error(f"MongoDB 컬렉션 설정 실패: {e}")

    async def _create_mongodb_indexes(self, db) -> None:
        """MongoDB 인덱스 생성"""
        try:
            # 주식 가격 인덱스
            db.stock_prices.create_index([("symbol", 1), ("timestamp", -1)])
            db.stock_prices.create_index([("timestamp", -1)])

            # 호가 인덱스
            db.orderbooks.create_index([("symbol", 1), ("timestamp", -1)])

            # 거래 인덱스
            db.trades.create_index([("symbol", 1), ("timestamp", -1)])

            # 신호 인덱스
            db.signals.create_index([("symbol", 1), ("timestamp", -1)])
            db.signals.create_index([("signal_type", 1)])

            logger.info("MongoDB 인덱스 생성 완료")

        except Exception as e:
            logger.error(f"MongoDB 인덱스 생성 실패: {e}")

    async def create_tables(self) -> None:
        """테이블 생성"""
        try:
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError("SQLAlchemy가 설치되지 않았습니다.")

            logger.info("테이블 생성 시작")

            # 메타데이터 생성
            metadata = MetaData()

            # 주식 가격 테이블
            stock_prices = Table(
                'stock_prices',
                metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('symbol', String(20), nullable=False, index=True),
                Column('timestamp', DateTime, nullable=False, index=True),
                Column('current_price', Float, nullable=False),
                Column('open_price', Float),
                Column('high_price', Float),
                Column('low_price', Float),
                Column('prev_close', Float),
                Column('change_rate', Float),
                Column('volume', Integer),
                Column('category', String(20)),
                Column('created_at', DateTime, default=datetime.utcnow)
            )

            # 호가 테이블
            orderbooks = Table(
                'orderbooks',
                metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('symbol', String(20), nullable=False, index=True),
                Column('timestamp', DateTime, nullable=False, index=True),
                Column('bid_prices', JSON),
                Column('ask_prices', JSON),
                Column('bid_volumes', JSON),
                Column('ask_volumes', JSON),
                Column('category', String(20)),
                Column('created_at', DateTime, default=datetime.utcnow)
            )

            # 거래 신호 테이블
            signals = Table(
                'signals',
                metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('signal_id', String(50), unique=True, nullable=False),
                Column('symbol', String(20), nullable=False, index=True),
                Column('signal_type', String(10), nullable=False, index=True),
                Column('strategy_type', String(20), nullable=False),
                Column('confidence_score', Float),
                Column('target_price', Float),
                Column('stop_loss', Float),
                Column('take_profit', Float),
                Column('reasoning', Text),
                Column('status', String(20), default='pending'),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('executed_at', DateTime)
            )

            # 백테스트 결과 테이블
            backtest_results = Table(
                'backtest_results',
                metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('strategy_name', String(50), nullable=False, index=True),
                Column('start_date', DateTime, nullable=False),
                Column('end_date', DateTime, nullable=False),
                Column('initial_capital', Float, nullable=False),
                Column('final_capital', Float, nullable=False),
                Column('total_return', Float, nullable=False),
                Column('annual_return', Float),
                Column('max_drawdown', Float),
                Column('sharpe_ratio', Float),
                Column('win_rate', Float),
                Column('total_trades', Integer),
                Column('parameters', JSON),
                Column('created_at', DateTime, default=datetime.utcnow)
            )

            # 성능 메트릭 테이블
            performance_metrics = Table(
                'performance_metrics',
                metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('metric_name', String(50), nullable=False, index=True),
                Column('metric_value', Float, nullable=False),
                Column('metric_unit', String(20)),
                Column('timestamp', DateTime, nullable=False, index=True),
                Column('category', String(20)),
                Column('metadata', JSON),
                Column('created_at', DateTime, default=datetime.utcnow)
            )

            # 테이블 생성
            metadata.create_all(self.postgres_engine)

            logger.info("테이블 생성 완료")

        except Exception as e:
            logger.error(f"테이블 생성 실패: {e}")
            raise

    async def create_indexes(self) -> None:
        """인덱스 생성"""
        try:
            logger.info("인덱스 생성 시작")

            # 복합 인덱스 생성
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_timestamp ON stock_prices (symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_timestamp ON stock_prices (timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_orderbooks_symbol_timestamp ON orderbooks (symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON signals (symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_signals_type_status ON signals (signal_type, status)",
                "CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy_date ON backtest_results (strategy_name, start_date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_timestamp ON performance_metrics (metric_name, timestamp DESC)"
            ]

            with self.postgres_engine.connect() as conn:
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                conn.commit()

            logger.info("인덱스 생성 완료")

        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            raise

    async def create_views(self) -> None:
        """뷰 생성"""
        try:
            logger.info("뷰 생성 시작")

            views = [
                """
                CREATE OR REPLACE VIEW daily_stock_summary AS
                SELECT 
                    symbol,
                    DATE(timestamp) as date,
                    AVG(current_price) as avg_price,
                    MAX(current_price) as max_price,
                    MIN(current_price) as min_price,
                    SUM(volume) as total_volume,
                    COUNT(*) as data_points
                FROM stock_prices
                GROUP BY symbol, DATE(timestamp)
                ORDER BY symbol, date DESC
                """,
                """
                CREATE OR REPLACE VIEW signal_performance AS
                SELECT 
                    signal_type,
                    strategy_type,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN status = 'executed' THEN 1 END) as executed_signals,
                    AVG(confidence_score) as avg_confidence,
                    AVG(CASE WHEN status = 'executed' THEN confidence_score END) as avg_executed_confidence
                FROM signals
                GROUP BY signal_type, strategy_type
                ORDER BY total_signals DESC
                """,
                """
                CREATE OR REPLACE VIEW strategy_performance AS
                SELECT 
                    strategy_name,
                    COUNT(*) as total_backtests,
                    AVG(total_return) as avg_return,
                    AVG(annual_return) as avg_annual_return,
                    AVG(sharpe_ratio) as avg_sharpe_ratio,
                    AVG(win_rate) as avg_win_rate,
                    MAX(final_capital) as best_final_capital,
                    MIN(max_drawdown) as best_max_drawdown
                FROM backtest_results
                GROUP BY strategy_name
                ORDER BY avg_return DESC
                """
            ]

            with self.postgres_engine.connect() as conn:
                for view_sql in views:
                    conn.execute(text(view_sql))
                conn.commit()

            logger.info("뷰 생성 완료")

        except Exception as e:
            logger.error(f"뷰 생성 실패: {e}")
            raise

    async def backup_database(self) -> None:
        """데이터베이스 백업"""
        try:
            if not self.config.backup_existing:
                return

            logger.info("데이터베이스 백업 시작")

            # 백업 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_trading_data_{timestamp}.sql"

            # pg_dump 명령어 실행
            import subprocess
            cmd = [
                'pg_dump',
                '-h', self.config.postgres_host,
                '-p', str(self.config.postgres_port),
                '-U', self.config.postgres_user,
                '-d', self.config.postgres_db,
                '-f', backup_file,
                '--no-password'
            ]

            # 환경변수 설정
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.postgres_password

            # 백업 실행
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"데이터베이스 백업 완료: {backup_file}")
            else:
                logger.error(f"데이터베이스 백업 실패: {result.stderr}")

        except Exception as e:
            logger.error(f"데이터베이스 백업 실패: {e}")

    async def restore_database(self, backup_file: str) -> None:
        """데이터베이스 복원"""
        try:
            logger.info(f"데이터베이스 복원 시작: {backup_file}")

            if not os.path.exists(backup_file):
                raise FileNotFoundError(f"백업 파일을 찾을 수 없습니다: {backup_file}")

            # psql 명령어 실행
            import subprocess
            cmd = [
                'psql',
                '-h', self.config.postgres_host,
                '-p', str(self.config.postgres_port),
                '-U', self.config.postgres_user,
                '-d', self.config.postgres_db,
                '-f', backup_file,
                '--no-password'
            ]

            # 환경변수 설정
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.postgres_password

            # 복원 실행
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("데이터베이스 복원 완료")
            else:
                logger.error(f"데이터베이스 복원 실패: {result.stderr}")

        except Exception as e:
            logger.error(f"데이터베이스 복원 실패: {e}")

    async def check_health(self) -> Dict[str, Any]:
        """데이터베이스 상태 확인"""
        health_status = {
            'postgresql': False,
            'redis': False,
            'mongodb': False,
            'overall': False
        }

        try:
            # PostgreSQL 상태 확인
            with self.postgres_engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.fetchone():
                    health_status['postgresql'] = True

            # Redis 상태 확인
            if self.redis_client:
                await self.redis_client.ping()
                health_status['redis'] = True

            # MongoDB 상태 확인
            if self.mongo_client:
                self.mongo_client.admin.command('ping')
                health_status['mongodb'] = True

            # 전체 상태
            health_status['overall'] = all([
                health_status['postgresql'],
                health_status['redis'],
                health_status['mongodb']
            ])

            logger.info(f"데이터베이스 상태: {health_status}")

        except Exception as e:
            logger.error(f"데이터베이스 상태 확인 실패: {e}")

        return health_status

    async def cleanup(self) -> None:
        """리소스 정리"""
        try:
            if self.postgres_engine:
                self.postgres_engine.dispose()

            if self.async_postgres_engine:
                await self.async_postgres_engine.dispose()

            if self.redis_client:
                await self.redis_client.close()

            if self.mongo_client:
                self.mongo_client.close()

            logger.info("데이터베이스 리소스 정리 완료")

        except Exception as e:
            logger.error(f"리소스 정리 실패: {e}")


async def main() -> None:
    """메인 함수"""
    print("🚀 데이터베이스 설정 시작")
    print("=" * 60)

    # 설정 생성
    config = DatabaseConfig()

    # 설정기 생성
    setup = DatabaseSetup(config)

    try:
        # 모든 데이터베이스 설정
        await setup.setup_all()

        # 상태 확인
        health = await setup.check_health()
        print(f"📊 데이터베이스 상태: {health}")

        if health['overall']:
            print("✅ 모든 데이터베이스가 정상적으로 설정되었습니다.")
        else:
            print("⚠️ 일부 데이터베이스에 문제가 있습니다.")

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        # 리소스 정리
        await setup.cleanup()
        print("✅ 데이터베이스 설정 완료")


if __name__ == "__main__":
    asyncio.run(main())

