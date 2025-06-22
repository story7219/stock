"""
Ultra 고성능 데이터베이스 매니저 - 비동기 커넥션 풀링, 배치 처리, 샤딩 지원
"""
import asyncio
import contextlib
from typing import Any, Dict, List, Optional, AsyncGenerator, Union, Tuple, Callable
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading

import asyncpg
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index, text, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql import func

from config.settings import settings
from core.cache_manager import UltraCacheManager

logger = structlog.get_logger(__name__)

# 메타데이터 설정
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

Base = declarative_base(metadata=metadata)


class TransactionIsolationLevel(Enum):
    """트랜잭션 격리 수준"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class DatabaseShardType(Enum):
    """데이터베이스 샤드 타입"""
    PRIMARY = "primary"
    REPLICA = "replica"
    ANALYTICS = "analytics"


@dataclass
class ConnectionPoolStats:
    """커넥션 풀 통계"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    pool_size: int = 0
    max_overflow: int = 0
    checked_out_connections: int = 0
    overflow_connections: int = 0
    invalidated_connections: int = 0


@dataclass
class QueryMetrics:
    """쿼리 성능 메트릭"""
    query_hash: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    last_executed: Optional[datetime] = None
    error_count: int = 0
    
    def update(self, execution_time: float, error: bool = False):
        """메트릭 업데이트"""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.execution_count
        self.max_time = max(self.max_time, execution_time)
        self.min_time = min(self.min_time, execution_time)
        self.last_executed = datetime.now()
        
        if error:
            self.error_count += 1


@dataclass
class BatchOperation:
    """배치 작업 정의"""
    operation_type: str  # insert, update, delete
    table_name: str
    data: List[Dict[str, Any]]
    batch_size: int = 1000
    on_conflict: str = "ignore"  # ignore, update, replace


class StockData(Base):
    """주식 데이터 테이블 - 파티션 지원"""
    __tablename__ = "stock_data"
    
    code = Column(String(20), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    market = Column(String(20), nullable=False)
    sector = Column(String(50))
    price = Column(Float, nullable=False)
    change_rate = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    market_cap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    shard_key = Column(String(32), nullable=False, index=True)  # 샤딩 키
    
    __table_args__ = (
        Index('idx_stock_code_date', 'code', 'created_at'),
        Index('idx_stock_market', 'market'),
        Index('idx_stock_sector', 'sector'),
        Index('idx_stock_shard', 'shard_key'),
        Index('idx_stock_composite', 'market', 'sector', 'created_at'),
    )


class PriceHistory(Base):
    """주가 이력 테이블 - 시계열 최적화"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    partition_key = Column(String(10), nullable=False, index=True)  # YYYY-MM 형식
    
    __table_args__ = (
        Index('idx_price_stock_date', 'stock_code', 'date'),
        Index('idx_price_partition', 'partition_key', 'stock_code'),
        Index('idx_price_composite', 'stock_code', 'date', 'volume'),
    )


class AnalysisResult(Base):
    """분석 결과 테이블 - JSON 최적화"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)
    result_data = Column(Text, nullable=False)  # JSON 문자열
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, index=True)
    version = Column(Integer, default=1)  # 결과 버전 관리
    checksum = Column(String(64))  # 데이터 무결성 검증
    
    __table_args__ = (
        Index('idx_analysis_stock_type', 'stock_code', 'analysis_type'),
        Index('idx_analysis_created', 'created_at'),
        Index('idx_analysis_expires', 'expires_at'),
        Index('idx_analysis_composite', 'stock_code', 'analysis_type', 'created_at'),
    )


class UltraDatabaseManager:
    """Ultra 고성능 데이터베이스 매니저"""
    
    def __init__(self):
        self.engines: Dict[DatabaseShardType, Any] = {}
        self.session_factories: Dict[DatabaseShardType, Any] = {}
        self.connection_pools: Dict[DatabaseShardType, Any] = {}
        self.cache_manager = UltraCacheManager()
        
        # 성능 모니터링
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.connection_stats: Dict[DatabaseShardType, ConnectionPoolStats] = {}
        
        # 배치 처리
        self.batch_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.batch_processor_task: Optional[asyncio.Task] = None
        
        # 스레드 안전성
        self._lock = threading.RLock()
        self._initialized = False
        
        # 약한 참조로 세션 추적
        self.active_sessions: weakref.WeakSet = weakref.WeakSet()
        
        # 백그라운드 작업자
        self.background_executor = ThreadPoolExecutor(
            max_workers=settings.performance.max_workers,
            thread_name_prefix="db_worker"
        )
    
    async def initialize(self) -> None:
        """Ultra 데이터베이스 초기화"""
        if self._initialized:
            return
        
        try:
            await self._initialize_cache_manager()
            await self._initialize_database_shards()
            await self._initialize_connection_pools()
            await self._create_tables()
            await self._start_batch_processor()
            await self._setup_monitoring()
            
            self._initialized = True
            logger.info("Ultra 데이터베이스 매니저 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            await self._cleanup_on_error()
            raise
    
    async def _initialize_cache_manager(self) -> None:
        """캐시 매니저 초기화"""
        await self.cache_manager.initialize()
    
    async def _initialize_database_shards(self) -> None:
        """데이터베이스 샤드 초기화"""
        shard_configs = {
            DatabaseShardType.PRIMARY: {
                "url": settings.database.postgres_url if not (settings.is_development or settings.is_test) else settings.database.sqlite_url,
                "pool_size": settings.database.pool_size,
                "max_overflow": settings.database.max_overflow
            },
            DatabaseShardType.REPLICA: {
                "url": settings.database.postgres_replica_url if hasattr(settings.database, 'postgres_replica_url') else settings.database.postgres_url,
                "pool_size": max(5, settings.database.pool_size // 2),
                "max_overflow": settings.database.max_overflow // 2
            },
            DatabaseShardType.ANALYTICS: {
                "url": settings.database.analytics_url if hasattr(settings.database, 'analytics_url') else settings.database.postgres_url,
                "pool_size": max(3, settings.database.pool_size // 3),
                "max_overflow": settings.database.max_overflow // 3
            }
        }
        
        for shard_type, config in shard_configs.items():
            connect_args = {"check_same_thread": False} if "sqlite" in config["url"] else {}
            
            engine = create_async_engine(
                config["url"],
                pool_size=config["pool_size"],
                max_overflow=config["max_overflow"],
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.debug,
                future=True,
                connect_args=connect_args,
                poolclass=QueuePool if "postgresql" in config["url"] else StaticPool,
                pool_timeout=30,
                pool_reset_on_return='commit'
            )
            
            session_factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,  # 성능 최적화
                autocommit=False
            )
            
            self.engines[shard_type] = engine
            self.session_factories[shard_type] = session_factory
            
            # 연결 통계 초기화
            self.connection_stats[shard_type] = ConnectionPoolStats(
                pool_size=config["pool_size"],
                max_overflow=config["max_overflow"]
            )
    
    async def _initialize_connection_pools(self) -> None:
        """PostgreSQL 원시 연결 풀 초기화"""
        if not (settings.is_development or settings.is_test):
            try:
                # Primary 풀
                self.connection_pools[DatabaseShardType.PRIMARY] = await asyncpg.create_pool(
                    host=settings.database.postgres_host,
                    port=settings.database.postgres_port,
                    user=settings.database.postgres_user,
                    password=settings.database.postgres_password,
                    database=settings.database.postgres_db,
                    min_size=10,
                    max_size=settings.database.pool_size,
                    command_timeout=60,
                    server_settings={
                        'application_name': 'ultra_hts_primary',
                        'tcp_keepalives_idle': '600',
                        'tcp_keepalives_interval': '30',
                        'tcp_keepalives_count': '3',
                    }
                )
                
                # Replica 풀 (읽기 전용)
                if hasattr(settings.database, 'postgres_replica_host'):
                    self.connection_pools[DatabaseShardType.REPLICA] = await asyncpg.create_pool(
                        host=settings.database.postgres_replica_host,
                        port=getattr(settings.database, 'postgres_replica_port', settings.database.postgres_port),
                        user=settings.database.postgres_user,
                        password=settings.database.postgres_password,
                        database=settings.database.postgres_db,
                        min_size=5,
                        max_size=max(10, settings.database.pool_size // 2),
                        command_timeout=60,
                        server_settings={
                            'application_name': 'ultra_hts_replica',
                            'default_transaction_read_only': 'on'
                        }
                    )
            except Exception as e:
                logger.warning(f"PostgreSQL 연결 풀 초기화 실패: {e}")
    
    async def _create_tables(self) -> None:
        """테이블 생성 및 인덱스 최적화"""
        for shard_type, engine in self.engines.items():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
                # PostgreSQL 전용 최적화
                if "postgresql" in str(engine.url):
                    await self._create_postgresql_optimizations(conn)
    
    async def _create_postgresql_optimizations(self, conn) -> None:
        """PostgreSQL 전용 최적화"""
        optimizations = [
            # 파티션 테이블 생성 (시계열 데이터용)
            """
            CREATE TABLE IF NOT EXISTS price_history_partitioned (
                LIKE price_history INCLUDING ALL
            ) PARTITION BY RANGE (date);
            """,
            
            # 인덱스 최적화
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_data_btree ON stock_data USING btree (code, created_at DESC);",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_brin ON price_history USING brin (date);",
            
            # 통계 업데이트
            "ANALYZE stock_data;",
            "ANALYZE price_history;",
            "ANALYZE analysis_results;",
        ]
        
        for sql in optimizations:
            try:
                await conn.execute(text(sql))
            except Exception as e:
                logger.warning(f"PostgreSQL 최적화 실행 실패: {sql[:50]}... - {e}")
    
    async def _start_batch_processor(self) -> None:
        """배치 프로세서 시작"""
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
    
    async def _batch_processor_loop(self) -> None:
        """배치 처리 루프"""
        batch_operations: List[BatchOperation] = []
        last_process_time = asyncio.get_event_loop().time()
        
        while True:
            try:
                # 배치 작업 수집 (최대 1초 대기)
                try:
                    operation = await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)
                    batch_operations.append(operation)
                except asyncio.TimeoutError:
                    pass
                
                current_time = asyncio.get_event_loop().time()
                
                # 배치 처리 조건 확인
                should_process = (
                    len(batch_operations) >= settings.database.batch_size or
                    (batch_operations and current_time - last_process_time > 5.0)  # 5초마다 강제 처리
                )
                
                if should_process and batch_operations:
                    await self._process_batch_operations(batch_operations)
                    batch_operations.clear()
                    last_process_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"배치 처리 오류: {e}")
                batch_operations.clear()
    
    async def _process_batch_operations(self, operations: List[BatchOperation]) -> None:
        """배치 작업 처리"""
        # 테이블별로 그룹화
        grouped_ops = {}
        for op in operations:
            key = (op.table_name, op.operation_type)
            if key not in grouped_ops:
                grouped_ops[key] = []
            grouped_ops[key].extend(op.data)
        
        # 각 그룹별로 배치 실행
        for (table_name, op_type), data_list in grouped_ops.items():
            try:
                if op_type == "insert":
                    await self._batch_insert(table_name, data_list)
                elif op_type == "update":
                    await self._batch_update(table_name, data_list)
                elif op_type == "delete":
                    await self._batch_delete(table_name, data_list)
            except Exception as e:
                logger.error(f"배치 {op_type} 실행 실패 ({table_name}): {e}")
    
    async def _setup_monitoring(self) -> None:
        """모니터링 설정"""
        # 주기적 통계 수집 태스크 시작
        asyncio.create_task(self._collect_stats_periodically())
    
    async def _collect_stats_periodically(self) -> None:
        """주기적 통계 수집"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다
                await self._update_connection_stats()
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"통계 수집 오류: {e}")
    
    def _get_shard_key(self, stock_code: str) -> str:
        """샤딩 키 생성"""
        return hashlib.md5(stock_code.encode()).hexdigest()
    
    def _get_partition_key(self, date: datetime) -> str:
        """파티션 키 생성 (YYYY-MM)"""
        return date.strftime("%Y-%m")
    
    def _calculate_checksum(self, data: str) -> str:
        """데이터 체크섬 계산"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _get_query_hash(self, query: str) -> str:
        """쿼리 해시 생성"""
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    @contextlib.asynccontextmanager
    async def get_session(self, 
                         shard_type: DatabaseShardType = DatabaseShardType.PRIMARY,
                         isolation_level: Optional[TransactionIsolationLevel] = None) -> AsyncGenerator[AsyncSession, None]:
        """Ultra 비동기 세션 컨텍스트 매니저"""
        if not self._initialized:
            await self.initialize()
        
        session_factory = self.session_factories[shard_type]
        
        async with session_factory() as session:
            try:
                # 약한 참조로 세션 추적
                self.active_sessions.add(session)
                
                # 격리 수준 설정
                if isolation_level and shard_type == DatabaseShardType.PRIMARY:
                    await session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.value}"))
                
                yield session
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                logger.error(f"세션 오류 ({shard_type.value}): {e}")
                raise
            finally:
                # 세션 정리는 WeakSet이 자동으로 처리
                pass
    
    @contextlib.asynccontextmanager
    async def get_connection(self, 
                           shard_type: DatabaseShardType = DatabaseShardType.PRIMARY) -> AsyncGenerator[asyncpg.Connection, None]:
        """Ultra 원시 연결 컨텍스트 매니저"""
        if not self._initialized:
            await self.initialize()
        
        pool = self.connection_pools.get(shard_type)
        
        if pool:  # PostgreSQL 환경
            async with pool.acquire() as connection:
                yield connection
        else:  # SQLite 환경 - 세션 사용
            async with self.get_session(shard_type) as session:
                connection = await session.connection()
                yield connection.connection
    
    async def execute_query_with_metrics(self, 
                                       query: str, 
                                       params: Optional[Dict[str, Any]] = None,
                                       shard_type: DatabaseShardType = DatabaseShardType.REPLICA) -> List[Dict[str, Any]]:
        """메트릭 수집이 포함된 쿼리 실행"""
        start_time = asyncio.get_event_loop().time()
        query_hash = self._get_query_hash(query)
        params = params or {}
        
        try:
            if settings.is_development or settings.is_test:  # SQLite 환경
                async with self.get_session(shard_type) as session:
                    result = await session.execute(text(query), params)
                    rows = [dict(row._mapping) for row in result.fetchall()]
            else:  # PostgreSQL 환경
                async with self.get_connection(shard_type) as conn:
                    if params:
                        rows = await conn.fetch(query, *params.values())
                    else:
                        rows = await conn.fetch(query)
                    rows = [dict(row) for row in rows]
            
            # 메트릭 업데이트
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_query_metrics(query_hash, execution_time, False)
            
            return rows
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_query_metrics(query_hash, execution_time, True)
            logger.error(f"쿼리 실행 실패: {e}")
            raise
    
    def _update_query_metrics(self, query_hash: str, execution_time: float, error: bool) -> None:
        """쿼리 메트릭 업데이트"""
        with self._lock:
            if query_hash not in self.query_metrics:
                self.query_metrics[query_hash] = QueryMetrics(query_hash=query_hash)
            
            self.query_metrics[query_hash].update(execution_time, error)
    
    async def add_batch_operation(self, operation: BatchOperation) -> None:
        """배치 작업 추가"""
        try:
            await self.batch_queue.put(operation)
        except asyncio.QueueFull:
            logger.warning("배치 큐가 가득참, 즉시 처리")
            await self._process_batch_operations([operation])
    
    async def _batch_insert(self, table_name: str, data_list: List[Dict[str, Any]]) -> None:
        """배치 삽입"""
        if not data_list:
            return
        
        # 데이터 전처리
        for data in data_list:
            if table_name == "stock_data" and "code" in data:
                data["shard_key"] = self._get_shard_key(data["code"])
            elif table_name == "price_history" and "date" in data:
                data["partition_key"] = self._get_partition_key(data["date"])
            elif table_name == "analysis_results" and "result_data" in data:
                data["checksum"] = self._calculate_checksum(str(data["result_data"]))
        
        async with self.get_session(DatabaseShardType.PRIMARY) as session:
            if settings.is_development or settings.is_test:  # SQLite
                # SQLite용 배치 삽입
                table = Base.metadata.tables[table_name]
                stmt = sqlite_insert(table).values(data_list)
                stmt = stmt.on_conflict_do_nothing()
                await session.execute(stmt)
            else:  # PostgreSQL
                # PostgreSQL용 고성능 배치 삽입
                async with self.get_connection(DatabaseShardType.PRIMARY) as conn:
                    columns = list(data_list[0].keys())
                    values = [list(data.values()) for data in data_list]
                    
                    await conn.copy_records_to_table(
                        table_name,
                        records=values,
                        columns=columns,
                        timeout=60
                    )
    
    async def get_stock_data_ultra(self, 
                                 code: Optional[str] = None,
                                 market: Optional[str] = None,
                                 limit: int = 100,
                                 use_cache: bool = True) -> List[Dict[str, Any]]:
        """Ultra 주식 데이터 조회 (멀티레벨 캐싱)"""
        # 캐시 키 생성
        cache_key = f"stock_data:{code or 'all'}:{market or 'all'}:{limit}"
        
        if use_cache:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
        
        # 쿼리 구성
        conditions = []
        params = {}
        
        if code:
            conditions.append("code = :code")
            params["code"] = code
        
        if market:
            conditions.append("market = :market")
            params["market"] = market
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT code, name, market, sector, price, change_rate, volume, 
               market_cap, created_at, updated_at
        FROM stock_data 
        WHERE {where_clause}
        ORDER BY created_at DESC 
        LIMIT :limit
        """
        
        params["limit"] = limit
        
        result = await self.execute_query_with_metrics(
            query, params, DatabaseShardType.REPLICA
        )
        
        # 캐시 저장
        if use_cache and result:
            await self.cache_manager.set(cache_key, result, ttl=300)
        
        return result
    
    async def bulk_insert_stock_data_ultra(self, stock_data_list: List[Dict[str, Any]]) -> None:
        """Ultra 주식 데이터 배치 삽입"""
        if not stock_data_list:
            return
        
        # 현재 시간 추가
        current_time = datetime.utcnow()
        for data in stock_data_list:
            data.setdefault("created_at", current_time)
            data.setdefault("updated_at", current_time)
        
        batch_op = BatchOperation(
            operation_type="insert",
            table_name="stock_data",
            data=stock_data_list,
            batch_size=1000
        )
        
        await self.add_batch_operation(batch_op)
        
        # 관련 캐시 무효화
        await self.cache_manager.delete_pattern("stock_data:*")
    
    async def get_price_history_ultra(self, 
                                    stock_code: str,
                                    days: int = 100,
                                    use_cache: bool = True) -> List[Dict[str, Any]]:
        """Ultra 주가 이력 조회"""
        cache_key = f"price_history:{stock_code}:{days}"
        
        if use_cache:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
        
        query = """
        SELECT stock_code, date, open_price, high_price, low_price, 
               close_price, volume, adjusted_close
        FROM price_history 
        WHERE stock_code = :stock_code
        ORDER BY date DESC 
        LIMIT :limit
        """
        
        params = {"stock_code": stock_code, "limit": days}
        
        result = await self.execute_query_with_metrics(
            query, params, DatabaseShardType.REPLICA
        )
        
        if use_cache and result:
            await self.cache_manager.set(cache_key, result, ttl=600)
        
        return result
    
    async def save_analysis_result_ultra(self, 
                                       stock_code: str,
                                       analysis_type: str,
                                       result_data: Dict[str, Any],
                                       confidence_score: Optional[float] = None,
                                       expires_in_hours: int = 24) -> None:
        """Ultra 분석 결과 저장"""
        current_time = datetime.utcnow()
        expires_at = current_time + timedelta(hours=expires_in_hours)
        result_json = json.dumps(result_data, ensure_ascii=False)
        
        data = {
            "stock_code": stock_code,
            "analysis_type": analysis_type,
            "result_data": result_json,
            "confidence_score": confidence_score,
            "created_at": current_time,
            "expires_at": expires_at,
            "version": 1,
            "checksum": self._calculate_checksum(result_json)
        }
        
        batch_op = BatchOperation(
            operation_type="insert",
            table_name="analysis_results",
            data=[data],
            on_conflict="update"
        )
        
        await self.add_batch_operation(batch_op)
        
        # 캐시에도 저장
        cache_key = f"analysis:{stock_code}:{analysis_type}"
        await self.cache_manager.set(cache_key, result_data, ttl=expires_in_hours * 3600)
    
    async def get_analysis_result_ultra(self, 
                                      stock_code: str,
                                      analysis_type: str,
                                      use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Ultra 분석 결과 조회"""
        cache_key = f"analysis:{stock_code}:{analysis_type}"
        
        if use_cache:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
        
        query = """
        SELECT result_data, confidence_score, created_at, expires_at
        FROM analysis_results 
        WHERE stock_code = :stock_code 
          AND analysis_type = :analysis_type
          AND (expires_at IS NULL OR expires_at > :current_time)
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        params = {
            "stock_code": stock_code,
            "analysis_type": analysis_type,
            "current_time": datetime.utcnow()
        }
        
        result = await self.execute_query_with_metrics(
            query, params, DatabaseShardType.REPLICA
        )
        
        if result:
            data = result[0]
            parsed_result = {
                "result_data": json.loads(data["result_data"]),
                "confidence_score": data["confidence_score"],
                "created_at": data["created_at"],
                "expires_at": data["expires_at"]
            }
            
            # 캐시 저장
            if use_cache:
                ttl = 1800  # 30분
                if data["expires_at"]:
                    remaining_seconds = (data["expires_at"] - datetime.utcnow()).total_seconds()
                    ttl = min(ttl, max(60, int(remaining_seconds)))
                
                await self.cache_manager.set(cache_key, parsed_result, ttl=ttl)
            
            return parsed_result
        
        return None
    
    async def cleanup_expired_data_ultra(self) -> Dict[str, int]:
        """Ultra 만료 데이터 정리"""
        current_time = datetime.utcnow()
        cleanup_results = {}
        
        # 만료된 분석 결과 삭제
        query = "DELETE FROM analysis_results WHERE expires_at < :current_time"
        
        async with self.get_session(DatabaseShardType.PRIMARY) as session:
            result = await session.execute(text(query), {"current_time": current_time})
            cleanup_results["analysis_results"] = result.rowcount
        
        # 오래된 주가 이력 정리 (6개월 이상)
        old_date = current_time - timedelta(days=180)
        query = "DELETE FROM price_history WHERE created_at < :old_date"
        
        async with self.get_session(DatabaseShardType.PRIMARY) as session:
            result = await session.execute(text(query), {"old_date": old_date})
            cleanup_results["price_history"] = result.rowcount
        
        # 캐시 정리
        await self.cache_manager.cleanup()
        
        logger.info(f"데이터 정리 완료: {cleanup_results}")
        return cleanup_results
    
    async def get_database_stats_ultra(self) -> Dict[str, Any]:
        """Ultra 데이터베이스 통계"""
        stats = {
            "connection_pools": {},
            "query_metrics": {},
            "table_stats": {},
            "cache_stats": await self.cache_manager.get_stats(),
            "active_sessions": len(self.active_sessions)
        }
        
        # 연결 풀 통계
        for shard_type, pool_stats in self.connection_stats.items():
            stats["connection_pools"][shard_type.value] = {
                "total_connections": pool_stats.total_connections,
                "active_connections": pool_stats.active_connections,
                "idle_connections": pool_stats.idle_connections,
                "pool_size": pool_stats.pool_size,
                "max_overflow": pool_stats.max_overflow
            }
        
        # 쿼리 메트릭 (상위 10개)
        sorted_metrics = sorted(
            self.query_metrics.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )[:10]
        
        for query_hash, metrics in sorted_metrics:
            stats["query_metrics"][query_hash] = {
                "execution_count": metrics.execution_count,
                "avg_time": metrics.avg_time,
                "total_time": metrics.total_time,
                "error_count": metrics.error_count
            }
        
        # 테이블 통계
        table_queries = {
            "stock_data": "SELECT COUNT(*) as count FROM stock_data",
            "price_history": "SELECT COUNT(*) as count FROM price_history",
            "analysis_results": "SELECT COUNT(*) as count FROM analysis_results"
        }
        
        for table_name, query in table_queries.items():
            try:
                result = await self.execute_query_with_metrics(
                    query, {}, DatabaseShardType.REPLICA
                )
                stats["table_stats"][table_name] = result[0]["count"] if result else 0
            except Exception as e:
                stats["table_stats"][table_name] = f"Error: {e}"
        
        return stats
    
    async def _update_connection_stats(self) -> None:
        """연결 통계 업데이트"""
        for shard_type, engine in self.engines.items():
            if hasattr(engine.pool, 'size'):
                pool = engine.pool
                stats = self.connection_stats[shard_type]
                stats.total_connections = pool.size()
                stats.active_connections = pool.checkedout()
                stats.idle_connections = stats.total_connections - stats.active_connections
    
    async def _cleanup_old_metrics(self) -> None:
        """오래된 메트릭 정리"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            to_remove = []
            for query_hash, metrics in self.query_metrics.items():
                if metrics.last_executed and metrics.last_executed < cutoff_time:
                    to_remove.append(query_hash)
            
            for query_hash in to_remove:
                del self.query_metrics[query_hash]
    
    async def _cleanup_on_error(self) -> None:
        """오류 발생 시 정리"""
        try:
            if self.batch_processor_task:
                self.batch_processor_task.cancel()
            
            for pool in self.connection_pools.values():
                if pool:
                    await pool.close()
            
            for engine in self.engines.values():
                if engine:
                    await engine.dispose()
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")
    
    async def close(self) -> None:
        """Ultra 데이터베이스 매니저 종료"""
        logger.info("Ultra 데이터베이스 매니저 종료 시작")
        
        try:
            # 배치 프로세서 중지
            if self.batch_processor_task:
                self.batch_processor_task.cancel()
                try:
                    await self.batch_processor_task
                except asyncio.CancelledError:
                    pass
            
            # 연결 풀 종료
            for pool in self.connection_pools.values():
                if pool:
                    await pool.close()
            
            # 엔진 종료
            for engine in self.engines.values():
                if engine:
                    await engine.dispose()
            
            # 캐시 매니저 종료
            await self.cache_manager.close()
            
            # 백그라운드 실행자 종료
            self.background_executor.shutdown(wait=True)
            
            self._initialized = False
            logger.info("Ultra 데이터베이스 매니저 종료 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 매니저 종료 중 오류: {e}")


# 전역 인스턴스
ultra_db_manager = UltraDatabaseManager()


async def initialize_database() -> None:
    """데이터베이스 초기화"""
    await ultra_db_manager.initialize()


async def cleanup_database() -> None:
    """데이터베이스 정리"""
    await ultra_db_manager.close()


# 하위 호환성을 위한 별칭
DatabaseManager = UltraDatabaseManager
db_manager = ultra_db_manager 