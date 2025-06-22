"""
데이터베이스 매니저 - 비동기 커넥션 풀링 및 트랜잭션 관리
"""
import asyncio
import contextlib
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from datetime import datetime
import json

import asyncpg
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import settings
from core.cache_manager import cached

logger = structlog.get_logger(__name__)

Base = declarative_base()


class StockData(Base):
    """주식 데이터 테이블"""
    __tablename__ = "stock_data"
    
    code = Column(String(20), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    market = Column(String(20), nullable=False)
    sector = Column(String(50))
    price = Column(Float, nullable=False)
    change_rate = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    market_cap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_stock_code_date', 'code', 'created_at'),
        Index('idx_stock_market', 'market'),
        Index('idx_stock_sector', 'sector'),
    )


class PriceHistory(Base):
    """주가 이력 테이블"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_price_stock_date', 'stock_code', 'date'),
    )


class AnalysisResult(Base):
    """분석 결과 테이블"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)
    # SQLite 호환성을 위해 Text 타입 사용 (JSON 문자열 저장)
    result_data = Column(Text, nullable=False)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_analysis_stock_type', 'stock_code', 'analysis_type'),
        Index('idx_analysis_created', 'created_at'),
    )


class DatabaseManager:
    """데이터베이스 매니저 - 비동기 커넥션 풀링 및 세션 관리"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._connection_pool = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """데이터베이스 초기화"""
        if self._initialized:
            return
        
        try:
            # SQLAlchemy 비동기 엔진 생성 (개발/테스트 환경에서는 SQLite 사용)
            db_url = settings.database.sqlite_url if settings.is_development or settings.is_test else settings.database.postgres_url
            
            connect_args = {"check_same_thread": False} if settings.is_development or settings.is_test else {}
            
            self.engine = create_async_engine(
                db_url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1시간마다 연결 재생성
                echo=settings.debug,
                future=True,
                connect_args=connect_args
            )
            
            # 세션 팩토리 생성
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # 테이블 생성
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # PostgreSQL 환경에서만 원시 연결 풀 생성
            if not (settings.is_development or settings.is_test):
                self._connection_pool = await asyncpg.create_pool(
                    host=settings.database.postgres_host,
                    port=settings.database.postgres_port,
                    user=settings.database.postgres_user,
                    password=settings.database.postgres_password,
                    database=settings.database.postgres_db,
                    min_size=10,
                    max_size=settings.database.pool_size,
                    command_timeout=60
                )
            
            self._initialized = True
            logger.info("데이터베이스 매니저 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """비동기 세션 컨텍스트 매니저"""
        if not self._initialized:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    @contextlib.asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """원시 PostgreSQL 연결 컨텍스트 매니저 (고성능 쿼리용)"""
        if not self._connection_pool:
            await self.initialize()
        
        if self._connection_pool:  # PostgreSQL 환경
            async with self._connection_pool.acquire() as connection:
                yield connection
        else:  # SQLite 환경 - 세션 사용
            async with self.get_session() as session:
                yield session.connection()
    
    async def execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """원시 SQL 쿼리 실행"""
        if settings.is_development or settings.is_test:  # SQLite 환경
            async with self.get_session() as session:
                result = await session.execute(text(query), params)
                return [dict(row._mapping) for row in result.fetchall()]
        else:  # PostgreSQL 환경
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *params.values())
                return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, *args) -> str:
        """원시 SQL 명령 실행 (INSERT, UPDATE, DELETE)"""
        if settings.is_development or settings.is_test:  # SQLite 환경
            async with self.get_session() as session:
                result = await session.execute(text(command), args)
                await session.commit()
                return f"Affected rows: {result.rowcount}"
        else:  # PostgreSQL 환경
            async with self.get_connection() as conn:
                return await conn.execute(command, *args)
    
    @cached(ttl=300, key_prefix="db_stock")
    async def get_stock_data(self, code: Optional[str] = None, 
                           market: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """주식 데이터 조회 (캐시됨)"""
        params = {}
        if settings.is_development or settings.is_test:  # SQLite 환경
            query = """
                SELECT code, name, market, sector, price, change_rate, volume, market_cap,
                       created_at, updated_at
                FROM stock_data
                WHERE 1=1
            """
            if code:
                query += " AND code = :code"
                params["code"] = code
            
            if market:
                query += " AND market = :market"
                params["market"] = market
            
            query += " ORDER BY updated_at DESC LIMIT :limit"
            params["limit"] = limit
            
            return await self.execute_query(query, params)
        else:  # PostgreSQL 환경
            query = """
                SELECT code, name, market, sector, price, change_rate, volume, market_cap,
                       created_at, updated_at
                FROM stock_data
                WHERE 1=1
            """
            pg_params = []
            if code:
                query += f" AND code = ${len(pg_params) + 1}"
                pg_params.append(code)
            
            if market:
                query += f" AND market = ${len(pg_params) + 1}"
                pg_params.append(market)
            
            query += f" ORDER BY updated_at DESC LIMIT ${len(pg_params) + 1}"
            pg_params.append(limit)
            
            # execute_query가 딕셔너리를 받도록 수정되었으므로, 여기서는 직접 실행
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *pg_params)
                return [dict(row) for row in rows]
    
    async def bulk_insert_stock_data(self, stock_data_list: List[Dict[str, Any]]) -> None:
        """주식 데이터 대량 삽입 (고성능)"""
        if not stock_data_list:
            return
        
        if settings.is_development or settings.is_test:  # SQLite 환경
            async with self.get_session() as session:
                for data in stock_data_list:
                    # code를 기준으로 이미 존재하는지 확인
                    existing = await session.get(StockData, data['code'])
                    if not existing:
                        session.add(StockData(**data))
                await session.commit()
        else:  # PostgreSQL 환경
            async with self.get_connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO stock_data (code, name, market, sector, price, change_rate, volume, market_cap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (code) DO UPDATE SET
                        name = EXCLUDED.name,
                        market = EXCLUDED.market,
                        sector = EXCLUDED.sector,
                        price = EXCLUDED.price,
                        change_rate = EXCLUDED.change_rate,
                        volume = EXCLUDED.volume,
                        market_cap = EXCLUDED.market_cap,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    [
                        (
                            data['code'], data['name'], data['market'], data.get('sector'),
                            data['price'], data['change_rate'], data['volume'], data.get('market_cap')
                        )
                        for data in stock_data_list
                    ]
                )
    
    @cached(ttl=600, key_prefix="db_price_history")
    async def get_price_history(self, stock_code: str, 
                              days: int = 100) -> List[Dict[str, Any]]:
        """주가 이력 조회 (캐시됨)"""
        if settings.is_development or settings.is_test:  # SQLite 환경
            query = """
                SELECT stock_code, date, open_price, high_price, low_price, 
                       close_price, volume, adjusted_close
                FROM price_history
                WHERE stock_code = :stock_code
                ORDER BY date DESC
                LIMIT :days
            """
            return await self.execute_query(query, {"stock_code": stock_code, "days": days})
        else:  # PostgreSQL 환경
            query = """
                SELECT stock_code, date, open_price, high_price, low_price, 
                       close_price, volume, adjusted_close
                FROM price_history
                WHERE stock_code = $1
                ORDER BY date DESC
                LIMIT $2
            """
            # PostgreSQL은 위치 기반 파라미터를 사용하므로 직접 실행
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, stock_code, days)
                return [dict(row) for row in rows]
    
    async def bulk_insert_price_history(self, price_data_list: List[Dict[str, Any]]) -> None:
        """주가 이력 대량 삽입"""
        if not price_data_list:
            return
        
        if settings.is_development or settings.is_test:  # SQLite 환경
            async with self.get_session() as session:
                session.add_all([PriceHistory(**data) for data in price_data_list])
                await session.commit()
        else:  # PostgreSQL 환경
            async with self.get_connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO price_history 
                    (stock_code, date, open_price, high_price, low_price, close_price, volume, adjusted_close)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (stock_code, date) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        adjusted_close = EXCLUDED.adjusted_close
                    """,
                    [
                        (
                            data['stock_code'], data['date'], data['open_price'],
                            data['high_price'], data['low_price'], data['close_price'],
                            data['volume'], data.get('adjusted_close')
                        )
                        for data in price_data_list
                    ]
                )
    
    async def save_analysis_result(self, stock_code: str, analysis_type: str,
                                 result_data: Dict[str, Any], 
                                 confidence_score: Optional[float] = None,
                                 expires_in_hours: int = 24) -> None:
        """분석 결과 저장"""
        expires_at = datetime.utcnow().replace(hour=datetime.utcnow().hour + expires_in_hours)
        
        # SQLite 환경에서는 JSON 직렬화 필요
        if settings.is_development or settings.is_test:
            serialized_data = json.dumps(result_data, ensure_ascii=False, default=str)
        else:
            serialized_data = result_data
        
        async with self.get_session() as session:
            analysis = AnalysisResult(
                stock_code=stock_code,
                analysis_type=analysis_type,
                result_data=serialized_data,
                confidence_score=confidence_score,
                expires_at=expires_at
            )
            session.add(analysis)
    
    @cached(ttl=1800, key_prefix="db_analysis")
    async def get_analysis_result(self, stock_code: str, 
                                analysis_type: str) -> Optional[Dict[str, Any]]:
        """분석 결과 조회 (캐시됨)"""
        if settings.is_development or settings.is_test:  # SQLite 환경
            query = """
                SELECT result_data, confidence_score, created_at
                FROM analysis_results
                WHERE stock_code = ? AND analysis_type = ?
                  AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY created_at DESC
                LIMIT 1
            """
        else:  # PostgreSQL 환경
            query = """
                SELECT result_data, confidence_score, created_at
                FROM analysis_results
                WHERE stock_code = $1 AND analysis_type = $2
                  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY created_at DESC
                LIMIT 1
            """
        results = await self.execute_query(query, {"stock_code": stock_code, "analysis_type": analysis_type})
        
        if results:
            result = results[0]
            # SQLite 환경에서는 JSON 역직렬화 필요
            if (settings.is_development or settings.is_test) and isinstance(result.get('result_data'), str):
                try:
                    result['result_data'] = json.loads(result['result_data'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"JSON 역직렬화 실패: {result.get('result_data')}")
                    result['result_data'] = {}
            return result
        
        return None
    
    async def cleanup_expired_data(self) -> None:
        """만료된 데이터 정리"""
        if settings.is_development or settings.is_test:  # SQLite 환경
            # 만료된 분석 결과 삭제
            await self.execute_command(
                "DELETE FROM analysis_results WHERE expires_at < datetime('now')"
            )
            
            # 오래된 주가 이력 삭제 (1년 이상)
            await self.execute_command(
                "DELETE FROM price_history WHERE created_at < datetime('now', '-1 year')"
            )
        else:  # PostgreSQL 환경
            async with self.get_connection() as conn:
                # 만료된 분석 결과 삭제
                await conn.execute(
                    "DELETE FROM analysis_results WHERE expires_at < CURRENT_TIMESTAMP"
                )
                
                # 오래된 주가 이력 삭제 (1년 이상)
                await conn.execute(
                    "DELETE FROM price_history WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 year'"
                )
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보"""
        if settings.is_development or settings.is_test:  # SQLite 환경
            stats_query = """
                SELECT 
                    (SELECT COUNT(*) FROM stock_data) as stock_count,
                    (SELECT COUNT(*) FROM price_history) as price_history_count,
                    (SELECT COUNT(*) FROM analysis_results) as analysis_count,
                    'SQLite' as db_type
            """
        else:  # PostgreSQL 환경
            stats_query = """
                SELECT 
                    (SELECT COUNT(*) FROM stock_data) as stock_count,
                    (SELECT COUNT(*) FROM price_history) as price_history_count,
                    (SELECT COUNT(*) FROM analysis_results) as analysis_count,
                    (SELECT pg_size_pretty(pg_database_size(current_database()))) as db_size
            """
        results = await self.execute_query(stats_query, {})
        return results[0] if results else {}
    
    async def close(self) -> None:
        """데이터베이스 연결 종료"""
        if self._connection_pool:
            await self._connection_pool.close()
        
        if self.engine:
            await self.engine.dispose()
        
        logger.info("데이터베이스 매니저 종료 완료")


# 전역 데이터베이스 매니저 인스턴스
db_manager = DatabaseManager()


async def initialize_database() -> None:
    """데이터베이스 초기화"""
    await db_manager.initialize()


async def cleanup_database() -> None:
    """데이터베이스 정리"""
    await db_manager.close() 