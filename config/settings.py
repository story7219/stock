#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra HTS 시스템 환경 설정 관리 v5.0
고성능 비동기 처리, 멀티레벨 캐싱, 커넥션풀링 최적화

이 모듈은 환경변수와 설정값들을 중앙에서 관리합니다.
Pydantic을 사용하여 타입 안전성과 검증을 보장합니다.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from pydantic import Field, validator, field_validator, BaseModel
from pydantic_settings import BaseSettings

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent


class Environment(str, Enum):
    """실행 환경 열거형"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class CacheLevel(str, Enum):
    """캐시 레벨 열거형"""
    L1_MEMORY = "memory"
    L2_REDIS = "redis"
    L3_DISK = "disk"


@dataclass(frozen=True)
class PerformanceProfile:
    """성능 프로필 설정"""
    max_workers: int
    chunk_size: int
    memory_limit_mb: int
    connection_pool_size: int
    cache_size: int
    
    @classmethod
    def high_performance(cls) -> 'PerformanceProfile':
        """고성능 프로필"""
        return cls(
            max_workers=16,
            chunk_size=500,
            memory_limit_mb=4096,
            connection_pool_size=50,
            cache_size=50000
        )
    
    @classmethod
    def balanced(cls) -> 'PerformanceProfile':
        """균형 프로필"""
        return cls(
            max_workers=8,
            chunk_size=200,
            memory_limit_mb=2048,
            connection_pool_size=20,
            cache_size=20000
        )
    
    @classmethod
    def low_resource(cls) -> 'PerformanceProfile':
        """저사양 프로필"""
        return cls(
            max_workers=4,
            chunk_size=100,
            memory_limit_mb=1024,
            connection_pool_size=10,
            cache_size=5000
        )


class DatabaseSettings(BaseSettings):
    """🗄️ 데이터베이스 설정 - 고성능 커넥션 풀링"""
    
    # PostgreSQL 설정 (운영환경)
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="hts_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="hts_db", env="POSTGRES_DB")
    
    # SQLite 설정 (개발환경)
    sqlite_path: str = Field(default="data/hts.db", env="SQLITE_PATH")
    
    # 고성능 연결 풀 설정
    pool_size: int = Field(default=30, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=50, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")  # 1시간
    pool_pre_ping: bool = Field(default=True, env="DB_POOL_PRE_PING")
    
    # 비동기 설정
    async_pool_size: int = Field(default=20, env="DB_ASYNC_POOL_SIZE")
    async_max_overflow: int = Field(default=30, env="DB_ASYNC_MAX_OVERFLOW")
    
    # 쿼리 최적화
    query_timeout: int = Field(default=30, env="DB_QUERY_TIMEOUT")
    enable_query_cache: bool = Field(default=True, env="DB_ENABLE_QUERY_CACHE")
    
    @property
    def postgres_url(self) -> str:
        """PostgreSQL 비동기 연결 URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def sqlite_url(self) -> str:
        """SQLite 비동기 연결 URL"""
        return f"sqlite+aiosqlite:///{self.sqlite_path}"
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class RedisSettings(BaseSettings):
    """⚡ Redis 설정 - 고성능 분산 캐시"""
    
    # Redis 활성화 여부
    enable_redis: bool = Field(default=True, env="REDIS_ENABLE")
    
    # Redis 클러스터 설정
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # 고성능 연결 풀 설정
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    min_connections: int = Field(default=10, env="REDIS_MIN_CONNECTIONS")
    connection_pool_kwargs: Dict[str, Any] = Field(default_factory=lambda: {
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "socket_keepalive_options": {},
        "health_check_interval": 30,
    })
    
    # TTL 설정 (계층별)
    l2_cache_ttl: int = Field(default=3600, env="REDIS_L2_TTL")  # 1시간
    session_ttl: int = Field(default=86400, env="REDIS_SESSION_TTL")  # 24시간
    hot_data_ttl: int = Field(default=300, env="REDIS_HOT_DATA_TTL")  # 5분
    
    # 성능 최적화
    pipeline_size: int = Field(default=1000, env="REDIS_PIPELINE_SIZE")
    enable_clustering: bool = Field(default=False, env="REDIS_ENABLE_CLUSTERING")
    
    @property
    def url(self) -> str:
        """Redis URL 생성"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"
        case_sensitive = False


class PerformanceSettings(BaseSettings):
    """🚀 성능 최적화 설정 - 비동기 고속 병렬처리"""
    
    # 비동기 처리 설정
    max_workers: int = Field(default=16, env="MAX_WORKERS")
    max_concurrent_tasks: int = Field(default=100, env="MAX_CONCURRENT_TASKS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")  # 5분
    
    # 배치 처리 최적화
    batch_size: int = Field(default=500, env="BATCH_SIZE")
    chunk_size: int = Field(default=200, env="CHUNK_SIZE")
    parallel_chunks: int = Field(default=8, env="PARALLEL_CHUNKS")
    
    # 메모리 관리 최적화
    memory_limit_mb: int = Field(default=4096, env="MEMORY_LIMIT_MB")
    gc_threshold: int = Field(default=5000, env="GC_THRESHOLD")
    enable_memory_profiling: bool = Field(default=False, env="ENABLE_MEMORY_PROFILING")
    
    # 네트워크 최적화
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    read_timeout: int = Field(default=60, env="READ_TIMEOUT")
    max_retries: int = Field(default=5, env="MAX_RETRIES")
    retry_backoff: float = Field(default=1.5, env="RETRY_BACKOFF")
    
    # 커넥션 풀링
    http_pool_connections: int = Field(default=20, env="HTTP_POOL_CONNECTIONS")
    http_pool_maxsize: int = Field(default=50, env="HTTP_POOL_MAXSIZE")
    http_pool_block: bool = Field(default=True, env="HTTP_POOL_BLOCK")
    
    @property
    def performance_profile(self) -> PerformanceProfile:
        """현재 성능 프로필 반환"""
        if self.memory_limit_mb >= 4096:
            return PerformanceProfile.high_performance()
        elif self.memory_limit_mb >= 2048:
            return PerformanceProfile.balanced()
        else:
            return PerformanceProfile.low_resource()
    
    class Config:
        env_prefix = "PERF_"
        case_sensitive = False


class CacheSettings(BaseSettings):
    """💾 멀티레벨 캐싱 설정 - L1/L2/L3 캐시 최적화"""
    
    # L1 캐시 (메모리) - 초고속
    l1_enabled: bool = Field(default=True, env="CACHE_L1_ENABLED")
    l1_max_size: int = Field(default=50000, env="CACHE_L1_MAX_SIZE")
    l1_ttl: int = Field(default=300, env="CACHE_L1_TTL")  # 5분
    l1_eviction_policy: str = Field(default="lru", env="CACHE_L1_EVICTION")
    
    # L2 캐시 (Redis) - 고속 분산
    l2_enabled: bool = Field(default=True, env="CACHE_L2_ENABLED")
    l2_max_size: int = Field(default=1000000, env="CACHE_L2_MAX_SIZE")
    l2_ttl: int = Field(default=3600, env="CACHE_L2_TTL")  # 1시간
    l2_compression: bool = Field(default=True, env="CACHE_L2_COMPRESSION")
    
    # L3 캐시 (디스크) - 대용량 영구
    l3_enabled: bool = Field(default=True, env="CACHE_L3_ENABLED")
    l3_directory: str = Field(default="cache", env="CACHE_L3_DIR")
    l3_max_size_gb: int = Field(default=10, env="CACHE_L3_MAX_SIZE_GB")
    l3_ttl: int = Field(default=86400, env="CACHE_L3_TTL")  # 24시간
    
    # 캐시 전략
    write_through: bool = Field(default=True, env="CACHE_WRITE_THROUGH")
    write_back: bool = Field(default=False, env="CACHE_WRITE_BACK")
    read_through: bool = Field(default=True, env="CACHE_READ_THROUGH")
    
    # 성능 최적화
    prefetch_enabled: bool = Field(default=True, env="CACHE_PREFETCH_ENABLED")
    prefetch_size: int = Field(default=100, env="CACHE_PREFETCH_SIZE")
    async_write: bool = Field(default=True, env="CACHE_ASYNC_WRITE")
    
    class Config:
        env_prefix = "CACHE_"
        case_sensitive = False


class APISettings(BaseSettings):
    """🔗 외부 API 설정 - 고성능 비동기 호출"""
    
    # Gemini AI 설정
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=8192, env="GEMINI_MAX_TOKENS")
    gemini_timeout: int = Field(default=60, env="GEMINI_TIMEOUT")
    
    # 금융 데이터 API
    alpha_vantage_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_KEY")
    financial_modeling_prep_key: Optional[str] = Field(default=None, env="FMP_KEY")
    yahoo_finance_timeout: int = Field(default=30, env="YAHOO_TIMEOUT")
    
    # 비동기 요청 최적화
    max_concurrent_requests: int = Field(default=20, env="API_MAX_CONCURRENT")
    requests_per_minute: int = Field(default=300, env="API_REQUESTS_PER_MINUTE")
    burst_limit: int = Field(default=50, env="API_BURST_LIMIT")
    
    # 재시도 정책
    max_retries: int = Field(default=3, env="API_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="API_RETRY_DELAY")
    exponential_backoff: bool = Field(default=True, env="API_EXPONENTIAL_BACKOFF")
    
    # 응답 캐싱
    cache_responses: bool = Field(default=True, env="API_CACHE_RESPONSES")
    cache_ttl: int = Field(default=300, env="API_CACHE_TTL")  # 5분
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """📝 로깅 설정 - 구조화된 고성능 로깅"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # 비동기 로깅
    async_logging: bool = Field(default=True, env="LOG_ASYNC")
    buffer_size: int = Field(default=10000, env="LOG_BUFFER_SIZE")
    flush_interval: float = Field(default=5.0, env="LOG_FLUSH_INTERVAL")
    
    # 파일 로깅
    file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    file_path: str = Field(default="logs/hts.log", env="LOG_FILE_PATH")
    file_max_size: int = Field(default=52428800, env="LOG_FILE_MAX_SIZE")  # 50MB
    file_backup_count: int = Field(default=10, env="LOG_FILE_BACKUP_COUNT")
    
    # 구조화된 로깅
    structured: bool = Field(default=True, env="LOG_STRUCTURED")
    json_format: bool = Field(default=True, env="LOG_JSON_FORMAT")
    include_extra: bool = Field(default=True, env="LOG_INCLUDE_EXTRA")
    
    # 성능 로깅
    performance_logging: bool = Field(default=True, env="LOG_PERFORMANCE")
    slow_query_threshold: float = Field(default=1.0, env="LOG_SLOW_QUERY_THRESHOLD")
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'로그 레벨은 {valid_levels} 중 하나여야 합니다')
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class UISettings(BaseSettings):
    """🎨 UI 설정 - 반응형 고성능 인터페이스"""
    
    # 윈도우 설정
    window_width: int = Field(default=1920, env="WINDOW_WIDTH")
    window_height: int = Field(default=1080, env="WINDOW_HEIGHT")
    window_title: str = Field(default="🚀 Ultra HTS v5.0 - Professional Trading System", env="WINDOW_TITLE")
    
    # 테마 설정
    theme: str = Field(default="dark", env="UI_THEME")
    font_family: str = Field(default="맑은 고딕", env="UI_FONT_FAMILY")
    font_size: int = Field(default=11, env="UI_FONT_SIZE")
    
    # 성능 최적화
    update_interval: float = Field(default=0.5, env="UI_UPDATE_INTERVAL")  # 500ms
    chart_update_interval: float = Field(default=1.0, env="CHART_UPDATE_INTERVAL")  # 1초
    async_ui_updates: bool = Field(default=True, env="UI_ASYNC_UPDATES")
    
    # 차트 설정
    chart_style: str = Field(default="professional", env="CHART_STYLE")
    show_volume: bool = Field(default=True, env="CHART_SHOW_VOLUME")
    chart_cache_size: int = Field(default=1000, env="CHART_CACHE_SIZE")
    
    # 반응성 설정
    debounce_delay: float = Field(default=0.3, env="UI_DEBOUNCE_DELAY")
    throttle_interval: float = Field(default=0.1, env="UI_THROTTLE_INTERVAL")
    
    class Config:
        env_prefix = "UI_"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """🔒 보안 설정 - 엔터프라이즈급 보안"""
    
    # 암호화 키
    secret_key: str = Field(default="ultra-hts-secret-key-v5", env="SECRET_KEY")
    encryption_algorithm: str = Field(default="AES-256-GCM", env="ENCRYPTION_ALGORITHM")
    
    # 세션 관리
    session_timeout: int = Field(default=7200, env="SESSION_TIMEOUT")  # 2시간
    max_concurrent_sessions: int = Field(default=10, env="MAX_CONCURRENT_SESSIONS")
    
    # API 보안
    api_rate_limit: int = Field(default=5000, env="API_RATE_LIMIT")
    api_burst_limit: int = Field(default=200, env="API_BURST_LIMIT")
    enable_api_key_rotation: bool = Field(default=True, env="ENABLE_API_KEY_ROTATION")
    
    # 감사 로깅
    audit_logging: bool = Field(default=True, env="AUDIT_LOGGING")
    audit_retention_days: int = Field(default=90, env="AUDIT_RETENTION_DAYS")
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class Settings(BaseSettings):
    """🎯 메인 설정 클래스 - 모든 설정을 통합 관리"""
    
    # 환경 설정
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # 프로젝트 정보
    project_name: str = Field(default="Ultra HTS", env="PROJECT_NAME")
    version: str = Field(default="5.0.0", env="VERSION")
    description: str = Field(default="🚀 Ultra High-Performance Trading System", env="DESCRIPTION")
    
    # 하위 설정들
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    performance: PerformanceSettings = PerformanceSettings()
    cache: CacheSettings = CacheSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    ui: UISettings = UISettings()
    security: SecuritySettings = SecuritySettings()
    
    # 디렉토리 설정
    data_dir: str = Field(default="data", env="DATA_DIR")
    cache_dir: str = Field(default="cache", env="CACHE_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        self._optimize_settings()
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.data_dir, self.cache_dir, self.logs_dir, self.temp_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _optimize_settings(self):
        """환경에 따른 설정 최적화"""
        if self.is_production:
            # 프로덕션 최적화
            self.debug = False
            self.logging.level = "WARNING"
            self.performance.max_workers = 32
            self.cache.l1_max_size = 100000
        elif self.is_development:
            # 개발 환경 최적화
            self.debug = True
            self.logging.level = "DEBUG"
            self.performance.max_workers = 8
    
    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """테스트 환경 여부"""
        return self.environment == Environment.TESTING
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == Environment.PRODUCTION
    
    def get_performance_config(self) -> Dict[str, Any]:
        """성능 설정 딕셔너리 반환"""
        return {
            "max_workers": self.performance.max_workers,
            "max_concurrent_tasks": self.performance.max_concurrent_tasks,
            "batch_size": self.performance.batch_size,
            "chunk_size": self.performance.chunk_size,
            "memory_limit_mb": self.performance.memory_limit_mb,
            "cache_config": {
                "l1_size": self.cache.l1_max_size,
                "l2_enabled": self.cache.l2_enabled,
                "l3_enabled": self.cache.l3_enabled,
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """데이터베이스 설정 딕셔너리 반환"""
        return {
            "url": self.database.postgres_url if self.is_production else self.database.sqlite_url,
            "pool_size": self.database.pool_size,
            "max_overflow": self.database.max_overflow,
            "pool_timeout": self.database.pool_timeout,
            "pool_recycle": self.database.pool_recycle,
            "pool_pre_ping": self.database.pool_pre_ping,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # 추가 필드 허용
        
        # 성능 최적화
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True


# 전역 설정 인스턴스
settings = Settings()

# 편의를 위한 단축 접근자들
db_settings = settings.database
redis_settings = settings.redis
perf_settings = settings.performance
api_settings = settings.api
log_settings = settings.logging
ui_settings = settings.ui
security_settings = settings.security
cache_settings = settings.cache 