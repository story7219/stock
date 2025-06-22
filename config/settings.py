#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTS 시스템 환경 설정 관리

이 모듈은 환경변수와 설정값들을 중앙에서 관리합니다.
Pydantic을 사용하여 타입 안전성과 검증을 보장합니다.
"""

import os
from typing import Optional, List
from pathlib import Path

from pydantic import Field, validator, field_validator
from pydantic_settings import BaseSettings

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

class DatabaseSettings(BaseSettings):
    """데이터베이스 설정"""
    
    # PostgreSQL 설정
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="hts_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="hts_db", env="POSTGRES_DB")
    
    # SQLite 설정 (개발용)
    sqlite_path: str = Field(default="data/hts.db", env="SQLITE_PATH")
    
    # 연결 풀 설정
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    @property
    def postgres_url(self) -> str:
        """PostgreSQL 연결 URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def sqlite_url(self) -> str:
        """SQLite 연결 URL"""
        return f"sqlite+aiosqlite:///{self.sqlite_path}"
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class RedisSettings(BaseSettings):
    """Redis 설정"""
    
    # Redis 활성화 여부
    enable_redis: bool = Field(default=False, env="REDIS_ENABLE")
    
    # Redis 연결 설정
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # 연결 풀 설정
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    # TTL 설정
    default_ttl: int = Field(default=3600, env="REDIS_DEFAULT_TTL")  # 1시간
    
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
    """성능 최적화 설정"""
    
    # 멀티프로세싱 설정
    max_workers: int = Field(default=8, env="MAX_WORKERS")
    chunk_size: int = Field(default=100, env="CHUNK_SIZE")
    
    # 메모리 관리
    memory_limit_mb: int = Field(default=2048, env="MEMORY_LIMIT_MB")
    gc_threshold: int = Field(default=1000, env="GC_THRESHOLD")
    
    # 캐시 설정
    memory_cache_size: int = Field(default=10000, env="MEMORY_CACHE_SIZE")
    disk_cache_size_mb: int = Field(default=1024, env="DISK_CACHE_SIZE_MB")
    
    # 네트워크 설정
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    read_timeout: int = Field(default=60, env="READ_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    class Config:
        env_prefix = "PERF_"
        case_sensitive = False


class APISettings(BaseSettings):
    """외부 API 설정"""
    
    # Gemini AI 설정
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=8192, env="GEMINI_MAX_TOKENS")
    
    # 금융 데이터 API
    alpha_vantage_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_KEY")
    financial_modeling_prep_key: Optional[str] = Field(default=None, env="FMP_KEY")
    
    # 요청 제한 설정
    requests_per_minute: int = Field(default=60, env="REQUESTS_PER_MINUTE")
    concurrent_requests: int = Field(default=10, env="CONCURRENT_REQUESTS")
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """로깅 설정"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # 파일 로깅
    file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    file_path: str = Field(default="logs/hts.log", env="LOG_FILE_PATH")
    file_max_size: int = Field(default=10485760, env="LOG_FILE_MAX_SIZE")  # 10MB
    file_backup_count: int = Field(default=5, env="LOG_FILE_BACKUP_COUNT")
    
    # 구조화된 로깅
    structured: bool = Field(default=True, env="LOG_STRUCTURED")
    json_format: bool = Field(default=False, env="LOG_JSON_FORMAT")
    
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
    """UI 설정"""
    
    # 윈도우 설정
    window_width: int = Field(default=1920, env="WINDOW_WIDTH")
    window_height: int = Field(default=1080, env="WINDOW_HEIGHT")
    window_title: str = Field(default="고성능 HTS v5.0", env="WINDOW_TITLE")
    
    # 테마 설정
    theme: str = Field(default="dark", env="UI_THEME")
    font_family: str = Field(default="Segoe UI", env="UI_FONT_FAMILY")
    font_size: int = Field(default=11, env="UI_FONT_SIZE")
    
    # 업데이트 간격 (초)
    update_interval: int = Field(default=1, env="UI_UPDATE_INTERVAL")
    chart_update_interval: int = Field(default=5, env="CHART_UPDATE_INTERVAL")
    
    # 차트 설정
    chart_style: str = Field(default="charles", env="CHART_STYLE")
    show_volume: bool = Field(default=True, env="CHART_SHOW_VOLUME")
    
    class Config:
        env_prefix = "UI_"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """보안 설정"""
    
    # 암호화 키
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    
    # 세션 설정
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")  # 1시간
    
    # API 보안
    api_rate_limit: int = Field(default=1000, env="API_RATE_LIMIT")
    api_burst_limit: int = Field(default=100, env="API_BURST_LIMIT")
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class CacheSettings(BaseSettings):
    """캐시 설정"""
    
    # 메모리 캐시 설정
    memory_max_size: int = Field(default=1000, env="CACHE_MEMORY_MAX_SIZE")
    memory_ttl: int = Field(default=3600, env="CACHE_MEMORY_TTL")  # 1시간
    
    # 디스크 캐시 설정
    disk_cache_dir: str = Field(default="cache", env="CACHE_DISK_DIR")
    disk_max_size: int = Field(default=1073741824, env="CACHE_DISK_MAX_SIZE")  # 1GB
    disk_ttl: int = Field(default=86400, env="CACHE_DISK_TTL")  # 24시간
    
    # 캐시 정책
    enable_memory_cache: bool = Field(default=True, env="CACHE_ENABLE_MEMORY")
    enable_redis_cache: bool = Field(default=True, env="CACHE_ENABLE_REDIS")
    enable_disk_cache: bool = Field(default=True, env="CACHE_ENABLE_DISK")
    
    class Config:
        env_prefix = "CACHE_"
        case_sensitive = False


class Settings(BaseSettings):
    """메인 설정 클래스 - 모든 설정을 통합 관리"""
    
    # 환경 설정
    environment: str = Field(default="development", description="실행 환경")
    debug: bool = Field(default=True, description="디버그 모드")
    
    # 프로젝트 설정
    project_name: str = Field(default="HTS", env="PROJECT_NAME")
    version: str = Field(default="5.0.0", env="VERSION")
    
    # 하위 설정들
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    performance: PerformanceSettings = PerformanceSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    ui: UISettings = UISettings()
    security: SecuritySettings = SecuritySettings()
    cache: CacheSettings = CacheSettings()
    
    # 데이터 디렉토리
    data_dir: str = Field(default="data", env="DATA_DIR")
    cache_dir: str = Field(default="cache", env="CACHE_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = ['development', 'testing', 'production', 'test']  # 'test' 추가
        if v not in allowed_envs:
            raise ValueError(f"환경은 {allowed_envs} 중 하나여야 합니다")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """필요한 디렉토리들을 생성"""
        dirs = [self.data_dir, self.cache_dir, self.logs_dir]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment == "development"
    
    @property
    def is_test(self) -> bool:
        """테스트 환경 여부"""
        return self.environment == "test"
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # 추가 필드 허용


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