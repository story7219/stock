#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ 투자 분석 시스템 설정 관리 모듈 (System Configuration Manager)
================================================================

투자 분석 시스템의 모든 설정을 중앙에서 관리하는 핵심 모듈입니다.
데이터베이스, 캐시, API, 성능 최적화 등 시스템 운영에 필요한 
모든 설정을 체계적으로 관리합니다.

주요 구성 요소:
1. DatabaseConfig: 데이터베이스 연결 및 풀 설정
   - PostgreSQL 연결 정보
   - 연결 풀 크기 및 타임아웃 설정
   - 트랜잭션 관리 옵션

2. CacheConfig: 캐시 시스템 설정
   - Redis 캐시 서버 설정
   - 메모리 캐시 크기 및 TTL
   - 다단계 캐시 활성화 옵션

3. APIConfig: 외부 API 연동 설정
   - Gemini AI API 키 관리
   - Yahoo Finance API 설정
   - 요청 제한 및 재시도 정책

4. PerformanceConfig: 성능 최적화 설정
   - 비동기 처리 활성화
   - 워커 스레드 수 설정
   - 메모리 사용량 제한
   - 프로파일링 옵션

설정 로드 순서:
1. 기본값 설정
2. config.json 파일에서 로드
3. 환경 변수로 오버라이드
4. 설정 검증 수행

이 모듈은 시스템의 모든 설정을 일관되게 관리하며,
운영 환경에 따른 유연한 설정 변경을 지원합니다.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """데이터베이스 설정 클래스"""
    host: str = "localhost"
    port: int = 5432
    database: str = "investment_system"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """캐시 시스템 설정 클래스"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    memory_cache_size: int = 1000
    cache_ttl: int = 3600
    multi_level_enabled: bool = True


@dataclass
class APIConfig:
    """API 연동 설정 클래스"""
    gemini_api_key: Optional[str] = None
    yahoo_finance_timeout: int = 30
    max_concurrent_requests: int = 50
    rate_limit_per_minute: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class PerformanceConfig:
    """성능 최적화 설정 클래스"""
    async_enabled: bool = True
    max_workers: int = 10
    batch_size: int = 100
    memory_limit_mb: int = 2048
    gc_threshold: int = 1000
    profiling_enabled: bool = False


@dataclass
class SystemConfig:
    """시스템 전체 설정"""
    debug_mode: bool = False
    mock_mode: bool = True
    log_level: str = "INFO"
    max_workers: int = 4
    request_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """환경변수에서 설정 로드"""
        return cls(
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            mock_mode=os.getenv('IS_MOCK', 'true').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30'))
        )


class Config:
    """메인 설정 관리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """설정 초기화"""
        self.config_path = config_path or self._get_default_config_path()
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.api = APIConfig()
        self.performance = PerformanceConfig()
        
        self._load_config()
        self._load_env_variables()
    
    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로 반환"""
        return os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.json")
    
    def _load_config(self) -> None:
        """설정 파일에서 설정 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self._update_from_dict(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
    
    def _load_env_variables(self) -> None:
        """환경 변수에서 설정 로드"""
        # API Keys
        self.api.gemini_api_key = (
            os.getenv('GEMINI_API_KEY') or 
            os.getenv('GOOGLE_GEMINI_API_KEY') or 
            self.api.gemini_api_key
        )
        
        # Database
        if os.getenv('DB_HOST'):
            self.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            self.database.database = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            self.database.username = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self.database.password = os.getenv('DB_PASSWORD')
        
        # Cache
        if os.getenv('REDIS_HOST'):
            self.cache.redis_host = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            self.cache.redis_port = int(os.getenv('REDIS_PORT'))
        if os.getenv('REDIS_PASSWORD'):
            self.cache.redis_password = os.getenv('REDIS_PASSWORD')
        
        # Performance
        if os.getenv('MAX_WORKERS'):
            self.performance.max_workers = int(os.getenv('MAX_WORKERS'))
        if os.getenv('MEMORY_LIMIT_MB'):
            self.performance.memory_limit_mb = int(os.getenv('MEMORY_LIMIT_MB'))
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """딕셔너리에서 설정 업데이트"""
        if 'database' in config_data:
            for key, value in config_data['database'].items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)
        
        if 'cache' in config_data:
            for key, value in config_data['cache'].items():
                if hasattr(self.cache, key):
                    setattr(self.cache, key, value)
        
        if 'api' in config_data:
            for key, value in config_data['api'].items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)
        
        if 'performance' in config_data:
            for key, value in config_data['performance'].items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
    
    def save_config(self) -> None:
        """현재 설정을 파일에 저장"""
        try:
            config_data = {
                'database': self.database.__dict__,
                'cache': self.cache.__dict__,
                'api': {k: v for k, v in self.api.__dict__.items() if not k.endswith('_key')},
                'performance': self.performance.__dict__
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        errors = []
        
        # Validate performance settings
        if self.performance.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if self.performance.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.performance.memory_limit_mb <= 0:
            errors.append("memory_limit_mb must be positive")
        
        # Validate cache settings
        if self.cache.cache_ttl <= 0:
            errors.append("cache_ttl must be positive")
        
        if self.cache.memory_cache_size <= 0:
            errors.append("memory_cache_size must be positive")
        
        if errors:
            logger.error(f"Configuration validation errors: {errors}")
            return False
        
        return True
    
    def get_database_url(self) -> str:
        """데이터베이스 연결 URL 반환"""
        return (
            f"postgresql://{self.database.username}:{self.database.password}"
            f"@{self.database.host}:{self.database.port}/{self.database.database}"
        )
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.cache.redis_password}@" if self.cache.redis_password else ""
        return f"redis://{auth}{self.cache.redis_host}:{self.cache.redis_port}/{self.cache.redis_db}"


# Global configuration instance
config = Config()

# 전역 설정 인스턴스
system_config = SystemConfig.from_env() 