from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, List, Optional, Union
import os
import pydantic
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: settings.py
모듈: Pydantic Settings 기반 설정 관리
목적: 환경변수 기반 설정 및 검증 강화

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - pydantic==2.5.0
    - pydantic-settings==2.1.0
    - python-dotenv==1.0.0

Architecture:
    - Pydantic Settings
    - Environment Variables
    - Configuration Validation
    - Type Safety

License: MIT
"""





class DatabaseSettings(BaseSettings):
    """데이터베이스 설정"""

    model_config = SettingsConfigDict(env_prefix="DB_", case_sensitive=False)

    url: str = Field(default="sqlite:///./trading_data.db", description="DB 연결 URL")
    pool_size: int = Field(default=10, ge=1, le=100, description="커넥션 풀 크기")
    max_overflow: int = Field(default=20, ge=0, le=100, description="최대 오버플로우")
    echo: bool = Field(default=False, description="SQL 로깅")
    pool_pre_ping: bool = Field(default=True, description="커넥션 풀 사전 핑")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """URL 검증"""
        if not v:
            raise ValueError("데이터베이스 URL은 필수입니다")
        return v


class APISettings(BaseSettings):
    """API 설정"""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    # KIS API 설정 (모의)
    mock_kis_app_key: str = Field(..., alias="MOCK_KIS_APP_KEY", description="모의 KIS 앱 키")
    mock_kis_app_secret: str = Field(..., alias="MOCK_KIS_APP_SECRET", description="모의 KIS 앱 시크릿")
    mock_kis_account_number: str = Field(..., alias="MOCK_KIS_ACCOUNT_NUMBER", description="모의 KIS 계좌번호")

    # KIS API 설정 (실거래)
    live_kis_app_key: str = Field(..., alias="LIVE_KIS_APP_KEY", description="실거래 KIS 앱 키")
    live_kis_app_secret: str = Field(..., alias="LIVE_KIS_APP_SECRET", description="실거래 KIS 앱 시크릿")
    live_kis_account_number: str = Field(..., alias="LIVE_KIS_ACCOUNT_NUMBER", description="실거래 KIS 계좌번호")

    # DART API 설정
    dart_api_key: str = Field(..., alias="DART_API_KEY", description="DART API 키")

    # 기타 API 설정
    # ... (나머지 코드 생략)


# ... (나머지 코드)
