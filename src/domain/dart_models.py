#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pydantic import BaseModel
import Field
from typing import Optional
"""
파일명: dart_models.py
모듈: DART 데이터 모델
목적: DART 공시 및 기업정보 데이터의 타입 안전성 및 검증

Author: AI Assistant
Created: 2025-07-10
Version: 1.0.0

Dependencies:
    - pydantic>=2.11.0

License: MIT
"""


class DartCorpInfo(BaseModel):
    """DART 기업 정보 모델"""
    corp_code: str = Field(..., min_length=8, max_length=8, description="DART 고유 기업코드")
    corp_name: str = Field(..., min_length=1, max_length=100, description="기업명")
    stock_code: Optional[str] = Field(None, min_length=0, max_length=6, description="상장종목코드")
    modify_date: Optional[str] = Field(None, min_length=8, max_length=8, description="수정일자(YYYYMMDD)")

class DartDisclosure(BaseModel):
    """DART 공시 데이터 모델"""
    corp_code: str = Field(..., min_length=8, max_length=8, description="DART 고유 기업코드")
    corp_name: str = Field(..., min_length=1, max_length=100, description="기업명")
    stock_code: Optional[str] = Field(None, min_length=0, max_length=6, description="상장종목코드")
    corp_cls: Optional[str] = Field(None, min_length=1, max_length=1, description="기업구분 (Y:유가, K:코스닥, N:코넥스, E:기타)")
    report_nm: str = Field(..., min_length=1, max_length=200, description="보고서명")
    rcept_no: str = Field(..., min_length=14, max_length=14, description="접수번호")
    flr_nm: str = Field(..., min_length=1, max_length=100, description="공시제출인")
    rcept_dt: str = Field(..., min_length=8, max_length=8, description="접수일자(YYYYMMDD)")
    rm: Optional[str] = Field(None, max_length=200, description="비고")

    class Config:
        extra = "ignore"
        validate_assignment = True
        str_strip_whitespace = True
