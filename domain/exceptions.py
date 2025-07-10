from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
import traceback
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: exceptions.py
모듈: 도메인 예외 정의
목적: 비즈니스 로직 예외 처리 및 에러 컨텍스트 관리

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - typing-extensions==4.8.0
    - dataclasses
    - datetime

Architecture:
    - Exception Hierarchy
    - Error Context
    - Structured Error Handling

License: MIT
"""




class ErrorSeverity(Enum):
    """에러 심각도 열거형"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    operation: str = field()  # 수행 중이던 작업
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # 타임스탬프
    severity: ErrorSeverity = field(default=ErrorSeverity.MEDIUM)  # 에러 심각도
    user_id: Optional[str] = field(default=None)  # 사용자 ID
    request_id: Optional[str] = field(default=None)  # 요청 ID
    extra: Optional[Dict[str, Any]] = field(default=None)  # 기타 정보
    session_id: Optional[str] = field(default=None)  # 세션 ID
    symbol: Optional[str] = field(default=None)  # 관련 종목 코드
    strategy_type: Optional[str] = field(default=None)  # 관련 전략 타입
    portfolio_id: Optional[str] = field(default=None)  # 포트폴리오 ID
    function_name: Optional[str] = field(default=None)  # 함수명
    line_number: Optional[int] = field(default=None)  # 라인 번호
    stack_trace: Optional[str] = field(default=None)  # 스택 트레이스
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.name,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'symbol': self.symbol,
            'strategy_type': self.strategy_type,
            'portfolio_id': self.portfolio_id,
            'function_name': self.function_name,
            'line_number': self.line_number,
            'stack_trace': self.stack_trace,
            'metadata': self.metadata
        }


class DomainException(Exception):
    def __init__(:
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext(operation="unknown")
        self.cause = cause
        self._capture_stack_trace()

    def _capture_stack_trace(self):
        if not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message} (Operation: {self.context.operation})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'context': self.context.to_dict(),
            'cause': str(self.cause) if self.cause else None
        }

# ... (rest of the code)
