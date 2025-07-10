from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: __init__.py
모듈: 애플리케이션 계층 초기화
목적: 사용 사례 및 애플리케이션 서비스

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - pydantic==2.5.0
    - typing-extensions==4.8.0

Architecture:
    - Application Layer
    - Use Cases
    - Application Services
    - Command/Query Separation

License: MIT
"""

from .cli import CLIService
from .dashboard import DashboardService
from .services import TradingSystemService

# 명령/쿼리 핸들러들 (필요시 주석 해제)
# from .commands import (
#     GenerateSignalCommand,
#     ExecuteTradeCommand,
#     UpdateRiskCommand
# )

# from .queries import (
#     GetPortfolioQuery,
#     GetSignalsQuery,
#     GetRiskMetricsQuery
# )

__all__ = [
    'TradingSystemService',
    'CLIService',
    'DashboardService',
    # 'GenerateSignalCommand',
    # 'ExecuteTradeCommand',
    # 'UpdateRiskCommand',
    # 'GetPortfolioQuery',
    # 'GetSignalsQuery',
    # 'GetRiskMetricsQuery'
]

