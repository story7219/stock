from __future__ import annotations
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from core.config import Config
from core.logger import setup_logging
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from service.command_service import CommandService
from service.query_service import QueryService
from typing import Any, Dict, List
import asyncio
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_system.py
모듈: 시스템 통합 테스트
목적: 전체 시스템 기능 테스트 및 데모

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio
    - datetime

Performance:
    - 전체 테스트: < 2분
    - 메모리사용량: < 200MB

Security:
    - 테스트 데이터 사용
    - 에러 처리
    - 로깅

License: MIT
"""





console = Console()


async def test_news_momentum_strategy() -> Dict[str, Any]:
    """뉴스 모멘텀 전략 테스트"""
    console.print("\n📰 뉴스 모멘텀 전략 테스트", style="bold blue")
    query_service = QueryService()
    keywords = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '반도체', '배터리']
    target_stocks = ['005930', '000660', '373220', '051910', '006400']
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("뉴스 모멘텀 분석 중...", total=None)
            result = await query_service.analyze_news_momentum(keywords=keywords, target_stocks=target_stocks, start_date=start_date, end_date=end_date)
            progress.update(task, completed=True)
        table = Table(title="📰 뉴스 모멘텀 분석 결과")
        table.add_column("종목", style="cyan")
        table.add_column("신호", style="magenta")
        table.add_column("신뢰도", style="green")
        table.add_column("근거", style="yellow")
        for signal in result['signals'][:5]:
            table.add_row(signal.stock_code, signal.signal_type.value, f"{signal.confidence_score:.2f}", signal.reasoning[:50] + "..." if len(signal.reasoning) > 50 else signal.reasoning)
        console.print(table)
        return {'strategy': 'news_momentum', 'signals_count': len(result['signals']), 'avg_confidence': sum(s.confidence_score for s in result['signals']) / len(result['signals']) if result['signals'] else 0}
    except Exception as e:
        console.print(f"❌ 뉴스 모멘텀 테스트 실패: {e}", style="red")
        return {'error': str(e)}
# ... (나머지 함수들) ...
