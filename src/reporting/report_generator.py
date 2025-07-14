#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: report_generator.py
모듈: 리포트 생성/포맷팅/저장/시각화
목적: 자동매매 백테스트 리포트 생성/저장/시각화

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Dict
import logging
import os

logger = logging.getLogger(__name__)

class ReportGenerator:
    """리포트 생성/포맷팅/저장/시각화 담당 클래스"""
    def __init__(self):
        pass

    def generate(self, perf: Dict[str, Any], risk: Dict[str, Any], checklist: Dict[str, Any]) -> str:
        """Markdown 리포트 생성: 성능/리스크/체크리스트 요약"""
        try:
            lines = [
                "# 자동매매 백테스트 리포트",
                "",
                "## 1. 성능 요약",
                f"- 총수익률: {perf['total_return']:.2%}",
                f"- 연환산수익률: {perf['annualized_return']:.2%}",
                f"- 샤프비율: {perf['sharpe']:.2f}",
                f"- 최대낙폭: {perf['max_drawdown']:.2%}",
                f"- 변동성: {perf['volatility']:.2%}",
                "",
                "## 2. 리스크 요약",
                f"- p-value: {risk['p_value']:.3f}",
                f"- 스트레스테스트(5%): {risk['stress_loss']:.2%}",
                "",
                "## 3. 실전 체크리스트",
            ]
            for k, v in checklist.items():
                status = '✅' if v['passed'] else '❌'
                lines.append(f"- {k}: {status} ({v['reason']})")
            return '\n'.join(lines)
        except Exception as e:
            logger.error(f"Report generate error: {e}")
            raise

    def save(self, report: str, filename: str = "backtest_report.md") -> str:
        """리포트 파일 저장"""
        try:
            os.makedirs('reports', exist_ok=True)
            path = f"reports/{filename}"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved: {path}")
            return path
        except Exception as e:
            logger.error(f"Report save error: {e}")
            raise 