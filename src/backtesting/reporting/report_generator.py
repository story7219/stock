#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/reporting/report_generator.py
"""
from __future__ import annotations
import json
from datetime import datetime
from typing import Dict
import Any
from ..utils.helpers import generate_report_filename

class ReportGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_full_report(self, results: Dict[str, Any]) -> str:
        summary = self._create_summary(results)

        full_report = {
            "summary": summary,
            "details": results,
            "config": self.config,
            "generated_at": datetime.now().isoformat(),
        }

        filename = generate_report_filename()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)

        print(f"\n📄 상세 보고서 저장: {filename}")
        return summary

    def _create_summary(self, results: Dict[str, Any]) -> str:
        base = results.get("base", {})
        stat = results.get("stat", {})
        stress = results.get("stress", {})

        summary = f"""
        ================================================================
        🏆 백테스팅 시스템 결과 요약
        ================================================================
        생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        📈 기본 백테스트:
           총 수익률: {base.get('total_return', 0):.2%}
           샤프 비율: {base.get('sharpe_ratio', 0):.2f}
           최대 낙폭: {base.get('max_drawdown', 0):.2%}
           승률: {base.get('win_rate', 0):.2%}

        🔬 통계적 검증:
           p-value: {stat.get('p_value', 1):.4f}
           통계적 유의성: {'✅ 통과' if stat.get('significant', False) else '❌ 미달'}

        🛡️ 스트레스 테스트:
           평균 손익: {stress.get('average_pnl', 0):,.0f} 원
           생존률: {stress.get('survival_rate', 0):.2%}
        ================================================================
        """
        return summary.strip()
