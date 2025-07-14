#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: report.py
모듈: 백테스팅 리포트 생성
목적: 자동 리포트 생성 (Executive Summary, Technical Report, Risk Report 등)

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from .utils import format_percentage, format_currency, generate_report_filename

class ReportGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """경영진용 요약 보고서를 생성합니다."""
        summary = []
        summary.append("=" * 80)
        summary.append("🏆 완전 자동화 백테스팅 시스템 - Executive Summary")
        summary.append("=" * 80)
        summary.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        # 기본 백테스트 결과
        if "base" in results and results["base"]:
            base = results["base"]
            summary.append("📈 기본 백테스트 성과:")
            summary.append(f"   총 수익률: {format_percentage(base.get('total_return', 0))}")
            summary.append(f"   샤프 비율: {base.get('sharpe', 0):.2f}")
            summary.append(f"   최대 낙폭: {format_percentage(base.get('max_drawdown', 0))}")
            summary.append(f"   승률: {format_percentage(base.get('win_rate', 0))}")
            summary.append(f"   손익비: {base.get('profit_factor', 0):.2f}")
            summary.append("")

        # 통계적 검증 결과
        if "stat" in results and results["stat"]:
            stat = results["stat"]
            summary.append("🔬 통계적 검증:")
            summary.append(f"   p-value: {stat.get('p_value', 1):.4f}")
            summary.append(f"   통계적 유의성: {'✅ 통과' if stat.get('significant', False) else '❌ 미달'}")
            summary.append(f"   효과 크기: {stat.get('effect_size', 0):.2f}")
            summary.append("")

        # 스트레스 테스트 결과
        if "stress" in results and results["stress"]:
            stress = results["stress"]
            summary.append("🛡️ 스트레스 테스트:")
            summary.append(f"   평균 수익률: {format_percentage(stress.get('total_return', 0))}")
            summary.append(f"   생존률: {format_percentage(stress.get('survival_rate', 0))}")
            summary.append("")

        # 최종 권고사항
        summary.append("🎯 최종 권고사항:")
        if self._is_system_ready(results):
            summary.append("   ✅ 실전 투자 준비 완료")
            summary.append("   ✅ 모든 검증 기준 통과")
            summary.append("   ✅ 안전한 투자 가능")
        else:
            summary.append("   ⚠️ 추가 검증 필요")
            summary.append("   ⚠️ 실전 투자 전 보완 필요")

        summary.append("=" * 80)
        return "\n".join(summary)

    def generate_technical_report(self, results: Dict[str, Any]) -> str:
        """기술적 상세 분석 보고서를 생성합니다."""
        report = []
        report.append("🔬 완전 자동화 백테스팅 시스템 - Technical Report")
        report.append("=" * 80)
        report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 상세 성과 분석
        if "base" in results and results["base"]:
            base = results["base"]
            report.append("📊 상세 성과 분석:")
            report.append(f"   총 수익률: {format_percentage(base.get('total_return', 0))}")
            report.append(f"   샤프 비율: {base.get('sharpe', 0):.4f}")
            report.append(f"   소르티노 비율: {base.get('sortino_ratio', 0):.4f}")
            report.append(f"   칼마 비율: {base.get('calmar_ratio', 0):.4f}")
            report.append(f"   오메가 비율: {base.get('omega_ratio', 0):.4f}")
            report.append(f"   최대 낙폭: {format_percentage(base.get('max_drawdown', 0))}")
            report.append(f"   승률: {format_percentage(base.get('win_rate', 0))}")
            report.append(f"   손익비: {base.get('profit_factor', 0):.4f}")
            report.append(f"   변동성: {format_percentage(base.get('volatility', 0))}")
            report.append(f"   VaR (95%): {format_percentage(base.get('var_95', 0))}")
            report.append(f"   CVaR (95%): {format_percentage(base.get('cvar_95', 0))}")
            report.append("")

        # Walk-Forward 분석
        if "walkforward" in results and results["walkforward"]:
            wf = results["walkforward"]
            report.append("🔄 Walk-Forward 분석:")
            report.append(f"   평균 수익률: {format_percentage(wf.get('total_return', 0))}")
            report.append(f"   일관성 (표준편차): {format_percentage(wf.get('consistency', 0))}")
            report.append(f"   분석 기간 수: {wf.get('periods', 0)}")
            report.append("")

        # Monte Carlo 분석
        if "montecarlo" in results and results["montecarlo"]:
            mc = results["montecarlo"]
            report.append("🎲 Monte Carlo 분석:")
            report.append(f"   평균 수익률: {format_percentage(mc.get('total_return', 0))}")
            report.append(f"   수익률 표준편차: {format_percentage(mc.get('return_std', 0))}")
            report.append(f"   양의 수익률 확률: {format_percentage(mc.get('positive_probability', 0))}")
            report.append(f"   시뮬레이션 횟수: {mc.get('n_simulations', 0):,}")
            report.append("")

        return "\n".join(report)

    def generate_risk_report(self, results: Dict[str, Any]) -> str:
        """리스크 관리 보고서를 생성합니다."""
        report = []
        report.append("🛡️ 완전 자동화 백테스팅 시스템 - Risk Report")
        report.append("=" * 80)
        report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 스트레스 테스트 상세
        if "stress" in results and results["stress"]:
            stress = results["stress"]
            report.append("🌪️ 스트레스 테스트 결과:")
            report.append(f"   평균 수익률: {format_percentage(stress.get('total_return', 0))}")
            report.append(f"   생존률: {format_percentage(stress.get('survival_rate', 0))}")

            if "scenario_results" in stress:
                report.append("   📋 위기 상황별 결과:")
                for scenario, result in stress["scenario_results"].items():
                    report.append(f"     {scenario}: {format_percentage(result.get('total_return', 0))} (생존: {'✅' if result.get('survived', False) else '❌'})")
            report.append("")

        # 리스크 지표
        if "base" in results and results["base"]:
            base = results["base"]
            report.append("📊 리스크 지표:")
            report.append(f"   최대 낙폭: {format_percentage(base.get('max_drawdown', 0))}")
            report.append(f"   VaR (95%): {format_percentage(base.get('var_95', 0))}")
            report.append(f"   CVaR (95%): {format_percentage(base.get('cvar_95', 0))}")
            report.append(f"   변동성: {format_percentage(base.get('volatility', 0))}")
            report.append("")

        return "\n".join(report)

    def generate_full_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """전체 보고서를 생성합니다."""
        full_report = {
            "executive_summary": self.generate_executive_summary(results),
            "technical_report": self.generate_technical_report(results),
            "risk_report": self.generate_risk_report(results),
            "raw_results": results,
            "generated_at": datetime.now().isoformat(),
        }

        # 파일로 저장
        filename = generate_report_filename()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)

        return full_report

    def _is_system_ready(self, results: Dict[str, Any]) -> bool:
        """시스템이 실전 투자 준비가 되었는지 확인합니다."""
        thresholds = self.config.get("performance_thresholds", {})

        if "base" not in results or not results["base"]:
            return False

        base = results["base"]

        # 기본 성과 기준 확인
        if (base.get("sharpe", 0) < thresholds.get("min_sharpe_ratio", 1.5) or:
            base.get("max_drawdown", 0) < thresholds.get("max_drawdown", -0.15) or:
            base.get("win_rate", 0) < thresholds.get("min_win_rate", 0.55)):
            return False

        # 통계적 유의성 확인
        if "stat" in results and results["stat"]:
            if not results["stat"].get("significant", False):
                return False

        # 스트레스 테스트 생존률 확인
        if "stress" in results and results["stress"]:
            if results["stress"].get("survival_rate", 0) < 0.8:  # 80% 이상 생존
                return False

        return True
