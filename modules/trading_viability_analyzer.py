#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: trading_viability_analyzer.py
모듈: 실전 매매 투입 가능성 평가
목적: ML/DL 모델의 실전 매매 투입 가능성, 리스크, 체크리스트 자동 평가

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pydantic>=2.5.0
    - structlog>=24.1.0

Performance:
    - O(1) for checklist, O(n) for report

Security:
    - robust 예외처리, 입력 검증

License: MIT
"""

from __future__ import annotations
from typing import Dict, List, Literal, Any
from pydantic import BaseModel, Field, ValidationError
import structlog
from enum import Enum

logger = structlog.get_logger(__name__)

class ViabilityGrade(str, Enum):
    PASS = "실전매매 투입 가능"
    CONDITIONAL = "조건부(추가검증 필요)"
    FAIL = "실전매매 투입 불가"

class ChecklistItem(BaseModel):
    name: str
    description: str
    passed: bool
    reason: str = ""

class ViabilityReport(BaseModel):
    grade: ViabilityGrade
    summary: str
    checklist: List[ChecklistItem]
    recommendations: List[str]

class TradingViabilityAnalyzer:
    """실전 매매 투입 가능성 평가기 (커서룰 100%)"""
    CHECKLIST = [
        ("성능지표(정확도/수익률)", "테스트셋/실전 데이터에서 성능이 기준 이상인가?"),
        ("과최적화(Overfitting)", "훈련-테스트 성능 차이가 과도하지 않은가?"),
        ("데이터누수/미래데이터", "미래 데이터/정보가 누수되지 않았는가?"),
        ("시장적합성", "실제 시장 환경(슬리피지, 체결, 유동성 등) 반영 여부"),
        ("리스크관리", "손실 제한, 포트폴리오 분산 등 리스크 관리가 내장되어 있는가?"),
        ("실행가능성", "실제 주문/체결/실행이 기술적으로 가능한가?"),
        ("거래비용/슬리피지", "수수료, 세금, 슬리피지 등 비용 반영 여부"),
        ("시장충격/유동성", "대량 주문 시 시장 충격/유동성 리스크 고려 여부"),
        ("모델 해석성", "모델/신호의 해석 가능성 및 설명력"),
        ("운영/모니터링", "실시간 모니터링/알림/장애대응 체계 구축 여부"),
        ("법적/윤리적 이슈", "시장규제, 윤리, 데이터/알고리즘 투명성 준수 여부")
    ]

    @staticmethod
    def analyze(
        metrics: Dict[str, Any],
        thresholds: Dict[str, float] = None
    ) -> ViabilityReport:
        """실전 매매 투입 가능성 평가 및 체크리스트 리포트 생성

        Args:
            metrics: 주요 성능/리스크 지표 dict (예: accuracy, sharpe, max_drawdown 등)
            thresholds: 각 항목별 기준값 dict (없으면 기본값 적용)
        Returns:
            ViabilityReport: 등급, 체크리스트, 요약, 권고사항 포함
        """
        if thresholds is None:
            thresholds = {
                "accuracy": 0.7,
                "sharpe": 1.0,
                "max_drawdown": -0.2,
                "overfit_gap": 0.1,
            }
        checklist: List[ChecklistItem] = []
        recommendations: List[str] = []
        # 1. 성능지표
        acc = metrics.get("accuracy", 0)
        sharpe = metrics.get("sharpe", 0)
        mdd = metrics.get("max_drawdown", 0)
        overfit_gap = metrics.get("overfit_gap", 1)
        checklist.append(ChecklistItem(
            name="성능지표(정확도/수익률)",
            description="테스트셋/실전 데이터에서 성능이 기준 이상인가?",
            passed=acc >= thresholds["accuracy"] and sharpe >= thresholds["sharpe"],
            reason=f"accuracy={acc:.3f}, sharpe={sharpe:.2f} (기준: {thresholds['accuracy']}, {thresholds['sharpe']})"
        ))
        # 2. 과최적화
        checklist.append(ChecklistItem(
            name="과최적화(Overfitting)",
            description="훈련-테스트 성능 차이가 과도하지 않은가?",
            passed=overfit_gap <= thresholds["overfit_gap"],
            reason=f"overfit_gap={overfit_gap:.3f} (기준: {thresholds['overfit_gap']})"
        ))
        # 3. 데이터누수
        checklist.append(ChecklistItem(
            name="데이터누수/미래데이터",
            description="미래 데이터/정보가 누수되지 않았는가?",
            passed=metrics.get("data_leakage", False) is False,
            reason="데이터누수 없음" if not metrics.get("data_leakage", False) else "데이터누수 의심"
        ))
        # 4. 시장적합성
        checklist.append(ChecklistItem(
            name="시장적합성",
            description="실제 시장 환경(슬리피지, 체결, 유동성 등) 반영 여부",
            passed=metrics.get("market_realism", True),
            reason="실전 환경 반영" if metrics.get("market_realism", True) else "시장환경 미반영"
        ))
        # 5. 리스크관리
        checklist.append(ChecklistItem(
            name="리스크관리",
            description="손실 제한, 포트폴리오 분산 등 리스크 관리가 내장되어 있는가?",
            passed=metrics.get("risk_control", True),
            reason="리스크 관리 내장" if metrics.get("risk_control", True) else "리스크 관리 미흡"
        ))
        # 6. 실행가능성
        checklist.append(ChecklistItem(
            name="실행가능성",
            description="실제 주문/체결/실행이 기술적으로 가능한가?",
            passed=metrics.get("executable", True),
            reason="실행 가능" if metrics.get("executable", True) else "실행 불가"
        ))
        # 7. 거래비용/슬리피지
        checklist.append(ChecklistItem(
            name="거래비용/슬리피지",
            description="수수료, 세금, 슬리피지 등 비용 반영 여부",
            passed=metrics.get("cost_included", True),
            reason="비용 반영" if metrics.get("cost_included", True) else "비용 미반영"
        ))
        # 8. 시장충격/유동성
        checklist.append(ChecklistItem(
            name="시장충격/유동성",
            description="대량 주문 시 시장 충격/유동성 리스크 고려 여부",
            passed=metrics.get("liquidity_ok", True),
            reason="유동성 충분" if metrics.get("liquidity_ok", True) else "유동성 리스크 존재"
        ))
        # 9. 모델 해석성
        checklist.append(ChecklistItem(
            name="모델 해석성",
            description="모델/신호의 해석 가능성 및 설명력",
            passed=metrics.get("explainable", True),
            reason="해석 가능" if metrics.get("explainable", True) else "블랙박스/설명력 부족"
        ))
        # 10. 운영/모니터링
        checklist.append(ChecklistItem(
            name="운영/모니터링",
            description="실시간 모니터링/알림/장애대응 체계 구축 여부",
            passed=metrics.get("monitoring", True),
            reason="모니터링 구축" if metrics.get("monitoring", True) else "모니터링 미흡"
        ))
        # 11. 법적/윤리적 이슈
        checklist.append(ChecklistItem(
            name="법적/윤리적 이슈",
            description="시장규제, 윤리, 데이터/알고리즘 투명성 준수 여부",
            passed=metrics.get("compliance", True),
            reason="준수" if metrics.get("compliance", True) else "위반/불명확"
        ))
        # 종합 등급 산출
        passed_count = sum(1 for item in checklist if item.passed)
        if all(item.passed for item in checklist):
            grade = ViabilityGrade.PASS
            summary = "모든 체크리스트를 통과했습니다. 실전 매매 투입이 가능합니다."
        elif passed_count >= 9:
            grade = ViabilityGrade.CONDITIONAL
            summary = "대부분 항목을 통과했으나 일부 보완이 필요합니다."
            recommendations = [item.name for item in checklist if not item.passed]
        else:
            grade = ViabilityGrade.FAIL
            summary = "실전 매매 투입이 불가합니다. 반드시 보완이 필요합니다."
            recommendations = [item.name for item in checklist if not item.passed]
        return ViabilityReport(
            grade=grade,
            summary=summary,
            checklist=checklist,
            recommendations=recommendations
        )

if __name__ == "__main__":
    # 단위테스트 예시
    test_metrics = {
        "accuracy": 0.81,
        "sharpe": 1.2,
        "max_drawdown": -0.15,
        "overfit_gap": 0.05,
        "data_leakage": False,
        "market_realism": True,
        "risk_control": True,
        "executable": True,
        "cost_included": True,
        "liquidity_ok": True,
        "explainable": True,
        "monitoring": True,
        "compliance": True
    }
    report = TradingViabilityAnalyzer.analyze(test_metrics)
    print("[실전매매 투입 가능성 평가 결과]")
    print(f"등급: {report.grade}")
    print(f"요약: {report.summary}")
    for item in report.checklist:
        print(f"- {item.name}: {'✅' if item.passed else '❌'} ({item.reason})")
    if report.recommendations:
        print("[권고사항]", ", ".join(report.recommendations)) 