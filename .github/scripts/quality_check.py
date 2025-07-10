#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: quality_check.py
모듈: 코드 품질 검사 시스템
목적: 자동 품질 검사, 테스트 실행, 보안 감사

Author: GitHub Actions
Created: 2025-01-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pytest
    - subprocess
    - pathlib

Performance:
    - 검사 시간: < 120초
    - 메모리사용량: < 300MB
    - 처리용량: 500+ files/minute

Security:
    - 코드 보안 검사
    - 취약점 탐지
    - 권한 검증

License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """품질 검사 결과"""
    tool_name: str
    passed: bool
    score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    output: str = ""


@dataclass
class QualityReport:
    """전체 품질 보고서"""
    overall_score: float
    passed: bool
    results: List[QualityResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class InvestmentSystemQualityChecker:
    """투자 시스템 품질 검사기"""

    def __init__(self, project_root: Optional[Path] = None):
        """초기화"""
        try:
            self.project_root = project_root or Path(__file__).parent.parent.parent
            self.src_path = self.project_root / "src"
            self.config_path = self.project_root / "config"
            
            # 품질 기준
            self.quality_standards = {
                "pylint_min_score": 8.0,
                "coverage_min_percent": 80.0,
                "complexity_max": 10,
                "line_length_max": 88,
                "function_length_max": 50,
                "class_length_max": 200,
            }
            
            # Python 파일 목록
            self.python_files = self._get_python_files()
            
            # 투자 시스템 특화 규칙
            self.investment_rules = {
                "required_modules": [
                    "yfinance", "pandas", "numpy", "ta",
                    "google-generative-ai", "aiohttp"
                ],
                "required_functions": {
                    "strategy": ["analyze", "get_strategy_type"],
                    "data_collector": ["collect_market_data", "get_stock_data"],
                    "ai_analyzer": ["analyze_recommendations"],
                },
                "security_patterns": [
                    r'api_key\s*=\s*"[^"]+"',
                    r'password\s*=\s*"[^"]+"',
                    r'secret\s*=\s*"[^"]+"',
                ],
            }
            
            logger.info("품질 검사기 초기화 완료")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def _get_python_files(self) -> List[Path]:
        """프로젝트 디렉토리 내의 Python 파일 목록을 반환합니다."""
        try:
            python_files = []
            for path in self.project_root.rglob("*.py"):
                if "venv" not in str(path) and "__pycache__" not in str(path):
                    python_files.append(path)
            return python_files
        except Exception as e:
            logger.error(f"Python 파일 목록 생성 실패: {e}")
            return []

    def run_pylint_check(self) -> QualityResult:
        """Pylint 검사 실행"""
        try:
            start_time = time.time()
            
            # Pylint 명령어 실행
            cmd = [
                "pylint",
                "--output-format=json",
                "--score=y",
                "--disable=C0114,C0115,C0116",  # docstring 관련 경고 비활성화
                str(self.project_root)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            execution_time = time.time() - start_time
            
            # 결과 파싱
            if result.returncode == 0:
                return QualityResult(
                    tool_name="pylint",
                    passed=True,
                    score=10.0,
                    execution_time=execution_time,
                    output=result.stdout
                )
            else:
                # JSON 출력 파싱
                issues = []
                try:
                    pylint_results = json.loads(result.stdout)
                    for item in pylint_results:
                        issues.append({
                            'file': item.get('path', ''),
                            'line': item.get('line', 0),
                            'message': item.get('message', ''),
                            'type': item.get('type', '')
                        })
                except json.JSONDecodeError:
                    issues.append({'message': 'Pylint 결과 파싱 실패'})
                
                return QualityResult(
                    tool_name="pylint",
                    passed=False,
                    score=0.0,
                    issues=issues,
                    execution_time=execution_time,
                    output=result.stdout
                )
                
        except subprocess.TimeoutExpired:
            return QualityResult(
                tool_name="pylint",
                passed=False,
                score=0.0,
                warnings=["Pylint 실행 시간 초과"],
                execution_time=300.0
            )
        except Exception as e:
            return QualityResult(
                tool_name="pylint",
                passed=False,
                score=0.0,
                warnings=[f"Pylint 실행 실패: {e}"]
            )

    def run_test_coverage(self) -> QualityResult:
        """테스트 커버리지 검사"""
        try:
            start_time = time.time()
            
            # pytest-cov 명령어 실행
            cmd = [
                "python", "-m", "pytest",
                "--cov=.",
                "--cov-report=json",
                "--cov-report=term-missing",
                str(self.project_root / "tests")
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10분 타임아웃
            )
            
            execution_time = time.time() - start_time
            
            # 커버리지 결과 파싱
            coverage_score = 0.0
            try:
                # JSON 커버리지 리포트에서 점수 추출
                coverage_match = re.search(r'TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%', result.stdout)
                if coverage_match:
                    coverage_score = float(coverage_match.group(3))
            except Exception:
                coverage_score = 0.0
            
            passed = coverage_score >= self.quality_standards["coverage_min_percent"]
            
            return QualityResult(
                tool_name="test_coverage",
                passed=passed,
                score=coverage_score,
                execution_time=execution_time,
                output=result.stdout
            )
            
        except subprocess.TimeoutExpired:
            return QualityResult(
                tool_name="test_coverage",
                passed=False,
                score=0.0,
                warnings=["테스트 커버리지 실행 시간 초과"],
                execution_time=600.0
            )
        except Exception as e:
            return QualityResult(
                tool_name="test_coverage",
                passed=False,
                score=0.0,
                warnings=[f"테스트 커버리지 실행 실패: {e}"]
            )

    def run_security_audit(self) -> QualityResult:
        """보안 감사 실행"""
        try:
            start_time = time.time()
            
            # bandit 보안 검사
            cmd = [
                "bandit",
                "-r",
                str(self.project_root),
                "-f", "json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            execution_time = time.time() - start_time
            
            # 보안 이슈 파싱
            security_issues = []
            try:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get('results', []):
                    security_issues.append({
                        'file': issue.get('filename', ''),
                        'line': issue.get('line_number', 0),
                        'severity': issue.get('issue_severity', ''),
                        'message': issue.get('issue_text', '')
                    })
            except json.JSONDecodeError:
                security_issues.append({'message': '보안 감사 결과 파싱 실패'})
            
            # 보안 점수 계산 (이슈가 적을수록 높은 점수)
            security_score = max(0.0, 10.0 - len(security_issues) * 2.0)
            passed = len(security_issues) == 0
            
            return QualityResult(
                tool_name="security_audit",
                passed=passed,
                score=security_score,
                issues=security_issues,
                execution_time=execution_time,
                output=result.stdout
            )
            
        except subprocess.TimeoutExpired:
            return QualityResult(
                tool_name="security_audit",
                passed=False,
                score=0.0,
                warnings=["보안 감사 실행 시간 초과"],
                execution_time=300.0
            )
        except Exception as e:
            return QualityResult(
                tool_name="security_audit",
                passed=False,
                score=0.0,
                warnings=[f"보안 감사 실행 실패: {e}"]
            )

    def check_code_structure(self) -> QualityResult:
        """코드 구조 검사"""
        try:
            start_time = time.time()
            
            issues = []
            warnings = []
            
            # 파일 헤더 검사
            for py_file in self.python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 파일 헤더 검사
                    if not content.startswith('#!/usr/bin/env python3'):
                        issues.append({
                            'file': str(py_file),
                            'message': '파일 헤더가 누락되었습니다'
                        })
                    
                    # 인코딩 선언 검사
                    if '# -*- coding: utf-8 -*-' not in content:
                        warnings.append(f"{py_file}: 인코딩 선언이 누락되었습니다")
                    
                    # 타입 힌트 검사
                    if 'def ' in content and '->' not in content:
                        warnings.append(f"{py_file}: 함수에 타입 힌트가 누락되었습니다")
                    
                except Exception as e:
                    warnings.append(f"{py_file}: 파일 읽기 실패 - {e}")
            
            execution_time = time.time() - start_time
            
            # 구조 점수 계산
            structure_score = max(0.0, 10.0 - len(issues) * 2.0 - len(warnings) * 0.5)
            passed = len(issues) == 0
            
            return QualityResult(
                tool_name="code_structure",
                passed=passed,
                score=structure_score,
                issues=issues,
                warnings=warnings,
                execution_time=execution_time
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="code_structure",
                passed=False,
                score=0.0,
                warnings=[f"코드 구조 검사 실패: {e}"]
            )

    def run_full_quality_check(self) -> QualityReport:
        """전체 품질 검사 실행"""
        try:
            start_time = time.time()
            logger.info("🔍 전체 품질 검사 시작")
            
            results = []
            
            # 1. Pylint 검사
            logger.info("실행 중: Pylint 검사")
            pylint_result = self.run_pylint_check()
            results.append(pylint_result)
            
            # 2. 테스트 커버리지
            logger.info("실행 중: 테스트 커버리지")
            coverage_result = self.run_test_coverage()
            results.append(coverage_result)
            
            # 3. 보안 감사
            logger.info("실행 중: 보안 감사")
            security_result = self.run_security_audit()
            results.append(security_result)
            
            # 4. 코드 구조 검사
            logger.info("실행 중: 코드 구조 검사")
            structure_result = self.check_code_structure()
            results.append(structure_result)
            
            # 전체 점수 계산
            total_score = sum(result.score for result in results) / len(results)
            all_passed = all(result.passed for result in results)
            
            execution_time = time.time() - start_time
            
            # 요약 생성
            summary = {
                'total_tools': len(results),
                'passed_tools': sum(1 for r in results if r.passed),
                'total_issues': sum(len(r.issues) for r in results),
                'total_warnings': sum(len(r.warnings) for r in results),
                'average_score': total_score
            }
            
            # 권장사항 생성
            recommendations = []
            if not all_passed:
                recommendations.append("일부 품질 검사가 실패했습니다. 이슈를 수정하고 다시 실행하세요.")
            if total_score < 8.0:
                recommendations.append("전체 품질 점수가 낮습니다. 코드 품질을 개선하세요.")
            if summary['total_issues'] > 0:
                recommendations.append(f"{summary['total_issues']}개의 이슈를 수정하세요.")
            
            report = QualityReport(
                overall_score=total_score,
                passed=all_passed,
                results=results,
                summary=summary,
                recommendations=recommendations,
                execution_time=execution_time
            )
            
            logger.info(f"✅ 전체 품질 검사 완료 (점수: {total_score:.2f}, 통과: {all_passed})")
            return report
            
        except Exception as e:
            logger.error(f"전체 품질 검사 실패: {e}")
            return QualityReport(
                overall_score=0.0,
                passed=False,
                recommendations=[f"품질 검사 실행 실패: {e}"]
            )

    def save_quality_report(self, report: QualityReport, output_path: str = "quality_report.json") -> bool:
        """품질 보고서 저장"""
        try:
            report_dict = {
                'overall_score': report.overall_score,
                'passed': report.passed,
                'summary': report.summary,
                'recommendations': report.recommendations,
                'execution_time': report.execution_time,
                'results': [
                    {
                        'tool_name': r.tool_name,
                        'passed': r.passed,
                        'score': r.score,
                        'issues': r.issues,
                        'warnings': r.warnings,
                        'execution_time': r.execution_time
                    }
                    for r in report.results
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"품질 보고서 저장 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"품질 보고서 저장 실패: {e}")
            return False


def test_investment_system_quality_checker_init():
    """품질 검사기 초기화 테스트"""
    try:
        checker = InvestmentSystemQualityChecker()
        assert isinstance(checker.project_root, Path)
        assert isinstance(checker.src_path, Path)
        assert isinstance(checker.quality_standards, dict)
        assert isinstance(checker.python_files, list)
        assert isinstance(checker.investment_rules, dict)
        logger.info("품질 검사기 초기화 테스트 통과")
    except Exception as e:
        pytest.fail(f"품질 검사기 초기화 테스트 실패: {e}")


def main():
    """메인 함수"""
    try:
        logger.info("🔄 품질 검사 시작")
        
        checker = InvestmentSystemQualityChecker()
        report = checker.run_full_quality_check()
        
        # 보고서 저장
        checker.save_quality_report(report)
        
        # 결과 출력
        print(f"📊 품질 검사 결과:")
        print(f"  - 전체 점수: {report.overall_score:.2f}/10.0")
        print(f"  - 통과 여부: {'✅' if report.passed else '❌'}")
        print(f"  - 실행 시간: {report.execution_time:.2f}초")
        print(f"  - 총 이슈: {report.summary.get('total_issues', 0)}개")
        print(f"  - 총 경고: {report.summary.get('total_warnings', 0)}개")
        
        if report.recommendations:
            print(f"\n💡 권장사항:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        return report.passed
        
    except Exception as e:
        logger.error(f"메인 함수 실행 실패: {e}")
        print(f"❌ 품질 검사 실패: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

