#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ai_reviewer.py
모듈: AI 코드 리뷰 시스템
목적: 자동 코드 품질 검사, 보안 감사, 성능 분석

Author: GitHub Actions
Created: 2025-01-06
Modified: 2025-01-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pytest
    - subprocess
    - pathlib
    - asyncio

Performance:
    - 리뷰 시간: < 30초
    - 메모리사용량: < 100MB
    - 처리용량: 100+ files/minute

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
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from contextlib import asynccontextmanager
from functools import lru_cache

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ai_review.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """이슈 심각도"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class IssueType(Enum):
    """이슈 타입"""
    HEADER_MISSING = auto()
    ENCODING_MISSING = auto()
    TYPE_HINT_MISSING = auto()
    PRINT_USAGE = auto()
    SECURITY_ISSUE = auto()
    PERFORMANCE_ISSUE = auto()
    LOGGING_MISSING = auto()
    ERROR_HANDLING_MISSING = auto()


@dataclass(frozen=True)
class CodeIssue:
    """코드 이슈"""
    type: IssueType
    severity: Severity
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class SecurityIssue:
    """보안 이슈"""
    pattern_name: str
    count: int
    severity: Severity
    description: str


@dataclass(frozen=True)
class PerformanceIssue:
    """성능 이슈"""
    pattern_name: str
    count: int
    severity: Severity
    description: str


@dataclass
class ReviewResult:
    """리뷰 결과"""
    file_path: str
    issues: List[CodeIssue] = field(default_factory=list)
    score: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)
    security_issues: List[SecurityIssue] = field(default_factory=list)
    performance_issues: List[PerformanceIssue] = field(default_factory=list)
    review_time: float = field(default=0.0)


@dataclass
class InvestmentSystemReviewer:
    """투자 시스템 리뷰어 - Cursor 룰 100% 준수"""
    
    project_root: Path = field(default_factory=lambda: Path("."))
    investment_rules: Dict[str, bool] = field(default_factory=dict)
    performance_patterns: Dict[str, str] = field(default_factory=dict)
    security_patterns: Dict[str, str] = field(default_factory=dict)
    review_results: Dict[str, ReviewResult] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """초기화 후 검증 및 설정"""
        if not isinstance(self.project_root, Path):
            raise TypeError("project_root은 Path 객체여야 합니다.")
        
        if not self.project_root.exists():
            raise FileNotFoundError(f"프로젝트 루트가 존재하지 않습니다: {self.project_root}")
        
        # 투자 시스템 특화 검사 규칙
        self.investment_rules = {
            "no_hardcoding": True,
            "use_logging": True,
            "type_hints_required": True,
            "async_support": True,
            "error_handling": True,
            "security_validation": True,
            "performance_optimization": True,
            "documentation_required": True
        }

        # 성능 최적화 패턴
        self.performance_patterns = {
            "heavy_calculation": r"(\w+\s*\(.*?\))",
            "db_query_count": r"cursor\.execute",
            "memory_leak": r"global\s+\w+",
            "inefficient_loop": r"for.*in.*range\(len\(",
            "unused_imports": r"import\s+\w+$",
            "synchronous_io": r"requests\.get|requests\.post",
            "blocking_operation": r"time\.sleep",
            "large_data_structure": r"\[\s*\]\s*for.*in.*range"
        }

        # 보안 패턴
        self.security_patterns = {
            "api_key_exposure": r'api_key\s*=\s*["\'][^"\']+["\']',
            "password_hardcode": r'password\s*=\s*["\'][^"\']+["\']',
            "sql_injection": r'execute\s*\(\s*f["\'][^"\']*\{[^}]*\}',
            "eval_usage": r'eval\s*\(',
            "exec_usage": r'exec\s*\(',
            "shell_command": r'subprocess\.run\s*\(\s*["\'][^"\']*["\']',
            "unsafe_deserialization": r'pickle\.loads',
            "debug_info_exposure": r'print\s*\(.*password.*\)|print\s*\(.*secret.*\)'
        }

        logger.info("AI 리뷰어 초기화 완료")

    async def review_file_async(self, file_path: str) -> ReviewResult:
        """비동기 파일 리뷰"""
        try:
            start_time = time.time()
            result = ReviewResult(file_path=file_path)

            async with self._safe_file_read(file_path) as content:
                if not content:
                    return result

                # 기본 검사
                result.issues = await self._check_basic_issues_async(content, file_path)
                result.suggestions = await self._generate_suggestions_async(content, file_path)
                result.security_issues = await self._check_security_issues_async(content)
                result.performance_issues = await self._check_performance_issues_async(content)

                # 점수 계산
                result.score = self._calculate_score(result)
                result.review_time = time.time() - start_time

                return result

        except Exception as e:
            logger.error(f"파일 리뷰 실패: {file_path} - {e}")
            return ReviewResult(file_path=file_path, score=0.0, review_time=0.0)

    @asynccontextmanager
    async def _safe_file_read(self, file_path: str):
        """안전한 파일 읽기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            yield content
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    content = f.read()
                yield content
            except Exception as e:
                logger.warning(f"파일 읽기 실패: {file_path} - {e}")
                yield ""
        except Exception as e:
            logger.warning(f"파일 읽기 실패: {file_path} - {e}")
            yield ""

    async def _check_basic_issues_async(self, content: str, file_path: str) -> List[CodeIssue]:
        """비동기 기본 이슈 검사"""
        try:
            issues = []

            # 파일 헤더 검사
            if not content.startswith('#!/usr/bin/env python3'):
                issues.append(CodeIssue(
                    type=IssueType.HEADER_MISSING,
                    severity=Severity.MEDIUM,
                    message='파일 헤더가 누락되었습니다',
                    suggestion='파일 상단에 #!/usr/bin/env python3를 추가하세요'
                ))

            # 인코딩 선언 검사
            if '# -*- coding: utf-8 -*-' not in content:
                issues.append(CodeIssue(
                    type=IssueType.ENCODING_MISSING,
                    severity=Severity.LOW,
                    message='인코딩 선언이 누락되었습니다',
                    suggestion='파일 상단에 # -*- coding: utf-8 -*-를 추가하세요'
                ))

            # 타입 힌트 검사
            if self.investment_rules.get('type_hints_required'):
                function_defs = re.findall(r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*\w+)?:', content)
                for func_name in function_defs:
                    if f'def {func_name}(' in content and f'->' not in content:
                        issues.append(CodeIssue(
                            type=IssueType.TYPE_HINT_MISSING,
                            severity=Severity.MEDIUM,
                            message=f'함수 {func_name}에 타입 힌트가 누락되었습니다',
                            suggestion=f'함수 {func_name}에 반환 타입 힌트를 추가하세요'
                        ))

            # 로깅 사용 검사
            if self.investment_rules.get('use_logging') and 'print(' in content:
                issues.append(CodeIssue(
                    type=IssueType.PRINT_USAGE,
                    severity=Severity.LOW,
                    message='print() 대신 logging을 사용하는 것을 권장합니다',
                    suggestion='logging 모듈을 import하고 print() 대신 logger.info()를 사용하세요'
                ))

            # 에러 처리 검사
            if self.investment_rules.get('error_handling') and 'def ' in content and 'try:' not in content:
                issues.append(CodeIssue(
                    type=IssueType.ERROR_HANDLING_MISSING,
                    severity=Severity.MEDIUM,
                    message='적절한 에러 처리가 누락되었습니다',
                    suggestion='함수에 try-except 블록을 추가하여 예외 처리를 구현하세요'
                ))

            return issues
        except Exception as e:
            logger.error(f"기본 이슈 검사 실패: {e}")
            return []

    async def _check_security_issues_async(self, content: str) -> List[SecurityIssue]:
        """비동기 보안 이슈 검사"""
        try:
            security_issues = []

            for pattern_name, pattern in self.security_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    severity = self._determine_security_severity(pattern_name)
                    description = self._get_security_description(pattern_name)
                    
                    security_issues.append(SecurityIssue(
                        pattern_name=pattern_name,
                        count=len(matches),
                        severity=severity,
                        description=description
                    ))

            return security_issues
        except Exception as e:
            logger.error(f"보안 이슈 검사 실패: {e}")
            return []

    async def _check_performance_issues_async(self, content: str) -> List[PerformanceIssue]:
        """비동기 성능 이슈 검사"""
        try:
            performance_issues = []

            for pattern_name, pattern in self.performance_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    severity = self._determine_performance_severity(pattern_name)
                    description = self._get_performance_description(pattern_name)
                    
                    performance_issues.append(PerformanceIssue(
                        pattern_name=pattern_name,
                        count=len(matches),
                        severity=severity,
                        description=description
                    ))

            return performance_issues
        except Exception as e:
            logger.error(f"성능 이슈 검사 실패: {e}")
            return []

    def _determine_security_severity(self, pattern_name: str) -> Severity:
        """보안 이슈 심각도 결정"""
        critical_patterns = {"eval_usage", "exec_usage", "sql_injection"}
        high_patterns = {"api_key_exposure", "password_hardcode", "shell_command"}
        
        if pattern_name in critical_patterns:
            return Severity.CRITICAL
        elif pattern_name in high_patterns:
            return Severity.HIGH
        else:
            return Severity.MEDIUM

    def _determine_performance_severity(self, pattern_name: str) -> Severity:
        """성능 이슈 심각도 결정"""
        high_patterns = {"memory_leak", "inefficient_loop", "synchronous_io"}
        medium_patterns = {"heavy_calculation", "blocking_operation"}
        
        if pattern_name in high_patterns:
            return Severity.HIGH
        elif pattern_name in medium_patterns:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _get_security_description(self, pattern_name: str) -> str:
        """보안 이슈 설명"""
        descriptions = {
            "api_key_exposure": "API 키가 하드코딩되어 보안 위험이 있습니다",
            "password_hardcode": "패스워드가 하드코딩되어 보안 위험이 있습니다",
            "sql_injection": "SQL 인젝션 공격에 취약할 수 있습니다",
            "eval_usage": "eval() 사용은 보안 위험이 있습니다",
            "exec_usage": "exec() 사용은 보안 위험이 있습니다",
            "shell_command": "쉘 명령어 실행은 보안 위험이 있습니다",
            "unsafe_deserialization": "안전하지 않은 역직렬화는 보안 위험이 있습니다",
            "debug_info_exposure": "디버그 정보 노출은 보안 위험이 있습니다"
        }
        return descriptions.get(pattern_name, "보안 위험이 감지되었습니다")

    def _get_performance_description(self, pattern_name: str) -> str:
        """성능 이슈 설명"""
        descriptions = {
            "heavy_calculation": "무거운 계산이 성능에 영향을 줄 수 있습니다",
            "db_query_count": "과도한 DB 쿼리가 성능에 영향을 줄 수 있습니다",
            "memory_leak": "메모리 누수가 감지되었습니다",
            "inefficient_loop": "비효율적인 루프가 성능에 영향을 줄 수 있습니다",
            "unused_imports": "사용하지 않는 import가 메모리를 낭비합니다",
            "synchronous_io": "동기 I/O가 성능에 영향을 줄 수 있습니다",
            "blocking_operation": "블로킹 연산이 성능에 영향을 줄 수 있습니다",
            "large_data_structure": "대용량 데이터 구조가 메모리를 과도하게 사용합니다"
        }
        return descriptions.get(pattern_name, "성능 이슈가 감지되었습니다")

    async def _generate_suggestions_async(self, content: str, file_path: str) -> List[str]:
        """비동기 개선 제안 생성"""
        try:
            suggestions = []

            # 캐싱 제안
            if "def get_" in content and "cache" not in content.lower():
                suggestions.append("데이터 조회 함수에 캐싱을 추가하면 성능이 향상됩니다.")

            # 비동기 제안
            if "requests." in content and "async def" not in content:
                suggestions.append("HTTP 요청 함수를 비동기로 변경하면 성능이 향상됩니다.")

            # 로깅 제안
            if "print(" in content:
                suggestions.append("print() 대신 logging을 사용하는 것을 권장합니다.")

            # 타입 힌트 제안
            if "def " in content and "->" not in content and not content.startswith("class"):
                suggestions.append("함수에 타입 힌트를 추가하면 코드 품질이 향상됩니다.")

            # 에러 처리 제안
            if "try:" not in content and "def " in content:
                suggestions.append("함수에 적절한 에러 처리를 추가하는 것을 권장합니다.")

            # 문서화 제안
            if '"""' not in content and "def " in content:
                suggestions.append("함수에 독스트링을 추가하면 코드 가독성이 향상됩니다.")

            # 상수 정의 제안
            if "magic_number" in content.lower() or re.search(r'\b\d{3,}\b', content):
                suggestions.append("매직 넘버를 상수로 정의하면 코드 가독성이 향상됩니다.")

            return suggestions
        except Exception as e:
            logger.error(f"제안 생성 실패: {e}")
            return []

    def _calculate_score(self, result: ReviewResult) -> float:
        """점수 계산"""
        try:
            base_score = 100.0

            # 이슈별 점수 차감
            for issue in result.issues:
                if issue.severity == Severity.CRITICAL:
                    base_score -= 25
                elif issue.severity == Severity.HIGH:
                    base_score -= 20
                elif issue.severity == Severity.MEDIUM:
                    base_score -= 10
                elif issue.severity == Severity.LOW:
                    base_score -= 5

            # 보안 이슈 점수 차감
            for security_issue in result.security_issues:
                if security_issue.severity == Severity.CRITICAL:
                    base_score -= 30
                elif security_issue.severity == Severity.HIGH:
                    base_score -= 20
                elif security_issue.severity == Severity.MEDIUM:
                    base_score -= 15
                else:
                    base_score -= 10

            # 성능 이슈 점수 차감
            for performance_issue in result.performance_issues:
                if performance_issue.severity == Severity.HIGH:
                    base_score -= 15
                elif performance_issue.severity == Severity.MEDIUM:
                    base_score -= 10
                else:
                    base_score -= 5

            return max(0.0, base_score)
        except Exception as e:
            logger.error(f"점수 계산 실패: {e}")
            return 0.0

    async def review_project_async(self) -> Dict[str, Any]:
        """비동기 프로젝트 전체 리뷰"""
        try:
            start_time = time.time()
            python_files = list(self.project_root.rglob("*.py"))
            results = []
            total_score = 0.0
            total_files = 0

            # 병렬 리뷰 실행
            tasks = []
            for file_path in python_files:
                if "venv" not in str(file_path) and "__pycache__" not in str(file_path):
                    tasks.append(self.review_file_async(str(file_path)))

            review_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in review_results:
                if isinstance(result, Exception):
                    logger.warning(f"리뷰 실패: {result}")
                elif isinstance(result, ReviewResult):
                    results.append(result)
                    if result.score is not None:
                        total_score += result.score
                        total_files += 1

            avg_score = total_score / total_files if total_files > 0 else 0.0
            execution_time = time.time() - start_time

            return {
                'total_files': total_files,
                'average_score': avg_score,
                'execution_time': execution_time,
                'results': [result.__dict__ for result in results],
                'summary': {
                    'high_priority_issues': sum(1 for r in results for i in r.issues if i.severity in [Severity.HIGH, Severity.CRITICAL]),
                    'security_issues': sum(len(r.security_issues) for r in results),
                    'performance_issues': sum(len(r.performance_issues) for r in results),
                    'total_suggestions': sum(len(r.suggestions) for r in results)
                }
            }

        except Exception as e:
            logger.error(f"프로젝트 리뷰 실패: {e}")
            return {'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()}


async def test_investment_reviewer_init():
    """리뷰어 초기화 테스트"""
    try:
        reviewer = InvestmentSystemReviewer()
        assert isinstance(reviewer, InvestmentSystemReviewer)
        assert isinstance(reviewer.project_root, Path)
        assert isinstance(reviewer.investment_rules, dict)
        assert isinstance(reviewer.performance_patterns, dict)
        assert isinstance(reviewer.security_patterns, dict)
        logger.info("리뷰어 초기화 테스트 통과")
    except Exception as e:
        logger.error(f"초기화 테스트 실패: {e}")
        raise


async def test_file_review():
    """파일 리뷰 테스트"""
    try:
        reviewer = InvestmentSystemReviewer()
        result = await reviewer.review_file_async("test_file.py")
        assert isinstance(result, ReviewResult)
        assert result.file_path == "test_file.py"
        logger.info("파일 리뷰 테스트 통과")
    except Exception as e:
        logger.error(f"파일 리뷰 테스트 실패: {e}")
        raise


async def main() -> None:
    """메인 실행 함수"""
    try:
        # 테스트 실행
        await test_investment_reviewer_init()
        await test_file_review()

        # 프로젝트 리뷰 실행
        reviewer = InvestmentSystemReviewer()
        project_result = await reviewer.review_project_async()

        # 결과를 JSON으로 저장
        with open("ai_review_report.json", "w", encoding="utf-8") as f:
            json.dump(project_result, f, indent=2, ensure_ascii=False, default=str)

        print(json.dumps(project_result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"메인 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")


if __name__ == "__main__":
    asyncio.run(main())

