#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 투자 분석 시스템 AI 코드 리뷰어
=================================

GitHub Actions에서 실행되는 AI 기반 코드 리뷰 시스템입니다.
투자 분석 시스템의 코드 품질, 보안, 성능을 자동으로 검토합니다.

주요 기능:
1. 코드 품질 검사
2. 보안 취약점 검사
3. 성능 최적화 제안
4. 투자 전략 로직 검증
5. API 사용량 최적화 검토
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import subprocess
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """리뷰 결과 데이터 클래스"""
    file_path: str
    issues: List[Dict[str, Any]]
    score: float
    suggestions: List[str]
    security_issues: List[str] = None
    performance_issues: List[str] = None


class InvestmentSystemReviewer:
    """투자 분석 시스템 전용 AI 리뷰어"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self.config_path = self.project_root / "config"
        
        # 투자 시스템 특화 검사 규칙
        self.investment_rules = {
            'api_key_security': r'(api_key|API_KEY)\s*=\s*["\'][^"\']+["\']',
            'hardcoded_credentials': r'(password|secret|token)\s*=\s*["\'][^"\']+["\']',
            'unsafe_eval': r'eval\s*\(',
            'sql_injection': r'(execute|query)\s*\(\s*["\'].*%.*["\']',
            'gemini_api_usage': r'gemini.*generate',
            'yfinance_calls': r'yf\.(download|Ticker)',
            'async_without_await': r'async\s+def.*\n(?!.*await)',
            'missing_error_handling': r'requests\.(get|post).*\n(?!.*except)',
        }
        
        # 성능 최적화 패턴
        self.performance_patterns = {
            'sync_in_async': r'def.*\n.*requests\.',
            'missing_connection_pool': r'requests\.(get|post)',
            'inefficient_loop': r'for.*in.*range\(len\(',
            'pandas_inefficiency': r'\.iterrows\(\)',
            'missing_cache': r'def.*fetch.*\n(?!.*cache)',
        }
    
    async def review_changed_files(self, changed_files: List[str]) -> List[ReviewResult]:
        """변경된 파일들을 리뷰"""
        results = []
        
        for file_path in changed_files:
            if self._should_review_file(file_path):
                result = await self._review_file(file_path)
                if result:
                    results.append(result)
        
        return results
    
    def _should_review_file(self, file_path: str) -> bool:
        """리뷰 대상 파일인지 확인"""
        if not file_path.endswith('.py'):
            return False
        
        # 제외할 파일들
        exclude_patterns = [
            '__pycache__',
            '.pyc',
            'test_',
            'tests/',
            '.git/',
            'venv/',
            '.venv/'
        ]
        
        return not any(pattern in file_path for pattern in exclude_patterns)
    
    async def _review_file(self, file_path: str) -> Optional[ReviewResult]:
        """개별 파일 리뷰"""
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                return None
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues = []
            security_issues = []
            performance_issues = []
            suggestions = []
            
            # 보안 검사
            security_issues.extend(self._check_security(content, file_path))
            
            # 성능 검사
            performance_issues.extend(self._check_performance(content, file_path))
            
            # 투자 시스템 특화 검사
            issues.extend(self._check_investment_logic(content, file_path))
            
            # 코드 품질 검사
            issues.extend(await self._check_code_quality(content, file_path))
            
            # 제안사항 생성
            suggestions = self._generate_suggestions(content, file_path)
            
            # 점수 계산 (0-100)
            score = self._calculate_score(issues, security_issues, performance_issues)
            
            return ReviewResult(
                file_path=file_path,
                issues=issues,
                score=score,
                suggestions=suggestions,
                security_issues=security_issues,
                performance_issues=performance_issues
            )
            
        except Exception as e:
            logger.error(f"Error reviewing file {file_path}: {e}")
            return None
    
    def _check_security(self, content: str, file_path: str) -> List[str]:
        """보안 취약점 검사"""
        issues = []
        
        for rule_name, pattern in self.investment_rules.items():
            if 'security' in rule_name or 'credentials' in rule_name:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    issues.append(f"🔒 보안 이슈 ({rule_name}): {len(matches)}개 발견")
        
        # API 키 하드코딩 검사
        if 'GEMINI_API_KEY' in content and '"' in content:
            issues.append("🔒 Gemini API 키가 하드코딩되어 있을 수 있습니다. 환경변수 사용을 권장합니다.")
        
        return issues
    
    def _check_performance(self, content: str, file_path: str) -> List[str]:
        """성능 이슈 검사"""
        issues = []
        
        for pattern_name, pattern in self.performance_patterns.items():
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                if pattern_name == 'sync_in_async':
                    issues.append("⚡ 비동기 함수에서 동기 HTTP 요청을 사용하고 있습니다. aiohttp 사용을 권장합니다.")
                elif pattern_name == 'missing_connection_pool':
                    issues.append("⚡ 연결 풀 없이 HTTP 요청을 하고 있습니다. 성능 최적화가 필요합니다.")
                elif pattern_name == 'inefficient_loop':
                    issues.append("⚡ 비효율적인 반복문 패턴이 발견되었습니다. enumerate() 사용을 권장합니다.")
                elif pattern_name == 'pandas_inefficiency':
                    issues.append("⚡ pandas iterrows() 사용이 비효율적입니다. vectorized 연산 사용을 권장합니다.")
        
        return issues
    
    def _check_investment_logic(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """투자 로직 특화 검사"""
        issues = []
        
        # 투자 전략 파일 검사
        if 'strategy' in file_path.lower():
            if 'def analyze' not in content:
                issues.append({
                    'type': 'missing_method',
                    'message': '💡 투자 전략 클래스에 analyze 메서드가 없습니다.',
                    'severity': 'warning'
                })
            
            if 'risk' not in content.lower():
                issues.append({
                    'type': 'missing_risk',
                    'message': '💡 위험 관리 로직이 없습니다. 리스크 계산을 추가하세요.',
                    'severity': 'warning'
                })
        
        # 데이터 수집 파일 검사
        if 'data' in file_path.lower() or 'collector' in file_path.lower():
            if 'try:' not in content or 'except:' not in content:
                issues.append({
                    'type': 'missing_exception',
                    'message': '📊 데이터 수집에서 예외 처리가 부족합니다.',
                    'severity': 'error'
                })
        
        # AI 분석 파일 검사
        if 'ai' in file_path.lower() or 'gemini' in file_path.lower():
            if 'rate_limit' not in content.lower():
                issues.append({
                    'type': 'missing_rate_limit',
                    'message': '🤖 AI API 호출에 rate limiting이 없습니다.',
                    'severity': 'warning'
                })
        
        return issues
    
    async def _check_code_quality(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """코드 품질 검사"""
        issues = []
        
        # 함수 길이 검사
        functions = re.findall(r'def\s+\w+.*?(?=\n\S|\Z)', content, re.DOTALL)
        for func in functions:
            lines = func.count('\n')
            if lines > 50:
                issues.append({
                    'type': 'long_function',
                    'message': f'📏 함수가 너무 깁니다 ({lines}줄). 분할을 고려하세요.',
                    'severity': 'warning'
                })
        
        # 주석 부족 검사
        comment_lines = len([line for line in content.split('\n') if line.strip().startswith('#')])
        total_lines = len([line for line in content.split('\n') if line.strip()])
        
        if total_lines > 20 and comment_lines / total_lines < 0.1:
            issues.append({
                'type': 'insufficient_comments',
                'message': '📝 주석이 부족합니다. 코드 가독성을 위해 주석을 추가하세요.',
                'severity': 'info'
            })
        
        return issues
    
    def _generate_suggestions(self, content: str, file_path: str) -> List[str]:
        """개선 제안사항 생성"""
        suggestions = []
        
        # 캐싱 제안
        if 'def get_' in content and 'cache' not in content.lower():
            suggestions.append("🚀 데이터 조회 함수에 캐싱을 추가하면 성능이 향상됩니다.")
        
        # 비동기 처리 제안
        if 'requests.' in content and 'async def' not in content:
            suggestions.append("⚡ HTTP 요청 함수를 비동기로 변경하면 성능이 향상됩니다.")
        
        # 로깅 제안
        if 'print(' in content:
            suggestions.append("📝 print() 대신 logging을 사용하는 것을 권장합니다.")
        
        # 타입 힌트 제안
        if 'def ' in content and '->' not in content:
            suggestions.append("🔍 함수에 타입 힌트를 추가하면 코드 품질이 향상됩니다.")
        
        return suggestions
    
    def _calculate_score(self, issues: List[Dict], security_issues: List[str], 
                        performance_issues: List[str]) -> float:
        """코드 품질 점수 계산 (0-100)"""
        base_score = 100.0
        
        # 보안 이슈 페널티
        base_score -= len(security_issues) * 15
        
        # 성능 이슈 페널티
        base_score -= len(performance_issues) * 10
        
        # 일반 이슈 페널티
        for issue in issues:
            if issue.get('severity') == 'error':
                base_score -= 20
            elif issue.get('severity') == 'warning':
                base_score -= 10
            else:
                base_score -= 5
        
        return max(0.0, min(100.0, base_score))
    
    def generate_review_comment(self, results: List[ReviewResult]) -> str:
        """리뷰 댓글 생성"""
        if not results:
            return "✅ **AI 코드 리뷰 완료** - 검토할 Python 파일이 없습니다."
        
        comment_lines = [
            "🤖 **AI 투자 시스템 코드 리뷰 결과**",
            "",
            f"📊 **총 {len(results)}개 파일 검토 완료**",
            ""
        ]
        
        total_score = sum(r.score for r in results) / len(results)
        
        # 전체 점수 표시
        if total_score >= 90:
            comment_lines.append(f"🎉 **전체 점수: {total_score:.1f}/100** - 우수한 코드 품질!")
        elif total_score >= 70:
            comment_lines.append(f"✅ **전체 점수: {total_score:.1f}/100** - 양호한 코드 품질")
        else:
            comment_lines.append(f"⚠️ **전체 점수: {total_score:.1f}/100** - 개선이 필요합니다")
        
        comment_lines.append("")
        
        # 파일별 상세 리뷰
        for result in results:
            comment_lines.append(f"### 📁 `{result.file_path}`")
            comment_lines.append(f"**점수: {result.score:.1f}/100**")
            comment_lines.append("")
            
            # 보안 이슈
            if result.security_issues:
                comment_lines.append("🔒 **보안 이슈:**")
                for issue in result.security_issues:
                    comment_lines.append(f"- {issue}")
                comment_lines.append("")
            
            # 성능 이슈
            if result.performance_issues:
                comment_lines.append("⚡ **성능 이슈:**")
                for issue in result.performance_issues:
                    comment_lines.append(f"- {issue}")
                comment_lines.append("")
            
            # 일반 이슈
            if result.issues:
                comment_lines.append("📋 **코드 품질 이슈:**")
                for issue in result.issues:
                    if isinstance(issue, dict):
                        comment_lines.append(f"- {issue['message']}")
                    else:
                        comment_lines.append(f"- {issue}")
                comment_lines.append("")
            
            # 제안사항
            if result.suggestions:
                comment_lines.append("💡 **개선 제안:**")
                for suggestion in result.suggestions:
                    comment_lines.append(f"- {suggestion}")
                comment_lines.append("")
            
            comment_lines.append("---")
            comment_lines.append("")
        
        # 마무리 메시지
        comment_lines.extend([
            "🚀 **투자 분석 시스템 최적화 팁:**",
            "- 모든 API 호출에 적절한 에러 핸들링과 재시도 로직을 구현하세요",
            "- 데이터 수집 함수에는 캐싱을 적용하여 성능을 최적화하세요",
            "- 비동기 처리를 활용하여 동시성을 향상시키세요",
            "- API 키는 반드시 환경변수로 관리하세요",
            "- 투자 전략에는 리스크 관리 로직을 포함시키세요",
            "",
            "*🤖 AI 리뷰어가 자동으로 생성한 리뷰입니다.*"
        ])
        
        return "\n".join(comment_lines)


async def main():
    """메인 실행 함수"""
    try:
        # GitHub Actions 환경에서 변경된 파일 목록 얻기
        changed_files = []
        
        # PR의 변경된 파일들 가져오기
        if os.getenv('GITHUB_EVENT_NAME') == 'pull_request':
            # GitHub API를 통해 변경된 파일 목록 가져오기
            result = subprocess.run([
                'gh', 'pr', 'diff', '--name-only'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                changed_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        
        # 로컬 테스트용 (모든 Python 파일)
        if not changed_files:
            project_root = Path(__file__).parent.parent.parent
            changed_files = [
                str(p.relative_to(project_root)) 
                for p in project_root.rglob('*.py')
                if 'venv' not in str(p) and '__pycache__' not in str(p)
            ]
        
        # AI 리뷰 실행
        reviewer = InvestmentSystemReviewer()
        results = await reviewer.review_changed_files(changed_files)
        
        # 결과 출력
        comment = reviewer.generate_review_comment(results)
        print(comment)
        
        # GitHub PR에 댓글 추가 (GitHub Actions 환경에서만)
        if os.getenv('GITHUB_TOKEN') and os.getenv('GITHUB_EVENT_NAME') == 'pull_request':
            with open('review_comment.md', 'w', encoding='utf-8') as f:
                f.write(comment)
        
        return 0 if all(r.score >= 70 for r in results) else 1
        
    except Exception as e:
        logger.error(f"AI 리뷰 실행 중 오류: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
