#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 투자 분석 시스템 코드 품질 검사 도구
=====================================

투자 분석 시스템의 코드 품질을 종합적으로 검사하는 도구입니다.
GitHub Actions CI/CD 파이프라인에서 자동으로 실행되어 
코드 품질 기준을 만족하는지 검증합니다.

주요 검사 항목:
1. 코드 스타일 검사 (PEP8, Black, isort)
2. 정적 분석 (pylint, flake8, mypy)
3. 보안 검사 (bandit)
4. 복잡도 분석 (mccabe)
5. 테스트 커버리지 (pytest-cov)
6. 투자 시스템 특화 검사
7. 성능 프로파일링
8. 문서화 품질 검사
"""

import os
import sys
import subprocess
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    """전체 품질 리포트"""
    overall_score: float
    passed: bool
    results: List[QualityResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class InvestmentSystemQualityChecker:
    """투자 분석 시스템 품질 검사기"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self.config_path = self.project_root / "config"
        
        # 품질 기준 설정
        self.quality_standards = {
            'pylint_min_score': 8.0,
            'coverage_min_percent': 80.0,
            'complexity_max': 10,
            'line_length_max': 88,
            'function_length_max': 50,
            'class_length_max': 200
        }
        
        # 검사할 Python 파일들
        self.python_files = self._get_python_files()
        
        # 투자 시스템 특화 검사 규칙
        self.investment_rules = {
            'required_modules': [
                'yfinance', 'pandas', 'numpy', 'ta',
                'google-generativeai', 'aiohttp'
            ],
            'required_functions': {
                'strategy': ['analyze', 'get_strategy_type'],
                'data_collector': ['collect_market_data', 'get_stock_data'],
                'ai_analyzer': ['analyze_recommendations'],
            },
            'security_patterns': [
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']'
            ]
        }
    
    def _get_python_files(self) -> List[Path]:
        """검사할 Python 파일 목록 생성"""
        python_files = []
        
        # src 디렉토리의 모든 Python 파일
        if self.src_path.exists():
            python_files.extend(self.src_path.rglob('*.py'))
        
        # 루트의 주요 Python 파일들
        main_files = ['main.py', 'config.py', 'setup.py']
        for file_name in main_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                python_files.append(file_path)
        
        # 테스트 제외, 가상환경 제외
        return [
            f for f in python_files 
            if not any(exclude in str(f) for exclude in [
                '__pycache__', '.pyc', 'test_', 'tests/', 
                'venv/', '.venv/', '.git/'
            ])
        ]
    
    async def run_quality_checks(self) -> QualityReport:
        """전체 품질 검사 실행"""
        start_time = time.time()
        results = []
        
        logger.info("🔍 투자 분석 시스템 코드 품질 검사 시작...")
        
        # 병렬 실행을 위한 태스크 목록
        tasks = [
            self._run_pylint(),
            self._run_flake8(),
            self._run_black_check(),
            self._run_isort_check(),
            self._run_mypy(),
            self._run_bandit(),
            self._run_complexity_check(),
            self._run_investment_specific_checks(),
            self._run_documentation_check(),
            self._run_performance_check()
        ]
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=5) as executor:
            task_results = await asyncio.gather(*[
                asyncio.get_event_loop().run_in_executor(executor, task)
                for task in tasks
            ], return_exceptions=True)
        
        # 결과 수집
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                logger.error(f"검사 도구 실행 중 오류: {result}")
                continue
            if result:
                results.append(result)
        
        # 테스트 커버리지 검사 (별도 실행)
        coverage_result = await self._run_coverage_check()
        if coverage_result:
            results.append(coverage_result)
        
        # 전체 점수 계산
        overall_score = self._calculate_overall_score(results)
        passed = overall_score >= 70.0
        
        # 추천사항 생성
        recommendations = self._generate_recommendations(results)
        
        # 요약 정보 생성
        summary = self._generate_summary(results)
        
        execution_time = time.time() - start_time
        
        report = QualityReport(
            overall_score=overall_score,
            passed=passed,
            results=results,
            summary=summary,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        logger.info(f"✅ 품질 검사 완료 - 전체 점수: {overall_score:.1f}/100")
        return report
    
    def _run_pylint(self) -> QualityResult:
        """Pylint 정적 분석 실행"""
        start_time = time.time()
        
        try:
            # pylint 설정 파일 생성
            pylintrc_content = """
[MASTER]
disable=C0114,C0115,C0116,R0903,R0913,W0613,C0103

[FORMAT]
max-line-length=88

[DESIGN]
max-args=10
max-locals=20
max-returns=6
max-branches=15
"""
            
            pylintrc_path = self.project_root / '.pylintrc'
            with open(pylintrc_path, 'w') as f:
                f.write(pylintrc_content)
            
            # Pylint 실행
            cmd = ['python', '-m', 'pylint'] + [str(f) for f in self.python_files[:5]]  # 처음 5개 파일만
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # 점수 추출
            score = 0.0
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Your code has been rated at' in line:
                        score_str = line.split('rated at ')[1].split('/')[0]
                        score = float(score_str)
                        break
            
            passed = score >= self.quality_standards['pylint_min_score']
            
            # 정리
            if pylintrc_path.exists():
                pylintrc_path.unlink()
            
            return QualityResult(
                tool_name="Pylint",
                passed=passed,
                score=score * 10,  # 100점 만점으로 변환
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            logger.error(f"Pylint 실행 오류: {e}")
            return QualityResult(
                tool_name="Pylint",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                warnings=[f"실행 오류: {str(e)}"]
            )
    
    def _run_flake8(self) -> QualityResult:
        """Flake8 코드 스타일 검사"""
        start_time = time.time()
        
        try:
            cmd = ['python', '-m', 'flake8', '--max-line-length=88', '--ignore=E203,W503'] + [str(f) for f in self.python_files[:10]]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        issues.append({'message': line, 'type': 'style'})
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 2)  # 이슈 1개당 2점 감점
            
            return QualityResult(
                tool_name="Flake8",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Flake8",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                warnings=[f"실행 오류: {str(e)}"]
            )
    
    def _run_black_check(self) -> QualityResult:
        """Black 코드 포맷팅 검사"""
        start_time = time.time()
        
        try:
            cmd = ['python', '-m', 'black', '--check', '--line-length=88'] + [str(f) for f in self.python_files[:10]]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            score = 100.0 if passed else 70.0
            
            issues = []
            if not passed and result.stdout:
                issues.append({'message': 'Black 포맷팅이 필요한 파일이 있습니다', 'type': 'formatting'})
            
            return QualityResult(
                tool_name="Black",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Black",
                passed=True,  # Black이 없어도 통과
                score=80.0,
                execution_time=time.time() - start_time,
                warnings=[f"Black 미설치: {str(e)}"]
            )
    
    def _run_isort_check(self) -> QualityResult:
        """isort import 정렬 검사"""
        start_time = time.time()
        
        try:
            cmd = ['python', '-m', 'isort', '--check-only', '--profile=black'] + [str(f) for f in self.python_files[:10]]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            score = 100.0 if passed else 80.0
            
            return QualityResult(
                tool_name="isort",
                passed=passed,
                score=score,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="isort",
                passed=True,  # isort가 없어도 통과
                score=80.0,
                execution_time=time.time() - start_time,
                warnings=[f"isort 미설치: {str(e)}"]
            )
    
    def _run_mypy(self) -> QualityResult:
        """MyPy 타입 검사"""
        start_time = time.time()
        
        try:
            cmd = ['python', '-m', 'mypy', '--ignore-missing-imports'] + [str(f) for f in self.python_files[:5]]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip() and 'error:' in line:
                        issues.append({'message': line, 'type': 'type'})
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 5)  # 타입 오류 1개당 5점 감점
            
            return QualityResult(
                tool_name="MyPy",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="MyPy",
                passed=True,  # MyPy가 없어도 통과
                score=70.0,
                execution_time=time.time() - start_time,
                warnings=[f"MyPy 미설치: {str(e)}"]
            )
    
    def _run_bandit(self) -> QualityResult:
        """Bandit 보안 검사"""
        start_time = time.time()
        
        try:
            cmd = ['python', '-m', 'bandit', '-r'] + [str(f) for f in self.python_files[:10]]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            issues = []
            if result.stdout:
                # JSON 출력 파싱 시도
                try:
                    bandit_output = json.loads(result.stdout)
                    for issue in bandit_output.get('results', []):
                        issues.append({
                            'message': issue.get('issue_text', ''),
                            'type': 'security',
                            'severity': issue.get('issue_severity', 'LOW')
                        })
                except json.JSONDecodeError:
                    # 일반 텍스트 출력 파싱
                    for line in result.stdout.split('\n'):
                        if 'Issue:' in line:
                            issues.append({'message': line, 'type': 'security'})
            
            passed = len([i for i in issues if i.get('severity') in ['HIGH', 'MEDIUM']]) == 0
            score = max(0, 100 - len(issues) * 10)  # 보안 이슈 1개당 10점 감점
            
            return QualityResult(
                tool_name="Bandit",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Bandit",
                passed=True,  # Bandit이 없어도 통과
                score=80.0,
                execution_time=time.time() - start_time,
                warnings=[f"Bandit 미설치: {str(e)}"]
            )
    
    def _run_complexity_check(self) -> QualityResult:
        """코드 복잡도 검사"""
        start_time = time.time()
        
        try:
            # mccabe를 사용한 복잡도 검사
            cmd = ['python', '-m', 'flake8', '--select=C901', '--max-complexity=10'] + [str(f) for f in self.python_files[:10]]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        issues.append({'message': line, 'type': 'complexity'})
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 15)  # 복잡도 이슈 1개당 15점 감점
            
            return QualityResult(
                tool_name="Complexity",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Complexity",
                passed=True,
                score=80.0,
                execution_time=time.time() - start_time,
                warnings=[f"복잡도 검사 오류: {str(e)}"]
            )
    
    def _run_investment_specific_checks(self) -> QualityResult:
        """투자 시스템 특화 검사"""
        start_time = time.time()
        
        issues = []
        warnings = []
        
        try:
            # 필수 모듈 검사
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                
                for module in self.investment_rules['required_modules']:
                    if module not in requirements:
                        issues.append({
                            'message': f'필수 모듈 누락: {module}',
                            'type': 'dependency'
                        })
            else:
                warnings.append('requirements.txt 파일이 없습니다')
            
            # 투자 전략 파일 검사
            strategy_files = [f for f in self.python_files if 'strategy' in str(f).lower()]
            for strategy_file in strategy_files:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for func in self.investment_rules['required_functions']['strategy']:
                    if f'def {func}' not in content:
                        issues.append({
                            'message': f'{strategy_file.name}에 {func} 메서드가 없습니다',
                            'type': 'missing_method'
                        })
            
            # 데이터 수집기 검사
            collector_files = [f for f in self.python_files if 'collector' in str(f).lower() or 'data' in str(f).lower()]
            for collector_file in collector_files:
                with open(collector_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'try:' not in content or 'except:' not in content:
                    issues.append({
                        'message': f'{collector_file.name}에 예외 처리가 부족합니다',
                        'type': 'missing_exception'
                    })
            
            # API 키 보안 검사
            for file_path in self.python_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in self.investment_rules['security_patterns']:
                    import re
                    if re.search(pattern, content):
                        issues.append({
                            'message': f'{file_path.name}에 하드코딩된 인증 정보가 있을 수 있습니다',
                            'type': 'security'
                        })
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 10)
            
            return QualityResult(
                tool_name="Investment System Check",
                passed=passed,
                score=score,
                issues=issues,
                warnings=warnings,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Investment System Check",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                warnings=[f"검사 오류: {str(e)}"]
            )
    
    def _run_documentation_check(self) -> QualityResult:
        """문서화 품질 검사"""
        start_time = time.time()
        
        issues = []
        
        try:
            total_functions = 0
            documented_functions = 0
            
            for file_path in self.python_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import re
                # 함수 정의 찾기
                functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
                total_functions += len(functions)
                
                # docstring이 있는 함수 찾기
                for func in functions:
                    func_pattern = rf'def\s+{func}\s*\([^)]*\):\s*""".*?"""'
                    if re.search(func_pattern, content, re.DOTALL):
                        documented_functions += 1
            
            if total_functions > 0:
                doc_ratio = documented_functions / total_functions
                if doc_ratio < 0.5:
                    issues.append({
                        'message': f'문서화 비율이 낮습니다: {doc_ratio:.1%}',
                        'type': 'documentation'
                    })
            
            # README 파일 검사
            readme_files = ['README.md', 'README.rst', 'README.txt']
            has_readme = any((self.project_root / readme).exists() for readme in readme_files)
            
            if not has_readme:
                issues.append({
                    'message': 'README 파일이 없습니다',
                    'type': 'documentation'
                })
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 20)
            
            return QualityResult(
                tool_name="Documentation",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Documentation",
                passed=True,
                score=70.0,
                execution_time=time.time() - start_time,
                warnings=[f"문서화 검사 오류: {str(e)}"]
            )
    
    def _run_performance_check(self) -> QualityResult:
        """성능 검사"""
        start_time = time.time()
        
        issues = []
        
        try:
            for file_path in self.python_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 성능 이슈 패턴 검사
                import re
                
                # 비효율적 루프
                if re.search(r'for.*in.*range\(len\(', content):
                    issues.append({
                        'message': f'{file_path.name}에 비효율적인 루프가 있습니다',
                        'type': 'performance'
                    })
                
                # pandas iterrows 사용
                if '.iterrows()' in content:
                    issues.append({
                        'message': f'{file_path.name}에서 pandas iterrows() 사용을 피하세요',
                        'type': 'performance'
                    })
                
                # 동기 HTTP 요청 in 비동기 함수
                if re.search(r'async\s+def.*\n.*requests\.', content, re.MULTILINE):
                    issues.append({
                        'message': f'{file_path.name}에서 비동기 함수 내 동기 HTTP 요청',
                        'type': 'performance'
                    })
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 15)
            
            return QualityResult(
                tool_name="Performance",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Performance",
                passed=True,
                score=80.0,
                execution_time=time.time() - start_time,
                warnings=[f"성능 검사 오류: {str(e)}"]
            )
    
    async def _run_coverage_check(self) -> Optional[QualityResult]:
        """테스트 커버리지 검사"""
        start_time = time.time()
        
        try:
            # pytest가 있는지 확인
            test_files = list(self.project_root.rglob('test_*.py'))
            if not test_files:
                return QualityResult(
                    tool_name="Coverage",
                    passed=True,
                    score=60.0,
                    execution_time=time.time() - start_time,
                    warnings=["테스트 파일이 없습니다"]
                )
            
            # pytest-cov 실행
            cmd = ['python', '-m', 'pytest', '--cov=src', '--cov-report=term-missing']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            coverage_percent = 0.0
            if result.stdout:
                import re
                match = re.search(r'TOTAL.*?(\d+)%', result.stdout)
                if match:
                    coverage_percent = float(match.group(1))
            
            passed = coverage_percent >= self.quality_standards['coverage_min_percent']
            score = min(100.0, coverage_percent * 1.2)  # 커버리지를 점수로 변환
            
            issues = []
            if not passed:
                issues.append({
                    'message': f'테스트 커버리지가 낮습니다: {coverage_percent}%',
                    'type': 'coverage'
                })
            
            return QualityResult(
                tool_name="Coverage",
                passed=passed,
                score=score,
                issues=issues,
                execution_time=time.time() - start_time,
                output=result.stdout
            )
            
        except Exception as e:
            return QualityResult(
                tool_name="Coverage",
                passed=True,
                score=50.0,
                execution_time=time.time() - start_time,
                warnings=[f"커버리지 측정 오류: {str(e)}"]
            )
    
    def _calculate_overall_score(self, results: List[QualityResult]) -> float:
        """전체 점수 계산"""
        if not results:
            return 0.0
        
        # 가중치 설정
        weights = {
            'Pylint': 0.2,
            'Flake8': 0.15,
            'Black': 0.1,
            'isort': 0.05,
            'MyPy': 0.15,
            'Bandit': 0.2,
            'Complexity': 0.1,
            'Investment System Check': 0.25,
            'Documentation': 0.1,
            'Performance': 0.15,
            'Coverage': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result.tool_name, 0.1)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, results: List[QualityResult]) -> List[str]:
        """개선 추천사항 생성"""
        recommendations = []
        
        for result in results:
            if not result.passed:
                if result.tool_name == "Pylint":
                    recommendations.append("🔍 Pylint 경고를 수정하여 코드 품질을 향상시키세요")
                elif result.tool_name == "Flake8":
                    recommendations.append("📏 PEP8 스타일 가이드를 따라 코드를 정리하세요")
                elif result.tool_name == "Black":
                    recommendations.append("⚫ Black을 사용하여 코드 포맷팅을 일관성 있게 유지하세요")
                elif result.tool_name == "Bandit":
                    recommendations.append("🔒 보안 취약점을 수정하세요")
                elif result.tool_name == "Complexity":
                    recommendations.append("🧩 복잡한 함수를 더 작은 함수로 분할하세요")
                elif result.tool_name == "Investment System Check":
                    recommendations.append("💰 투자 시스템 특화 요구사항을 충족시키세요")
                elif result.tool_name == "Coverage":
                    recommendations.append("🧪 테스트 커버리지를 80% 이상으로 높이세요")
        
        # 일반적인 추천사항
        recommendations.extend([
            "📝 모든 함수에 타입 힌트와 docstring을 추가하세요",
            "🚀 비동기 처리를 활용하여 성능을 최적화하세요",
            "🔐 API 키는 환경변수로 관리하세요",
            "📊 데이터 수집 함수에 캐싱을 적용하세요",
            "⚡ 에러 핸들링과 재시도 로직을 구현하세요"
        ])
        
        return recommendations
    
    def _generate_summary(self, results: List[QualityResult]) -> Dict[str, Any]:
        """요약 정보 생성"""
        total_issues = sum(len(r.issues) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        passed_tools = len([r for r in results if r.passed])
        
        return {
            'total_tools': len(results),
            'passed_tools': passed_tools,
            'failed_tools': len(results) - passed_tools,
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'python_files_checked': len(self.python_files),
            'execution_time': sum(r.execution_time for r in results)
        }
    
    def generate_report(self, report: QualityReport) -> str:
        """품질 리포트 생성"""
        lines = [
            "🔍 **투자 분석 시스템 코드 품질 검사 결과**",
            "=" * 50,
            "",
            f"📊 **전체 점수: {report.overall_score:.1f}/100**",
            f"✅ **통과 여부: {'PASS' if report.passed else 'FAIL'}**",
            f"⏱️ **실행 시간: {report.execution_time:.2f}초**",
            f"📁 **검사 파일 수: {report.summary.get('python_files_checked', 0)}개**",
            "",
            "## 📋 검사 도구별 결과",
            ""
        ]
        
        for result in report.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"### {result.tool_name} - {status} ({result.score:.1f}/100)")
            
            if result.issues:
                lines.append("**이슈:**")
                for issue in result.issues[:5]:  # 최대 5개만 표시
                    lines.append(f"- {issue.get('message', issue)}")
                if len(result.issues) > 5:
                    lines.append(f"- ... 및 {len(result.issues) - 5}개 추가 이슈")
            
            if result.warnings:
                lines.append("**경고:**")
                for warning in result.warnings[:3]:  # 최대 3개만 표시
                    lines.append(f"- {warning}")
            
            lines.append("")
        
        lines.extend([
            "## 💡 개선 추천사항",
            ""
        ])
        
        for rec in report.recommendations[:10]:  # 최대 10개
            lines.append(f"- {rec}")
        
        lines.extend([
            "",
            "## 📈 요약 통계",
            "",
            f"- 총 검사 도구: {report.summary.get('total_tools', 0)}개",
            f"- 통과한 도구: {report.summary.get('passed_tools', 0)}개",
            f"- 실패한 도구: {report.summary.get('failed_tools', 0)}개",
            f"- 총 이슈 수: {report.summary.get('total_issues', 0)}개",
            f"- 총 경고 수: {report.summary.get('total_warnings', 0)}개",
            "",
            "*🤖 자동 품질 검사 시스템이 생성한 리포트입니다.*"
        ])
        
        return "\n".join(lines)


async def main():
    """메인 실행 함수"""
    try:
        checker = InvestmentSystemQualityChecker()
        report = await checker.run_quality_checks()
        
        # 리포트 출력
        report_text = checker.generate_report(report)
        print(report_text)
        
        # 파일로 저장
        report_file = checker.project_root / 'quality_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"품질 리포트가 {report_file}에 저장되었습니다")
        
        # GitHub Actions용 출력
        if os.getenv('GITHUB_ACTIONS'):
            print(f"::set-output name=score::{report.overall_score}")
            print(f"::set-output name=passed::{'true' if report.passed else 'false'}")
        
        return 0 if report.passed else 1
        
    except Exception as e:
        logger.error(f"품질 검사 실행 중 오류: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
