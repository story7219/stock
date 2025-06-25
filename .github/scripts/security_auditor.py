#!/usr/bin/env python3
"""
🔒 보안 감사 시스템
투자 시스템의 보안 취약점을 자동으로 검사하고 보고하는 도구
"""

import os
import sys
import json
import re
import ast
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
import hashlib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityAuditor:
    """보안 감사 도구"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.vulnerabilities = []
        self.security_score = 100
        
        # 보안 패턴 정의
        self.security_patterns = {
            'hardcoded_secrets': [
                r'(?i)(password|pwd|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                r'(?i)(api_key|apikey|access_key)\s*=\s*["\'][^"\']{10,}["\']',
                r'(?i)(private_key|secret_key)\s*=\s*["\'][^"\']{20,}["\']',
            ],
            'sql_injection': [
                r'(?i)execute\s*\(\s*["\'].*%s.*["\']',
                r'(?i)cursor\.execute\s*\(\s*["\'].*\+.*["\']',
                r'(?i)query\s*=\s*["\'].*%.*["\']',
            ],
            'command_injection': [
                r'(?i)os\.system\s*\(',
                r'(?i)subprocess\.call\s*\(',
                r'(?i)eval\s*\(',
                r'(?i)exec\s*\(',
            ],
            'insecure_random': [
                r'(?i)random\.random\s*\(',
                r'(?i)random\.randint\s*\(',
                r'(?i)random\.choice\s*\(',
            ],
            'debug_info': [
                r'(?i)print\s*\(\s*.*password.*\)',
                r'(?i)print\s*\(\s*.*secret.*\)',
                r'(?i)print\s*\(\s*.*token.*\)',
                r'(?i)logging\.debug\s*\(\s*.*password.*\)',
            ]
        }
        
        # 민감한 파일 패턴
        self.sensitive_files = [
            '.env', '.env.local', '.env.production',
            'config.ini', 'settings.py', 'secrets.json',
            '*.pem', '*.key', '*.p12', '*.pfx'
        ]
        
        logger.info(f"🔒 보안 감사 시스템 초기화 (프로젝트: {self.project_root})")
    
    def scan_file_for_patterns(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일에서 보안 패턴 검사"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for category, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            issues.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'line': line_num,
                                'category': category,
                                'pattern': pattern,
                                'match': match.group(),
                                'severity': self._get_severity(category),
                                'description': self._get_description(category)
                            })
                            
        except Exception as e:
            logger.warning(f"⚠️ 파일 스캔 오류 {file_path}: {e}")
            
        return issues
    
    def _get_severity(self, category: str) -> str:
        """카테고리별 심각도 반환"""
        severity_map = {
            'hardcoded_secrets': 'CRITICAL',
            'sql_injection': 'HIGH',
            'command_injection': 'HIGH',
            'insecure_random': 'MEDIUM',
            'debug_info': 'LOW'
        }
        return severity_map.get(category, 'MEDIUM')
    
    def _get_description(self, category: str) -> str:
        """카테고리별 설명 반환"""
        descriptions = {
            'hardcoded_secrets': '하드코딩된 비밀정보 발견',
            'sql_injection': 'SQL 인젝션 취약점 가능성',
            'command_injection': '명령어 인젝션 취약점 가능성',
            'insecure_random': '보안에 취약한 난수 생성기 사용',
            'debug_info': '민감한 정보를 포함한 디버그 코드'
        }
        return descriptions.get(category, '알 수 없는 보안 문제')
    
    def check_file_permissions(self) -> List[Dict[str, Any]]:
        """파일 권한 검사"""
        issues = []
        
        for pattern in self.sensitive_files:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    # Windows에서는 파일 권한 검사가 제한적
                    if os.name == 'posix':
                        stat_info = file_path.stat()
                        mode = oct(stat_info.st_mode)[-3:]
                        
                        if mode != '600':  # 소유자만 읽기/쓰기
                            issues.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'category': 'file_permissions',
                                'severity': 'MEDIUM',
                                'description': f'민감한 파일의 권한이 안전하지 않음 ({mode})',
                                'recommendation': '파일 권한을 600으로 설정하세요'
                            })
                    else:
                        # Windows에서는 파일 존재만 확인
                        issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'category': 'sensitive_file',
                            'severity': 'LOW',
                            'description': '민감한 파일이 감지됨',
                            'recommendation': '파일이 버전 관리에서 제외되었는지 확인하세요'
                        })
        
        return issues
    
    def check_dependencies(self) -> List[Dict[str, Any]]:
        """의존성 보안 검사"""
        issues = []
        
        # requirements.txt 검사
        req_file = self.project_root / 'requirements.txt'
        if req_file.exists():
            try:
                # bandit으로 의존성 검사 (설치되어 있다면)
                result = subprocess.run(
                    ['bandit', '-f', 'json', '-r', str(self.project_root)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get('results', []):
                        issues.append({
                            'file': issue['filename'],
                            'line': issue['line_number'],
                            'category': 'bandit_' + issue['test_id'],
                            'severity': issue['issue_severity'],
                            'description': issue['issue_text'],
                            'confidence': issue['issue_confidence']
                        })
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("⚠️ Bandit 실행 실패 (설치되지 않았거나 오류 발생)")
                
        return issues
    
    def check_environment_variables(self) -> List[Dict[str, Any]]:
        """환경 변수 보안 검사"""
        issues = []
        
        # .env 파일들 검사
        for env_file in self.project_root.rglob('.env*'):
            if env_file.is_file():
                try:
                    with open(env_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            
                            # 빈 값 검사
                            if not value or value in ['""', "''"]:
                                issues.append({
                                    'file': str(env_file.relative_to(self.project_root)),
                                    'line': i,
                                    'category': 'empty_env_var',
                                    'severity': 'LOW',
                                    'description': f'환경 변수 {key}가 비어있음',
                                    'key': key
                                })
                            
                            # 약한 값 검사
                            if len(value.strip('"\'')) < 8 and 'password' in key.lower():
                                issues.append({
                                    'file': str(env_file.relative_to(self.project_root)),
                                    'line': i,
                                    'category': 'weak_password',
                                    'severity': 'MEDIUM',
                                    'description': f'환경 변수 {key}의 값이 너무 짧음',
                                    'key': key
                                })
                                
                except Exception as e:
                    logger.warning(f"⚠️ 환경 변수 파일 검사 오류 {env_file}: {e}")
        
        return issues
    
    def analyze_code_quality(self) -> List[Dict[str, Any]]:
        """코드 품질 보안 관점 분석"""
        issues = []
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # AST 파싱으로 코드 분석
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        # try-except 없는 외부 호출 검사
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, 'attr'):
                                func_name = node.func.attr
                                if func_name in ['requests.get', 'requests.post', 'urlopen']:
                                    # 상위 노드에서 try-except 확인
                                    parent = getattr(node, 'parent', None)
                                    if not self._is_in_try_block(node, tree):
                                        issues.append({
                                            'file': str(py_file.relative_to(self.project_root)),
                                            'line': node.lineno,
                                            'category': 'unhandled_request',
                                            'severity': 'MEDIUM',
                                            'description': f'예외 처리 없는 외부 요청: {func_name}',
                                            'recommendation': 'try-except 블록으로 감싸세요'
                                        })
                        
                        # 하드코딩된 URL 검사
                        if isinstance(node, ast.Str):
                            if node.s.startswith(('http://', 'https://')):
                                issues.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'category': 'hardcoded_url',
                                    'severity': 'LOW',
                                    'description': f'하드코딩된 URL: {node.s[:50]}...',
                                    'recommendation': '환경 변수나 설정 파일을 사용하세요'
                                })
                                
                except Exception as e:
                    logger.warning(f"⚠️ 코드 분석 오류 {py_file}: {e}")
        
        return issues
    
    def _is_in_try_block(self, node: ast.AST, tree: ast.AST) -> bool:
        """노드가 try 블록 안에 있는지 확인"""
        # 간단한 구현 - 실제로는 더 복잡한 AST 분석 필요
        return False
    
    def calculate_security_score(self, issues: List[Dict[str, Any]]) -> int:
        """보안 점수 계산"""
        severity_weights = {
            'CRITICAL': 25,
            'HIGH': 15,
            'MEDIUM': 10,
            'LOW': 5
        }
        
        total_deduction = 0
        for issue in issues:
            severity = issue.get('severity', 'LOW')
            total_deduction += severity_weights.get(severity, 5)
        
        score = max(0, 100 - total_deduction)
        return score
    
    def generate_security_report(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """보안 보고서 생성"""
        # 심각도별 분류
        by_severity = {}
        for issue in issues:
            severity = issue.get('severity', 'LOW')
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)
        
        # 카테고리별 분류
        by_category = {}
        for issue in issues:
            category = issue.get('category', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(issue)
        
        security_score = self.calculate_security_score(issues)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'total_issues': len(issues),
            'security_score': security_score,
            'grade': self._get_security_grade(security_score),
            'issues_by_severity': by_severity,
            'issues_by_category': by_category,
            'recommendations': self._generate_recommendations(by_category),
            'summary': {
                'critical': len(by_severity.get('CRITICAL', [])),
                'high': len(by_severity.get('HIGH', [])),
                'medium': len(by_severity.get('MEDIUM', [])),
                'low': len(by_severity.get('LOW', []))
            }
        }
        
        return report
    
    def _get_security_grade(self, score: int) -> str:
        """점수에 따른 보안 등급"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, by_category: Dict[str, List]) -> List[str]:
        """카테고리별 권장사항 생성"""
        recommendations = []
        
        if 'hardcoded_secrets' in by_category:
            recommendations.append("🔑 하드코딩된 비밀정보를 환경 변수로 이동하세요")
        
        if 'sql_injection' in by_category:
            recommendations.append("🛡️ 매개변수화된 쿼리를 사용하여 SQL 인젝션을 방지하세요")
        
        if 'command_injection' in by_category:
            recommendations.append("⚠️ 사용자 입력을 직접 명령어에 사용하지 마세요")
        
        if 'insecure_random' in by_category:
            recommendations.append("🎲 보안이 중요한 경우 secrets 모듈을 사용하세요")
        
        if 'file_permissions' in by_category:
            recommendations.append("📁 민감한 파일의 권한을 적절히 설정하세요")
        
        return recommendations
    
    def run_security_audit(self) -> Dict[str, Any]:
        """전체 보안 감사 실행"""
        logger.info("🔒 보안 감사 시작")
        
        all_issues = []
        
        # 1. 소스 코드 패턴 검사
        logger.info("📝 소스 코드 패턴 검사")
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                issues = self.scan_file_for_patterns(py_file)
                all_issues.extend(issues)
        
        # 2. 파일 권한 검사
        logger.info("📁 파일 권한 검사")
        permission_issues = self.check_file_permissions()
        all_issues.extend(permission_issues)
        
        # 3. 의존성 보안 검사
        logger.info("📦 의존성 보안 검사")
        dependency_issues = self.check_dependencies()
        all_issues.extend(dependency_issues)
        
        # 4. 환경 변수 검사
        logger.info("🌍 환경 변수 검사")
        env_issues = self.check_environment_variables()
        all_issues.extend(env_issues)
        
        # 5. 코드 품질 보안 분석
        logger.info("🔍 코드 품질 보안 분석")
        quality_issues = self.analyze_code_quality()
        all_issues.extend(quality_issues)
        
        # 보고서 생성
        report = self.generate_security_report(all_issues)
        
        logger.info(f"✅ 보안 감사 완료: {len(all_issues)}개 이슈 발견, 보안 점수: {report['security_score']}")
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = "security_audit_report.json"):
        """보고서 파일로 저장"""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 보안 보고서 저장: {output_path}")
        
        # 마크다운 요약 보고서도 생성
        self.save_markdown_report(report, str(output_path).replace('.json', '.md'))
    
    def save_markdown_report(self, report: Dict[str, Any], output_file: str):
        """마크다운 형식 보고서 저장"""
        md_content = f"""# 🔒 보안 감사 보고서

**생성 시간**: {report['timestamp']}  
**프로젝트**: {report['project_root']}  
**보안 점수**: {report['security_score']}/100 ({report['grade']} 등급)

## 📊 요약

- **총 이슈**: {report['total_issues']}개
- **심각**: {report['summary']['critical']}개
- **높음**: {report['summary']['high']}개  
- **중간**: {report['summary']['medium']}개
- **낮음**: {report['summary']['low']}개

## 🎯 주요 권장사항

"""
        
        for rec in report['recommendations']:
            md_content += f"- {rec}\n"
        
        md_content += "\n## 📋 상세 이슈\n\n"
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            issues = report['issues_by_severity'].get(severity, [])
            if issues:
                md_content += f"### {severity} ({len(issues)}개)\n\n"
                for issue in issues:
                    md_content += f"- **{issue['file']}:{issue.get('line', '?')}** - {issue['description']}\n"
                md_content += "\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"📄 마크다운 보고서 저장: {output_file}")

def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description='보안 감사 도구')
    parser.add_argument('--project-root', default='.', help='프로젝트 루트 디렉토리')
    parser.add_argument('--output', default='security_audit_report.json', help='출력 파일명')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 보안 감사 실행
    auditor = SecurityAuditor(args.project_root)
    report = auditor.run_security_audit()
    auditor.save_report(report, args.output)
    
    # 결과 출력
    print(f"\n🔒 보안 감사 완료!")
    print(f"📊 보안 점수: {report['security_score']}/100 ({report['grade']} 등급)")
    print(f"🚨 총 {report['total_issues']}개 이슈 발견")
    
    if report['summary']['critical'] > 0:
        print(f"⚠️  심각한 보안 이슈 {report['summary']['critical']}개 발견!")
        sys.exit(1)
    elif report['summary']['high'] > 0:
        print(f"⚠️  높은 위험도 이슈 {report['summary']['high']}개 발견")
        sys.exit(1)
    else:
        print("✅ 심각한 보안 이슈 없음")

if __name__ == "__main__":
    main() 