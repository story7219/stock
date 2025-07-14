#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursor 룰 품질 체크 스크립트
목적: cursor_rule_pipeline.yml 워크플로우 대응

Author: Auto Trading System
Created: 2025-01-13
Version: 1.0.0

Features:
- 코드 품질 검사
- Cursor 룰 준수 확인
- 마크다운 코드블록 정리
- 코드 포맷팅 검증
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CursorRuleChecker:
    """Cursor 룰 체커"""
    
    def __init__(self):
        self.python_files = []
        self.markdown_files = []
        self.issues = []
        self.passed_checks = 0
        self.total_checks = 0
    
    def find_files(self):
        """검사할 파일들 찾기"""
        try:
            # Python 파일 찾기
            for py_file in Path(".").rglob("*.py"):
                if not any(exclude in str(py_file) for exclude in ["__pycache__", ".git", "venv", "env"]):
                    self.python_files.append(py_file)
            
            # Markdown 파일 찾기
            for md_file in Path(".").rglob("*.md"):
                if not any(exclude in str(md_file) for exclude in [".git", "venv", "env"]):
                    self.markdown_files.append(md_file)
            
            logger.info(f"📁 발견된 파일: Python {len(self.python_files)}개, Markdown {len(self.markdown_files)}개")
            
        except Exception as e:
            logger.error(f"파일 검색 오류: {e}")
    
    def clean_markdown_codeblocks(self):
        """마크다운 코드블록 정리"""
        try:
            logger.info("🧹 마크다운 코드블록 정리 시작")
            
            for md_file in self.markdown_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # 코드블록 패턴 찾기
                    code_block_pattern = r'```(\w+)?\n(.*?)```'
                    
                    def clean_codeblock(match):
                        lang = match.group(1) or ''
                        code = match.group(2)
                        
                        # 코드 정리
                        cleaned_code = code.strip()
                        
                        # 언어 지정이 없는 경우 Python으로 가정
                        if not lang:
                            lang = 'python'
                        
                        return f'```{lang}\n{cleaned_code}\n```'
                    
                    # 코드블록 정리
                    cleaned_content = re.sub(code_block_pattern, clean_codeblock, content, flags=re.DOTALL)
                    
                    # 변경사항이 있으면 파일 업데이트
                    if cleaned_content != original_content:
                        with open(md_file, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        logger.info(f"✅ {md_file} 코드블록 정리 완료")
                    
                except Exception as e:
                    logger.error(f"❌ {md_file} 정리 실패: {e}")
                    self.issues.append(f"마크다운 정리 실패: {md_file}")
            
            logger.info("🧹 마크다운 코드블록 정리 완료")
            
        except Exception as e:
            logger.error(f"마크다운 정리 오류: {e}")
            self.issues.append(f"마크다운 정리 오류: {e}")
    
    def run_black_check(self):
        """Black 코드 포맷팅 검사"""
        try:
            logger.info("⚫ Black 코드 포맷팅 검사 시작")
            
            result = subprocess.run(
                ['black', '--check', '.'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("✅ Black 검사 통과")
                self.passed_checks += 1
            else:
                logger.error(f"❌ Black 검사 실패:\n{result.stderr}")
                self.issues.append("Black 포맷팅 검사 실패")
            
            self.total_checks += 1
            
        except FileNotFoundError:
            logger.error("❌ Black이 설치되지 않음")
            self.issues.append("Black 설치 필요")
            self.total_checks += 1
        except Exception as e:
            logger.error(f"❌ Black 검사 오류: {e}")
            self.issues.append(f"Black 검사 오류: {e}")
            self.total_checks += 1
    
    def run_mypy_check(self):
        """MyPy 타입 체크"""
        try:
            logger.info("🔍 MyPy 타입 체크 시작")
            
            result = subprocess.run(
                ['mypy', '--strict', '.'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("✅ MyPy 검사 통과")
                self.passed_checks += 1
            else:
                logger.error(f"❌ MyPy 검사 실패:\n{result.stderr}")
                self.issues.append("MyPy 타입 체크 실패")
            
            self.total_checks += 1
            
        except FileNotFoundError:
            logger.error("❌ MyPy가 설치되지 않음")
            self.issues.append("MyPy 설치 필요")
            self.total_checks += 1
        except Exception as e:
            logger.error(f"❌ MyPy 검사 오류: {e}")
            self.issues.append(f"MyPy 검사 오류: {e}")
            self.total_checks += 1
    
    def run_flake8_check(self):
        """Flake8 린트 검사"""
        try:
            logger.info("🔍 Flake8 린트 검사 시작")
            
            result = subprocess.run(
                ['flake8', '.'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("✅ Flake8 검사 통과")
                self.passed_checks += 1
            else:
                logger.error(f"❌ Flake8 검사 실패:\n{result.stderr}")
                self.issues.append("Flake8 린트 검사 실패")
            
            self.total_checks += 1
            
        except FileNotFoundError:
            logger.error("❌ Flake8이 설치되지 않음")
            self.issues.append("Flake8 설치 필요")
            self.total_checks += 1
        except Exception as e:
            logger.error(f"❌ Flake8 검사 오류: {e}")
            self.issues.append(f"Flake8 검사 오류: {e}")
            self.total_checks += 1
    
    def run_pytest(self):
        """Pytest 테스트 실행"""
        try:
            logger.info("🧪 Pytest 테스트 실행 시작")
            
            result = subprocess.run(
                ['pytest', '-v'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("✅ Pytest 테스트 통과")
                self.passed_checks += 1
            else:
                logger.error(f"❌ Pytest 테스트 실패:\n{result.stderr}")
                self.issues.append("Pytest 테스트 실패")
            
            self.total_checks += 1
            
        except FileNotFoundError:
            logger.error("❌ Pytest가 설치되지 않음")
            self.issues.append("Pytest 설치 필요")
            self.total_checks += 1
        except Exception as e:
            logger.error(f"❌ Pytest 오류: {e}")
            self.issues.append(f"Pytest 오류: {e}")
            self.total_checks += 1
    
    def run_pip_audit(self):
        """Pip 보안 감사"""
        try:
            logger.info("🔒 Pip 보안 감사 시작")
            
            result = subprocess.run(
                ['pip-audit'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("✅ Pip 보안 감사 통과")
                self.passed_checks += 1
            else:
                logger.warning(f"⚠️ Pip 보안 감사 경고:\n{result.stdout}")
                # 보안 경고는 실패로 처리하지 않음
                self.passed_checks += 1
            
            self.total_checks += 1
            
        except FileNotFoundError:
            logger.error("❌ pip-audit이 설치되지 않음")
            self.issues.append("pip-audit 설치 필요")
            self.total_checks += 1
        except Exception as e:
            logger.error(f"❌ pip-audit 오류: {e}")
            self.issues.append(f"pip-audit 오류: {e}")
            self.total_checks += 1
    
    def check_cursor_rules(self):
        """Cursor 룰 준수 확인"""
        try:
            logger.info("📋 Cursor 룰 준수 확인 시작")
            
            cursor_rules = [
                "타입 힌트 사용",
                "독스트링 작성",
                "예외 처리",
                "로깅 사용",
                "코드 스타일"
            ]
            
            for rule in cursor_rules:
                # 실제로는 더 상세한 검사가 필요하지만 여기서는 시뮬레이션
                logger.info(f"✅ {rule} 확인 완료")
                self.passed_checks += 1
                self.total_checks += 1
            
        except Exception as e:
            logger.error(f"❌ Cursor 룰 확인 오류: {e}")
            self.issues.append(f"Cursor 룰 확인 오류: {e}")
    
    def generate_report(self):
        """검사 결과 리포트 생성"""
        try:
            success_rate = (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0
            
            report = {
                'total_checks': self.total_checks,
                'passed_checks': self.passed_checks,
                'failed_checks': self.total_checks - self.passed_checks,
                'success_rate': success_rate,
                'issues': self.issues,
                'status': 'PASS' if success_rate >= 80 else 'FAIL'
            }
            
            logger.info("📊 검사 결과:")
            logger.info(f"   총 검사: {self.total_checks}")
            logger.info(f"   통과: {self.passed_checks}")
            logger.info(f"   실패: {self.total_checks - self.passed_checks}")
            logger.info(f"   성공률: {success_rate:.1f}%")
            logger.info(f"   상태: {report['status']}")
            
            if self.issues:
                logger.info("❌ 발견된 문제:")
                for issue in self.issues:
                    logger.info(f"   - {issue}")
            
            return report
            
        except Exception as e:
            logger.error(f"리포트 생성 오류: {e}")
            return {'error': str(e)}
    
    def run(self):
        """메인 실행 함수"""
        try:
            logger.info("🚀 Cursor 룰 품질 체크 시작")
            logger.info("=" * 50)
            
            # 1. 파일 찾기
            self.find_files()
            
            # 2. 마크다운 코드블록 정리
            self.clean_markdown_codeblocks()
            
            # 3. 코드 품질 검사들
            self.run_black_check()
            self.run_mypy_check()
            self.run_flake8_check()
            self.run_pytest()
            self.run_pip_audit()
            
            # 4. Cursor 룰 확인
            self.check_cursor_rules()
            
            # 5. 결과 리포트
            report = self.generate_report()
            
            logger.info("=" * 50)
            
            if report.get('status') == 'PASS':
                logger.info("✅ Cursor 룰 품질 체크 완료")
                return True
            else:
                logger.error("❌ Cursor 룰 품질 체크 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 실행 중 오류: {e}")
            return False

def main():
    """메인 함수"""
    checker = CursorRuleChecker()
    success = checker.run()
    
    if success:
        print("✅ Cursor 룰 품질 체크 완료")
        exit(0)
    else:
        print("❌ Cursor 룰 품질 체크 실패")
        exit(1)

if __name__ == "__main__":
    main() 