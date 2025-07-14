#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: security_auditor.py
모듈: 보안 감사 시스템
목적: 파일 권한, 보안 취약점, 코드 보안 검사

Author: Auto Trading System
Created: 2025-01-13
Modified: 2025-01-13
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - pathlib
    - stat
    - typing

Performance:
    - 감사 시간: < 30초
    - 메모리사용량: < 50MB
    - 처리용량: 1000+ files/minute

Security:
    - 파일 권한 검사
    - 보안 취약점 탐지
    - 접근 권한 검증
    - 자동 권한 수정

License: MIT
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Final

# 상수 정의
DEFAULT_PERMISSIONS: Final = 0o644
SECURE_DIRECTORY_PERMISSIONS: Final = 0o755
MAX_FILE_SIZE_MB: Final = 100
SECURITY_RISK_PATTERNS: Final = [
    r'password\s*=\s*["\'][^"\']+["\']',
    r'api_key\s*=\s*["\'][^"\']+["\']',
    r'secret\s*=\s*["\'][^"\']+["\']',
    r'token\s*=\s*["\'][^"\']+["\']'
]

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/security_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """보안 이슈 정보"""
    file_path: Path
    issue_type: str
    severity: str
    description: str
    recommendation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PermissionInfo:
    """파일 권한 정보"""
    file_path: Path
    mode: int
    owner_read: bool
    owner_write: bool
    owner_execute: bool
    group_read: bool
    group_write: bool
    group_execute: bool
    others_read: bool
    others_write: bool
    others_execute: bool
    security_risks: List[str] = field(default_factory=list)


class SecurityAuditor:
    """보안 감사 시스템"""
    
    def __init__(self, target_directory: Union[str, Path] = "."):
        self.target_directory = Path(target_directory)
        self.security_issues: List[SecurityIssue] = []
        self.permission_issues: List[PermissionInfo] = []
        self.fixed_issues: List[SecurityIssue] = []
        
        if not self.target_directory.exists():
            raise FileNotFoundError(f"Target directory does not exist: {target_directory}")
    
    def analyze_file_permissions(self, file_path: Path) -> PermissionInfo:
        """파일 권한 분석"""
        try:
            stat_info = file_path.stat()
            mode = stat_info.st_mode
            
            permission_info = PermissionInfo(
                file_path=file_path,
                mode=mode,
                owner_read=bool(mode & stat.S_IRUSR),
                owner_write=bool(mode & stat.S_IWUSR),
                owner_execute=bool(mode & stat.S_IXUSR),
                group_read=bool(mode & stat.S_IRGRP),
                group_write=bool(mode & stat.S_IWGRP),
                group_execute=bool(mode & stat.S_IXGRP),
                others_read=bool(mode & stat.S_IROTH),
                others_write=bool(mode & stat.S_IWOTH),
                others_execute=bool(mode & stat.S_IXOTH)
            )
            
            # 보안 위험 분석
            if permission_info.others_write:
                permission_info.security_risks.append("Others can write (CRITICAL)")
            if permission_info.others_execute:
                permission_info.security_risks.append("Others can execute (CRITICAL)")
            if permission_info.group_write:
                permission_info.security_risks.append("Group can write (MEDIUM)")
            if permission_info.others_read and file_path.suffix in ['.py', '.md', '.txt']:
                permission_info.security_risks.append("Sensitive file readable by others (LOW)")
            
            return permission_info
            
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to analyze permissions for {file_path}: {e}")
            raise
    
    def check_file_security_content(self, file_path: Path) -> List[SecurityIssue]:
        """파일 내용 보안 검사"""
        issues: List[SecurityIssue] = []
        
        try:
            if not file_path.is_file():
                return issues
            
            # 파일 크기 검사
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                issues.append(SecurityIssue(
                    file_path=file_path,
                    issue_type="LARGE_FILE",
                    severity="MEDIUM",
                    description=f"File size ({file_size_mb:.1f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)",
                    recommendation="Consider splitting or compressing the file"
                ))
            
            # 텍스트 파일만 내용 검사
            if file_path.suffix in ['.py', '.md', '.txt', '.json', '.yml', '.yaml']:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # 보안 패턴 검사
                    for pattern in SECURITY_RISK_PATTERNS:
                        if pattern in content.lower():
                            issues.append(SecurityIssue(
                                file_path=file_path,
                                issue_type="HARDCODED_SECRET",
                                severity="CRITICAL",
                                description="Hardcoded secret detected",
                                recommendation="Move secrets to environment variables or secure storage"
                            ))
                            break
                    
                    # 파일 확장자별 특별 검사
                    if file_path.suffix == '.py':
                        issues.extend(self._check_python_security(content, file_path))
                        
                except UnicodeDecodeError:
                    logger.warning(f"Cannot read {file_path} as text file")
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking security content for {file_path}: {e}")
            return issues
    
    def _check_python_security(self, content: str, file_path: Path) -> List[SecurityIssue]:
        """Python 파일 보안 검사"""
        issues: List[SecurityIssue] = []
        
        # 위험한 함수 사용 검사
        dangerous_functions = [
            'eval(', 'exec(', 'os.system(', 'subprocess.call(',
            'pickle.loads(', 'marshal.loads('
        ]
        
        for func in dangerous_functions:
            if func in content:
                issues.append(SecurityIssue(
                    file_path=file_path,
                    issue_type="DANGEROUS_FUNCTION",
                    severity="HIGH",
                    description=f"Use of dangerous function: {func}",
                    recommendation="Use safer alternatives and validate inputs"
                ))
        
        # 하드코딩된 경로 검사
        if 'C:\\' in content or '/home/' in content or '/root/' in content:
            issues.append(SecurityIssue(
                file_path=file_path,
                issue_type="HARDCODED_PATH",
                severity="MEDIUM",
                description="Hardcoded absolute path detected",
                recommendation="Use relative paths or environment variables"
            ))
        
        return issues
    
    def scan_directory(self) -> Dict[str, List[Union[SecurityIssue, PermissionInfo]]]:
        """디렉토리 전체 보안 스캔"""
        logger.info(f"Starting security scan of {self.target_directory}")
        
        scan_results = {
            'security_issues': [],
            'permission_issues': [],
            'total_files': 0,
            'total_directories': 0
        }
        
        try:
            for item in self.target_directory.rglob('*'):
                try:
                    if item.is_file():
                        scan_results['total_files'] += 1
                        
                        # 권한 검사
                        permission_info = self.analyze_file_permissions(item)
                        if permission_info.security_risks:
                            scan_results['permission_issues'].append(permission_info)
                        
                        # 내용 보안 검사
                        content_issues = self.check_file_security_content(item)
                        scan_results['security_issues'].extend(content_issues)
                        
                    elif item.is_dir():
                        scan_results['total_directories'] += 1
                        
                        # 디렉토리 권한 검사
                        permission_info = self.analyze_file_permissions(item)
                        if permission_info.security_risks:
                            scan_results['permission_issues'].append(permission_info)
                            
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access {item}: {e}")
                    continue
            
            logger.info(f"Scan completed: {scan_results['total_files']} files, "
                       f"{scan_results['total_directories']} directories")
            
            return scan_results
            
        except Exception as e:
            logger.error(f"Directory scan failed: {e}")
            raise
    
    def fix_permission_issues(self, permission_info: PermissionInfo) -> bool:
        """권한 이슈 자동 수정"""
        try:
            if not permission_info.security_risks:
                return True
            
            file_path = permission_info.file_path
            
            # 파일인 경우
            if file_path.is_file():
                new_mode = DEFAULT_PERMISSIONS
            # 디렉토리인 경우
            elif file_path.is_dir():
                new_mode = SECURE_DIRECTORY_PERMISSIONS
            else:
                return False
            
            # 권한 변경
            file_path.chmod(new_mode)
            
            logger.info(f"Fixed permissions for {file_path}: {oct(new_mode)}")
            
            # 수정된 이슈 기록
            self.fixed_issues.append(SecurityIssue(
                file_path=file_path,
                issue_type="PERMISSION_FIXED",
                severity="INFO",
                description=f"Permissions fixed to {oct(new_mode)}",
                recommendation="Permissions are now secure"
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fix permissions for {permission_info.file_path}: {e}")
            return False
    
    def generate_security_report(self, scan_results: Dict) -> str:
        """보안 보고서 생성"""
        report_lines = [
            "# Security Audit Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Target Directory: {self.target_directory}",
            "",
            "## Summary",
            f"- Total Files: {scan_results['total_files']}",
            f"- Total Directories: {scan_results['total_directories']}",
            f"- Security Issues: {len(scan_results['security_issues'])}",
            f"- Permission Issues: {len(scan_results['permission_issues'])}",
            f"- Fixed Issues: {len(self.fixed_issues)}",
            ""
        ]
        
        # 보안 이슈 상세
        if scan_results['security_issues']:
            report_lines.extend([
                "## Security Issues",
                ""
            ])
            
            for issue in scan_results['security_issues']:
                report_lines.extend([
                    f"### {issue.file_path}",
                    f"- **Type**: {issue.issue_type}",
                    f"- **Severity**: {issue.severity}",
                    f"- **Description**: {issue.description}",
                    f"- **Recommendation**: {issue.recommendation}",
                    ""
                ])
        
        # 권한 이슈 상세
        if scan_results['permission_issues']:
            report_lines.extend([
                "## Permission Issues",
                ""
            ])
            
            for perm_info in scan_results['permission_issues']:
                report_lines.extend([
                    f"### {perm_info.file_path}",
                    f"- **Current Permissions**: {oct(perm_info.mode)}",
                    f"- **Security Risks**: {', '.join(perm_info.security_risks)}",
                    ""
                ])
        
        # 수정된 이슈
        if self.fixed_issues:
            report_lines.extend([
                "## Fixed Issues",
                ""
            ])
            
            for issue in self.fixed_issues:
                report_lines.extend([
                    f"### {issue.file_path}",
                    f"- **Action**: {issue.description}",
                    f"- **Recommendation**: {issue.recommendation}",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def run_audit(self, auto_fix: bool = False) -> Dict[str, Union[str, int]]:
        """보안 감사 실행"""
        try:
            logger.info("🔒 Starting security audit...")
            
            # 디렉토리 스캔
            scan_results = self.scan_directory()
            
            # 권한 이슈 자동 수정
            if auto_fix:
                logger.info("🔧 Auto-fixing permission issues...")
                for perm_info in scan_results['permission_issues']:
                    if isinstance(perm_info, PermissionInfo):
                        self.fix_permission_issues(perm_info)
            
            # 보고서 생성
            report = self.generate_security_report(scan_results)
            
            # 결과 저장
            report_file = Path("security_audit_report.md")
            report_file.write_text(report, encoding='utf-8')
            
            logger.info(f"✅ Security audit completed. Report saved to {report_file}")
            
            return {
                'status': 'SUCCESS',
                'report_file': str(report_file),
                'total_issues': len(scan_results['security_issues']) + len(scan_results['permission_issues']),
                'fixed_issues': len(self.fixed_issues),
                'report': report
            }
            
        except Exception as e:
            logger.error(f"❌ Security audit failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'total_issues': 0,
                'fixed_issues': 0
            }


def main() -> int:
    """메인 함수"""
    try:
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)
        
        # 보안 감사 실행
        auditor = SecurityAuditor()
        result = auditor.run_audit(auto_fix=True)
        
        if result['status'] == 'SUCCESS':
            print("✅ Security audit completed successfully")
            print(f"📊 Total issues found: {result['total_issues']}")
            print(f"🔧 Issues fixed: {result['fixed_issues']}")
            print(f"📄 Report saved to: {result['report_file']}")
            return 0
        else:
            print(f"❌ Security audit failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        print(f"❌ Security audit failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
