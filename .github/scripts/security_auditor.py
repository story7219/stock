#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: security_auditor.py
모듈: 보안 감사 시스템
목적: 파일 권한, 보안 취약점, 코드 보안 검사

Author: GitHub Actions
Created: 2025-01-06
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - os
    - pathlib
    - stat

Performance:
    - 감사 시간: < 30초
    - 메모리사용량: < 50MB
    - 처리용량: 1000+ files/minute

Security:
    - 파일 권한 검사
    - 보안 취약점 탐지
    - 접근 권한 검증

License: MIT
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def check_file_permissions(filepath: str) -> str:
    """파일 권한을 검사합니다.

    Args:
        filepath: 검사할 파일 경로

    Returns:
        파일 권한 정보를 설명하는 문자열. 파일이 존재하지 않거나
        권한에 접근할 수 없는 경우 오류 메시지를 반환합니다.
    """
    try:
        file_info = os.stat(filepath)
        permissions = oct(file_info.st_mode & 0o777)
        
        # 권한 분석
        permission_analysis = analyze_permissions(file_info.st_mode)
        
        return (
            f"File: {filepath}\n"
            f"Permissions: {permissions[2:]}\n"
            f"Analysis: {permission_analysis}"
        )
        
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except PermissionError:
        return f"Error: Permission denied accessing '{filepath}'."
    except Exception as e:
        return f"Error accessing file permissions: {e}"


def analyze_permissions(mode: int) -> str:
    """파일 권한을 분석합니다.

    Args:
        mode: 파일 모드 (stat.st_mode)

    Returns:
        권한 분석 결과 문자열
    """
    analysis = []
    
    # 소유자 권한
    if mode & stat.S_IRUSR:
        analysis.append("Owner can read")
    if mode & stat.S_IWUSR:
        analysis.append("Owner can write")
    if mode & stat.S_IXUSR:
        analysis.append("Owner can execute")
    
    # 그룹 권한
    if mode & stat.S_IRGRP:
        analysis.append("Group can read")
    if mode & stat.S_IWGRP:
        analysis.append("Group can write")
    if mode & stat.S_IXGRP:
        analysis.append("Group can execute")
    
    # 기타 사용자 권한
    if mode & stat.S_IROTH:
        analysis.append("Others can read")
    if mode & stat.S_IWOTH:
        analysis.append("Others can write")
    if mode & stat.S_IXOTH:
        analysis.append("Others can execute")
    
    # 보안 위험 평가
    security_risks = []
    if mode & stat.S_IWOTH:
        security_risks.append("WARNING: Others can write (security risk)")
    if mode & stat.S_IXOTH:
        security_risks.append("WARNING: Others can execute (security risk)")
    if mode & stat.S_IWGRP:
        security_risks.append("INFO: Group can write")
    
    if security_risks:
        analysis.extend(security_risks)
    
    return "; ".join(analysis) if analysis else "No permissions"


def check_directory_security(directory_path: str) -> Dict[str, List[str]]:
    """디렉토리 보안을 검사합니다.

    Args:
        directory_path: 검사할 디렉토리 경로

    Returns:
        보안 검사 결과 딕셔너리
    """
    try:
        results = {
            'insecure_files': [],
            'insecure_directories': [],
            'warnings': [],
            'errors': []
        }
        
        directory = Path(directory_path)
        if not directory.exists():
            results['errors'].append(f"Directory does not exist: {directory_path}")
            return results
        
        if not directory.is_dir():
            results['errors'].append(f"Path is not a directory: {directory_path}")
            return results
        
        # 디렉토리 내 모든 파일 검사
        for file_path in directory.rglob('*'):
            try:
                if file_path.is_file():
                    file_info = os.stat(file_path)
                    mode = file_info.st_mode
                    
                    # 보안 위험 파일 검사
                    if mode & stat.S_IWOTH:  # 다른 사용자가 쓰기 가능
                        results['insecure_files'].append(str(file_path))
                    elif mode & stat.S_IXOTH:  # 다른 사용자가 실행 가능
                        results['insecure_files'].append(str(file_path))
                    
                elif file_path.is_dir():
                    dir_info = os.stat(file_path)
                    dir_mode = dir_info.st_mode
                    
                    # 보안 위험 디렉토리 검사
                    if dir_mode & stat.S_IWOTH:  # 다른 사용자가 쓰기 가능
                        results['insecure_directories'].append(str(file_path))
                    
            except (PermissionError, OSError) as e:
                results['warnings'].append(f"Cannot access {file_path}: {e}")
        
        return results
        
    except Exception as e:
        return {
            'insecure_files': [],
            'insecure_directories': [],
            'warnings': [],
            'errors': [f"Directory security check failed: {e}"]
        }


def generate_security_report(directory_path: str) -> str:
    """보안 보고서를 생성합니다.

    Args:
        directory_path: 검사할 디렉토리 경로

    Returns:
        보안 보고서 문자열
    """
    try:
        report_lines = [
            "# Security Audit Report",
            f"Directory: {directory_path}",
            f"Generated: {__import__('datetime').datetime.now().isoformat()}",
            ""
        ]
        
        # 디렉토리 보안 검사
        security_results = check_directory_security(directory_path)
        
        # 오류 보고
        if security_results['errors']:
            report_lines.append("## Errors")
            for error in security_results['errors']:
                report_lines.append(f"- ❌ {error}")
            report_lines.append("")
        
        # 경고 보고
        if security_results['warnings']:
            report_lines.append("## Warnings")
            for warning in security_results['warnings']:
                report_lines.append(f"- ⚠️ {warning}")
            report_lines.append("")
        
        # 보안 위험 파일 보고
        if security_results['insecure_files']:
            report_lines.append("## Insecure Files")
            report_lines.append("The following files have insecure permissions:")
            for file_path in security_results['insecure_files']:
                report_lines.append(f"- 🔴 {file_path}")
            report_lines.append("")
        else:
            report_lines.append("## Insecure Files")
            report_lines.append("✅ No insecure files found")
            report_lines.append("")
        
        # 보안 위험 디렉토리 보고
        if security_results['insecure_directories']:
            report_lines.append("## Insecure Directories")
            report_lines.append("The following directories have insecure permissions:")
            for dir_path in security_results['insecure_directories']:
                report_lines.append(f"- 🔴 {dir_path}")
            report_lines.append("")
        else:
            report_lines.append("## Insecure Directories")
            report_lines.append("✅ No insecure directories found")
            report_lines.append("")
        
        # 요약
        total_issues = (
            len(security_results['insecure_files']) +
            len(security_results['insecure_directories']) +
            len(security_results['errors'])
        )
        
        report_lines.append("## Summary")
        report_lines.append(f"- Total issues found: {total_issues}")
        report_lines.append(f"- Insecure files: {len(security_results['insecure_files'])}")
        report_lines.append(f"- Insecure directories: {len(security_results['insecure_directories'])}")
        report_lines.append(f"- Errors: {len(security_results['errors'])}")
        report_lines.append(f"- Warnings: {len(security_results['warnings'])}")
        
        if total_issues == 0:
            report_lines.append("\n🎉 Security audit passed! No issues found.")
        else:
            report_lines.append(f"\n⚠️ Security audit found {total_issues} issue(s) that need attention.")
        
        return "\n".join(report_lines)
        
    except Exception as e:
        return f"Security report generation failed: {e}"


def fix_file_permissions(filepath: str, permissions: str) -> bool:
    """파일 권한을 수정합니다.

    Args:
        filepath: 수정할 파일 경로
        permissions: 새로운 권한 (8진수 문자열, 예: "644")

    Returns:
        수정 성공 여부
    """
    try:
        # 8진수 문자열을 정수로 변환
        mode = int(permissions, 8)
        
        # 파일 권한 변경
        os.chmod(filepath, mode)
        
        print(f"✅ Successfully changed permissions of {filepath} to {permissions}")
        return True
        
    except (ValueError, OSError) as e:
        print(f"❌ Failed to change permissions of {filepath}: {e}")
        return False


def main():
    """메인 함수"""
    try:
        if len(sys.argv) < 2:
            print("Usage: python security_auditor.py <filepath|directory> [permissions]")
            print("Examples:")
            print("  python security_auditor.py /path/to/file")
            print("  python security_auditor.py /path/to/directory")
            print("  python security_auditor.py /path/to/file 644")
            sys.exit(1)
        
        target_path = sys.argv[1]
        
        # 권한 수정 모드
        if len(sys.argv) == 3:
            permissions = sys.argv[2]
            if os.path.isfile(target_path):
                success = fix_file_permissions(target_path, permissions)
                sys.exit(0 if success else 1)
            else:
                print(f"❌ {target_path} is not a file")
                sys.exit(1)
        
        # 검사 모드
        if os.path.isfile(target_path):
            # 단일 파일 검사
            result = check_file_permissions(target_path)
            print(result)
        elif os.path.isdir(target_path):
            # 디렉토리 보안 검사
            report = generate_security_report(target_path)
            print(report)
            
            # 보고서 파일로 저장
            report_file = f"security_audit_report_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n📄 Security report saved to: {report_file}")
            except Exception as e:
                print(f"⚠️ Failed to save report: {e}")
        else:
            print(f"❌ Path does not exist: {target_path}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Security audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
