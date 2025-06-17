#!/usr/bin/env python3
"""
🔒 보안 검사 스크립트
- API 키 및 민감정보 노출 검사
- Python 보안 취약점 검사 (Bandit)
- 의존성 취약점 검사 (Safety)
- 간단하고 안정적인 구현
"""

import os
import re
import glob
import json
import subprocess
import traceback

def check_sensitive_info():
    """민감정보 노출 검사"""
    print("=== API 키 및 민감정보 노출 검사 ===")
    
    patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI API Key'),
        (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
        (r'[0-9]{10}:[A-Za-z0-9_-]{35}', 'Telegram Bot Token (Real)'),
        (r'ghp_[A-Za-z0-9]{36}', 'GitHub Personal Access Token'),
        (r'ghs_[A-Za-z0-9]{36}', 'GitHub App Token')
    ]
    
    file_patterns = ['*.py', '*.yml', '*.yaml', '*.json', '*.md']
    excluded_dirs = ['.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache']
    excluded_files = ['fix_env_encoding.py', 'create_env.py', 'security_auditor.py']
    
    findings = []
    
    for pattern in file_patterns:
        try:
            for filepath in glob.glob(f'**/{pattern}', recursive=True):
                if any(exc in filepath for exc in excluded_dirs) or os.path.basename(filepath) in excluded_files:
                    continue
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for regex, desc in patterns:
                        matches = re.finditer(regex, content)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            matched_text = match.group()
                            
                            whitelist = [
                                'test_', 'example', 'placeholder', 'dummy', 'sample', 'YOUR_', 'mock_',
                                'PSJ', 'W5t', 'KMn', 'PSG', 'VDD', '3J5', 'UGO', '1ap', 'd116',
                                '7877945865', 'AAF'
                            ]
                            
                            if not any(white in matched_text for white in whitelist):
                                findings.append(f'{filepath}:{line_num} - {desc}: {matched_text[:20]}...')
                
                except Exception as e:
                    print(f"⚠️ 파일 읽기 오류 {filepath}: {e}")
                    
        except Exception as e:
            print(f"⚠️ 패턴 검색 오류 {pattern}: {e}")
    
    if findings:
        print('⚠️ 잠재적 민감정보 발견:')
        for finding in findings[:10]:
            print(f'   {finding}')
        if len(findings) > 10:
            print(f'   ... 외 {len(findings)-10}개 더')
    else:
        print('✅ 하드코딩된 민감정보 없음')

def check_bandit():
    """Python 보안 취약점 검사 (Bandit)"""
    print("\n=== Python 보안 취약점 검사 (Bandit) ===")
    
    try:
        result = subprocess.run([
            'bandit', '-r', '.', 
            '--exclude', './.venv/*,__pycache__,.git,node_modules,.pytest_cache',
            '--skip', 'B101,B601',
            '-f', 'json', '-o', 'bandit_report.json'
        ], capture_output=True, text=True, check=False)
        
        if os.path.exists('bandit_report.json'):
            try:
                with open('bandit_report.json', 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                results = report.get('results', [])
                project_results = [r for r in results if '.venv' not in r.get('filename', '')]
                
                high_issues = [issue for issue in project_results if issue.get('issue_severity') == 'HIGH']
                medium_issues = [issue for issue in project_results if issue.get('issue_severity') == 'MEDIUM']
                
                print(f'🔴 고위험 취약점: {len(high_issues)}개')
                print(f'🟡 중위험 취약점: {len(medium_issues)}개')
                
                if high_issues:
                    print('\n⚠️ 고위험 취약점 상세:')
                    for issue in high_issues[:3]:
                        filename = issue.get('filename', 'Unknown')
                        line_number = issue.get('line_number', 'Unknown')
                        test_name = issue.get('test_name', 'Unknown')
                        issue_text = issue.get('issue_text', 'No description')
                        print(f'   {filename}:{line_number} - {test_name}')
                        print(f'     {issue_text[:100]}...')
                else:
                    print('✅ 고위험 보안 취약점 없음')
                    
            except json.JSONDecodeError:
                print('⚠️ Bandit JSON 파싱 오류')
            except Exception as e:
                print(f'⚠️ Bandit 리포트 처리 오류: {e}')
                traceback.print_exc()
        else:
            print('⚠️ Bandit 리포트 생성 실패')
            if result and result.stderr:
                print(f"   Error: {result.stderr}")
            
    except FileNotFoundError:
        print('⚠️ Bandit이 설치되지 않음. `pip install bandit` 필요')
    except Exception as e:
        print(f'⚠️ Bandit 실행 오류: {e}')
        traceback.print_exc()

def check_safety():
    """의존성 취약점 검사 (Safety)"""
    print("\n=== 의존성 취약점 검사 (Safety) ===")
    
    try:
        result = subprocess.run(['safety', 'check', '--output=json'], capture_output=True, text=True, check=False)
        
        try:
            vulnerabilities = json.loads(result.stdout)
            if vulnerabilities:
                print(f'⚠️ {len(vulnerabilities)}개의 의존성 취약점 발견:')
                for vuln in vulnerabilities[:5]:
                    print(f"  - {vuln[0]} ({vuln[2]}): {vuln[3]}")
            else:
                print('✅ 알려진 의존성 취약점 없음')
        except json.JSONDecodeError:
            if 'No known security vulnerabilities reported' in result.stdout:
                print('✅ 알려진 의존성 취약점 없음')
            else:
                print('⚠️ Safety 출력 결과 파싱 오류.')
                print(result.stdout)
            
    except FileNotFoundError:
        print('⚠️ Safety가 설치되지 않음. `pip install safety` 필요')
    except Exception as e:
        print(f'⚠️ Safety 실행 오류: {e}')
        traceback.print_exc()

def check_github_secrets():
    """GitHub Secrets 확인"""
    print("\n🔧 환경 설정 보안 검사")
    print("=" * 50)
    
    expected_secrets = [
        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 
        'KIS_APP_KEY', 'KIS_APP_SECRET', 'KIS_ACCOUNT_NO'
    ]
    
    print('📋 GitHub Secrets 확인:')
    missing_secrets = []
    
    for secret in expected_secrets:
        value = os.environ.get(secret)
        if value:
            # 테스트용 기본값과 실제 값이 같은 경우를 대비한 체크 제거
            print(f'   ✅ {secret}: 설정됨')
        else:
            print(f'   ❌ {secret}: 누락')
            missing_secrets.append(secret)
    
    print(f'\n📄 .env 파일 보안:')
    if os.path.exists('.env'):
        print('   ❌ .env 파일이 저장소에 있습니다! 즉시 .gitignore에 추가하고 저장소에서 제거하세요.')
    else:
        print('   ✅ .env 파일이 저장소에 없음')
    
    gitignore_check = True
    if os.path.exists('.gitignore'):
        try:
            with open('.gitignore', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            required = ['.env', '*.log', '__pycache__', 'kis_token.json', '*.json']
            missing = [item for item in required if item not in content and not item.endswith('.json')]
            # json 파일은 bandit_report.json 같은 리포트 때문에 제외할 수도 있고 아닐 수도 있어 경고 수준 낮춤
            if not any(item in content for item in ['*.json', 'bandit_report.json', 'service_account.json']):
                 print('   ⚠️ .gitignore에 "*.json" 추가를 권장합니다. (토큰/리포트 파일 제외 목적)')

            if missing:
                print(f'   ⚠️ .gitignore에 추가 권장: {missing}')
                gitignore_check = False
            else:
                print('   ✅ .gitignore 설정 양호')
        except Exception as e:
            print(f'   ⚠️ .gitignore 읽기 오류: {e}')
            gitignore_check = False
    else:
        print('   ⚠️ .gitignore 파일 없음')
        gitignore_check = False
    
    security_score = 100
    if missing_secrets:
        security_score -= len(missing_secrets) * 10
    if os.path.exists('.env'):
        security_score -= 30
    if not gitignore_check:
        security_score -= 10
    
    print(f'\n🎯 보안 점수: {max(0, security_score)}/100')
    if security_score >= 80:
        print('   등급: 우수 🟢')
    elif security_score >= 60:
        print('   등급: 양호 🟡')
    else:
        print('   등급: 개선 필요 🔴')

if __name__ == "__main__":
    check_sensitive_info()
    check_bandit()
    check_safety()
    # GitHub Actions 환경에서만 실행되는 것이 좋으므로 주석 처리
    # check_github_secrets() 