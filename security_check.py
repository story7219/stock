#!/usr/bin/env python3
"""
🔒 보안 검사 스크립트
- API 키 및 민감정보 노출 검사
- 간단하고 안정적인 구현
"""

import os
import re
import glob

def check_sensitive_info():
    """민감정보 노출 검사"""
    print("=== API 키 및 민감정보 노출 검사 ===")
    
    # 민감정보 패턴들 (간단 버전)
    patterns = [
        (r'[A-Za-z0-9]{32,}', 'Long Token'),
        (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI API Key'),
        (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
        (r'[0-9]{10}:[A-Za-z0-9_-]{35}', 'Telegram Bot Token')
    ]
    
    # 검사할 파일들
    file_patterns = ['*.py', '*.yml', '*.yaml', '*.json', '*.md', '*.txt']
    excluded_dirs = ['.git', '__pycache__', '.venv', 'node_modules']
    
    findings = []
    
    for pattern in file_patterns:
        try:
            for filepath in glob.glob(f'**/{pattern}', recursive=True):
                # 제외 디렉토리 건너뛰기
                if any(exc in filepath for exc in excluded_dirs):
                    continue
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for regex, desc in patterns:
                        matches = re.finditer(regex, content)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            matched_text = match.group()
                            
                            # 화이트리스트 (테스트 값 제외)
                            whitelist = ['test_', 'example', 'placeholder', 'dummy', 'sample', 'YOUR_']
                            
                            if not any(white in matched_text for white in whitelist):
                                if len(matched_text) > 15:  # 충분히 긴 토큰만
                                    findings.append(f'{filepath}:{line_num} - {desc}: {matched_text[:20]}...')
                
                except Exception as e:
                    print(f"⚠️ 파일 읽기 오류 {filepath}: {e}")
                    
        except Exception as e:
            print(f"⚠️ 패턴 검색 오류 {pattern}: {e}")
    
    # 결과 출력
    if findings:
        print('⚠️ 잠재적 민감정보 발견:')
        for finding in findings[:10]:
            print(f'   {finding}')
        if len(findings) > 10:
            print(f'   ... 외 {len(findings)-10}개 더')
    else:
        print('✅ 하드코딩된 민감정보 없음')

def check_github_secrets():
    """GitHub Secrets 확인"""
    print("\n🔧 환경 설정 보안 검사")
    print("=" * 50)
    
    expected_secrets = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID', 
        'MOCK_KIS_APP_KEY',
        'MOCK_KIS_APP_SECRET',
        'MOCK_KIS_ACCOUNT_NUMBER'
    ]
    
    print('📋 GitHub Secrets 확인:')
    missing_secrets = []
    
    for secret in expected_secrets:
        value = os.environ.get(secret)
        if value:
            if value in ['test_telegram_token', 'test_key', 'test_secret']:
                print(f'   ⚠️ {secret}: 테스트 값 사용 중')
            else:
                print(f'   ✅ {secret}: 설정됨')
        else:
            print(f'   ❌ {secret}: 누락')
            missing_secrets.append(secret)
    
    # .env 파일 보안 검사
    print(f'\n📄 .env 파일 보안:')
    if os.path.exists('.env'):
        print('   ❌ .env 파일이 저장소에 있습니다! 즉시 제거하세요.')
    else:
        print('   ✅ .env 파일이 저장소에 없음')
    
    # 보안 점수 계산
    security_score = 100 - len(missing_secrets) * 10
    if os.path.exists('.env'):
        security_score -= 30
    
    print(f'\n🎯 보안 점수: {security_score}/100')
    if security_score >= 80:
        print('   등급: 우수 🟢')
    elif security_score >= 60:
        print('   등급: 양호 🟡')
    else:
        print('   등급: 개선 필요 🔴')

if __name__ == "__main__":
    check_sensitive_info()
    check_github_secrets() 