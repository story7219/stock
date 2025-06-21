#!/usr/bin/env python3
"""
🚀 GitHub Actions 워크플로우 수동 실행 스크립트
모든 워크플로우를 순차적으로 실행하여 워크플로우 런을 생성합니다.
"""

import requests
import json
import time
import os
from datetime import datetime

def trigger_workflow(owner, repo, workflow_file, token):
    """
    GitHub API를 통해 워크플로우를 수동 실행합니다.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    data = {
        "ref": "main"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response.status_code == 204, response.status_code
    except Exception as e:
        return False, str(e)

def main():
    print("🚀 GitHub Actions 워크플로우 수동 실행 시작")
    print("=" * 60)
    
    # GitHub 정보 설정 (실제 사용 시 환경변수나 입력으로 받아야 함)
    owner = "story7219"  # GitHub 사용자명
    repo = "stock"       # 리포지토리 이름
    
    # GitHub Personal Access Token이 필요합니다
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN 환경변수가 설정되지 않았습니다.")
        print("   Personal Access Token을 설정해주세요.")
        return
    
    # 실행할 워크플로우 목록
    workflows = [
        {
            "file": "trading.yml",
            "name": "🚀 고급 스캘핑 자동매매 시스템",
            "description": "스캘핑 트레이딩 시스템 테스트"
        },
        {
            "file": "code_review.yml", 
            "name": "🔍 AI 코드 품질 검사",
            "description": "AI 기반 코드 품질 및 보안 검사"
        },
        {
            "file": "report.yml",
            "name": "📊 일일 트레이딩 성과 리포트", 
            "description": "일일 성과 분석 및 리포트 생성"
        },
        {
            "file": "security_audit.yml",
            "name": "🔒 보안 감사 및 취약점 검사",
            "description": "보안 취약점 및 민감정보 검사"
        },
        {
            "file": "performance_monitor.yml",
            "name": "⚡ 통합 성능 모니터링",
            "description": "시스템 성능 및 헬스 체크"
        },
        {
            "file": "backup_recovery.yml", 
            "name": "💾 자동 백업 및 복구 관리",
            "description": "핵심 파일 백업 및 복구 시스템"
        }
    ]
    
    successful_runs = 0
    failed_runs = 0
    
    for i, workflow in enumerate(workflows, 1):
        print(f"\n{i}/6 실행 중: {workflow['name']}")
        print(f"     📄 파일: {workflow['file']}")
        print(f"     📝 설명: {workflow['description']}")
        
        success, result = trigger_workflow(owner, repo, workflow['file'], token)
        
        if success:
            print(f"     ✅ 성공: 워크플로우가 실행되었습니다")
            successful_runs += 1
        else:
            print(f"     ❌ 실패: {result}")
            failed_runs += 1
        
        # API 제한을 피하기 위해 잠시 대기
        if i < len(workflows):
            print("     ⏳ 잠시 대기 중...")
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("🎯 워크플로우 실행 완료 요약")
    print(f"   ✅ 성공: {successful_runs}개")
    print(f"   ❌ 실패: {failed_runs}개")
    print(f"   📊 총 실행: {len(workflows)}개")
    print(f"   📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_runs > 0:
        print(f"\n🔗 GitHub Actions 확인:")
        print(f"   https://github.com/{owner}/{repo}/actions")
        print("   잠시 후 워크플로우 실행 상태를 확인할 수 있습니다.")
    
    print("\n✅ 스크립트 실행 완료!")

if __name__ == "__main__":
    main() 