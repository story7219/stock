#!/usr/bin/env python3
"""
🎯 워크플로우 차례대로 생성 스크립트
각 워크플로우를 개별적으로 차례대로 실행하여 워크플로우 런을 생성합니다.
"""

import subprocess
import time
from datetime import datetime

def run_git_command(command):
    """Git 명령어를 실행합니다."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def create_workflow_trigger_file(workflow_name, index):
    """각 워크플로우별 트리거 파일을 생성합니다."""
    filename = f"trigger_{index}_{workflow_name.replace(' ', '_').replace('🚀', '').replace('🔍', '').replace('📊', '').replace('🔒', '').replace('⚡', '').replace('💾', '').strip()}.txt"
    
    content = f"""
워크플로우 트리거 파일
=====================
워크플로우: {workflow_name}
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
트리거 목적: 개별 워크플로우 실행
"""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 트리거 파일 생성: {filename}")
        return filename
    except Exception as e:
        print(f"❌ 파일 생성 실패: {e}")
        return None

def commit_and_push(workflow_name, trigger_file):
    """커밋하고 푸시합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"🎯 {workflow_name} 워크플로우 개별 실행 트리거 - {timestamp}"
    
    print(f"📝 Git 커밋: {workflow_name}")
    
    # 파일 추가
    success, stdout, stderr = run_git_command(f'git add {trigger_file}')
    if not success:
        print(f"❌ Git add 실패: {stderr}")
        return False
    
    # 커밋
    success, stdout, stderr = run_git_command(f'git commit -m "{commit_message}"')
    if not success:
        print(f"❌ Git commit 실패: {stderr}")
        return False
    
    # 푸시
    print("📤 GitHub 푸시 중...")
    success, stdout, stderr = run_git_command("git push origin main")
    if success:
        print("✅ 푸시 완료 - 워크플로우 트리거됨")
        return True
    else:
        print(f"❌ 푸시 실패: {stderr}")
        return False

def cleanup_trigger_file(filename):
    """트리거 파일을 정리합니다."""
    try:
        import os
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🧹 트리거 파일 삭제: {filename}")
    except Exception as e:
        print(f"⚠️ 파일 삭제 실패: {e}")

def main():
    print("🎯 워크플로우 차례대로 생성 시작")
    print("=" * 60)
    
    # 생성할 워크플로우 목록 (순서대로)
    workflows = [
        {
            "name": "🚀 고급 스캘핑 자동매매 시스템",
            "description": "스캘핑 트레이딩 자동 실행"
        },
        {
            "name": "🔒 보안 감사 및 취약점 검사", 
            "description": "시스템 보안 상태 점검"
        },
        {
            "name": "📊 일일 트레이딩 성과 리포트",
            "description": "트레이딩 성과 분석 리포트"
        },
        {
            "name": "⚡ 통합 성능 모니터링",
            "description": "시스템 성능 및 헬스 체크"
        },
        {
            "name": "💾 자동 백업 및 복구 관리",
            "description": "핵심 파일 백업 시스템"
        }
    ]
    
    print("📋 차례대로 생성할 워크플로우 목록:")
    for i, workflow in enumerate(workflows, 1):
        print(f"   {i}. {workflow['name']}")
        print(f"      📝 {workflow['description']}")
        print()
    
    print("🎯 실행 계획:")
    print("   1️⃣ 각 워크플로우별 개별 트리거 파일 생성")
    print("   2️⃣ Git 커밋 & 푸시로 순차 실행") 
    print("   3️⃣ 각 실행 간 충분한 대기 시간")
    print("   4️⃣ 트리거 파일 자동 정리")
    
    confirm = input("\n❓ 워크플로우를 차례대로 생성하시겠습니까? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes', '네', 'ㅇ']:
        print("❌ 워크플로우 실행을 취소했습니다.")
        return
    
    print(f"\n🚀 워크플로우 차례대로 생성 시작...")
    
    successful_runs = 0
    
    for i, workflow in enumerate(workflows, 1):
        print(f"\n{'='*60}")
        print(f"📊 진행상황: {i}/{len(workflows)}")
        print(f"🎯 실행 중: {workflow['name']}")
        print(f"📝 설명: {workflow['description']}")
        print(f"{'='*60}")
        
        # 1. 트리거 파일 생성
        trigger_file = create_workflow_trigger_file(workflow['name'], i)
        if not trigger_file:
            print(f"❌ {workflow['name']} 트리거 파일 생성 실패")
            continue
        
        # 2. 커밋 & 푸시
        if commit_and_push(workflow['name'], trigger_file):
            print(f"✅ {workflow['name']} 워크플로우 런 생성 완료")
            successful_runs += 1
        else:
            print(f"❌ {workflow['name']} 워크플로우 런 생성 실패")
        
        # 3. 트리거 파일 정리
        cleanup_trigger_file(trigger_file)
        
        # 4. 다음 워크플로우 전 대기
        if i < len(workflows):
            print(f"⏳ 다음 워크플로우 준비 중... (5초 대기)")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("🎉 워크플로우 차례대로 생성 완료!")
    print(f"✅ 성공: {successful_runs}/{len(workflows)}개")
    print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_runs > 0:
        print(f"\n🔗 GitHub Actions 확인:")
        print(f"   https://github.com/story7219/stock/actions")
        print("   각 워크플로우 런이 순차적으로 실행됩니다.")
    
    print(f"\n📊 생성된 워크플로우 런:")
    for i, workflow in enumerate(workflows[:successful_runs], 1):
        print(f"   {i}. {workflow['name']} ✅")
    
    print("\n✅ 스크립트 실행 완료!")

if __name__ == "__main__":
    main() 