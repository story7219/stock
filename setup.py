"""
원클릭 설치 및 설정
"""

import subprocess
import os
from datetime import datetime

def setup_auto_trading():
    """자동 트레이딩 설정"""
    
    print("🚀 초간단 트레이딩 시스템 설정")
    
    # Windows 작업 스케줄러 등록
    task_name = "SimpleTrading"
    python_path = subprocess.check_output("where python", shell=True).decode().strip()
    script_path = os.path.abspath("auto_run.py")
    
    # 매일 오전 9시 실행
    cmd = f'''schtasks /create /tn "{task_name}" /tr "{python_path} {script_path}" /sc daily /st 09:00 /f'''
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("✅ 매일 오전 9시 자동 실행 설정 완료")
        
        # 테스트 실행
        print("\n🧪 테스트 실행 중...")
        subprocess.run(f"{python_path} simple_trading.py", shell=True)
        
        print("\n✅ 설정 완료! 매일 오전 9시에 자동으로 실행됩니다.")
        
    except Exception as e:
        print(f"❌ 설정 실패: {e}")

if __name__ == "__main__":
    setup_auto_trading() 