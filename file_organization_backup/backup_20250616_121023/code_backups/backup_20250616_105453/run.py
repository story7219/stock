"""
🚀 간단한 실행 스크립트 (수정됨)
"""

import sys
import os
import asyncio

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

# 이제 절대 import 사용
from main import main

if __name__ == "__main__":
    print("🚀 고급 자동매매 시스템 시작")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 사용자에 의해 종료됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc() 