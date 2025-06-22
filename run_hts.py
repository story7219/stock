"""
HTS 스타일 주식 분석 시스템 실행기
"""

import sys
import os

# 필요한 라이브러리 확인
try:
    import tkinter as tk
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from hts_gui import HTSStyleGUI
    
    print("✅ 모든 라이브러리가 정상적으로 로드되었습니다.")
    
    # GUI 실행
    print("🚀 HTS 스타일 주식 분석 시스템을 시작합니다...")
    app = HTSStyleGUI()
    app.run()
    
except ImportError as e:
    print(f"❌ 라이브러리 오류: {e}")
    print("\n필요한 라이브러리를 설치해주세요:")
    print("pip install pandas matplotlib numpy tkinter")
    
except Exception as e:
    print(f"❌ 실행 오류: {e}")
    print("\n프로그램 실행 중 오류가 발생했습니다.")
    
input("\n프로그램을 종료하려면 Enter를 누르세요...") 