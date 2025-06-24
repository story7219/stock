#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import 테스트 스크립트 - 모든 주요 모듈들의 import 오류를 체크
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """모든 주요 import를 테스트"""
    print("🔍 모듈 Import 테스트 시작...")
    
    # 기본 라이브러리 테스트
    try:
        import pandas as pd
        import numpy as np
        import aiohttp
        import requests
        import yfinance as yf
        import google.generativeai as genai
        print("✅ 기본 라이브러리 import 성공")
    except ImportError as e:
        print(f"❌ 기본 라이브러리 import 실패: {e}")
        return False
    
    # 프로젝트 모듈 테스트
    try:
        from src.multi_data_collector import MultiDataCollector, StockData as CollectorStockData
        print("✅ MultiDataCollector import 성공")
    except ImportError as e:
        print(f"❌ MultiDataCollector import 실패: {e}")
    
    try:
        from src.gemini_analyzer import GeminiAnalyzer, StockData as AnalyzerStockData
        print("✅ GeminiAnalyzer import 성공")
    except ImportError as e:
        print(f"❌ GeminiAnalyzer import 실패: {e}")
    
    try:
        from src.strategies import ChartExpertManager
        print("✅ Strategies import 성공")
    except ImportError as e:
        print(f"❌ Strategies import 실패: {e}")
    
    try:
        from src.telegram_notifier import TelegramNotifier
        print("✅ TelegramNotifier import 성공")
    except ImportError as e:
        print(f"❌ TelegramNotifier import 실패: {e}")
    
    try:
        from src.google_sheets_manager import GoogleSheetsManager
        print("✅ GoogleSheetsManager import 성공")
    except ImportError as e:
        print(f"❌ GoogleSheetsManager import 실패: {e}")
    
    try:
        from src.scheduler import AutomatedScheduler
        print("✅ AutomatedScheduler import 성공")
    except ImportError as e:
        print(f"❌ AutomatedScheduler import 실패: {e}")
    
    try:
        from src.smart_data_storage import SmartDataStorage
        print("✅ SmartDataStorage import 성공")
    except ImportError as e:
        print(f"❌ SmartDataStorage import 실패: {e}")
    
    try:
        from src.sheets_dashboard import SheetsDashboard
        print("✅ SheetsDashboard import 성공")
    except ImportError as e:
        print(f"❌ SheetsDashboard import 실패: {e}")
    
    # Config 모듈 테스트
    try:
        from config import Config, config
        print("✅ Config import 성공")
    except ImportError as e:
        print(f"❌ Config import 실패: {e}")
    
    print("\n📊 Import 테스트 완료")
    return True

if __name__ == "__main__":
    test_imports() 