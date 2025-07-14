from datetime import datetime
import timedelta
from typing import Dict
import List
import Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
import time
import yfinance as yf
from pykrx import stock
import FinanceDataReader as fdr

warnings.filterwarnings('ignore')

# 1순위: pykrx (한국투자증권 공식 API)
try:
    PYKRX_AVAILABLE = True
    print("✅ pykrx 사용 가능")
except ImportError:
    PYKRX_AVAILABLE = False
    print("❌ pykrx 설치 필요: pip install pykrx")

# 2순위: FinanceDataReader (FDR)
try:
    FDR_AVAILABLE = True
    print("✅ FinanceDataReader 사용 가능")
except ImportError:
    FDR_AVAILABLE = False
    print("❌ FinanceDataReader 설치 필요: pip install finance-datareader")

# 3순위: yfinance (Yahoo Finance)
try:
    YFINANCE_AVAILABLE = True
    print("✅ yfinance 사용 가능")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("❌ yfinance 설치 필요: pip install yfinance")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RealDataCollector:
    def __init__(self):
        self.samsung_ticker = "005930"
        self.current_date = datetime.now().strftime("%Y%m%d")
        self.start_date = "19830610"

        self.data_sources = []
        if PYKRX_AVAILABLE:
            self.data_sources.append("pykrx")
        if FDR_AVAILABLE:
            self.data_sources.append("fdr")
        if YFINANCE_AVAILABLE:
            self.data_sources.append("yfinance")

        print(f"📊 사용 가능한 데이터 소스: {self.data_sources}")
        print(f"🎯 데이터 수집 기간: 1983-06-10 (삼성전자 상장일) ~ 현재")

    # ... (나머지 함수들)
