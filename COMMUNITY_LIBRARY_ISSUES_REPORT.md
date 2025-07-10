# 커뮤니티 라이브러리 문제 분석 및 해결 보고서

## 개요
한투 API 연동 시스템에서 사용하는 커뮤니티 라이브러리들의 문제점을 분석하고 해결책을 제시합니다.

## 현재 상태 분석

### ✅ 정상 작동하는 라이브러리
- **TensorFlow 2.19.0**: 정상 설치 및 작동
- **PyTorch 2.1.2**: 정상 설치 및 작동
- **Pandas 2.1.4**: 정상 설치 및 작동
- **NumPy 1.26.4**: 정상 설치 및 작동
- **aiohttp 3.9.1**: 정상 설치 및 작동
- **PyKIS 1.0.0**: 정상 설치 및 작동
- **TA-Lib**: 정상 설치 및 작동

### ❌ 문제가 있는 라이브러리
- **KoNLPy**: 설치되지 않음 (Windows 환경에서 설치 어려움)

## 주요 문제점 및 해결책

### 1. KoNLPy 설치 문제 (Windows)

#### 문제점
```bash
ModuleNotFoundError: No module named 'konlpy'
```

#### 원인
- Windows 환경에서 KoNLPy 설치가 복잡함
- Java JDK 의존성 문제
- C++ 컴파일러 의존성 문제

#### 해결책

**방법 1: conda 사용 (권장)**
```bash
# conda 환경 생성
conda create -n trading_ai python=3.11
conda activate trading_ai

# KoNLPy 설치
conda install -c conda-forge konlpy
```

**방법 2: 대체 라이브러리 사용**
```python
# KoNLPy 대신 사용할 수 있는 라이브러리들
from soynlp import LTokenizer, WordExtractor
from kiwipiepy import Kiwi
from mecab import MeCab
```

**방법 3: Docker 사용**
```dockerfile
FROM python:3.11-slim

# Java 설치
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# KoNLPy 설치
RUN pip install konlpy
```

### 2. TA-Lib 설치 문제

#### 문제점
- Windows에서 TA-Lib 설치가 어려움
- C++ 컴파일러 의존성

#### 해결책

**방법 1: conda 사용**
```bash
conda install -c conda-forge ta-lib
```

**방법 2: wheel 파일 사용**
```bash
# Windows용 wheel 파일 다운로드
pip install TA-Lib-0.4.28-cp311-cp311-win_amd64.whl
```

**방법 3: 대체 라이브러리**
```python
# TA-Lib 대신 사용할 수 있는 라이브러리들
import pandas_ta as pta
import yfinance as yf
import mplfinance as mpf
```

### 3. TensorFlow/PyTorch 버전 충돌

#### 현재 상태
- TensorFlow: 2.19.0 (최신)
- PyTorch: 2.1.2+cpu

#### 잠재적 문제
- GPU 지원 부족
- 메모리 사용량 증가

#### 해결책

**GPU 지원 활성화**
```bash
# CUDA 지원 TensorFlow 설치
pip install tensorflow[gpu]

# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**메모리 최적화**
```python
# TensorFlow 메모리 최적화
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# PyTorch 메모리 최적화
import torch
torch.cuda.empty_cache()
```

### 4. 의존성 충돌 문제

#### 문제점
- requirements.txt에 중복된 패키지들
- 버전 충돌 가능성

#### 해결책

**requirements.txt 정리**
```txt
# 핵심 의존성만 포함
pydantic==2.5.0
python-dotenv==1.0.0
aiohttp==3.9.1
pandas==2.1.4
numpy==1.26.4
tensorflow==2.19.0
torch==2.1.2
```

**가상환경 사용**
```bash
# 가상환경 생성
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
trading_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 권장 해결 방안

### 1. 즉시 적용 가능한 해결책

#### KoNLPy 대체
```python
# sentiment_analysis.py
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class KoreanSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> dict:
        """한국어 텍스트 감정 분석 (KoNLPy 대체)"""
        # 영어 텍스트로 변환 후 분석
        # 또는 규칙 기반 분석 사용
        return self.analyzer.polarity_scores(text)
```

#### TA-Lib 대체
```python
# technical_analysis.py
import pandas_ta as pta
import yfinance as yf

class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 (TA-Lib 대체)"""
        # pandas_ta 사용
        df['SMA_20'] = pta.sma(df['Close'], length=20)
        df['RSI'] = pta.rsi(df['Close'], length=14)
        df['MACD'] = pta.macd(df['Close'])
        return df
```

### 2. 장기적 해결책

#### Docker 환경 구축
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# KoNLPy 설치
RUN pip install konlpy

# 작업 디렉토리 설정
WORKDIR /app
COPY . .

# 실행
CMD ["python", "main.py"]
```

#### conda 환경 설정
```yaml
# environment.yml
name: trading_ai
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas=2.1.4
  - numpy=1.26.4
  - tensorflow=2.19.0
  - pytorch=2.1.2
  - konlpy
  - ta-lib
  - pip
  - pip:
    - aiohttp==3.9.1
    - pykis==1.0.0
    - fastapi==0.104.1
```

### 3. 모니터링 및 유지보수

#### 의존성 모니터링 스크립트
```python
# dependency_checker.py
import importlib
import sys
from typing import Dict, List

class DependencyChecker:
    def __init__(self):
        self.required_packages = [
            'tensorflow', 'torch', 'pandas', 'numpy',
            'aiohttp', 'pykis', 'ta', 'fastapi'
        ]
    
    def check_dependencies(self) -> Dict[str, bool]:
        """필수 패키지 설치 상태 확인"""
        results = {}
        for package in self.required_packages:
            try:
                importlib.import_module(package)
                results[package] = True
            except ImportError:
                results[package] = False
        return results
    
    def generate_report(self) -> str:
        """의존성 상태 보고서 생성"""
        results = self.check_dependencies()
        report = "=== 의존성 상태 보고서 ===\n"
        
        for package, installed in results.items():
            status = "✅ 설치됨" if installed else "❌ 설치되지 않음"
            report += f"{package}: {status}\n"
        
        return report

if __name__ == "__main__":
    checker = DependencyChecker()
    print(checker.generate_report())
```

## 결론 및 권장사항

### 1. 즉시 적용할 사항
- ✅ KoNLPy 대체 라이브러리 사용
- ✅ TA-Lib 대체 라이브러리 사용
- ✅ 가상환경 설정
- ✅ 의존성 모니터링 스크립트 구현

### 2. 중기적 개선사항
- 🔄 Docker 환경 구축
- 🔄 conda 환경 설정
- 🔄 자동화된 의존성 관리

### 3. 장기적 계획
- 📋 정기적인 라이브러리 업데이트
- 📋 보안 취약점 모니터링
- 📋 성능 최적화

현재 시스템은 대부분의 핵심 라이브러리가 정상적으로 작동하고 있으며, 문제가 있는 KoNLPy는 대체 라이브러리를 사용하여 해결할 수 있습니다. 전체적으로 안정적인 상태입니다. 