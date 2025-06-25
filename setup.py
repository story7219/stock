#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 투자 분석 시스템 패키지 설정 파일 (setup.py)
==============================================

코스피200·나스닥100·S&P500 투자 대가 전략 AI 분석 시스템의 
패키지 설정 및 배포를 관리하는 setuptools 설정 파일입니다.

이 파일의 주요 역할:
🔧 패키지 메타데이터 정의 (이름, 버전, 설명, 저자 등)
📋 의존성 라이브러리 목록 관리
📦 패키지 빌드 및 배포 설정
🏷️ PyPI 분류자 및 키워드 설정
⚙️ CLI 스크립트 진입점 정의
📄 패키지 데이터 파일 포함 설정

주요 기능:
- pip install을 통한 패키지 설치 지원
- 개발용 의존성과 운영용 의존성 분리
- 다양한 Python 버전 호환성 보장 (3.9+)
- 금융/AI 분야 패키지로 분류
- 콘솔 스크립트 자동 생성

설치 방법:
    # 개발용 설치 (편집 가능 모드)
    pip install -e .
    
    # 개발 도구 포함 설치
    pip install -e .[dev]
    
    # 모든 의존성 포함 설치
    pip install -e .[all]

빌드 및 배포:
    # 소스 배포판 생성
    python setup.py sdist
    
    # 휠 패키지 생성
    python setup.py bdist_wheel
    
    # PyPI 업로드 (테스트)
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    
    # PyPI 업로드 (실제)
    twine upload dist/*

패키지 구조:
- src/: 소스 코드 디렉토리
- modules/: 핵심 기능 모듈들
- tests/: 테스트 코드
- data/: 데이터 파일들
- docs/: 문서 파일들

이 설정 파일은 Python 패키징 표준을 준수하며,
현대적인 패키지 관리 도구들과 호환됩니다.
"""

from setuptools import setup, find_packages
from pathlib import Path

# README 파일 읽기 (패키지 설명용)
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# 🏷️ 버전 정보 (Semantic Versioning 준수)
VERSION = "5.0.0"

# 📋 필수 패키지 목록 (운영 환경에서 반드시 필요한 라이브러리들)
REQUIRED_PACKAGES = [
    # 📊 핵심 데이터 처리 라이브러리
    "pandas>=2.0.0",        # 데이터프레임 처리 및 분석
    "numpy>=1.24.0",        # 수치 계산 및 배열 처리
    
    # 🌐 비동기 HTTP 클라이언트
    "aiohttp>=3.8.5",       # 비동기 HTTP 요청
    "aiofiles>=23.0.0",     # 비동기 파일 I/O
    
    # 🕷️ 웹 스크래핑 도구
    "beautifulsoup4>=4.12.2",  # HTML 파싱
    "selenium>=4.15.0",        # 동적 웹페이지 스크래핑
    "requests>=2.31.0",        # HTTP 요청 처리
    
    # 💰 금융 데이터 수집
    "yfinance>=0.2.28",     # Yahoo Finance 데이터
    "investpy>=1.0.8",      # Investing.com 데이터
    
    # 🤖 AI/ML 라이브러리
    "google-generativeai>=0.3.0",  # Gemini AI API
    "scikit-learn>=1.3.0",         # 머신러닝 알고리즘
    
    # 📈 기술적 분석 도구
    "pandas-ta>=0.3.14b0",  # 기술적 지표 계산
    "finta>=1.3",           # 금융 기술적 분석
    
    # 📊 데이터 시각화
    "matplotlib>=3.7.2",    # 기본 차트 생성
    "seaborn>=0.12.2",      # 통계 시각화
    "plotly>=5.17.0",       # 인터랙티브 차트
    
    # 🔗 구글 서비스 연동
    "google-auth>=2.23.0",                    # Google 인증
    "google-auth-oauthlib>=1.1.0",           # OAuth 인증
    "google-auth-httplib2>=0.1.1",           # HTTP 라이브러리 연동
    "google-api-python-client>=2.100.0",     # Google API 클라이언트
    "gspread>=5.11.0",                       # Google Sheets API
    
    # 📱 텔레그램 봇 연동
    "python-telegram-bot>=20.6",  # 텔레그램 봇 API
    
    # ⏰ 스케줄링 도구
    "schedule>=1.2.0",      # 간단한 작업 스케줄링
    "APScheduler>=3.10.0",  # 고급 스케줄러
    
    # ⚙️ 환경 변수 관리
    "python-dotenv>=1.0.0",  # .env 파일 처리
    
    # 📝 로깅 및 모니터링
    "loguru>=0.7.0",        # 고급 로깅 시스템
    "psutil>=5.9.0",        # 시스템 모니터링
    
    # ✅ 데이터 검증
    "pydantic>=2.4.0",      # 데이터 모델 및 검증
    "cerberus>=1.3.4",      # 스키마 기반 검증
    
    # 🛠️ 코드 품질 도구
    "black>=23.9.0",        # 코드 포맷터
    "pylint>=2.17.0",       # 코드 린터
    "flake8>=6.1.0",        # 스타일 가이드 검사
    "mypy>=1.6.0",          # 정적 타입 검사
    
    # 🧪 테스팅 프레임워크
    "pytest>=7.4.0",        # 테스트 프레임워크
    "pytest-asyncio>=0.21.0",  # 비동기 테스트
    "pytest-cov>=4.1.0",    # 코드 커버리지
    "pytest-mock>=3.11.0",  # 모킹 도구
    
    # 🎨 유틸리티 라이브러리
    "tqdm>=4.66.0",         # 진행률 표시
    "colorama>=0.4.6",      # 터미널 색상
    "rich>=13.6.0",         # 리치 텍스트 출력
    "typer>=0.9.0",         # CLI 인터페이스
]

# 🔧 개발용 추가 패키지 (개발 환경에서만 필요)
DEV_PACKAGES = [
    # 📓 개발 도구
    "jupyter>=1.0.0",       # Jupyter 노트북
    "notebook>=7.0.0",      # 노트북 서버
    "ipywidgets>=8.1.0",    # 인터랙티브 위젯
    
    # 🔍 코드 품질 검사
    "pre-commit>=3.4.0",    # Git pre-commit 훅
    "bandit>=1.7.5",        # 보안 취약점 검사
    "safety>=2.3.0",        # 의존성 보안 검사
]

# 🏷️ PyPI 분류자 (패키지 카테고리 및 메타데이터)
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",           # 개발 상태: 안정
    "Intended Audience :: Financial and Insurance Industry", # 대상: 금융업계
    "Intended Audience :: Developers",                       # 대상: 개발자
    "License :: OSI Approved :: MIT License",                # 라이선스: MIT
    "Operating System :: OS Independent",                    # OS: 플랫폼 독립적
    "Programming Language :: Python :: 3",                  # 언어: Python 3
    "Programming Language :: Python :: 3.9",                # Python 3.9 지원
    "Programming Language :: Python :: 3.10",               # Python 3.10 지원
    "Programming Language :: Python :: 3.11",               # Python 3.11 지원
    "Programming Language :: Python :: 3.12",               # Python 3.12 지원
    "Topic :: Office/Business :: Financial :: Investment",   # 주제: 투자
    "Topic :: Scientific/Engineering :: Artificial Intelligence",  # 주제: AI
    "Topic :: Software Development :: Libraries :: Python Modules",  # 주제: Python 모듈
]

setup(
    # 📦 기본 패키지 정보
    name="stock-analysis-system",
    version=VERSION,
    author="AI Assistant",
    author_email="ai@example.com",
    description="코스피200·나스닥100·S&P500 투자 대가 전략 AI 분석 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # 🔗 프로젝트 URL
    url="https://github.com/user/stock-analysis-system",
    project_urls={
        "Bug Reports": "https://github.com/user/stock-analysis-system/issues",
        "Source": "https://github.com/user/stock-analysis-system",
        "Documentation": "https://github.com/user/stock-analysis-system/wiki",
    },
    
    # 📁 패키지 구성
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    python_requires=">=3.9",
    
    # 📋 의존성 관리
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "dev": DEV_PACKAGES,                    # 개발용 패키지
        "all": REQUIRED_PACKAGES + DEV_PACKAGES,  # 모든 패키지
    },
    
    # 🖥️ 콘솔 스크립트 정의
    entry_points={
        "console_scripts": [
            "stock-analysis=main:main",  # 'stock-analysis' 명령어로 실행 가능
        ],
    },
    
    # 📄 패키지 데이터 포함
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],  # 모든 설정 파일
        "src": ["*.py"],                                     # 소스 코드
    },
    
    # 🏷️ 메타데이터
    zip_safe=False,  # zip 파일로 설치 시 압축 해제 필요
    keywords=[
        # 📈 투자 관련 키워드
        "stock", "analysis", "ai", "gemini", "kospi", "nasdaq", "sp500",
        # 👨‍💼 투자 대가 키워드
        "warren-buffett", "peter-lynch", "benjamin-graham",
        # 🔧 기술 키워드
        "technical-analysis", "investment", "finance", "trading"
    ],
    platforms=["any"],  # 모든 플랫폼 지원
    license="MIT",      # MIT 라이선스
) 