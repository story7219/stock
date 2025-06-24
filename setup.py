#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup configuration for Stock Analysis System
코스피200·나스닥100·S&P500 투자 대가 전략 AI 분석 시스템 설정
"""

from setuptools import setup, find_packages
from pathlib import Path

# README 파일 읽기
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# 버전 정보
VERSION = "5.0.0"

# 필수 패키지 목록
REQUIRED_PACKAGES = [
    # 핵심 데이터 처리
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # 비동기 HTTP 클라이언트
    "aiohttp>=3.8.5",
    "aiofiles>=23.0.0",
    
    # 웹 스크래핑
    "beautifulsoup4>=4.12.2",
    "selenium>=4.15.0",
    "requests>=2.31.0",
    
    # 금융 데이터
    "yfinance>=0.2.28",
    "investpy>=1.0.8",
    
    # AI/ML
    "google-generativeai>=0.3.0",
    "scikit-learn>=1.3.0",
    
    # 기술적 분석
    "pandas-ta>=0.3.14b0",
    "finta>=1.3",
    
    # 데이터 시각화
    "matplotlib>=3.7.2",
    "seaborn>=0.12.2",
    "plotly>=5.17.0",
    
    # 구글 서비스
    "google-auth>=2.23.0",
    "google-auth-oauthlib>=1.1.0",
    "google-auth-httplib2>=0.1.1",
    "google-api-python-client>=2.100.0",
    "gspread>=5.11.0",
    
    # 텔레그램
    "python-telegram-bot>=20.6",
    
    # 스케줄링
    "schedule>=1.2.0",
    "APScheduler>=3.10.0",
    
    # 환경 변수 관리
    "python-dotenv>=1.0.0",
    
    # 로깅 및 모니터링
    "loguru>=0.7.0",
    "psutil>=5.9.0",
    
    # 데이터 검증
    "pydantic>=2.4.0",
    "cerberus>=1.3.4",
    
    # 코드 품질
    "black>=23.9.0",
    "pylint>=2.17.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    
    # 테스팅
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    
    # 유틸리티
    "tqdm>=4.66.0",
    "colorama>=0.4.6",
    "rich>=13.6.0",
    "typer>=0.9.0",
]

# 개발용 추가 패키지
DEV_PACKAGES = [
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "ipywidgets>=8.1.0",
    "pre-commit>=3.4.0",
    "bandit>=1.7.5",
    "safety>=2.3.0",
]

# 분류자
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Office/Business :: Financial :: Investment",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="stock-analysis-system",
    version=VERSION,
    author="AI Assistant",
    author_email="ai@example.com",
    description="코스피200·나스닥100·S&P500 투자 대가 전략 AI 분석 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user/stock-analysis-system",
    project_urls={
        "Bug Reports": "https://github.com/user/stock-analysis-system/issues",
        "Source": "https://github.com/user/stock-analysis-system",
        "Documentation": "https://github.com/user/stock-analysis-system/wiki",
    },
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    python_requires=">=3.9",
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "dev": DEV_PACKAGES,
        "all": REQUIRED_PACKAGES + DEV_PACKAGES,
    },
    entry_points={
        "console_scripts": [
            "stock-analysis=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
        "src": ["*.py"],
    },
    zip_safe=False,
    keywords=[
        "stock", "analysis", "ai", "gemini", "kospi", "nasdaq", "sp500",
        "warren-buffett", "peter-lynch", "benjamin-graham",
        "technical-analysis", "investment", "finance", "trading"
    ],
    platforms=["any"],
    license="MIT",
) 