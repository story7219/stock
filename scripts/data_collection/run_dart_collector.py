#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_dart_collector.py
모듈: DART 데이터 수집기 실행 스크립트
목적: DART API를 활용한 과거 데이터 수집 실행

Author: Trading AI System
Created: 2025-01-07
Version: 1.0.0
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dart_historical_data_collector import DARTHistoricalCollector, CollectionConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dart_collector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """환경 설정"""
    # DART API 키 확인
    api_key = os.environ.get('DART_API_KEY')
    if not api_key:
        logger.error("❌ DART_API_KEY 환경변수가 설정되지 않았습니다.")
        logger.info("🔧 환경변수 설정 방법:")
        logger.info("Windows PowerShell:")
        logger.info("  $env:DART_API_KEY='your_api_key_here'")
        logger.info("Windows Command Prompt:")
        logger.info("  set DART_API_KEY=your_api_key_here")
        logger.info("Linux/Mac:")
        logger.info("  export DART_API_KEY=your_api_key_here")
        logger.info("")
        logger.info("📝 DART API 키 발급 방법:")
        logger.info("  1. https://opendart.fss.or.kr/ 접속")
        logger.info("  2. 회원가입 및 로그인")
        logger.info("  3. '오픈API 신청' 메뉴에서 API 키 발급")
        return False
    
    logger.info("✅ DART API 키 확인 완료")
    return True


def create_collection_config():
    """수집 설정 생성"""
    config = CollectionConfig(
        api_key=os.environ.get('DART_API_KEY', ''),
        output_dir=Path('dart_historical_data'),
        start_year=2015,  # 2015년부터 수집
        end_year=datetime.now().year,
        include_disclosures=True,      # 공시 정보
        include_financials=True,       # 재무제표
        include_executives=True,       # 임원 정보
        include_dividends=True,        # 배당 정보
        include_auditors=True,         # 감사 정보
        include_corp_info=True,        # 기업 개황
        request_delay=0.1,            # API 호출 간격 (초)
        max_retries=3                 # 재시도 횟수
    )
    
    logger.info("📋 수집 설정:")
    logger.info(f"  - 출력 디렉토리: {config.output_dir}")
    logger.info(f"  - 수집 기간: {config.start_year}년 ~ {config.end_year}년")
    logger.info(f"  - API 호출 간격: {config.request_delay}초")
    logger.info(f"  - 재시도 횟수: {config.max_retries}회")
    
    return config


async def main():
    """메인 실행 함수"""
    logger.info("🚀 DART 데이터 수집기 시작")
    
    # 1. 환경 설정 확인
    if not setup_environment():
        return
    
    # 2. 수집 설정 생성
    config = create_collection_config()
    
    # 3. 수집기 실행
    try:
        async with DARTHistoricalCollector(config) as collector:
            await collector.collect_all_historical_data()
            
        logger.info("✅ DART 데이터 수집 완료")
        logger.info(f"📁 데이터 저장 위치: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ DART 데이터 수집 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 