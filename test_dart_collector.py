#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_dart_collector.py
모듈: DART 데이터 수집기 테스트 스크립트
목적: DART API 연결 및 기본 기능 테스트

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_connection():
    """DART API 연결 테스트"""
    try:
        import dart_fss as dart
        
        # API 키 확인
        api_key = os.environ.get('DART_API_KEY')
        if not api_key:
            logger.error("❌ DART_API_KEY 환경변수가 설정되지 않았습니다.")
            return False
            
        # API 초기화
        dart.set_api_key(api_key=api_key)
        
        # 기업 목록 조회 테스트
        corp_list = dart.get_corp_list()
        logger.info(f"✅ API 연결 성공: {len(corp_list)}개 기업 목록 조회")
        
        # 첫 번째 기업 정보 테스트
        if corp_list:
            first_corp = corp_list[0]
            logger.info(f"📋 첫 번째 기업: {first_corp.corp_name} ({first_corp.corp_code})")
            
            # 기업 개황 정보 테스트
            try:
                info = first_corp.info
                logger.info(f"✅ 기업 개황 정보 조회 성공: {len(info)}개 항목")
            except Exception as e:
                logger.warning(f"⚠️ 기업 개황 정보 조회 실패: {e}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ API 연결 테스트 실패: {e}")
        return False


async def test_single_corp_collection():
    """단일 기업 데이터 수집 테스트"""
    try:
        # 설정
        config = CollectionConfig(
            api_key=os.environ.get('DART_API_KEY', ''),
            output_dir=Path('test_dart_data'),
            start_year=2023,
            end_year=2024,
            include_disclosures=True,
            include_financials=True,
            include_executives=True,
            include_dividends=True,
            include_auditors=True,
            include_corp_info=True,
            request_delay=0.2,  # 테스트용으로 간격 증가
            max_retries=2
        )
        
        # 수집기 실행
        async with DARTHistoricalCollector(config) as collector:
            # 기업 목록 수집
            corp_list = await collector._collect_corp_list()
            
            if corp_list:
                # 첫 번째 기업만 테스트
                test_corp = corp_list[0]
                logger.info(f"🧪 테스트 대상: {test_corp.corp_name}")
                
                # 개별 데이터 수집 테스트
                await collector._collect_corp_data(test_corp)
                
                logger.info("✅ 단일 기업 데이터 수집 테스트 완료")
                return True
            else:
                logger.error("❌ 기업 목록이 비어있습니다.")
                return False
                
    except Exception as e:
        logger.error(f"❌ 단일 기업 수집 테스트 실패: {e}")
        return False


async def test_specific_corp():
    """특정 기업 데이터 수집 테스트 (삼성전자)"""
    try:
        import dart_fss as dart
        
        # 설정
        config = CollectionConfig(
            api_key=os.environ.get('DART_API_KEY', ''),
            output_dir=Path('test_samsung_data'),
            start_year=2023,
            end_year=2024,
            include_disclosures=True,
            include_financials=True,
            include_executives=True,
            include_dividends=True,
            include_auditors=True,
            include_corp_info=True,
            request_delay=0.2
        )
        
        # 삼성전자 찾기
        dart.set_api_key(api_key=config.api_key)
        corp_list = dart.get_corp_list()
        
        samsung_corp = None
        for corp in corp_list:
            if '삼성전자' in corp.corp_name:
                samsung_corp = corp
                break
                
        if not samsung_corp:
            logger.error("❌ 삼성전자를 찾을 수 없습니다.")
            return False
            
        logger.info(f"📱 삼성전자 테스트: {samsung_corp.corp_name} ({samsung_corp.corp_code})")
        
        # 수집기 실행
        async with DARTHistoricalCollector(config) as collector:
            await collector._collect_corp_data(samsung_corp)
            
        logger.info("✅ 삼성전자 데이터 수집 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 삼성전자 수집 테스트 실패: {e}")
        return False


async def main():
    """메인 테스트 함수"""
    logger.info("🧪 DART 데이터 수집기 테스트 시작")
    
    # 1. API 연결 테스트
    logger.info("1️⃣ API 연결 테스트")
    if not test_api_connection():
        logger.error("❌ API 연결 테스트 실패. 테스트를 중단합니다.")
        return
    
    # 2. 단일 기업 수집 테스트
    logger.info("2️⃣ 단일 기업 데이터 수집 테스트")
    if not await test_single_corp_collection():
        logger.error("❌ 단일 기업 수집 테스트 실패.")
    
    # 3. 특정 기업 (삼성전자) 테스트
    logger.info("3️⃣ 삼성전자 데이터 수집 테스트")
    if not await test_specific_corp():
        logger.error("❌ 삼성전자 수집 테스트 실패.")
    
    logger.info("✅ 모든 테스트 완료")


if __name__ == "__main__":
    asyncio.run(main()) 