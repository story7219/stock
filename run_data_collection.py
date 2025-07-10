#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_data_collection.py
모듈: 데이터 수집 실행 스크립트
목적: 데이터베이스 기반 고성능 데이터 수집 실행

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from database_data_collector import DatabaseDataCollector, DatabaseConfig, CollectionConfig
from database_setup import DatabaseSetup

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def setup_environment() -> bool:
    """환경 설정"""
    print("🔧 환경 설정 시작")

    # 환경변수 파일 로드
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ .env 파일 로드 완료")
    else:
        print("⚠️ .env 파일이 없습니다. env_example.txt를 참고하여 .env 파일을 생성하세요.")

    # 필수 환경변수 확인
    required_vars = [
        'LIVE_KIS_APP_KEY',
        'LIVE_KIS_APP_SECRET',
        'POSTGRES_PASSWORD'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ 필수 환경변수가 설정되지 않았습니다: {missing_vars}")
        return False

    print("✅ 환경 설정 완료")
    return True

async def setup_database() -> bool:
    """데이터베이스 설정"""
    print("🗄️ 데이터베이스 설정 시작")

    try:
        setup = DatabaseSetup()

        # 사용자 생성
        await setup.create_user()

        # 데이터베이스 설정
        await setup.setup_database()

        # 연결 테스트
        if await setup.test_connection():
            print("✅ 데이터베이스 설정 완료")
            return True
        else:
            print("❌ 데이터베이스 연결 실패")
            return False

    except Exception as e:
        print(f"❌ 데이터베이스 설정 실패: {e}")
        return False

async def start_collection(mode: str = "realtime"):
    """데이터 수집 시작"""
    print(f"🚀 {mode} 모드로 데이터 수집 시작")

    try:
        # 설정 생성
        db_config = DatabaseConfig()
        collection_config = CollectionConfig()

        # 모드별 설정 조정
        if mode == "test":
            # 테스트 모드: 적은 종목, 긴 간격
            collection_config.kospi_symbols = collection_config.kospi_symbols[:5]
            collection_config.kosdaq_symbols = collection_config.kosdaq_symbols[:5]
            collection_config.realtime_interval = 5.0
            collection_config.batch_size = 100
        elif mode == "production":
            # 프로덕션 모드: 모든 종목, 빠른 간격
            collection_config.realtime_interval = 1.0
            collection_config.batch_size = 1000
            collection_config.max_concurrent_requests = 50

        # 수집기 생성 및 시작
        collector = DatabaseDataCollector(db_config, collection_config)

        await collector.initialize()
        await collector.start_collection()

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        raise

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="데이터베이스 기반 고성능 데이터 수집기")
    parser.add_argument(
        "--mode",
        choices=["setup", "test", "realtime", "production"],
        default="realtime",
        help="실행 모드 (setup: DB 설정, test: 테스트, realtime: 실시간, production: 프로덕션)"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="데이터베이스 설정 건너뛰기"
    )

    args = parser.parse_args()

    print("🚀 데이터베이스 기반 고성능 데이터 수집기")
    print("=" * 60)

    async def run():
        # 1. 환경 설정
        if not await setup_environment():
            return

        # 2. 데이터베이스 설정 (setup 모드이거나 skip-setup이 아닌 경우)
        if args.mode == "setup":
            if await setup_database():
                print("✅ 데이터베이스 설정 완료")
            else:
                print("❌ 데이터베이스 설정 실패")
            return

        if not args.skip_setup:
            if not await setup_database():
                print("❌ 데이터베이스 설정 실패")
                return

        # 3. 데이터 수집 시작
        await start_collection(args.mode)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n⚠️ 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

