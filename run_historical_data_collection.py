#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_historical_data_collection.py
모듈: 한투 API + DART API 과거 데이터 수집 실행 스크립트
목적: 과거 최대치 데이터 수집 및 품질 체크

Author: Trading AI System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

사용법:
    python run_historical_data_collection.py

환경변수 설정:
    KIS_APP_KEY=your_kis_app_key
    KIS_APP_SECRET=your_kis_app_secret
    KIS_REAL_APP_KEY=your_kis_real_app_key
    KIS_REAL_APP_SECRET=your_kis_real_app_secret
    DART_API_KEY=your_dart_api_key

License: MIT
"""

import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.historical_data_collector import HistoricalDataConfig, HistoricalDataCollector


def setup_environment():
    """환경변수 설정 확인"""
    required_vars = [
        'KIS_APP_KEY',
        'KIS_APP_SECRET', 
        'KIS_REAL_APP_KEY',
        'KIS_REAL_APP_SECRET',
        'DART_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 필수 환경변수가 설정되지 않았습니다:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n환경변수 설정 방법:")
        print("1. .env 파일 생성:")
        print("   KIS_APP_KEY=your_kis_app_key")
        print("   KIS_APP_SECRET=your_kis_app_secret")
        print("   KIS_REAL_APP_KEY=your_kis_real_app_key")
        print("   KIS_REAL_APP_SECRET=your_kis_real_app_secret")
        print("   DART_API_KEY=your_dart_api_key")
        print("\n2. 또는 시스템 환경변수로 설정")
        return False
    
    print("✅ 환경변수 설정 확인 완료")
    return True


def create_config() -> HistoricalDataConfig:
    """설정 생성"""
    config = HistoricalDataConfig()
    
    # 추가 종목 설정 (필요시 수정)
    additional_symbols = [
        "035720",  # 카카오
        "051900",  # LG생활건강
        "068270",  # 셀트리온
        "207940",  # 삼성바이오로직스
        "323410",  # 카카오뱅크
        "373220",  # LG에너지솔루션
        "207940",  # 삼성바이오로직스
        "051910",  # LG화학
        "006400",  # 삼성SDI
        "035420",  # NAVER
    ]
    
    # 중복 제거
    all_symbols = list(set(config.symbols + additional_symbols))
    config.symbols = all_symbols
    
    # 기간 설정 (최대 10년)
    config.start_date = "2015-01-01"  # 10년 데이터
    config.end_date = "2024-12-31"
    
    # 저장 설정
    config.output_dir = "backtest/historical_data"
    config.save_format = "csv"
    
    print(f"📊 수집 설정:")
    print(f"   - 종목 수: {len(config.symbols)}개")
    print(f"   - 기간: {config.start_date} ~ {config.end_date}")
    print(f"   - 저장 형식: {config.save_format}")
    print(f"   - 출력 디렉토리: {config.output_dir}")
    
    return config


async def run_data_collection():
    """데이터 수집 실행"""
    print("\n🚀 과거 데이터 수집 시작")
    
    # 환경변수 확인
    if not setup_environment():
        return
    
    # 설정 생성
    config = create_config()
    
    # 수집기 초기화
    collector = HistoricalDataCollector(config)
    
    try:
        # 초기화
        print("\n📡 API 초기화 중...")
        if not await collector.initialize():
            print("❌ API 초기화 실패")
            return
        
        print("✅ API 초기화 완료")
        
        # 데이터 수집
        print(f"\n📊 {len(config.symbols)}개 종목 데이터 수집 중...")
        data = await collector.collect_all_data()
        
        # 결과 분석
        successful = sum(1 for df, _ in data.values() if not df.empty)
        failed = len(data) - successful
        total_points = sum(len(df) for df, _ in data.values())
        
        print(f"\n📈 수집 결과:")
        print(f"   - 성공: {successful}개 종목")
        print(f"   - 실패: {failed}개 종목")
        print(f"   - 총 데이터 포인트: {total_points:,}개")
        
        # 데이터 저장
        print("\n💾 데이터 저장 중...")
        collector.save_data(data)
        
        # 요약 리포트 생성
        summary = collector.generate_summary_report(data)
        
        # 리포트 저장
        report_filepath = f"{config.output_dir}/collection_summary.json"
        import json
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 데이터 저장 완료: {config.output_dir}")
        print(f"📋 요약 리포트: {report_filepath}")
        
        # 품질 점수 출력
        if summary['average_quality_score'] > 0:
            print(f"\n🎯 평균 품질 점수: {summary['average_quality_score']:.1f}/100")
            
            if summary['average_quality_score'] >= 90:
                print("   🟢 우수한 데이터 품질")
            elif summary['average_quality_score'] >= 70:
                print("   🟡 양호한 데이터 품질")
            else:
                print("   🔴 데이터 품질 개선 필요")
        
        # 실패한 종목 출력
        failed_symbols = [symbol for symbol, (df, _) in data.items() if df.empty]
        if failed_symbols:
            print(f"\n❌ 수집 실패 종목:")
            for symbol in failed_symbols:
                print(f"   - {symbol}")
        
        print("\n🎉 과거 데이터 수집 완료!")
        
    except Exception as e:
        print(f"❌ 데이터 수집 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await collector.close()


def main():
    """메인 함수"""
    print("=" * 60)
    print("📊 한투 API + DART API 과거 데이터 수집 시스템")
    print("=" * 60)
    
    # 비동기 실행
    asyncio.run(run_data_collection())


if __name__ == "__main__":
    main() 