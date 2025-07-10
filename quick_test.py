#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: quick_test.py
모듈: 빠른 시스템 테스트
목적: 최적화된 파이프라인 시스템의 기본 기능 테스트

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import aiohttp
    import numpy as np
    import pandas as pd
    import redis.asyncio as redis
    import torch
except ImportError:
    aiohttp = None
    np = None
    pd = None
    redis = None
    torch = None

def test_environment() -> bool:
    """환경 설정 테스트"""
    print("🔍 환경 설정 테스트...")

    # Python 버전 확인
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
        print(f"❌ Python 3.11+ 필요 (현재: {python_version.major}.{python_version.minor})")
        return False
    print(f"✅ Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 필수 패키지 확인
    required_packages = [
        'pandas', 'numpy', 'torch', 'aiohttp', 'redis'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 패키지 확인")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 패키지 누락")

    if missing_packages:
        print(f"\n❌ 누락된 패키지: {', '.join(missing_packages)}")
        print("pip install -r requirements_optimized.txt를 실행하세요.")
        return False

    # GPU 확인
    if torch is None:
        print("⚠️ PyTorch가 설치되지 않았습니다. GPU 테스트를 건너뜁니다.")
    else:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ GPU 확인: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        except ImportError:
            print("⚠️ PyTorch가 설치되지 않았습니다.")

    # 환경변수 확인
    env_vars = ['LIVE_KIS_APP_KEY', 'LIVE_KIS_APP_SECRET']
    missing_env = [var for var in env_vars if not os.getenv(var)]

    if missing_env:
        print(f"⚠️ 환경변수 누락: {', '.join(missing_env)}")
        print("실제 데이터 수집을 위해서는 환경변수를 설정하세요.")
    else:
        print("✅ 환경변수 설정 확인")

    return True

def test_redis_connection():
    """Redis 연결 테스트"""
    print("\n🔗 Redis 연결 테스트...")

    try:

        # Redis 연결 테스트
        redis_client = redis.from_url('redis://localhost:6379/0', decode_responses=True)

        # 연결 테스트
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(redis_client.ping())
            print("✅ Redis 연결 성공")
            loop.run_until_complete(redis_client.close())
            return True
        except Exception as e:
            print(f"❌ Redis 연결 실패: {e}")
            print("Redis 서버를 시작하세요: sudo systemctl start redis-server")
            return False
        finally:
            loop.close()

    except ImportError:
        print("⚠️ Redis 패키지가 설치되지 않았습니다.")
        return False

def test_gpu_functionality():
    """GPU 기능 테스트"""
    print("\n⚡ GPU 기능 테스트...")

    try:

        if torch is None or not torch.cuda.is_available():
            print("⚠️ GPU를 사용할 수 없습니다. CPU 테스트로 진행합니다.")
            return test_cpu_functionality()

        # GPU 메모리 테스트
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)

        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        print(f"✅ GPU 행렬 곱셈: {gpu_time:.4f}초")

        # CPU 비교
        x_cpu = x.cpu()
        y_cpu = y.cpu()

        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time

        print(f"✅ CPU 행렬 곱셈: {cpu_time:.4f}초")
        print(f"✅ GPU 가속 비율: {cpu_time/gpu_time:.1f}x")

        return True

    except Exception as e:
        print(f"❌ GPU 테스트 실패: {e}")
        return False

def test_cpu_functionality():
    """CPU 기능 테스트"""
    print("\n💻 CPU 기능 테스트...")

    try:

        # NumPy 테스트
        start_time = time.time()
        x = np.random.randn(1000, 1000)
        y = np.random.randn(1000, 1000)
        z = np.dot(x, y)
        numpy_time = time.time() - start_time

        print(f"✅ NumPy 행렬 곱셈: {numpy_time:.4f}초")

        # Pandas 테스트
        start_time = time.time()
        df = pd.DataFrame(np.random.randn(10000, 10))
        result = df.describe()
        pandas_time = time.time() - start_time

        print(f"✅ Pandas 데이터 분석: {pandas_time:.4f}초")

        return True

    except Exception as e:
        print(f"❌ CPU 테스트 실패: {e}")
        return False

def test_data_pipeline():
    """데이터 파이프라인 테스트"""
    print("\n📊 데이터 파이프라인 테스트...")

    try:
        # 샘플 데이터 생성

        # 가상 주식 데이터 생성
        dates = pd.date_range('2024-01-01', periods=1000, freq='D')
        symbols = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': np.random.uniform(50000, 100000),
                    'high': np.random.uniform(50000, 100000),
                    'low': np.random.uniform(50000, 100000),
                    'close': np.random.uniform(50000, 100000),
                    'volume': np.random.randint(1000000, 10000000)
                })

        df = pd.DataFrame(data)

        # 기술적 지표 계산
        start_time = time.time()

        # 이동평균
        df['ma_5'] = df.groupby('symbol')['close'].rolling(window=5).mean().reset_index(0, drop=True)
        df['ma_20'] = df.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)

        # RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi'] = df.groupby('symbol')['close'].apply(calculate_rsi)

        processing_time = time.time() - start_time

        print(f"✅ 데이터 처리 완료: {len(df):,} 레코드")
        print(f"✅ 처리 시간: {processing_time:.4f}초")
        print(f"✅ 처리 속도: {len(df)/processing_time:.0f} 레코드/초")

        # 결과 저장
        output_file = f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(output_file, compression='snappy')

        file_size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"✅ 파일 저장 완료: {output_file} ({file_size:.1f}MB)")

        return True

    except Exception as e:
        print(f"❌ 데이터 파이프라인 테스트 실패: {e}")
        return False

def test_async_functionality():
    """비동기 기능 테스트"""
    print("\n🔄 비동기 기능 테스트...")

    async def async_test():

        # 비동기 HTTP 요청 테스트
        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            # 여러 요청을 동시에 실행
            tasks = []
            for i in range(10):
                task = session.get('https://httpbin.org/delay/1')
                tasks.append(task)

            responses = await asyncio.gather(*tasks)
            async_time = time.time() - start_time

            print(f"✅ 비동기 HTTP 요청: {async_time:.4f}초 (10개 요청)")
            print(f"✅ 평균 응답 시간: {async_time/10:.4f}초/요청")

            return len(responses) == 10

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(async_test())
        loop.close()

        if result:
            print("✅ 비동기 기능 테스트 성공")
            return True
        else:
            print("❌ 비동기 기능 테스트 실패")
            return False

    except Exception as e:
        print(f"❌ 비동기 테스트 실패: {e}")
        return False

def generate_test_report(results):
    """테스트 리포트 생성"""
    print("\n📋 테스트 리포트")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test_name}: {status}")

    print(f"\n총 테스트: {total_tests}개")
    print(f"통과: {passed_tests}개")
    print(f"실패: {total_tests - passed_tests}개")
    print(f"성공률: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        print("\n다음 단계:")
        print("1. 환경변수를 설정하세요 (KIS API 키)")
        print("2. python run_optimized_pipeline.py를 실행하세요")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다.")
        print("실패한 항목을 확인하고 수정한 후 다시 테스트하세요.")

def main():
    """메인 함수"""
    print("🚀 최적화된 AI 트레이딩 파이프라인 - 빠른 테스트")
    print("=" * 60)

    start_time = time.time()

    # 테스트 실행
    test_results = {
        "환경 설정": test_environment(),
        "Redis 연결": test_redis_connection(),
        "GPU/CPU 기능": test_gpu_functionality(),
        "데이터 파이프라인": test_data_pipeline(),
        "비동기 기능": test_async_functionality()
    }

    # 테스트 리포트 생성
    generate_test_report(test_results)

    total_time = time.time() - start_time
    print(f"\n총 테스트 시간: {total_time:.2f}초")

if __name__ == "__main__":
    main()

