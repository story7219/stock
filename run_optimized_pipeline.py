#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_optimized_pipeline.py
모듈: 최적화된 파이프라인 실행 스크립트
목적: 데이터 수집부터 GPU 훈련까지 전체 파이프라인 실행

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

def check_environment() -> bool:
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")

    # 필수 환경변수 확인
    required_env_vars = [
        'LIVE_KIS_APP_KEY',
        'LIVE_KIS_APP_SECRET',
        'LIVE_KIS_ACCOUNT_NUMBER',
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ 누락된 환경변수: {', '.join(missing_vars)}")
        print("qubole_env_example.txt 파일을 참고하여 환경변수를 설정하세요.")
        return False

    # Python 패키지 확인
    required_packages = [
        'pandas', 'numpy', 'pyarrow', 'dask',
        'redis', 'sqlalchemy', 'aiohttp', 'pykis',
    ]
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
        print("pip install -r requirements.txt를 실행하세요.")
        return False

    # GPU 확인
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ GPU 확인: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        except Exception:
            print("⚠️ PyTorch가 설치되었으나 GPU 확인 실패. CPU 모드로 실행됩니다.")
    else:
        print("⚠️ PyTorch가 설치되지 않았습니다. CPU 모드로 실행됩니다.")

    print("✅ 환경 설정 확인 완료")
    return True

def run_data_collection() -> bool:
    """데이터 수집 실행"""
    print("\n📊 데이터 수집 시작...")
    try:
        result = subprocess.run([
            sys.executable, "optimized_data_pipeline.py"
        ], capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print("✅ 데이터 수집 완료")
            return True
        else:
            print(f"❌ 데이터 수집 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 데이터 수집 시간 초과")
        return False
    except Exception as e:
        print(f"❌ 데이터 수집 오류: {e}")
        return False

def run_gpu_training() -> bool:
    """GPU 훈련 실행"""
    print("\n🤖 GPU 훈련 시작...")
    data_files = list(Path(".").glob("optimized_data_*.parquet"))
    if not data_files:
        print("❌ 훈련할 데이터 파일을 찾을 수 없습니다.")
        return False
    latest_data_file = max(data_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 사용할 데이터 파일: {latest_data_file}")
    try:
        result = subprocess.run([
            sys.executable, "gpu_optimized_training.py"
        ], capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            print("✅ GPU 훈련 완료")
            return True
        else:
            print(f"❌ GPU 훈련 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ GPU 훈련 시간 초과")
        return False
    except Exception as e:
        print(f"❌ GPU 훈련 오류: {e}")
        return False

def generate_performance_report() -> None:
    """성능 리포트 생성"""
    print("\n📈 성능 리포트 생성...")
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_collection": {
            "status": "completed",
            "files": [str(f) for f in Path(".").glob("optimized_data_*.parquet")],
        },
        "gpu_training": {
            "status": "completed",
            "models": [str(f) for f in Path(".").glob("gpu_optimized_model_*.pth")],
        },
        "logs": {
            "pipeline": "optimized_pipeline.log",
            "training": "gpu_training.log",
        },
    }
    with open("performance_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("✅ 성능 리포트 생성 완료: performance_report.json")

def main() -> None:
    """메인 함수"""
    print("🚀 최적화된 AI 트레이딩 파이프라인 실행")
    print("=" * 60)
    start_time = time.time()
    if not check_environment():
        print("❌ 환경 설정 확인 실패")
        return
    if not run_data_collection():
        print("❌ 데이터 수집 실패")
        return
    if not run_gpu_training():
        print("❌ GPU 훈련 실패")
        return
    generate_performance_report()
    total_time = time.time() - start_time
    print(f"\n🎉 전체 파이프라인 완료!")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print("\n📁 생성된 파일들:")
    for pattern in ["optimized_data_*.parquet", "gpu_optimized_model_*.pth", "*.log"]:
        files = list(Path(".").glob(pattern))
        for file in files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.1f}MB)")

if __name__ == "__main__":
    main()

