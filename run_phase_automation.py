#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: run_phase_automation.py
목적: Phase 1~4 완전 자동화 파이프라인 실행 스크립트

Author: AI Trading System
Created: 2025-01-08
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - src/phase_automation_pipeline.py

License: MIT
"""

import asyncio
import logging
from pathlib import Path
import sys

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from phase_automation_pipeline import PhaseAutomationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

async def main():
    pipeline = PhaseAutomationPipeline()
    await pipeline.initialize()
    await pipeline.run(interval=60)

if __name__ == "__main__":
    asyncio.run(main()) 