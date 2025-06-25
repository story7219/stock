#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 스탠리 드러켄밀러 전략 독립 실행기
=========================================

스탠리 드러켄밀러의 투자 전략을 독립적으로 실행하는 전용 스크립트입니다.
거시경제 분석과 추세 추종을 기반으로 한 드러켄밀러의 투자 철학을 구현합니다.

주요 기능:
- 스탠리 드러켄밀러 전략 독립 실행
- 전체 시장(코스피200, 나스닥100, S&P500) 분석
- Top 5 종목 자동 선정
- 분석 결과 리포트 생성 및 저장

실행 방법:
    python run_druckenmiller.py

드러켄밀러 전략 특징:
- 거시경제 트렌드 분석 중심
- 높은 집중도의 포트폴리오 구성
- 추세 추종과 역추세 매매 병행
- 리스크 관리 우선주의

이 스크립트는 독립적으로 실행되며, 다른 전략과 분리되어 
순수하게 드러켄밀러 전략만의 관점에서 종목을 분석합니다.
"""

import asyncio
import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가 (수정)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.independent_strategy_runner import IndependentStrategyRunner

async def main():
    """🎯 메인 실행 함수"""
    runner = IndependentStrategyRunner()
    
    print("🎯 스탠리 드러켄밀러 전략 실행...")
    results = await runner.run_single_strategy('stanley_druckenmiller', 'all', 5)
    
    # 결과 출력
    report = runner.generate_report(results)
    print(report)
    
    # 결과 저장
    runner.save_results(results)

if __name__ == "__main__":
    """🚀 프로그램 진입점"""
    # Windows에서 asyncio 실행 시 ProactorEventLoopPolicy 사용
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main()) 