#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 분석기 간단 테스트 스크립트
"""

import asyncio
import sys
import os
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append('.')

from personal_blackrock.ai_analyzer import AIAnalyzer
from personal_blackrock.stock_data_manager import DataManager


async def simple_test():
    """간단한 AI 분석 테스트"""
    print("🚀 AI 분석기 간단 테스트 시작")
    print(f"⏰ 테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        # 1. DataManager 생성 (빠른 초기화)
        print("📊 1단계: DataManager 생성 중...")
        data_manager = DataManager(preload_data=False)
        print("✅ DataManager 생성 완료")
        
        # 2. AIAnalyzer 생성
        print("🤖 2단계: AIAnalyzer 생성 중...")
        analyzer = AIAnalyzer(data_manager=data_manager)
        print("✅ AIAnalyzer 생성 완료")
        
        # 3. 삼성전자 단일 분석 테스트
        print("📈 3단계: 삼성전자(005930) 윌리엄 오닐 전략 분석 중...")
        result = await analyzer.analyze_stock_with_strategy('005930', '윌리엄 오닐')
        
        # 4. 결과 출력
        print("📋 분석 결과:")
        print(f"   종목명: {result.get('name', 'N/A')}")
        print(f"   종목코드: {result.get('stock_code', 'N/A')}")
        
        if 'error' in result:
            print(f"   ❌ 오류: {result['error']}")
        else:
            print(f"   📊 점수: {result.get('점수', 0)}점")
            print(f"   🏆 등급: {result.get('추천 등급', 'N/A')}")
            print(f"   💡 결론: {result.get('결론', 'N/A')}")
            print(f"   🎯 신뢰도: {result.get('신뢰도', 0)}")
        
        print("-" * 60)
        print("✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_test()) 