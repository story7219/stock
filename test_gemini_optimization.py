#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Gemini AI 100% 성능 최적화 테스트
"""

import asyncio
import time
import logging
from typing import Dict, List, Any
import json
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_integration.gemini_optimizer import GeminiOptimizer, OptimizationConfig
from ai_integration.ultra_ai_analyzer import UltraAIAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiPerformanceTest:
    """Gemini AI 성능 테스트 클래스"""
    
    def __init__(self):
        self.test_stocks = [
            {
                "symbol": "005930",
                "name": "삼성전자",
                "price": 75000,
                "market_cap": 450000000000000,
                "pe_ratio": 15.2,
                "pb_ratio": 1.8,
                "roe": 12.5,
                "debt_ratio": 45.3,
                "sector": "반도체"
            },
            {
                "symbol": "000660",
                "name": "SK하이닉스",
                "price": 125000,
                "market_cap": 91000000000000,
                "pe_ratio": 18.5,
                "pb_ratio": 2.1,
                "roe": 8.9,
                "debt_ratio": 38.7,
                "sector": "반도체"
            },
            {
                "symbol": "035420",
                "name": "NAVER",
                "price": 195000,
                "market_cap": 32000000000000,
                "pe_ratio": 22.1,
                "pb_ratio": 1.9,
                "roe": 9.8,
                "debt_ratio": 25.4,
                "sector": "인터넷"
            }
        ]
        
        # 결과 저장
        self.test_results = {}
    
    async def test_optimizer_performance(self):
        """최적화기 성능 테스트"""
        print("\n🚀 Gemini 최적화기 성능 테스트 시작")
        print("=" * 60)
        
        try:
            # 올바른 파라미터명으로 설정
            config = OptimizationConfig(
                max_concurrent=100,
                batch_size=50,
                ultra_cache_ttl=14400,
                enable_turbo_mode=True
            )
            
            optimizer = GeminiOptimizer(config)
            
            # 단일 종목 분석 테스트
            start_time = time.time()
            result = await optimizer.ultra_analyze_stock(
                self.test_stocks[0], 
                strategy="ultra_comprehensive"
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            print(f"✅ 단일 분석 완료: {duration:.3f}초")
            print(f"📊 분석 결과 품질: {len(str(result))} 문자")
            
            # 성능 통계 조회
            stats = optimizer.get_ultra_performance_stats()
            print(f"📈 성능 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
            
            self.test_results['optimizer'] = {
                'duration': duration,
                'success': True,
                'result_size': len(str(result)),
                'stats': stats
            }
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['optimizer'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_ultra_analyzer(self):
        """울트라 분석기 테스트"""
        print("\n🎯 울트라 AI 분석기 테스트 시작")
        print("=" * 60)
        
        try:
            analyzer = UltraAIAnalyzer()
            
            # 배치 분석 테스트
            symbols = [stock["symbol"] for stock in self.test_stocks]
            
            start_time = time.time()
            results = await analyzer.analyze_stocks(
                symbols, 
                strategy='comprehensive',
                market='kr'
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            print(f"✅ 배치 분석 완료: {duration:.3f}초")
            print(f"📊 분석된 종목 수: {len(results)}")
            
            # 각 결과 요약 출력
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"  {i+1}. {result.get('symbol', 'N/A')}: 분석 성공")
                else:
                    print(f"  {i+1}. 분석 실패: {result.get('error', 'Unknown')}")
            
            # 시스템 정보 조회
            system_info = analyzer.get_system_info()
            print(f"🖥️ 시스템 정보: {json.dumps(system_info, indent=2, ensure_ascii=False)}")
            
            self.test_results['ultra_analyzer'] = {
                'duration': duration,
                'success': True,
                'analyzed_count': len(results),
                'system_info': system_info
            }
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['ultra_analyzer'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_concurrent_performance(self):
        """동시 처리 성능 테스트"""
        print("\n⚡ 동시 처리 성능 테스트 시작")
        print("=" * 60)
        
        try:
            config = OptimizationConfig(
                max_concurrent=50,
                batch_size=25,
                enable_turbo_mode=True
            )
            
            optimizer = GeminiOptimizer(config)
            
            # 여러 종목 동시 분석
            tasks = []
            for stock in self.test_stocks:
                task = optimizer.ultra_analyze_stock(stock, "comprehensive")
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            duration = end_time - start_time
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            print(f"✅ 동시 분석 완료: {duration:.3f}초")
            print(f"📊 성공률: {success_count}/{len(tasks)} ({success_count/len(tasks)*100:.1f}%)")
            print(f"⚡ 처리 속도: {len(tasks)/duration:.2f} 종목/초")
            
            self.test_results['concurrent'] = {
                'duration': duration,
                'success_rate': success_count/len(tasks),
                'throughput': len(tasks)/duration,
                'total_tasks': len(tasks),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['concurrent'] = {
                'success': False,
                'error': str(e)
            }
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🧪 Gemini AI 100% 성능 최적화 테스트 시작")
        print("=" * 80)
        
        # 개별 테스트 실행
        await self.test_optimizer_performance()
        await self.test_ultra_analyzer()
        await self.test_concurrent_performance()
        
        # 종합 결과 출력
        print("\n📋 테스트 종합 결과")
        print("=" * 80)
        
        for test_name, result in self.test_results.items():
            print(f"\n🎯 {test_name.upper()} 테스트:")
            if result.get('success', False):
                print(f"  ✅ 성공")
                if 'duration' in result:
                    print(f"  ⏱️ 소요시간: {result['duration']:.3f}초")
                if 'success_rate' in result:
                    print(f"  📊 성공률: {result['success_rate']*100:.1f}%")
                if 'throughput' in result:
                    print(f"  ⚡ 처리속도: {result['throughput']:.2f} 종목/초")
            else:
                print(f"  ❌ 실패: {result.get('error', 'Unknown error')}")
        
        # 전체 성능 평가
        success_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        total_tests = len(self.test_results)
        
        print(f"\n🏆 전체 성공률: {success_tests}/{total_tests} ({success_tests/total_tests*100:.1f}%)")
        
        if success_tests == total_tests:
            print("🎉 모든 테스트 통과! Gemini AI 100% 성능 최적화 완료!")
        else:
            print("⚠️ 일부 테스트 실패. 추가 최적화가 필요합니다.")

async def main():
    """메인 실행 함수"""
    tester = GeminiPerformanceTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 