#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Gemini AI 100% 성능 최적화 테스트
최고 성능 검증 및 벤치마크 시스템
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime
import statistics

from gemini_optimizer import GeminiOptimizer, OptimizationConfig
from ultra_ai_analyzer import UltraAIAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiPerformanceTest:
    """제미나이 성능 테스트 클래스"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {
            'response_times': [],
            'success_count': 0,
            'error_count': 0,
            'cache_hits': 0,
            'total_tokens': 0
        }
        
        # 테스트용 샘플 데이터
        self.sample_stocks = [
            {
                'symbol': '005930', 'name': '삼성전자', 'price': 75000,
                'market_cap': 450000000000000, 'pe_ratio': 15.5, 'pb_ratio': 1.2,
                'roe': 12.5, 'debt_ratio': 15.2, 'dividend_yield': 2.1,
                'revenue_growth': 8.5, 'profit_growth': 12.3, 'sector': '기술'
            },
            {
                'symbol': '000660', 'name': 'SK하이닉스', 'price': 125000,
                'market_cap': 90000000000000, 'pe_ratio': 22.1, 'pb_ratio': 1.8,
                'roe': 8.7, 'debt_ratio': 25.4, 'dividend_yield': 1.5,
                'revenue_growth': 15.2, 'profit_growth': 25.8, 'sector': '반도체'
            },
            {
                'symbol': '035420', 'name': 'NAVER', 'price': 180000,
                'market_cap': 30000000000000, 'pe_ratio': 28.5, 'pb_ratio': 2.1,
                'roe': 15.2, 'debt_ratio': 8.9, 'dividend_yield': 0.8,
                'revenue_growth': 22.1, 'profit_growth': 18.7, 'sector': '인터넷'
            },
            {
                'symbol': '207940', 'name': '삼성바이오로직스', 'price': 850000,
                'market_cap': 60000000000000, 'pe_ratio': 45.2, 'pb_ratio': 3.5,
                'roe': 18.9, 'debt_ratio': 12.1, 'dividend_yield': 0.0,
                'revenue_growth': 35.8, 'profit_growth': 42.1, 'sector': '바이오'
            },
            {
                'symbol': '051910', 'name': 'LG화학', 'price': 420000,
                'market_cap': 30000000000000, 'pe_ratio': 18.7, 'pb_ratio': 1.4,
                'roe': 9.8, 'debt_ratio': 32.1, 'dividend_yield': 1.2,
                'revenue_growth': 12.5, 'profit_growth': 8.9, 'sector': '화학'
            }
        ]
    
    async def test_individual_analysis(self) -> Dict[str, Any]:
        """개별 종목 분석 테스트"""
        logger.info("🧪 개별 종목 분석 테스트 시작...")
        
        optimizer = GeminiOptimizer()
        test_stock = self.sample_stocks[0]  # 삼성전자
        
        start_time = time.time()
        
        try:
            result = await optimizer.ultra_analyze_stock(test_stock)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            self.performance_metrics['response_times'].append(response_time)
            self.performance_metrics['success_count'] += 1
            
            test_result = {
                "테스트": "개별_종목_분석",
                "종목": f"{test_stock['name']} ({test_stock['symbol']})",
                "응답시간": f"{response_time:.3f}초",
                "상태": "✅ 성공",
                "AI등급": result.get('ultra_grade', 'N/A'),
                "AI점수": result.get('ai_score', 'N/A'),
                "투자의견": result.get('investment_opinion', 'N/A'),
                "신뢰도": result.get('confidence_level', 'N/A')
            }
            
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            test_result = {
                "테스트": "개별_종목_분석",
                "상태": f"❌ 실패: {str(e)}",
                "응답시간": f"{time.time() - start_time:.3f}초"
            }
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_ultra_analyzer(self) -> Dict[str, Any]:
        """울트라 분석기 테스트"""
        logger.info("🧪 울트라 분석기 테스트 시작...")
        
        analyzer = UltraAIAnalyzer()
        symbols = [stock['symbol'] for stock in self.sample_stocks[:3]]
        
        start_time = time.time()
        
        try:
            results = await analyzer.analyze_stocks(
                symbols=symbols,
                strategy='comprehensive'
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / len(symbols)
            
            self.performance_metrics['response_times'].extend([avg_time] * len(symbols))
            self.performance_metrics['success_count'] += len(results)
            
            test_result = {
                "테스트": "울트라_분석기",
                "종목수": len(symbols),
                "총_시간": f"{total_time:.3f}초",
                "평균_시간": f"{avg_time:.3f}초",
                "상태": "✅ 성공",
                "처리된_종목": len(results),
                "성공률": f"{len(results)/len(symbols)*100:.1f}%"
            }
            
        except Exception as e:
            self.performance_metrics['error_count'] += len(symbols)
            test_result = {
                "테스트": "울트라_분석기",
                "상태": f"❌ 실패: {str(e)}",
                "총_시간": f"{time.time() - start_time:.3f}초"
            }
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_concurrent_processing(self) -> Dict[str, Any]:
        """동시 처리 성능 테스트"""
        logger.info("🧪 동시 처리 성능 테스트 시작...")
        
        optimizer = GeminiOptimizer()
        
        # 동시 처리 태스크 생성
        tasks = [
            optimizer.ultra_analyze_stock(stock)
            for stock in self.sample_stocks
        ]
        
        start_time = time.time()
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 성공/실패 분류
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            self.performance_metrics['success_count'] += len(successful_results)
            self.performance_metrics['error_count'] += len(failed_results)
            
            test_result = {
                "테스트": "동시_처리_성능",
                "동시_처리수": len(tasks),
                "총_시간": f"{total_time:.3f}초",
                "평균_시간": f"{total_time/len(tasks):.3f}초",
                "성공": len(successful_results),
                "실패": len(failed_results),
                "성공률": f"{len(successful_results)/len(tasks)*100:.1f}%",
                "상태": "✅ 성공" if len(successful_results) > 0 else "❌ 실패",
                "처리량": f"{len(tasks)/total_time:.2f} 종목/초"
            }
            
        except Exception as e:
            test_result = {
                "테스트": "동시_처리_성능",
                "상태": f"❌ 실패: {str(e)}",
                "총_시간": f"{time.time() - start_time:.3f}초"
            }
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """캐시 성능 테스트"""
        logger.info("🧪 캐시 성능 테스트 시작...")
        
        optimizer = GeminiOptimizer()
        test_stock = self.sample_stocks[0]
        
        # 첫 번째 요청 (캐시 미스)
        start_time1 = time.time()
        result1 = await optimizer.ultra_analyze_stock(test_stock)
        time1 = time.time() - start_time1
        
        # 두 번째 요청 (캐시 히트)
        start_time2 = time.time()
        result2 = await optimizer.ultra_analyze_stock(test_stock)
        time2 = time.time() - start_time2
        
        cache_speedup = time1 / time2 if time2 > 0 else float('inf')
        
        test_result = {
            "테스트": "캐시_성능",
            "첫번째_요청": f"{time1:.3f}초 (캐시 미스)",
            "두번째_요청": f"{time2:.3f}초 (캐시 히트)",
            "속도_향상": f"{cache_speedup:.1f}배",
            "캐시_효율성": "✅ 우수" if cache_speedup > 5 else "⚠️ 보통",
            "상태": "✅ 성공"
        }
        
        if cache_speedup > 5:
            self.performance_metrics['cache_hits'] += 1
        
        self.test_results.append(test_result)
        return test_result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        logger.info("🚀 제미나이 AI 100% 성능 최적화 종합 테스트 시작!")
        
        overall_start = time.time()
        
        # 모든 테스트 실행
        test_functions = [
            self.test_individual_analysis,
            self.test_ultra_analyzer,
            self.test_concurrent_processing,
            self.test_cache_performance
        ]
        
        for test_func in test_functions:
            try:
                await test_func()
                await asyncio.sleep(0.1)  # 테스트 간 간격
            except Exception as e:
                logger.error(f"테스트 실패: {test_func.__name__} - {e}")
        
        overall_time = time.time() - overall_start
        
        # 종합 성능 분석
        total_requests = self.performance_metrics['success_count'] + self.performance_metrics['error_count']
        avg_response_time = statistics.mean(self.performance_metrics['response_times']) if self.performance_metrics['response_times'] else 0
        success_rate = (self.performance_metrics['success_count'] / total_requests * 100) if total_requests > 0 else 0
        
        comprehensive_result = {
            "🚀 제미나이 AI 100% 최적화 테스트 결과": {
                "전체_테스트_시간": f"{overall_time:.3f}초",
                "총_요청수": total_requests,
                "성공률": f"{success_rate:.1f}%",
                "평균_응답시간": f"{avg_response_time:.3f}초",
                "캐시_히트수": self.performance_metrics['cache_hits'],
                "성능_등급": self._calculate_performance_grade(success_rate, avg_response_time),
                "최적화_상태": "🟢 최고 성능" if success_rate > 90 and avg_response_time < 5 else "🟡 양호",
                "개별_테스트_결과": self.test_results
            }
        }
        
        return comprehensive_result
    
    def _calculate_performance_grade(self, success_rate: float, avg_time: float) -> str:
        """성능 등급 계산"""
        if success_rate >= 95 and avg_time <= 2:
            return "🏆 SSS+ (초월 성능)"
        elif success_rate >= 90 and avg_time <= 3:
            return "🥇 SS (최고 성능)"
        elif success_rate >= 85 and avg_time <= 5:
            return "🥈 S (우수 성능)"
        elif success_rate >= 80 and avg_time <= 8:
            return "🥉 A (양호 성능)"
        else:
            return "📈 개선 필요"

# 실행 함수들
async def run_quick_test():
    """빠른 테스트 실행"""
    tester = GeminiPerformanceTest()
    result = await tester.test_individual_analysis()
    print("🧪 빠른 테스트 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

async def run_full_test():
    """전체 테스트 실행"""
    tester = GeminiPerformanceTest()
    result = await tester.run_comprehensive_test()
    
    print("🚀 제미나이 AI 100% 성능 최적화 종합 테스트 완료!")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)
    
    return result

def run_sync_test():
    """동기 테스트 실행"""
    return asyncio.run(run_full_test())

if __name__ == "__main__":
    print("🧪 Gemini AI 성능 테스트 시작...")
    
    # 환경 변수 확인
    import os
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다!")
        print("💡 .env 파일에 GEMINI_API_KEY=your_api_key_here 를 추가하세요.")
    else:
        print("✅ GEMINI_API_KEY 확인됨")
        
        # 테스트 실행 선택
        test_type = input("\n테스트 유형 선택 (1: 빠른 테스트, 2: 전체 테스트): ").strip()
        
        if test_type == "1":
            asyncio.run(run_quick_test())
        else:
            asyncio.run(run_full_test()) 