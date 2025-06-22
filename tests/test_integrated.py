#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 통합 주식 분석 시스템 테스트
한국/미국/글로벌 포트폴리오 전문 분석 테스트

통합된 기능:
- 전문 애널리스트 수준 분석
- 한국/미국 혼합 포트폴리오
- 다양한 투자 전략 테스트
- 성능 벤치마킹
"""

import asyncio
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from ultra_ai_analyzer import UltraAIAnalyzer

class IntegratedAnalysisTest:
    """통합 분석 테스트 클래스"""
    
    def __init__(self):
        self.analyzer = UltraAIAnalyzer()
        self.test_results = []
        self.start_time = time.time()
        
    async def run_all_tests(self):
        """모든 통합 테스트 실행"""
        print("🏛️ INTEGRATED ANALYSIS | 통합 분석 시스템 테스트 시작")
        print("=" * 80)
        
        # 1. 한국 주식 전략 테스트
        await self.test_korean_strategies()
        
        # 2. 미국 주식 전략 테스트  
        await self.test_us_strategies()
        
        # 3. 혼합 포트폴리오 테스트
        await self.test_mixed_portfolios()
        
        # 4. 섹터별 분석 테스트
        await self.test_sector_analysis()
        
        # 5. 성능 벤치마킹
        await self.test_performance_benchmark()
        
        # 6. 최종 결과 요약
        self.print_final_summary()
    
    async def test_korean_strategies(self):
        """한국 주식 전략 테스트"""
        print("\n🇰🇷 한국 주식 전략 테스트")
        print("-" * 60)
        
        # 한국 대표 주식들
        kr_stocks = ["005930", "000660", "035420", "051910", "068270"]  # 삼성전자, SK하이닉스, NAVER, LG화학, 셀트리온
        strategies = ["warren_buffett", "peter_lynch", "william_oneil"]
        
        for strategy in strategies:
            print(f"\n📊 {strategy} 전략 테스트 중...")
            start_time = time.time()
            
            try:
                # 한국 주식 데이터 준비
                stock_data_list = [{'stock_code': code, 'market': 'KR'} for code in kr_stocks]
                results = await self.analyzer.analyze_stocks(stock_data_list, strategy)
                
                analysis_time = time.time() - start_time
                
                if results:
                    avg_score = sum(r.get('점수', r.get('score', 0)) for r in results) / len(results)
                    print(f"   ✅ 성공: {len(results)}개 종목, 평균 점수: {avg_score:.1f}점, 시간: {analysis_time:.2f}초")
                    
                    self.test_results.append({
                        'test_name': f'Korean {strategy}',
                        'market': 'KR',
                        'stocks_count': len(kr_stocks),
                        'analysis_time': analysis_time,
                        'avg_score': avg_score,
                        'success': True
                    })
                else:
                    print(f"   ❌ 실패: 결과 없음")
                    self.test_results.append({
                        'test_name': f'Korean {strategy}',
                        'market': 'KR',
                        'success': False
                    })
                    
            except Exception as e:
                print(f"   ❌ 오류: {e}")
                self.test_results.append({
                    'test_name': f'Korean {strategy}',
                    'market': 'KR',
                    'success': False,
                    'error': str(e)
                })
    
    async def test_us_strategies(self):
        """미국 주식 전략 테스트"""
        print("\n🇺🇸 미국 주식 전략 테스트")
        print("-" * 60)
        
        # 미국 대표 주식들 (섹터별)
        stock_groups = {
            'Growth': ['NVDA', 'TSLA', 'AMZN', 'GOOGL', 'MSFT'],
            'Value': ['BRK-B', 'KO', 'JNJ', 'PG', 'WMT'],
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX']
        }
        
        strategies = ["william_oneil", "peter_lynch", "warren_buffett"]
        
        for group_name, stocks in stock_groups.items():
            print(f"\n📈 {group_name} 그룹 테스트:")
            
            for strategy in strategies:
                start_time = time.time()
                
                try:
                    results = await self.analyzer.analyze_us_stocks(stocks, strategy)
                    analysis_time = time.time() - start_time
                    
                    if results:
                        avg_score = sum(r.get('점수', r.get('score', 0)) for r in results) / len(results)
                        print(f"   • {strategy}: {len(results)}개 종목, 평균 {avg_score:.1f}점, {analysis_time:.2f}초")
                        
                        self.test_results.append({
                            'test_name': f'US {group_name} {strategy}',
                            'market': 'US',
                            'group': group_name,
                            'stocks_count': len(stocks),
                            'analysis_time': analysis_time,
                            'avg_score': avg_score,
                            'success': True
                        })
                    else:
                        print(f"   • {strategy}: 실패")
                        
                except Exception as e:
                    print(f"   • {strategy}: 오류 - {e}")
    
    async def test_mixed_portfolios(self):
        """혼합 포트폴리오 테스트"""
        print("\n🌍 글로벌 혼합 포트폴리오 테스트")
        print("-" * 60)
        
        # 다양한 포트폴리오 조합
        portfolio_configs = [
            {
                'name': '대형주 포트폴리오',
                'kr_stocks': ["005930", "000660", "035420"],  # 삼성전자, SK하이닉스, NAVER
                'us_stocks': ["AAPL", "MSFT", "GOOGL"]        # 애플, 마이크로소프트, 구글
            },
            {
                'name': '성장주 포트폴리오', 
                'kr_stocks': ["035420", "068270", "207940"],  # NAVER, 셀트리온, 삼성바이오로직스
                'us_stocks': ["NVDA", "TSLA", "AMZN"]         # 엔비디아, 테슬라, 아마존
            },
            {
                'name': '가치주 포트폴리오',
                'kr_stocks': ["005930", "051910", "028260"],  # 삼성전자, LG화학, 삼성물산
                'us_stocks': ["BRK-B", "KO", "JNJ"]           # 버크셔, 코카콜라, J&J
            }
        ]
        
        strategies = ["warren_buffett", "peter_lynch"]
        
        for config in portfolio_configs:
            print(f"\n💼 {config['name']} 테스트:")
            
            for strategy in strategies:
                start_time = time.time()
                
                try:
                    results = await self.analyzer.analyze_mixed_portfolio(
                        kr_codes=config['kr_stocks'],
                        us_symbols=config['us_stocks'],
                        strategy=strategy
                    )
                    
                    analysis_time = time.time() - start_time
                    
                    if results and "summary" in results:
                        summary = results["summary"]
                        total_analyzed = summary.get('total_analyzed', 0)
                        avg_score = summary.get('average_score', 0)
                        
                        print(f"   • {strategy}: {total_analyzed}개 종목, 평균 {avg_score:.1f}점, {analysis_time:.2f}초")
                        
                        # TOP 종목 출력
                        top_picks = summary.get('top_5_picks', [])[:3]
                        for i, pick in enumerate(top_picks, 1):
                            flag = "🇰🇷" if pick['country'] == 'KR' else "🇺🇸"
                            print(f"     {i}. {flag} {pick['symbol']} - {pick['score']}점")
                        
                        self.test_results.append({
                            'test_name': f'Mixed {config["name"]} {strategy}',
                            'market': 'MIXED',
                            'portfolio': config['name'],
                            'total_analyzed': total_analyzed,
                            'analysis_time': analysis_time,
                            'avg_score': avg_score,
                            'success': True
                        })
                    else:
                        print(f"   • {strategy}: 실패")
                        
                except Exception as e:
                    print(f"   • {strategy}: 오류 - {e}")
    
    async def test_sector_analysis(self):
        """섹터별 분석 테스트"""
        print("\n🏭 섹터별 분석 테스트")
        print("-" * 60)
        
        # 미국 섹터별 대표 종목
        us_sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL'],
            'Healthcare': ['JNJ', 'PFE', 'UNH'],
            'Financial': ['JPM', 'BAC', 'WFC'],
            'Consumer': ['AMZN', 'TSLA', 'NKE']
        }
        
        # 한국 섹터별 대표 종목
        kr_sectors = {
            'Technology': ['005930', '000660', '035420'],  # 삼성전자, SK하이닉스, NAVER
            'Chemical': ['051910', '009830', '011170'],    # LG화학, 한화솔루션, 롯데케미칼
            'Bio': ['068270', '207940', '326030']          # 셀트리온, 삼성바이오로직스, 바이오니아
        }
        
        print("🇺🇸 미국 섹터 분석:")
        for sector, stocks in us_sectors.items():
            start_time = time.time()
            try:
                results = await self.analyzer.analyze_us_stocks(stocks, "comprehensive")
                analysis_time = time.time() - start_time
                
                if results:
                    avg_score = sum(r.get('점수', r.get('score', 0)) for r in results) / len(results)
                    print(f"   • {sector}: 평균 {avg_score:.1f}점, {analysis_time:.2f}초")
                    
            except Exception as e:
                print(f"   • {sector}: 오류 - {e}")
        
        print("\n🇰🇷 한국 섹터 분석:")
        for sector, stocks in kr_sectors.items():
            start_time = time.time()
            try:
                stock_data_list = [{'stock_code': code, 'market': 'KR'} for code in stocks]
                results = await self.analyzer.analyze_stocks(stock_data_list, "comprehensive")
                analysis_time = time.time() - start_time
                
                if results:
                    avg_score = sum(r.get('점수', r.get('score', 0)) for r in results) / len(results)
                    print(f"   • {sector}: 평균 {avg_score:.1f}점, {analysis_time:.2f}초")
                    
            except Exception as e:
                print(f"   • {sector}: 오류 - {e}")
    
    async def test_performance_benchmark(self):
        """성능 벤치마킹 테스트"""
        print("\n⚡ 성능 벤치마킹 테스트")
        print("-" * 60)
        
        # 대량 분석 테스트
        test_cases = [
            {'name': '소규모 (5종목)', 'count': 5},
            {'name': '중규모 (10종목)', 'count': 10},
            {'name': '대규모 (20종목)', 'count': 20}
        ]
        
        us_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "NFLX", "CRM", "ADBE",
                    "PYPL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC"]
        
        for test_case in test_cases:
            count = test_case['count']
            test_stocks = us_stocks[:count]
            
            print(f"\n📊 {test_case['name']} 테스트:")
            
            start_time = time.time()
            try:
                results = await self.analyzer.analyze_us_stocks(test_stocks, "warren_buffett")
                analysis_time = time.time() - start_time
                
                if results:
                    throughput = len(results) / analysis_time
                    avg_score = sum(r.get('점수', r.get('score', 0)) for r in results) / len(results)
                    
                    print(f"   • 분석 시간: {analysis_time:.2f}초")
                    print(f"   • 처리량: {throughput:.2f} 종목/초")
                    print(f"   • 평균 점수: {avg_score:.1f}점")
                    print(f"   • 성공률: {len(results)}/{count} ({len(results)/count*100:.1f}%)")
                    
                    self.test_results.append({
                        'test_name': f'Performance {test_case["name"]}',
                        'stocks_count': count,
                        'analysis_time': analysis_time,
                        'throughput': throughput,
                        'avg_score': avg_score,
                        'success_rate': len(results)/count,
                        'success': True
                    })
                    
            except Exception as e:
                print(f"   • 오류: {e}")
    
    def print_final_summary(self):
        """최종 결과 요약"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("📊 통합 테스트 최종 결과 요약")
        print("="*80)
        
        # 전체 통계
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get('success', False))
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"🎯 전체 테스트 결과:")
        print(f"   • 총 테스트: {total_tests}개")
        print(f"   • 성공: {successful_tests}개")
        print(f"   • 실패: {total_tests - successful_tests}개")
        print(f"   • 성공률: {success_rate:.1f}%")
        print(f"   • 총 소요시간: {total_time:.2f}초")
        
        # 시장별 통계
        markets = {}
        for result in self.test_results:
            if result.get('success'):
                market = result.get('market', 'Unknown')
                if market not in markets:
                    markets[market] = {'count': 0, 'avg_score': 0, 'total_time': 0}
                
                markets[market]['count'] += 1
                markets[market]['avg_score'] += result.get('avg_score', 0)
                markets[market]['total_time'] += result.get('analysis_time', 0)
        
        if markets:
            print(f"\n📈 시장별 성과:")
            for market, stats in markets.items():
                avg_score = stats['avg_score'] / stats['count'] if stats['count'] > 0 else 0
                avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                print(f"   • {market}: 평균 {avg_score:.1f}점, 평균 {avg_time:.2f}초/테스트")
        
        # 성능 통계
        performance_tests = [r for r in self.test_results if 'Performance' in r.get('test_name', '')]
        if performance_tests:
            print(f"\n⚡ 성능 벤치마크:")
            for perf in performance_tests:
                if perf.get('success'):
                    print(f"   • {perf['test_name']}: {perf.get('throughput', 0):.2f} 종목/초")
        
        # AI 분석기 통계
        ai_stats = self.analyzer.get_performance_stats()
        if ai_stats and '🚀 울트라 성능' in ai_stats:
            perf = ai_stats['🚀 울트라 성능']
            print(f"\n🤖 AI 분석기 성능:")
            print(f"   • 평균 응답시간: {perf.get('평균응답', 'N/A')}")
            print(f"   • 성공률: {perf.get('성공률', 'N/A')}")
            print(f"   • 캐시 적중률: {perf.get('캐시적중', 'N/A')}")
            print(f"   • 예상 비용: {perf.get('예상비용', 'N/A')}")
        
        # 결과 저장
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'detailed_results': self.test_results,
            'ai_stats': ai_stats
        }
        
        filename = f"logs/integrated_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 상세 테스트 리포트가 저장되었습니다: {filename}")
        print("="*80)

async def quick_test():
    """빠른 테스트 (주요 기능만)"""
    print("🚀 빠른 통합 테스트 시작")
    
    analyzer = UltraAIAnalyzer()
    
    # 1. 한국 주식 테스트
    print("\n🇰🇷 한국 주식 테스트...")
    kr_results = await analyzer.analyze_stocks(
        [{'stock_code': '005930', 'market': 'KR'}], 
        'warren_buffett'
    )
    print(f"   결과: {'성공' if kr_results else '실패'}")
    
    # 2. 미국 주식 테스트
    print("\n🇺🇸 미국 주식 테스트...")
    us_results = await analyzer.analyze_us_stocks(['AAPL'], 'warren_buffett')
    print(f"   결과: {'성공' if us_results else '실패'}")
    
    # 3. 혼합 포트폴리오 테스트
    print("\n🌍 혼합 포트폴리오 테스트...")
    mixed_results = await analyzer.analyze_mixed_portfolio(
        kr_codes=['005930'],
        us_symbols=['AAPL'],
        strategy='warren_buffett'
    )
    print(f"   결과: {'성공' if mixed_results else '실패'}")
    
    print("\n✅ 빠른 테스트 완료!")

async def main():
    """메인 실행 함수"""
    import os
    os.makedirs('logs', exist_ok=True)
    
    print("🔧 통합 테스트 모드를 선택하세요:")
    print("1. 전체 테스트 (완전한 분석, 시간 소요)")
    print("2. 빠른 테스트 (기본 기능만)")
    
    choice = input("선택 (1-2): ").strip()
    
    if choice == '1':
        test_runner = IntegratedAnalysisTest()
        await test_runner.run_all_tests()
    elif choice == '2':
        await quick_test()
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    asyncio.run(main()) 