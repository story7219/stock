#!/usr/bin/env python3
"""
⚡ 실시간 처리 성능 테스트
- 비동기 병렬처리 성능 측정
- Gemini AI 응답 속도 테스트
- 한투 API 실시간 데이터 처리 속도
"""

import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import aiohttp
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class RealTimePerformanceTester:
    def __init__(self):
        # Gemini AI 설정
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 한투 API 설정
        self.kis_base_url = "https://openapi.koreainvestment.com:9443"
        self.kis_app_key = os.getenv('LIVE_KIS_APP_KEY')
        self.kis_app_secret = os.getenv('LIVE_KIS_APP_SECRET')
        self.kis_access_token = None
        
        # 성능 측정 결과
        self.performance_results = {
            "data_collection": [],
            "gemini_analysis": [],
            "parallel_processing": [],
            "total_cycle_time": []
        }
    
    async def test_kis_api_speed(self) -> Dict[str, float]:
        """한투 API 응답 속도 테스트"""
        print("🚀 한투 API 속도 테스트 중...")
        
        # 토큰 발급 시간 측정
        start_time = time.time()
        token_success = await self.get_kis_token()
        token_time = time.time() - start_time
        
        if not token_success:
            return {"token_time": token_time, "data_fetch_time": 0, "error": "토큰 발급 실패"}
        
        # 데이터 조회 시간 측정 (병렬)
        start_time = time.time()
        
        # 여러 종목 병렬 조회
        symbols = ["005930", "000660", "035420", "051910", "207940"]  # 코스피 5개
        tasks = []
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                task = self.fetch_stock_data(session, symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_fetch_time = time.time() - start_time
        
        successful_fetches = len([r for r in results if not isinstance(r, Exception)])
        
        return {
            "token_time": token_time,
            "data_fetch_time": data_fetch_time,
            "symbols_count": len(symbols),
            "successful_fetches": successful_fetches,
            "avg_time_per_symbol": data_fetch_time / len(symbols) if symbols else 0
        }
    
    async def get_kis_token(self) -> bool:
        """한투 API 토큰 발급"""
        try:
            url = f"{self.kis_base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.kis_app_key,
                "appsecret": self.kis_app_secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.kis_access_token = result.get('access_token')
                        return True
                    return False
        except:
            return False
    
    async def fetch_stock_data(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """개별 종목 데이터 조회"""
        try:
            url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            headers = {
                "authorization": f"Bearer {self.kis_access_token}",
                "appkey": self.kis_app_key,
                "appsecret": self.kis_app_secret,
                "tr_id": "FHKST01010100"
            }
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol
            }
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"symbol": symbol, "data": data, "success": True}
                else:
                    return {"symbol": symbol, "error": response.status, "success": False}
        except Exception as e:
            return {"symbol": symbol, "error": str(e), "success": False}
    
    async def test_gemini_speed(self, test_count: int = 5) -> Dict[str, float]:
        """Gemini AI 응답 속도 테스트"""
        print(f"🤖 Gemini AI 속도 테스트 ({test_count}회)...")
        
        test_prompt = """
다음 주식 데이터를 빠르게 분석해주세요:
- 삼성전자: 73,000원 (+1.2%)
- SK하이닉스: 125,000원 (+2.1%)
- NAVER: 195,000원 (-0.8%)

매수/매도/보유 중 하나만 답변해주세요.
"""
        
        response_times = []
        
        for i in range(test_count):
            start_time = time.time()
            try:
                response = self.gemini_model.generate_content(test_prompt)
                response_time = time.time() - start_time
                response_times.append(response_time)
                print(f"   테스트 {i+1}: {response_time:.2f}초")
            except Exception as e:
                print(f"   테스트 {i+1} 실패: {str(e)}")
                response_times.append(30.0)  # 실패시 30초로 처리
        
        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_tests": test_count,
            "success_rate": len([t for t in response_times if t < 20]) / test_count
        }
    
    async def test_parallel_processing(self) -> Dict[str, float]:
        """병렬 처리 성능 테스트"""
        print("⚡ 병렬 처리 성능 테스트...")
        
        # 시나리오 1: 순차 처리
        start_time = time.time()
        
        # 데이터 수집 (순차)
        kis_result = await self.test_kis_api_speed()
        
        # Gemini 분석 (순차)
        gemini_result = await self.test_gemini_speed(1)
        
        sequential_time = time.time() - start_time
        
        # 시나리오 2: 병렬 처리
        start_time = time.time()
        
        # 데이터 수집과 Gemini 분석을 병렬로
        tasks = [
            self.test_kis_api_speed(),
            self.test_gemini_speed(1)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup_ratio": speedup,
            "time_saved": sequential_time - parallel_time
        }
    
    async def test_trading_cycle_speed(self) -> Dict[str, Any]:
        """실제 매매 사이클 속도 테스트"""
        print("🔄 매매 사이클 속도 테스트...")
        
        cycle_times = []
        
        for cycle in range(3):  # 3번 테스트
            print(f"   사이클 {cycle + 1}/3...")
            cycle_start = time.time()
            
            # 1단계: 데이터 수집 (병렬)
            data_start = time.time()
            kis_task = self.test_kis_api_speed()
            yahoo_task = self.fetch_yahoo_data()
            
            data_results = await asyncio.gather(kis_task, yahoo_task, return_exceptions=True)
            data_time = time.time() - data_start
            
            # 2단계: Gemini AI 분석
            analysis_start = time.time()
            analysis_result = await self.test_gemini_speed(1)
            analysis_time = time.time() - analysis_start
            
            # 3단계: 매매 결정 (시뮬레이션)
            decision_start = time.time()
            await asyncio.sleep(0.1)  # 매매 로직 시뮬레이션
            decision_time = time.time() - decision_start
            
            total_cycle_time = time.time() - cycle_start
            cycle_times.append(total_cycle_time)
            
            print(f"      데이터 수집: {data_time:.2f}초")
            print(f"      AI 분석: {analysis_time:.2f}초")
            print(f"      매매 결정: {decision_time:.2f}초")
            print(f"      총 시간: {total_cycle_time:.2f}초")
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        max_trades_per_hour = 3600 / avg_cycle_time if avg_cycle_time > 0 else 0
        max_trades_per_day = max_trades_per_hour * 13  # 13시간 (한국+미국 시장)
        
        return {
            "avg_cycle_time": avg_cycle_time,
            "min_cycle_time": min(cycle_times),
            "max_cycle_time": max(cycle_times),
            "max_trades_per_hour": max_trades_per_hour,
            "max_trades_per_day": max_trades_per_day,
            "realistic_trades_per_day": min(max_trades_per_day, 200)  # API 제한 고려
        }
    
    async def fetch_yahoo_data(self) -> Dict:
        """Yahoo Finance 데이터 조회 (비교용)"""
        try:
            symbols = ["AAPL", "MSFT", "GOOGL"]
            tasks = []
            
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    task = session.get(url)
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                return {"success": True, "count": len(symbols)}
        except:
            return {"success": False, "count": 0}
    
    async def run_full_performance_test(self):
        """전체 성능 테스트 실행"""
        print("=" * 70)
        print("⚡ 실시간 처리 성능 종합 테스트")
        print("=" * 70)
        
        # 1. 한투 API 속도 테스트
        kis_results = await self.test_kis_api_speed()
        
        # 2. Gemini AI 속도 테스트
        gemini_results = await self.test_gemini_speed(3)
        
        # 3. 병렬 처리 테스트
        parallel_results = await self.test_parallel_processing()
        
        # 4. 매매 사이클 테스트
        cycle_results = await self.test_trading_cycle_speed()
        
        # 결과 출력
        print("\n" + "=" * 70)
        print("📊 테스트 결과")
        print("=" * 70)
        
        print(f"\n🏢 한투 API 성능:")
        print(f"   토큰 발급: {kis_results.get('token_time', 0):.2f}초")
        print(f"   데이터 조회: {kis_results.get('data_fetch_time', 0):.2f}초")
        print(f"   종목당 평균: {kis_results.get('avg_time_per_symbol', 0):.2f}초")
        
        print(f"\n🤖 Gemini AI 성능:")
        print(f"   평균 응답: {gemini_results.get('avg_response_time', 0):.2f}초")
        print(f"   최소 응답: {gemini_results.get('min_response_time', 0):.2f}초")
        print(f"   성공률: {gemini_results.get('success_rate', 0):.1%}")
        
        print(f"\n⚡ 병렬 처리 성능:")
        print(f"   순차 처리: {parallel_results.get('sequential_time', 0):.2f}초")
        print(f"   병렬 처리: {parallel_results.get('parallel_time', 0):.2f}초")
        print(f"   속도 향상: {parallel_results.get('speedup_ratio', 0):.1f}배")
        print(f"   시간 절약: {parallel_results.get('time_saved', 0):.2f}초")
        
        print(f"\n🔄 매매 사이클 성능:")
        print(f"   평균 사이클: {cycle_results.get('avg_cycle_time', 0):.2f}초")
        print(f"   시간당 최대: {cycle_results.get('max_trades_per_hour', 0):.0f}회")
        print(f"   일일 이론치: {cycle_results.get('max_trades_per_day', 0):.0f}회")
        print(f"   일일 현실치: {cycle_results.get('realistic_trades_per_day', 0):.0f}회")
        
        # 최종 결론
        print("\n" + "=" * 70)
        print("💡 성능 분석 결론")
        print("=" * 70)
        
        avg_cycle = cycle_results.get('avg_cycle_time', 25)
        realistic_daily = cycle_results.get('realistic_trades_per_day', 200)
        
        if avg_cycle < 30:
            print("✅ 고속 처리 가능: 사이클 시간 30초 미만")
        else:
            print("⚠️  중속 처리: 사이클 시간 30초 이상")
        
        if realistic_daily >= 100:
            print("✅ 대량 매매 가능: 일일 100회 이상")
        else:
            print("⚠️  제한적 매매: 일일 100회 미만")
        
        if parallel_results.get('speedup_ratio', 0) > 1.5:
            print("✅ 병렬 처리 효과: 1.5배 이상 속도 향상")
        else:
            print("⚠️  병렬 처리 제한적: 속도 향상 미미")
        
        print(f"\n🎯 권장 매매 빈도: 하루 {min(10, realistic_daily // 20)}회")
        print(f"📊 예상 처리 시간: {avg_cycle:.0f}초/회")

async def main():
    tester = RealTimePerformanceTester()
    await tester.run_full_performance_test()

if __name__ == "__main__":
    asyncio.run(main()) 