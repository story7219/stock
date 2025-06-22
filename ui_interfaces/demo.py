#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Investing TOP5 데모 스크립트
완전한 데이터 파이프라인 테스트: 수집 → 정제 → AI 분석 → 전략 적용 → 추천
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 로컬 모듈 임포트
try:
    from utils.pipeline_manager import PipelineManager, run_korean_market_analysis
    from data.data_loader import DataLoader, load_korean_stocks
    from ai_integration.gemini_client import GeminiClient
    from strategies.buffett import BuffettStrategy
    from strategies.lynch import LynchStrategy
    from strategies.greenblatt import GreenblattStrategy
    from recommenders.recommender import InvestmentRecommender
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    print("필요한 모듈들이 설치되어 있는지 확인해주세요.")
    sys.exit(1)

class PipelineDemo:
    """파이프라인 데모 클래스"""
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.data_loader = DataLoader()
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    🚀 Investing TOP5 파이프라인 데모                             ║
║                                                                                  ║
║  📊 데이터 수집 → 🧹 정제 → 🤖 AI 분석 → 📈 전략 적용 → 🎯 TOP5 추천           ║
║                                                                                  ║
║  이 데모는 전체 투자 추천 파이프라인의 동작을 보여줍니다.                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    async def demo_data_collection(self):
        """1단계: 데이터 수집 데모"""
        print("\n" + "="*60)
        print("📊 1단계: 데이터 수집 데모")
        print("="*60)
        
        try:
            print("🔍 한국 주요 종목 데이터 수집 중...")
            
            # 테스트용 종목 리스트 (실제 존재하는 종목들)
            test_symbols = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
            
            print(f"📋 테스트 종목: {test_symbols}")
            
            # 데이터 로드 시뮬레이션
            print("⏳ 외부 API에서 데이터 수집 중...")
            await asyncio.sleep(1)  # 실제 API 호출 시뮬레이션
            
            print("✅ 데이터 수집 완료!")
            print(f"  • 요청 종목: {len(test_symbols)}개")
            print(f"  • 수집 성공: {len(test_symbols)}개")
            print(f"  • 수집 실패: 0개")
            
            # 수집된 데이터 예시
            print("\n📈 수집된 데이터 예시:")
            sample_data = {
                "005930": {"name": "삼성전자", "price": 75000, "market_cap": 4500000000000},
                "000660": {"name": "SK하이닉스", "price": 125000, "market_cap": 900000000000},
                "035420": {"name": "NAVER", "price": 180000, "market_cap": 300000000000}
            }
            
            for symbol, data in sample_data.items():
                print(f"  • {symbol}: {data['name']} - {data['price']:,}원 (시총: {data['market_cap']/1000000000000:.1f}조)")
            
        except Exception as e:
            print(f"❌ 데이터 수집 오류: {e}")
    
    async def demo_data_cleaning(self):
        """2단계: 데이터 정제 데모"""
        print("\n" + "="*60)
        print("🧹 2단계: 데이터 정제 데모")
        print("="*60)
        
        try:
            print("🔄 원시 데이터 정제 중...")
            
            # 정제 과정 시뮬레이션
            steps = [
                "결측값 처리",
                "이상치 제거",
                "데이터 타입 변환",
                "재무 지표 계산",
                "품질 점수 산출"
            ]
            
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}...")
                await asyncio.sleep(0.5)
            
            print("\n✅ 데이터 정제 완료!")
            print("📊 정제 결과:")
            print("  • 원시 데이터: 3개 종목")
            print("  • 정제 완료: 3개 종목")
            print("  • 품질 점수: 85.2/100")
            print("  • 고품질 데이터: 3개 (100%)")
            
            # 정제된 데이터 예시
            print("\n📈 정제된 데이터 예시:")
            cleaned_data = [
                {"symbol": "005930", "name": "삼성전자", "quality": 92.5, "pe_ratio": 12.3, "roe": 15.2},
                {"symbol": "000660", "name": "SK하이닉스", "quality": 88.7, "pe_ratio": 8.9, "roe": 18.4},
                {"symbol": "035420", "name": "NAVER", "quality": 84.3, "pe_ratio": 15.6, "roe": 12.8}
            ]
            
            for data in cleaned_data:
                print(f"  • {data['symbol']}: 품질={data['quality']:.1f} PER={data['pe_ratio']} ROE={data['roe']}%")
            
        except Exception as e:
            print(f"❌ 데이터 정제 오류: {e}")
    
    async def demo_ai_analysis(self):
        """3단계: AI 분석 데모"""
        print("\n" + "="*60)
        print("🤖 3단계: AI 분석 데모")
        print("="*60)
        
        try:
            print("🧠 Gemini AI 분석 요청 중...")
            
            # AI 분석 시뮬레이션
            stocks = ["삼성전자", "SK하이닉스", "NAVER"]
            
            for stock in stocks:
                print(f"  📊 {stock} AI 분석 중...")
                await asyncio.sleep(1)  # AI 분석 시뮬레이션
                print(f"  ✅ {stock} 분석 완료")
            
            print("\n🎯 AI 분석 결과:")
            ai_results = [
                {
                    "stock": "삼성전자",
                    "sentiment": "긍정적",
                    "score": 8.5,
                    "summary": "반도체 업사이클과 AI 수혜 기대"
                },
                {
                    "stock": "SK하이닉스",
                    "sentiment": "매우 긍정적",
                    "score": 9.2,
                    "summary": "메모리 반도체 시장 회복 및 HBM 수혜"
                },
                {
                    "stock": "NAVER",
                    "sentiment": "중립적",
                    "score": 7.1,
                    "summary": "플랫폼 성장 둔화, AI 투자 확대 필요"
                }
            ]
            
            for result in ai_results:
                print(f"  • {result['stock']}: {result['sentiment']} ({result['score']}/10)")
                print(f"    → {result['summary']}")
            
            print(f"\n✅ AI 분석 완료: {len(ai_results)}개 종목")
            
        except Exception as e:
            print(f"❌ AI 분석 오류: {e}")
    
    async def demo_strategy_application(self):
        """4단계: 투자 전략 적용 데모"""
        print("\n" + "="*60)
        print("📈 4단계: 투자 전략 적용 데모")
        print("="*60)
        
        try:
            strategies = ["워렌 버핏 전략", "피터 린치 전략", "조엘 그린블라트 전략"]
            stocks = ["삼성전자", "SK하이닉스", "NAVER"]
            
            print("🎯 투자 전략별 분석 중...")
            
            # 전략별 분석 결과
            strategy_results = {
                "워렌 버핏 전략": {
                    "삼성전자": 78.5,
                    "SK하이닉스": 65.2,
                    "NAVER": 72.8
                },
                "피터 린치 전략": {
                    "삼성전자": 82.1,
                    "SK하이닉스": 89.3,
                    "NAVER": 68.7
                },
                "조엘 그린블라트 전략": {
                    "삼성전자": 75.9,
                    "SK하이닉스": 81.6,
                    "NAVER": 70.4
                }
            }
            
            for strategy in strategies:
                print(f"\n📊 {strategy} 결과:")
                for stock in stocks:
                    score = strategy_results[strategy][stock]
                    print(f"  • {stock}: {score:.1f}/100")
                await asyncio.sleep(0.5)
            
            print("\n🏆 전략별 최고 점수:")
            for strategy in strategies:
                best_stock = max(strategy_results[strategy], key=strategy_results[strategy].get)
                best_score = strategy_results[strategy][best_stock]
                print(f"  • {strategy}: {best_stock} ({best_score:.1f}점)")
            
            print("\n✅ 투자 전략 적용 완료!")
            
        except Exception as e:
            print(f"❌ 전략 적용 오류: {e}")
    
    async def demo_final_recommendation(self):
        """5단계: 최종 추천 생성 데모"""
        print("\n" + "="*60)
        print("🎯 5단계: 최종 추천 생성 데모")
        print("="*60)
        
        try:
            print("🔄 종합 점수 계산 중...")
            
            # 종합 점수 계산 시뮬레이션
            final_scores = [
                {"stock": "SK하이닉스", "score": 86.7, "rank": 1},
                {"stock": "삼성전자", "score": 78.8, "rank": 2},
                {"stock": "NAVER", "score": 70.6, "rank": 3}
            ]
            
            await asyncio.sleep(1)
            
            print("\n🏆 최종 추천 결과:")
            print("┌─────┬──────────────┬──────────┬─────────────────────────────┐")
            print("│ 순위│   종목명     │ 종합점수 │         추천 이유           │")
            print("├─────┼──────────────┼──────────┼─────────────────────────────┤")
            
            reasons = [
                "메모리 반도체 회복 + AI 수혜",
                "반도체 업사이클 + 안정적 배당",
                "플랫폼 안정성 + AI 투자 확대"
            ]
            
            for i, (result, reason) in enumerate(zip(final_scores, reasons)):
                print(f"│ {result['rank']:^3} │ {result['stock']:^12} │ {result['score']:^8.1f} │ {reason:^27} │")
            
            print("└─────┴──────────────┴──────────┴─────────────────────────────┘")
            
            print(f"\n✅ TOP3 추천 완료!")
            print("📊 추천 품질:")
            print(f"  • 평균 점수: {sum(s['score'] for s in final_scores)/len(final_scores):.1f}/100")
            print(f"  • 데이터 신뢰도: 92.3%")
            print(f"  • AI 분석 활용: 100%")
            
        except Exception as e:
            print(f"❌ 최종 추천 오류: {e}")
    
    async def demo_pipeline_summary(self):
        """파이프라인 요약"""
        print("\n" + "="*60)
        print("📋 파이프라인 실행 요약")
        print("="*60)
        
        execution_time = 8.5  # 시뮬레이션 시간
        
        print("🎉 전체 파이프라인 실행 완료!")
        print(f"⏱️  총 실행 시간: {execution_time:.1f}초")
        print(f"📊 처리 단계: 5단계")
        print(f"🔍 분석 종목: 3개")
        print(f"🤖 AI 분석: 활성화")
        print(f"📈 적용 전략: 3개")
        print(f"🎯 최종 추천: 3개")
        
        print("\n🔄 파이프라인 흐름:")
        flow_steps = [
            "📊 데이터 수집 (외부 API)",
            "🧹 데이터 정제 (품질 향상)",
            "🤖 AI 분석 (Gemini Pro)",
            "📈 전략 적용 (3가지 전략)",
            "🎯 최종 추천 (TOP3 선정)"
        ]
        
        for i, step in enumerate(flow_steps, 1):
            print(f"  {i}. {step} ✅")
        
        print("\n💡 주요 특징:")
        features = [
            "비동기 처리로 빠른 실행",
            "다중 데이터 소스 활용",
            "AI 기반 지능형 분석",
            "투자 대가 전략 통합",
            "실시간 품질 모니터링"
        ]
        
        for feature in features:
            print(f"  • {feature}")
    
    async def run_full_demo(self):
        """전체 데모 실행"""
        try:
            start_time = datetime.now()
            
            # 각 단계별 데모 실행
            await self.demo_data_collection()
            await self.demo_data_cleaning()
            await self.demo_ai_analysis()
            await self.demo_strategy_application()
            await self.demo_final_recommendation()
            await self.demo_pipeline_summary()
            
            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n🎊 데모 완료! 실제 실행 시간: {execution_time:.1f}초")
            
        except Exception as e:
            logger.error(f"데모 실행 오류: {e}")
            print(f"❌ 데모 실행 중 오류가 발생했습니다: {e}")
    
    async def run_real_pipeline_test(self):
        """실제 파이프라인 테스트 (간단한 버전)"""
        print("\n" + "="*60)
        print("🚀 실제 파이프라인 테스트")
        print("="*60)
        
        try:
            print("⚠️  주의: 실제 API 호출이 발생할 수 있습니다.")
            choice = input("계속 진행하시겠습니까? (y/n): ").strip().lower()
            
            if choice != 'y':
                print("❌ 테스트를 취소했습니다.")
                return
            
            print("\n🔄 실제 파이프라인 실행 중...")
            
            # 실제 파이프라인 실행 (AI 분석 비활성화)
            custom_config = {
                'enable_ai_analysis': False,  # 데모를 위해 AI 분석 비활성화
                'max_recommendations': 3
            }
            
            # 테스트 종목 (실제 존재하는 종목)
            test_symbols = ['005930', '000660']  # 삼성전자, SK하이닉스
            
            result = await self.pipeline_manager.run_full_pipeline(
                market='KR',
                symbols=test_symbols,
                custom_config=custom_config
            )
            
            # 결과 출력
            if result.success:
                print(f"\n✅ 실제 파이프라인 테스트 성공!")
                print(f"  • 처리 종목: {result.processed_stocks}개")
                print(f"  • 실행 시간: {result.execution_time:.2f}초")
                print(f"  • 품질 점수: {result.quality_score:.1f}/100")
                print(f"  • 추천 개수: {len(result.top_recommendations)}개")
            else:
                print(f"❌ 실제 파이프라인 테스트 실패:")
                for error in result.errors:
                    print(f"  • {error}")
            
        except Exception as e:
            logger.error(f"실제 파이프라인 테스트 오류: {e}")
            print(f"❌ 테스트 중 오류가 발생했습니다: {e}")

async def main():
    """메인 함수"""
    demo = PipelineDemo()
    
    print("데모 옵션을 선택하세요:")
    print("1. 시뮬레이션 데모 (빠른 실행)")
    print("2. 실제 파이프라인 테스트 (실제 API 호출)")
    print("3. 둘 다 실행")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == "1":
        await demo.run_full_demo()
    elif choice == "2":
        await demo.run_real_pipeline_test()
    elif choice == "3":
        await demo.run_full_demo()
        print("\n" + "="*60)
        await demo.run_real_pipeline_test()
    else:
        print("❌ 올바른 옵션을 선택해주세요.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 데모를 종료합니다.")
    except Exception as e:
        print(f"❌ 데모 실행 오류: {e}")
        sys.exit(1) 