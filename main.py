#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Investing TOP5 - 메인 실행 파일
데이터 수집 → 정제 → AI 분석 → 전략 적용 → 추천 생성 완전한 파이프라인
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging.config

# 로컬 모듈 임포트
from utils.pipeline_manager import PipelineManager, PipelineResult
from data.data_loader import DataLoader
from configs.settings import SYSTEM_CONFIG, LOGGING_CONFIG
from recommenders.recommender import InvestmentRecommender
from ai_integration.gemini_client import GeminiClient

# 로깅 설정
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class InvestingTOP5:
    """🎯 Investing TOP5 메인 시스템"""
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.data_loader = DataLoader()
        self.recommender = InvestmentRecommender()
        
        # 시스템 정보
        self.version = "2.0.0"
        self.last_update = "2024-12-19"
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           🎯 Investing TOP5 v{self.version}                           ║
║                        AI 기반 투자 추천 시스템                                  ║
║                                                                                  ║
║  📊 데이터 수집 → 🧹 정제 → 🤖 AI 분석 → 📈 전략 적용 → 🎯 TOP5 추천           ║
║                                                                                  ║
║  지원 시장: 🇰🇷 한국 | 🇺🇸 미국                                                 ║
║  투자 전략: 워렌 버핏 | 피터 린치 | 조엘 그린블라트                              ║
║  AI 분석: Gemini Pro 통합                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def display_menu(self):
        """메인 메뉴 표시"""
        print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            📈 투자 대가 전략 메뉴                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1️⃣  윌리엄 오닐 (William O'Neil)                                               │
│  2️⃣  로버트 아놀드 (Robert Arnold)                                              │
│  3️⃣  리처드 데니스 (Richard Dennis)                                             │
│  4️⃣  조엘 그린블라트 (Joel Greenblatt)                                          │
│  5️⃣  제시 리버모어 (Jesse Livermore)                                            │
│  6️⃣  블랙록 기관 (BlackRock)                                                    │
│  7️⃣  업종순위                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
        """)
    
    async def run_korean_market_analysis(self):
        """한국 시장 전체 파이프라인 실행"""
        try:
            print("\n🇰🇷 한국 시장 TOP5 추천 분석을 시작합니다...")
            print("📊 데이터 수집 → 🧹 정제 → 🤖 AI 분석 → 📈 전략 적용 → 🎯 추천 생성")
            
            # AI 분석 필수로 설정
            custom_config = {'enable_ai_analysis': True}
            
            # 파이프라인 실행
            result = await self.pipeline_manager.run_full_pipeline(
                market='KR',
                symbols=None,
                custom_config=custom_config
            )
            
            # 결과 출력
            self._display_pipeline_result(result)
            
        except Exception as e:
            logger.error(f"한국 시장 분석 오류: {e}")
            print(f"❌ 분석 중 오류가 발생했습니다: {e}")
    
    async def run_us_market_analysis(self):
        """미국 시장 전체 파이프라인 실행"""
        try:
            print("\n🇺🇸 미국 시장 TOP5 추천 분석을 시작합니다...")
            print("📊 데이터 수집 → 🧹 정제 → 🤖 AI 분석 → 📈 전략 적용 → 🎯 추천 생성")
            
            # AI 분석 필수로 설정
            custom_config = {'enable_ai_analysis': True}
            
            # 파이프라인 실행
            result = await self.pipeline_manager.run_full_pipeline(
                market='US',
                symbols=None,
                custom_config=custom_config
            )
            
            # 결과 출력
            self._display_pipeline_result(result)
            
        except Exception as e:
            logger.error(f"미국 시장 분석 오류: {e}")
            print(f"❌ 분석 중 오류가 발생했습니다: {e}")
    
    async def run_individual_analysis(self):
        """개별 종목 빠른 분석"""
        try:
            print("\n🔍 개별 종목 분석")
            
            # 시장 선택
            print("시장을 선택하세요:")
            print("1. 한국 (KR)")
            print("2. 미국 (US)")
            
            market_choice = input("선택하세요 (1-2): ").strip()
            market = 'KR' if market_choice == '1' else 'US'
            
            # 종목 코드 입력
            if market == 'KR':
                symbol = input("한국 종목 코드를 입력하세요 (예: 005930): ").strip()
            else:
                symbol = input("미국 종목 코드를 입력하세요 (예: AAPL): ").strip().upper()
            
            if not symbol:
                print("❌ 종목 코드를 입력해주세요.")
                return
            
            print(f"\n🔍 {symbol} 종목 분석 중...")
            
            # 빠른 분석 실행
            result = await self.pipeline_manager.run_quick_analysis(symbol, market)
            
            if 'error' in result:
                print(f"❌ 분석 오류: {result['error']}")
                return
            
            # 결과 출력
            self._display_individual_analysis(result)
            
        except Exception as e:
            logger.error(f"개별 종목 분석 오류: {e}")
            print(f"❌ 분석 중 오류가 발생했습니다: {e}")
    
    async def run_sector_analysis(self):
        """섹터별 분석"""
        try:
            print("\n📊 섹터별 분석")
            
            # 시장 선택
            print("시장을 선택하세요:")
            print("1. 한국 (KR)")
            print("2. 미국 (US)")
            
            market_choice = input("선택하세요 (1-2): ").strip()
            market = 'KR' if market_choice == '1' else 'US'
            
            # 섹터 입력
            if market == 'KR':
                print("한국 주요 섹터: IT, 바이오, 자동차, 화학, 금융, 건설")
                sector = input("분석할 섹터를 입력하세요: ").strip()
            else:
                print("미국 주요 섹터: Technology, Healthcare, Finance, Consumer, Energy")
                sector = input("분석할 섹터를 입력하세요: ").strip()
            
            if not sector:
                print("❌ 섹터를 입력해주세요.")
                return
            
            print(f"\n📊 {market} 시장 {sector} 섹터 분석 중...")
            
            # 섹터 데이터 로드
            sector_data = await self.data_loader.load_sector_data(sector, market, limit=20)
            
            if not sector_data:
                print(f"❌ {sector} 섹터 데이터를 찾을 수 없습니다.")
                return
            
            # 섹터 분석 결과 출력
            self._display_sector_analysis(sector, sector_data)
                
        except Exception as e:
            logger.error(f"섹터 분석 오류: {e}")
            print(f"❌ 분석 중 오류가 발생했습니다: {e}")
    
    async def show_market_overview(self):
        """시장 개요 표시"""
        try:
            print("\n📈 시장 개요 분석")
            
            # 시장 선택
            print("시장을 선택하세요:")
            print("1. 한국 (KR)")
            print("2. 미국 (US)")
            print("3. 양쪽 모두")
            
            choice = input("선택하세요 (1-3): ").strip()
            
            if choice == "1":
                markets = ['KR']
            elif choice == "2":
                markets = ['US']
            else:
                markets = ['KR', 'US']
            
            for market in markets:
                print(f"\n🌐 {market} 시장 개요 분석 중...")
                overview = await self.data_loader.get_market_overview(market)
                self._display_market_overview(overview)
                
        except Exception as e:
            logger.error(f"시장 개요 분석 오류: {e}")
            print(f"❌ 분석 중 오류가 발생했습니다: {e}")
    
    def show_pipeline_status(self):
        """파이프라인 상태 표시"""
        try:
            print("\n⚙️ 파이프라인 상태 확인")
            
            status = self.pipeline_manager.get_pipeline_status()
            
            print("\n📊 시스템 구성 요소:")
            for component, status_text in status['components'].items():
                print(f"  • {component}: {status_text}")
            
            print("\n🔧 현재 설정:")
            config = status['config']
            print(f"  • AI 분석: {'활성화' if config['enable_ai_analysis'] else '비활성화'}")
            print(f"  • 캐시 사용: {'활성화' if config['enable_caching'] else '비활성화'}")
            print(f"  • 최소 데이터 품질: {config['min_data_quality']}")
            print(f"  • 최대 추천 수: {config['max_recommendations']}")
            
            print("\n📈 전략 가중치:")
            for strategy, weight in config['strategy_weights'].items():
                print(f"  • {strategy}: {weight:.1%}")
            
            print("\n📋 최근 실행 결과:")
            recent_results = status['last_results']
            if recent_results:
                for result in recent_results[:3]:
                    success_icon = "✅" if result['success'] else "❌"
                    print(f"  {success_icon} {result['timestamp'][:19]} | {result['market']} | "
                          f"{result['total_stocks']}종목 | 품질:{result['quality_score']:.1f}")
            else:
                print("  • 최근 실행 결과가 없습니다.")
            
        except Exception as e:
            logger.error(f"상태 확인 오류: {e}")
            print(f"❌ 상태 확인 중 오류가 발생했습니다: {e}")
    
    def show_system_settings(self):
        """시스템 설정 표시"""
        print("\n⚙️ 시스템 설정")
        print("현재 설정값들:")
        print(f"  • 버전: {self.version}")
        print(f"  • 마지막 업데이트: {self.last_update}")
        print(f"  • 로그 레벨: {LOGGING_CONFIG['level']}")
        print(f"  • 로그 파일: {LOGGING_CONFIG['file']}")
        
        print("\n🔧 설정 변경 옵션:")
        print("1. 캐시 삭제")
        print("2. 로그 파일 보기")
        print("3. 시스템 정보")
        print("0. 돌아가기")
        
        choice = input("\n선택하세요 (0-3): ").strip()
        
        if choice == "1":
            self._clear_cache()
        elif choice == "2":
            self._show_logs()
        elif choice == "3":
            self._show_system_info()
    
    def _clear_cache(self):
        """캐시 삭제"""
        try:
            self.data_loader.clear_cache()
            print("✅ 캐시가 삭제되었습니다.")
        except Exception as e:
            print(f"❌ 캐시 삭제 오류: {e}")
    
    def _show_logs(self):
        """로그 파일 표시"""
        try:
            log_file = LOGGING_CONFIG['file']
            if os.path.exists(log_file):
                print(f"\n📋 최근 로그 ({log_file}):")
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:  # 최근 20줄
                        print(f"  {line.strip()}")
            else:
                print("📋 로그 파일이 없습니다.")
        except Exception as e:
            print(f"❌ 로그 읽기 오류: {e}")
    
    def _show_system_info(self):
        """시스템 정보 표시"""
        print(f"\n🖥️ 시스템 정보:")
        print(f"  • Python 버전: {sys.version}")
        print(f"  • 작업 디렉토리: {os.getcwd()}")
        print(f"  • 시스템 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _display_pipeline_result(self, result: PipelineResult):
        """파이프라인 결과 출력"""
        if not result.success:
            print(f"\n❌ 파이프라인 실행 실패:")
            for error in result.errors:
                print(f"  • {error}")
            return
        
        print(f"\n🎉 {result.market} 시장 분석 완료!")
        print(f"📊 분석 통계:")
        print(f"  • 총 종목 수: {result.total_stocks}")
        print(f"  • 처리된 종목: {result.processed_stocks}")
        print(f"  • AI 분석: {'완료' if result.ai_analysis_completed else '건너뜀'}")
        print(f"  • 적용된 전략: {', '.join(result.strategies_applied)}")
        print(f"  • 실행 시간: {result.execution_time:.2f}초")
        print(f"  • 품질 점수: {result.quality_score:.1f}/100")
        
        print(f"\n🎯 TOP {len(result.top_recommendations)} 추천 종목:")
        print("┌─────┬──────────┬──────────────┬──────────┬──────────┬─────────────────────────────┐")
        print("│ 순위│ 종목코드 │    종목명    │   점수   │ 시가총액 │         추천 이유           │")
        print("├─────┼──────────┼──────────────┼──────────┼──────────┼─────────────────────────────┤")
        
        for i, rec in enumerate(result.top_recommendations, 1):
            symbol = rec['symbol'][:8]
            name = rec['name'][:12]
            score = f"{rec['final_score']:.1f}"
            market_cap = f"{rec['market_cap']/100000000:.0f}억" if rec['market_cap'] else "N/A"
            reason = rec.get('recommendation_reason', '')[:25]
            
            print(f"│ {i:^3} │ {symbol:^8} │ {name:^12} │ {score:^8} │ {market_cap:^8} │ {reason:^27} │")
        
        print("└─────┴──────────┴──────────────┴──────────┴──────────┴─────────────────────────────┘")
        
        # 자동으로 상세 분석 제공
            self._display_detailed_recommendations(result.top_recommendations)
    
    def _display_detailed_recommendations(self, recommendations: List[Dict[str, Any]]):
        """상세 추천 정보 출력"""
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{'='*60}")
            print(f"🏆 {i}위: {rec['name']} ({rec['symbol']})")
            print(f"{'='*60}")
            
            print(f"💰 기본 정보:")
            print(f"  • 현재가: {rec['price']:,}원")
            print(f"  • 시가총액: {rec['market_cap']/100000000:.0f}억원")
            print(f"  • 섹터: {rec.get('sector', '미분류')}")
            
            print(f"\n📊 점수 분석:")
            print(f"  • 최종 점수: {rec['final_score']:.1f}/100")
            print(f"  • 전략 점수: {rec['strategy_score']:.1f}/100")
            print(f"  • 데이터 품질: {rec['data_quality']:.1f}/100")
            
            if 'base_scores' in rec:
                base_scores = rec['base_scores']
                print(f"  • 가치 점수: {base_scores.get('value_score', 0):.1f}/100")
                print(f"  • 성장 점수: {base_scores.get('growth_score', 0):.1f}/100")
                print(f"  • 품질 점수: {base_scores.get('quality_score', 0):.1f}/100")
            
            if rec.get('ai_analysis'):
                print(f"\n🤖 AI 분석:")
                ai_text = rec['ai_analysis'][:200] + "..." if len(rec['ai_analysis']) > 200 else rec['ai_analysis']
                print(f"  {ai_text}")
    
    def _display_individual_analysis(self, result: Dict[str, Any]):
        """개별 종목 분석 결과 출력"""
        print(f"\n📊 {result['name']} ({result['symbol']}) 분석 결과")
        print("="*60)
        
        print(f"💰 기본 정보:")
        print(f"  • 현재가: {result['price']:,}원")
        print(f"  • 시가총액: {result['market_cap']/100000000:.0f}억원")
        print(f"  • 섹터: {result.get('sector', '미분류')}")
        print(f"  • 데이터 품질: {result['data_quality']:.1f}/100")
        
        print(f"\n📈 종합 점수:")
        base_scores = result['base_scores']
        print(f"  • 종합 점수: {base_scores['comprehensive_score']:.1f}/100")
        print(f"  • 가치 점수: {base_scores['value_score']:.1f}/100")
        print(f"  • 성장 점수: {base_scores['growth_score']:.1f}/100")
        print(f"  • 품질 점수: {base_scores['quality_score']:.1f}/100")
        print(f"  • 모멘텀 점수: {base_scores['momentum_score']:.1f}/100")
        
        print(f"\n🎯 전략별 점수:")
        for strategy, score in result['strategy_scores'].items():
            print(f"  • {strategy}: {score:.1f}/100")
        
        if result.get('ai_analysis'):
            print(f"\n🤖 AI 분석:")
            print(f"  {result['ai_analysis']}")
    
    def _display_sector_analysis(self, sector: str, sector_data: List):
        """섹터 분석 결과 출력"""
        print(f"\n📊 {sector} 섹터 분석 결과")
        print("="*60)
        
        if not sector_data:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        print(f"📈 섹터 통계:")
        print(f"  • 분석 종목 수: {len(sector_data)}")
        
        # 평균 지표 계산
        avg_quality = sum(stock.data_quality for stock in sector_data) / len(sector_data)
        avg_market_cap = sum(stock.market_cap for stock in sector_data if stock.market_cap) / len([s for s in sector_data if s.market_cap])
        
        print(f"  • 평균 데이터 품질: {avg_quality:.1f}/100")
        print(f"  • 평균 시가총액: {avg_market_cap/100000000:.0f}억원")
        
        print(f"\n🏆 상위 5개 종목:")
        top_stocks = sorted(sector_data, key=lambda x: x.market_cap or 0, reverse=True)[:5]
        
        for i, stock in enumerate(top_stocks, 1):
            print(f"  {i}. {stock.name} ({stock.symbol}) - {stock.market_cap/100000000:.0f}억원")
    
    def _display_market_overview(self, overview: Dict[str, Any]):
        """시장 개요 출력"""
        if 'error' in overview:
            print(f"❌ {overview['error']}")
            return
        
        print(f"\n🌐 {overview['시장']} 시장 개요")
        print("="*50)
        
        print(f"📊 기본 정보:")
        print(f"  • 분석 시간: {overview['분석_시간'][:19]}")
        print(f"  • 총 종목 수: {overview['총_종목수']}")
        
        stats = overview['데이터_통계']
        print(f"  • 평균 품질: {stats['평균_품질']}/100")
        print(f"  • 고품질 종목: {stats['고품질_종목']}개")
        
        print(f"\n🏆 시가총액 상위 종목:")
        for i, stock in enumerate(overview['상위_종목'][:5], 1):
            print(f"  {i}. {stock['종목명']} ({stock['종목코드']}) - {stock['시가총액']/100000000:.0f}억원")
    
    async def main_loop(self):
        """메인 실행 루프"""
            try:
                self.display_menu()
                choice = input("메뉴를 선택하세요: ").strip()
                
                if choice == '1':
                print("\n🎯 윌리엄 오닐 (William O'Neil) 전략 분석 중...")
                    await self.run_korean_market_analysis()
                elif choice == '2':
                print("\n🎯 로버트 아놀드 (Robert Arnold) 전략 분석 중...")
                    await self.run_us_market_analysis()
                elif choice == '3':
                print("\n🎯 리처드 데니스 (Richard Dennis) 전략 분석 중...")
                    await self.run_individual_analysis()
                elif choice == '4':
                print("\n🎯 조엘 그린블라트 (Joel Greenblatt) 전략 분석 중...")
                    await self.run_sector_analysis()
                elif choice == '5':
                print("\n🎯 제시 리버모어 (Jesse Livermore) 전략 분석 중...")
                    await self.show_market_overview()
                elif choice == '6':
                print("\n🎯 블랙록 기관 (BlackRock) 전략 분석 중...")
                    self.show_pipeline_status()
                elif choice == '7':
                print("\n📊 업종순위 분석 중...")
                    self.show_system_settings()
                else:
                    print("❌ 올바른 메뉴 번호를 선택해주세요.")
                
            print("\n✅ 분석이 완료되었습니다.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 사용자가 프로그램을 종료했습니다.")
        except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")

async def main():
    """메인 함수"""
    try:
        app = InvestingTOP5()
        await app.main_loop()
    except Exception as e:
        logger.error(f"메인 함수 오류: {e}")
        print(f"❌ 시스템 오류: {e}")

if __name__ == "__main__":
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        sys.exit(1) 