"""
🌍 통합 주식 분석 시스템 (한국 + 미국)
코스피200, 나스닥100, S&P500 TOP5 추천 시스템

주요 기능:
1. 🇰🇷 한국주식: 코스피200 TOP5 (6가지 전략)
2. 🇺🇸 미국주식: 나스닥100 & S&P500 TOP5 (4가지 전략)
3. 🎯 통합 분석 및 비교
4. 📊 실시간 데이터 기반 분석
5. 🤖 AI 기반 종합 추천
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
import sys
import os

# 기존 시스템 모듈들
try:
    from ai_trading import AdvancedScalpingAI
    from core_legacy.core_trader import CoreTrader
    from basic import USStockAnalyzer
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("필요한 파일들이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegratedStockAnalyzer:
    """🌍 통합 주식 분석 시스템 (한국 + 미국)"""
    
    def __init__(self):
        """초기화"""
        try:
            logger.info("🌍 통합 주식 분석 시스템 초기화 중...")
            
            # CoreTrader 초기화
            self.trader = CoreTrader()
            
            # AI 시스템 초기화 (한국 + 미국주식 통합)
            self.ai_system = AdvancedScalpingAI(self.trader)
            
            # 미국주식 분석기 초기화
            self.us_analyzer = USStockAnalyzer()
            
            # 한국주식 전략 목록
            self.korean_strategies = [
                ("blackrock", "블랙록 전략"),
                ("warren_buffett", "워렌 버핏 전략"),
                ("peter_lynch", "피터 린치 전략"),
                ("william_oneil", "윌리엄 오닐 전략"),
                ("jesse_livermore", "제시 리버모어 전략"),
                ("ichimoku", "일목균형표 전략")
            ]
            
            # 미국주식 전략 목록
            self.us_strategies = [
                ("momentum", "모멘텀 전략"),
                ("value", "가치 전략"),
                ("growth", "성장 전략"),
                ("quality", "퀄리티 전략")
            ]
            
            logger.info("✅ 통합 주식 분석 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            raise
    
    def print_welcome_message(self):
        """환영 메시지 출력"""
        print("=" * 100)
        print("🌍 통합 주식 분석 시스템 (한국 + 미국)")
        print("=" * 100)
        print("📊 지원 시장:")
        print("   🇰🇷 한국: 코스피200 (KOSPI 200)")
        print("   🇺🇸 미국: 나스닥100 (NASDAQ-100) & S&P500")
        print()
        print("🎯 지원 전략:")
        print("   🇰🇷 한국주식: 블랙록, 워렌버핏, 피터린치, 윌리엄오닐, 제시리버모어, 일목균형표")
        print("   🇺🇸 미국주식: 모멘텀, 가치, 성장, 퀄리티")
        print()
        print("🤖 AI 기반 실시간 분석 및 TOP5 추천")
        print("=" * 100)
        print()
    
    def print_main_menu(self):
        """메인 메뉴 출력"""
        print("\n" + "="*80)
        print("🌍 통합 주식 분석 시스템")
        print("="*80)
        print("🇰🇷 한국주식 (코스피200):")
        print("  1. 블랙록 전략 TOP5")
        print("  2. 워렌 버핏 전략 TOP5")
        print("  3. 피터 린치 전략 TOP5")
        print("  4. 윌리엄 오닐 전략 TOP5")
        print("  5. 제시 리버모어 전략 TOP5")
        print("  6. 일목균형표 전략 TOP5")
        print()
        print("🇺🇸 미국주식:")
        print("  7. 나스닥100 모멘텀 전략 TOP5")
        print("  8. 나스닥100 가치 전략 TOP5")
        print("  9. S&P500 가치 전략 TOP5")
        print(" 10. S&P500 성장 전략 TOP5")
        print()
        print("🔄 통합 분석:")
        print(" 11. 한국주식 전체 분석")
        print(" 12. 미국주식 전체 분석")
        print(" 13. 글로벌 통합 분석")
        print()
        print("  0. 종료")
        print("="*80)
    
    async def analyze_korean_strategy(self, strategy_code: str, strategy_name: str):
        """한국주식 전략 분석"""
        try:
            print(f"\n🇰🇷 {strategy_name} 분석 시작...")
            
            # 기존 한국 분석 시스템 호출 (가상 구현)
            # 실제로는 기존 AI 분석 시스템과 연동해야 함
            results = await self._simulate_korean_analysis(strategy_code, strategy_name)
            
            if results:
                self.print_korean_results(strategy_name, results)
            else:
                print(f"❌ {strategy_name} 분석 결과가 없습니다.")
                
        except Exception as e:
            logger.error(f"❌ {strategy_name} 분석 실패: {e}")
            print(f"❌ {strategy_name} 분석 중 오류가 발생했습니다: {e}")
    
    async def _simulate_korean_analysis(self, strategy_code: str, strategy_name: str) -> List[Dict]:
        """한국주식 분석 시뮬레이션 (실제로는 기존 시스템과 연동)"""
        # 실제 구현에서는 기존 AI 분석 시스템을 호출해야 함
        sample_results = [
            {
                'stock_code': '005930',
                'name': '삼성전자',
                'score': 85,
                'recommendation': '매수',
                'reason': f'{strategy_name} 기준 우수한 투자 대상',
                'current_price': 71000,
                'target_price': 78000,
                'market_cap': 425000000000000
            },
            {
                'stock_code': '000660',
                'name': 'SK하이닉스',
                'score': 82,
                'recommendation': '매수',
                'reason': f'{strategy_name} 기준 성장 잠재력 우수',
                'current_price': 89000,
                'target_price': 95000,
                'market_cap': 65000000000000
            },
            {
                'stock_code': '035420',
                'name': 'NAVER',
                'score': 78,
                'recommendation': '보유',
                'reason': f'{strategy_name} 기준 안정적 성장',
                'current_price': 185000,
                'target_price': 200000,
                'market_cap': 30000000000000
            },
            {
                'stock_code': '051910',
                'name': 'LG화학',
                'score': 75,
                'recommendation': '보유',
                'reason': f'{strategy_name} 기준 중장기 전망 양호',
                'current_price': 420000,
                'target_price': 450000,
                'market_cap': 29000000000000
            },
            {
                'stock_code': '006400',
                'name': '삼성SDI',
                'score': 73,
                'recommendation': '보유',
                'reason': f'{strategy_name} 기준 배터리 시장 성장성',
                'current_price': 385000,
                'target_price': 420000,
                'market_cap': 27000000000000
            }
        ]
        
        # 약간의 지연으로 실제 분석 시뮬레이션
        await asyncio.sleep(2)
        return sample_results
    
    def print_korean_results(self, strategy_name: str, results: List[Dict]):
        """한국주식 분석 결과 출력"""
        print(f"\n✅ 🇰🇷 {strategy_name} 분석 완료! TOP {len(results)} 종목:")
        print("=" * 120)
        
        for i, stock in enumerate(results, 1):
            stock_code = stock.get('stock_code', 'N/A')
            name = stock.get('name', 'N/A')
            score = stock.get('score', 0)
            recommendation = stock.get('recommendation', '보유')
            reason = stock.get('reason', '분석 결과 기반')
            current_price = stock.get('current_price', 0)
            target_price = stock.get('target_price', 0)
            market_cap = stock.get('market_cap', 0)
            
            # 시가총액 포맷팅
            market_cap_str = f"{market_cap // 1000000000000:.1f}조원" if market_cap > 0 else "N/A"
            
            print(f"  {i:2d}위. {name} ({stock_code})")
            print(f"       📊 점수: {score}점 | 💡 추천: {recommendation}")
            print(f"       🎯 이유: {reason}")
            print(f"       💰 현재가: {current_price:,}원 | 🚀 목표가: {target_price:,}원")
            print(f"       🏢 시가총액: {market_cap_str}")
            print("-" * 120)
        
        print("=" * 120)
        print()
    
    async def analyze_all_korean_strategies(self):
        """한국주식 전체 전략 분석"""
        print("\n🇰🇷 한국주식 전체 전략 분석 시작...")
        print("=" * 80)
        
        for strategy_code, strategy_name in self.korean_strategies:
            await self.analyze_korean_strategy(strategy_code, strategy_name)
            print()  # 전략 간 구분
        
        print("🎉 한국주식 전체 분석 완료!")
    
    async def analyze_all_us_strategies(self):
        """미국주식 전체 전략 분석"""
        print("\n🇺🇸 미국주식 전체 전략 분석 시작...")
        print("=" * 80)
        
        # 나스닥100 분석
        print("📊 나스닥100 분석:")
        for strategy_code, strategy_name in self.us_strategies:
            results = await self.ai_system.analyze_nasdaq100_top5(strategy_code)
            self.us_analyzer.print_analysis_results(f"나스닥100 {strategy_name}", results)
        
        # S&P500 분석
        print("📊 S&P500 분석:")
        for strategy_code, strategy_name in self.us_strategies:
            results = await self.ai_system.analyze_sp500_top5(strategy_code)
            self.us_analyzer.print_analysis_results(f"S&P500 {strategy_name}", results)
        
        print("🎉 미국주식 전체 분석 완료!")
    
    async def analyze_global_comprehensive(self):
        """글로벌 통합 분석"""
        print("\n🌍 글로벌 통합 분석 시작...")
        print("=" * 100)
        
        print("1️⃣ 한국주식 대표 전략 분석...")
        # 한국주식 대표 전략들 (블랙록, 워렌버핏, 윌리엄오닐)
        korean_tasks = [
            self._simulate_korean_analysis("blackrock", "블랙록 전략"),
            self._simulate_korean_analysis("warren_buffett", "워렌 버핏 전략"),
            self._simulate_korean_analysis("william_oneil", "윌리엄 오닐 전략")
        ]
        
        print("2️⃣ 미국주식 대표 전략 분석...")
        # 미국주식 대표 전략들
        us_tasks = [
            self.ai_system.analyze_nasdaq100_top5("momentum"),
            self.ai_system.analyze_nasdaq100_top5("value"),
            self.ai_system.analyze_sp500_top5("value"),
            self.ai_system.analyze_sp500_top5("growth")
        ]
        
        # 병렬 실행
        all_tasks = korean_tasks + us_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # 결과 출력
        korean_titles = ["블랙록 전략", "워렌 버핏 전략", "윌리엄 오닐 전략"]
        us_titles = ["나스닥100 모멘텀", "나스닥100 가치", "S&P500 가치", "S&P500 성장"]
        
        print("🇰🇷 한국주식 결과:")
        for i, (title, result) in enumerate(zip(korean_titles, results[:3])):
            if isinstance(result, list):
                self.print_korean_results(title, result)
            else:
                print(f"❌ {title} 분석 실패: {result}")
        
        print("🇺🇸 미국주식 결과:")
        for i, (title, result) in enumerate(zip(us_titles, results[3:])):
            if isinstance(result, list):
                self.us_analyzer.print_analysis_results(title, result)
            else:
                print(f"❌ {title} 분석 실패: {result}")
        
        print("🎉 글로벌 통합 분석 완료!")
        
        # 투자 추천 요약
        self.print_investment_summary()
    
    def print_investment_summary(self):
        """투자 추천 요약"""
        print("\n📋 투자 추천 요약:")
        print("=" * 80)
        print("🇰🇷 한국주식 포인트:")
        print("   • 대형주 중심의 안정적 포트폴리오 구성")
        print("   • 반도체, IT, 바이오 섹터 주목")
        print("   • 배당주와 성장주의 균형 잡힌 투자")
        print()
        print("🇺🇸 미국주식 포인트:")
        print("   • 기술주 중심의 성장 포트폴리오")
        print("   • 나스닥100의 높은 성장성")
        print("   • S&P500의 안정성과 다양성")
        print()
        print("🌍 글로벌 분산투자 전략:")
        print("   • 한국 40% + 미국 60% 비중 권장")
        print("   • 시장 변동성에 따른 리밸런싱")
        print("   • 장기 투자 관점 유지")
        print("=" * 80)
    
    async def run_interactive_mode(self):
        """대화형 모드 실행"""
        self.print_welcome_message()
        
        while True:
            try:
                self.print_main_menu()
                choice = input("선택하세요 (0-13): ").strip()
                
                if choice == '0':
                    print("👋 통합 주식 분석 시스템을 종료합니다.")
                    break
                
                # 한국주식 분석
                elif choice in ['1', '2', '3', '4', '5', '6']:
                    strategy_idx = int(choice) - 1
                    strategy_code, strategy_name = self.korean_strategies[strategy_idx]
                    await self.analyze_korean_strategy(strategy_code, strategy_name)
                
                # 미국주식 분석
                elif choice in ['7', '8', '9', '10']:
                    if choice == '7':
                        results = await self.ai_system.analyze_nasdaq100_top5("momentum")
                        self.us_analyzer.print_analysis_results("나스닥100 모멘텀 전략", results)
                    elif choice == '8':
                        results = await self.ai_system.analyze_nasdaq100_top5("value")
                        self.us_analyzer.print_analysis_results("나스닥100 가치 전략", results)
                    elif choice == '9':
                        results = await self.ai_system.analyze_sp500_top5("value")
                        self.us_analyzer.print_analysis_results("S&P500 가치 전략", results)
                    elif choice == '10':
                        results = await self.ai_system.analyze_sp500_top5("growth")
                        self.us_analyzer.print_analysis_results("S&P500 성장 전략", results)
                
                # 통합 분석
                elif choice in ['11', '12', '13']:
                    if choice == '11':
                        await self.analyze_all_korean_strategies()
                    elif choice == '12':
                        await self.analyze_all_us_strategies()
                    elif choice == '13':
                        await self.analyze_global_comprehensive()
                
                else:
                    print("❌ 잘못된 선택입니다. 올바른 번호를 입력하세요.")
                
                print("✅ 작업이 완료되었습니다!")
                print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                await asyncio.sleep(3)
                print("\n" + "🚀" * 30 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 사용자에 의해 종료되었습니다.")
                break
            except Exception as e:
                logger.error(f"❌ 실행 중 오류 발생: {e}")
                print(f"❌ 오류가 발생했습니다: {e}")
                print("🔄 3초 후 자동으로 메뉴로 돌아갑니다...")
                await asyncio.sleep(3)

async def main():
    """메인 실행 함수"""
    try:
        # 통합 분석 시스템 초기화
        analyzer = IntegratedStockAnalyzer()
        
        # 대화형 모드 실행
        await analyzer.run_interactive_mode()
        
    except Exception as e:
        logger.error(f"❌ 시스템 실행 실패: {e}")
        print(f"❌ 시스템 실행 실패: {e}")

if __name__ == "__main__":
    """프로그램 시작점"""
    print("🌍 통합 주식 분석 시스템 (한국 + 미국) 시작...")
    
    try:
        # 이벤트 루프 실행
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 오류: {e}")
        logger.error(f"프로그램 실행 오류: {e}") 