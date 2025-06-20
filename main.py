#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_legacy.config import *
from core.auth import KISAuth
from core_legacy.core_trader import CoreTrader
from personal_blackrock.ai_analyzer import AIAnalyzer
from personal_blackrock.real_time_monitor import RealTimeMonitor
from personal_blackrock.notifier import Notifier
from ui.menu import create_main_menu
from services.code_analyzer import CodeAnalyzer, CodeQualityReport
from personal_blackrock.stock_data_manager import DataManager

class StockAnalysisSystem:
    """🚀 주식 분석 시스템 메인 클래스"""
    
    def __init__(self):
        self.trader = None
        self.ai_analyzer = None
        self.monitor = None
        self.notifier = None
        self.auth = None
        self.code_analyzer = None
        print("🚀 주식 분석 시스템 초기화...")
    
    async def initialize(self):
        """시스템 초기화 - 성능 최적화 버전"""
        try:
            print("📊 시스템 구성 요소 초기화 중...")
            
            # 1. 공통 데이터 관리자 초기화 (한 번만)
            print("🔄 공통 데이터 관리자 초기화 중...")
            shared_data_manager = DataManager()
            print("✅ 공통 데이터 관리자 초기화 완료")
            
            # 2. KISAuth 초기화 (인증 관리)
            self.auth = KISAuth(app_key=KIS_APP_KEY, app_secret=KIS_APP_SECRET)
            print("✅ KISAuth (인증 관리) 초기화 완료")
            
            # 3. Core Trader 초기화
            self.trader = CoreTrader()
            print("✅ Core Trader 초기화 완료")
            
            # 4. AI Analyzer 초기화 (공통 데이터 관리자 사용)
            self.ai_analyzer = AIAnalyzer(data_manager=shared_data_manager)
            print("✅ AI Analyzer 초기화 완료")

            # 5. Code Analyzer 초기화
            self.code_analyzer = CodeAnalyzer(target_directory=".")
            print("✅ Code Analyzer 초기화 완료")
            
            # 6. Notifier 초기화
            self.notifier = Notifier()
            print("✅ Notifier 초기화 완료")
            
            # 7. Real Time Monitor 초기화 (공통 데이터 관리자 사용)
            self.monitor = RealTimeMonitor(self.trader, self.notifier, data_manager=shared_data_manager)
            print("✅ Real Time Monitor (통합 모니터링 시스템) 초기화 완료")
            
            print("🎉 모든 시스템 구성 요소 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    def _print_top5_results(self, strategy_name: str, results: List[Dict]):
        """TOP 5 결과를 포맷팅하여 출력합니다."""
        if results:
            print(f"✅ {strategy_name} 분석 완료! TOP {len(results)} 종목:")
            print("="*100)
            for i, stock in enumerate(results, 1):
                company_name = stock.get('name', 'N/A')
                stock_code = stock.get('stock_code', 'N/A')
                score = stock.get('점수', 'N/A')
                recommendation = stock.get('추천 등급', 'N/A')
                reason = stock.get('추천 이유', '분석 결과 기반')
                entry_price = stock.get('진입 가격', '현재가 기준')
                target_price = stock.get('목표 가격', '목표가 미설정')
                
                print(f"  {i:2d}위. {company_name} ({stock_code})")
                print(f"       📊 점수: {score}점 | 💡 추천: {recommendation}")
                print(f"       🎯 이유: {reason}")
                print(f"       💰 진입가: {entry_price} | 🚀 목표가: {target_price}")
                print("-" * 100)
            print("="*100)
        else:
            print("❌ 분석 결과를 가져올 수 없습니다.")

    async def analyze_strategy(self, strategy_name: str, kor_strategy_name: str):
        """특정 전략으로 KOSPI 200 종목을 분석합니다."""
        print(f"\n🔍 {kor_strategy_name} 분석 시작 (코스피 200 대상)...")
        try:
            token = await self.auth.get_valid_token()
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            results = await self.ai_analyzer.analyze_strategy_for_kospi200(strategy_name)
            self._print_top5_results(kor_strategy_name, results)
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")

    async def analyze_william_oneil(self):
        """윌리엄 오닐 TOP 5 종목 추천"""
        await self.analyze_strategy("윌리엄 오닐", "윌리엄 오닐")

    async def analyze_jesse_livermore(self):
        """제시 리버모어 TOP 5 종목 추천"""
        await self.analyze_strategy("제시 리버모어", "제시 리버모어")

    async def analyze_warren_buffett(self):
        """워렌 버핏 TOP 5 종목 추천"""
        await self.analyze_strategy("워렌 버핏", "워렌 버핏")

    async def analyze_peter_lynch(self):
        """피터 린치 TOP 5 종목 추천"""
        await self.analyze_strategy("피터 린치", "피터 린치")

    async def analyze_ichimoku(self):
        """일목균형표 TOP 5 종목 추천"""
        await self.analyze_strategy("일목균형표", "일목균형표")

    async def analyze_blackrock(self):
        """블랙록 TOP 5 종목 추천"""
        await self.analyze_strategy("블랙록", "블랙록")

    async def analyze_individual_stock(self):
        """개별 종목 분석"""
        print("\n📊 개별 종목 분석")
        
        stock_code = input("종목 코드를 입력하세요 (예: 005930): ").strip()
        if not stock_code:
            print("❌ 종목 코드가 입력되지 않았습니다.")
            return
        
        strategy_name = input("분석할 투자 전략을 입력하세요 (예: 워렌 버핏): ").strip()
        if not strategy_name:
            print("❌ 투자 전략이 입력되지 않았습니다.")
            return

        try:
            # 토큰 유효성 확인
            token = await self.auth.get_valid_token()
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            print(f"🔍 {stock_code} 종목을 '{strategy_name}' 전략으로 분석 중...")
            
            # 개별 종목 분석 실행 (리팩토링된 AIAnalyzer 사용)
            result = await self.ai_analyzer.analyze_stock_with_strategy(stock_code, strategy_name)
            
            if result and 'error' not in result:
                print(f"✅ {result.get('name', stock_code)} 분석 완료!")
                print(f"📊 점수: {result.get('점수', 'N/A')}")
                print(f"💡 추천 등급: {result.get('추천 등급', 'N/A')}")
                print(f"🎯 추천 이유: {result.get('추천 이유', 'N/A')}")
                print(f"💰 진입 가격: {result.get('진입 가격', 'N/A')}")
                print(f"🚀 목표 가격: {result.get('목표 가격', 'N/A')}")
                print(f"🔍 신뢰도: {result.get('신뢰도', 'N/A')}")

                # 상세 분석 결과 표시
                print("\n📋 상세 분석:")
                print(result.get('분석', '상세 분석 결과가 없습니다.'))

            else:
                error_msg = result.get('error', '알 수 없는 오류')
                print(f"❌ 분석 결과를 가져올 수 없습니다: {error_msg}")
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
    
    async def start_monitoring(self):
        """실시간 모니터링 시작"""
        print("\n🔄 실시간 모니터링 시작...")
        
        try:
            # 토큰 유효성 확인
            token = await self.auth.get_valid_token()
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            print("📊 실시간 모니터링을 시작합니다...")
            print("⏹️ 중단하려면 Ctrl+C를 누르세요.")
            
            await self.monitor.start_monitoring()
            
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 모니터링을 중단했습니다.")
        except Exception as e:
            print(f"❌ 모니터링 중 오류 발생: {e}")
    
    async def show_token_status(self):
        """토큰 상태 확인"""
        print("\n🔑 토큰 상태 확인...")
        
        try:
            status = await self.auth.get_token_status()
            
            print(f"📊 토큰 상태: {status.get('status', 'N/A')}")
            if status.get('expires_at'):
                print(f"⏰ 만료 시간: {status['expires_at']}")
            if status.get('time_left'):
                print(f"⏳ 남은 시간: {status['time_left']}")
            
        except Exception as e:
            print(f"❌ 토큰 상태 확인 실패: {e}")
    
    async def manual_token_renewal(self):
        """수동 토큰 갱신"""
        print("\n🔄 수동 토큰 갱신...")
        
        try:
            # KISAuth는 자동으로 갱신하므로, 기존 토큰을 무효화하여 강제로 재발급을 유도
            await self.auth.invalidate_token()
            new_token = await self.auth.get_valid_token()

            if new_token:
                print("✅ 토큰 갱신 완료!")
            else:
                print("❌ 토큰 갱신 실패!")
            
        except Exception as e:
            print(f"❌ 토큰 갱신 중 오류 발생: {e}")
    
    async def run_quality_check(self):
        """코드 품질 검사 실행"""
        print("\n🔍 코드 품질 검사 실행...")
        
        try:
            report = self.code_analyzer.analyze()
            
            print(f"📊 검사 완료!")
            print(f"📁 검사 파일: {report.total_files}개")
            print(f"🎯 성능 점수: {report.performance_score}/100")
            print(f"❌ 구문 오류: {len(report.syntax_errors)}개")
            print(f"🔧 복잡도 이슈: {len(report.complexity_issues)}개")
            print(f"👃 코드 스멜: {len(report.code_smells)}개")
            print(f"🔒 보안 이슈: {len(report.security_issues)}개")
            
            if report.recommendations:
                print("\n💡 주요 권장사항:")
                for i, rec in enumerate(report.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
        except Exception as e:
            print(f"❌ 코드 품질 검사 중 오류 발생: {e}")
    
    async def start_trading_volume_analysis(self):
        """거래대금 TOP 20 실시간 전략 매칭 분석 시작"""
        print("\n🔍 거래대금 TOP 20 실시간 전략 매칭 분석...")
        
        try:
            # 토큰 유효성 확인
            token = await self.auth.get_valid_token()
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            print("📊 거래대금 TOP 20 종목 실시간 전략 매칭 분석을 시작합니다...")
            print("⚙️ 분석 설정:")
            print(f"   - 전략 분석 주기: {self.monitor.analysis_interval}초")
            print(f"   - 기본 모니터링 주기: {self.monitor.monitoring_interval}초")
            print(f"   - 최소 매칭 점수: {self.monitor.min_score_threshold}점")
            print(f"   - 분석 전략: {', '.join(self.monitor.strategies)}")
            print("⏹️ 중단하려면 Ctrl+C를 누르세요.")
            
            # 분석 설정 변경 옵션
            change_settings = input("\n분석 설정을 변경하시겠습니까? (y/N): ").strip().lower()
            if change_settings == 'y':
                await self._configure_trading_analysis()
            
            # 실시간 분석 시작 (통합 모니터링)
            await self.monitor.start_real_time_analysis()
            
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 거래대금 분석을 중단했습니다.")
            await self.monitor.stop_monitoring()
        except Exception as e:
            print(f"❌ 거래대금 분석 중 오류 발생: {e}")

    async def _configure_trading_analysis(self):
        """거래대금 분석 설정 변경"""
        try:
            print("\n⚙️ 분석 설정 변경")
            
            # 전략 분석 주기 변경
            interval_input = input(f"전략 분석 주기 (현재: {self.monitor.analysis_interval}초, 최소 60초): ").strip()
            if interval_input.isdigit():
                new_interval = int(interval_input)
                if new_interval >= 60:
                    await self.monitor.update_analysis_settings(interval=new_interval)
                    print(f"✅ 전략 분석 주기가 {new_interval}초로 변경되었습니다.")
                else:
                    print("⚠️ 전략 분석 주기는 최소 60초 이상이어야 합니다.")
            
            # 기본 모니터링 주기 변경
            monitoring_input = input(f"기본 모니터링 주기 (현재: {self.monitor.monitoring_interval}초, 최소 10초): ").strip()
            if monitoring_input.isdigit():
                new_monitoring = int(monitoring_input)
                if new_monitoring >= 10:
                    await self.monitor.update_analysis_settings(monitoring_interval=new_monitoring)
                    print(f"✅ 기본 모니터링 주기가 {new_monitoring}초로 변경되었습니다.")
                else:
                    print("⚠️ 기본 모니터링 주기는 최소 10초 이상이어야 합니다.")
            
            # 최소 매칭 점수 변경
            score_input = input(f"최소 매칭 점수 (현재: {self.monitor.min_score_threshold}점, 50-100): ").strip()
            if score_input.isdigit():
                new_score = int(score_input)
                if 50 <= new_score <= 100:
                    await self.monitor.update_analysis_settings(min_score=new_score)
                    print(f"✅ 최소 매칭 점수가 {new_score}점으로 변경되었습니다.")
                else:
                    print("⚠️ 최소 매칭 점수는 50-100 범위여야 합니다.")
            
        except Exception as e:
            print(f"❌ 설정 변경 중 오류: {e}")

    async def show_trading_analysis_status(self):
        """거래대금 분석 상태 확인"""
        try:
            if not self.monitor:
                print("❌ 통합 모니터링 시스템이 초기화되지 않았습니다.")
                return
            
            status = await self.monitor.get_current_analysis_status()
            
            print("\n📊 통합 모니터링 시스템 상태")
            print("-" * 50)
            print(f"🔄 실행 상태: {'실행 중' if status['is_running'] else '중지됨'}")
            print(f"⏱️ 전략 분석 주기: {status['analysis_interval']}초")
            print(f"⏰ 기본 모니터링 주기: {status['monitoring_interval']}초")
            print(f"🎯 최소 매칭 점수: {status['min_score_threshold']}점")
            print(f"📈 분석 전략: {', '.join(status['strategies'])}")
            print(f"📱 매칭 알림 기록: {status['notified_matches_count']}개")
            print(f"🔔 모니터링 알림 기록: {status['alert_history_count']}개")
            
            if status['last_analysis_time']:
                print(f"🕐 마지막 분석: {status['last_analysis_time']}")
                
        except Exception as e:
            print(f"❌ 상태 확인 중 오류: {e}")

    async def show_system_status(self):
        """시스템 상태 확인"""
        print("\n📊 시스템 상태 확인...")
        
        try:
            import psutil
            
            # 시스템 리소스
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            print(f"💾 메모리 사용률: {memory.percent:.1f}%")
            print(f"💿 디스크 사용률: {(disk.used/disk.total*100):.1f}%")
            
            # 토큰 상태
            token_status = await self.auth.get_token_status()
            print(f"🔑 토큰 상태: {token_status.get('status', 'N/A')}")
            
            # API 제한 설정
            print(f"⚙️ API 제한 설정:")
            print(f"  - 초당 호출: {TOTAL_API_CALLS_PER_SEC}회")
            print(f"  - 일일 한도: {DAILY_API_LIMIT:,}회")
            
            # 통합 모니터링 시스템 상태
            if self.monitor:
                await self.show_trading_analysis_status()
            
            print("✅ 시스템이 정상 운영 중입니다.")
                
        except Exception as e:
            print(f"❌ 시스템 상태 확인 중 오류 발생: {e}")
    
    async def run(self):
        """메인 실행 루프"""
        if not await self.initialize():
            print("❌ 시스템 초기화에 실패했습니다.")
            return
        
        print(f"\n🎉 시스템이 성공적으로 시작되었습니다!")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 매일 오전 7시에 자동으로 토큰 발행 및 코드 품질 검사가 실행됩니다.")
        
        main_menu = create_main_menu(self)
        
        while True:
            try:
                main_menu.display()
                should_continue = await main_menu.get_and_execute_choice()
                
                if not should_continue:
                    break
                
                input("\n계속하려면 Enter를 누르세요...")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다...")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                input("\n계속하려면 Enter를 누르세요...")

    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.monitor:
                await self.monitor.cleanup()
            print("✅ 시스템 정리 완료")
        except Exception as e:
            print(f"⚠️ 정리 중 오류: {e}")

async def main():
    """메인 함수"""
    system = StockAnalysisSystem()
    
    try:
        await system.run()
    finally:
        await system.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}") 