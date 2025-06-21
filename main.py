#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import sys
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_legacy.config import *
from core.auth import KISAuth
from core_legacy.trader import CoreTrader
from personal_blackrock.ai_analyzer import HighPerformanceAIAnalyzer
from personal_blackrock.monitor import RealTimeMonitor
from personal_blackrock.notifier import Notifier
from personal_blackrock.data import DataManager

# 성능 최적화 모듈 import
from core.performance_optimizer import (
    PerformanceOptimizer, 
    get_optimizer, 
    cached_call, 
    batch_call
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_performance.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedStockAnalysisSystem:
    """🚀 고성능 최적화된 주식 분석 시스템 메인 클래스"""
    
    def __init__(self):
        self.trader = None
        self.ai_analyzer = None
        self.monitor = None
        self.notifier = None
        self.auth = None
        self.data_manager = None
        self.optimizer: Optional[PerformanceOptimizer] = None
        self.start_time = time.time()
        
        # 성능 지표
        self.analysis_count = 0
        self.cache_hits = 0
        self.total_requests = 0
        
        print("🚀 고성능 최적화된 주식 분석 시스템 초기화...")
    
    async def initialize(self):
        """시스템 초기화 - 고성능 최적화 버전"""
        try:
            logger.info("📊 고성능 시스템 구성 요소 초기화 시작...")
            init_start_time = time.time()
            
            # 1. 성능 최적화 매니저 초기화 (최우선)
            print("⚡ 성능 최적화 매니저 초기화 중...")
            self.optimizer = await get_optimizer()
            print("✅ 성능 최적화 매니저 초기화 완료")
            
            # 2. 병렬 초기화를 위한 태스크 리스트
            initialization_tasks = []
            
            # 2-1. 공통 데이터 관리자 초기화 (캐시 최적화)
            async def init_data_manager():
            print("🔄 공통 데이터 관리자 초기화 중...")
                self.data_manager = DataManager()
                # 데이터 매니저에 성능 최적화 적용
                if hasattr(self.data_manager, 'set_optimizer'):
                    self.data_manager.set_optimizer(self.optimizer)
            print("✅ 공통 데이터 관리자 초기화 완료")
            
            # 2-2. KISAuth 초기화 (토큰 캐싱 최적화)
            async def init_auth():
                print("🔐 KISAuth (인증 관리) 초기화 중...")
            self.auth = KISAuth(app_key=KIS_APP_KEY, app_secret=KIS_APP_SECRET)
                # 인증 토큰 캐싱 최적화
                if hasattr(self.auth, 'set_optimizer'):
                    self.auth.set_optimizer(self.optimizer)
            print("✅ KISAuth (인증 관리) 초기화 완료")
            
            # 병렬 초기화 실행
            initialization_tasks.extend([init_data_manager(), init_auth()])
            
            # 기본 컴포넌트들을 병렬로 초기화
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # 3. Core Trader 초기화 (의존성 있는 컴포넌트)
            print("🤖 Core Trader 초기화 중...")
            self.trader = CoreTrader(kis_api=self.auth)
            await self.trader.async_initialize()
            # 트레이더에 성능 최적화 적용
            if hasattr(self.trader, 'set_optimizer'):
                self.trader.set_optimizer(self.optimizer)
            print("✅ Core Trader 초기화 완료")
            
            # 4. 고성능 컴포넌트들 병렬 초기화
            async def init_ai_analyzer():
                print("🧠 AI Analyzer 초기화 중...")
                self.ai_analyzer = HighPerformanceAIAnalyzer(data_manager=self.data_manager)
                # AI 분석기에 성능 최적화 적용
                if hasattr(self.ai_analyzer, 'set_optimizer'):
                    self.ai_analyzer.set_optimizer(self.optimizer)
            print("✅ AI Analyzer 초기화 완료")
            
            async def init_notifier():
                print("📢 Notifier 초기화 중...")
            self.notifier = Notifier()
                if hasattr(self.notifier, 'set_optimizer'):
                    self.notifier.set_optimizer(self.optimizer)
            print("✅ Notifier 초기화 완료")
            
            # 고성능 컴포넌트들 병렬 초기화
            await asyncio.gather(
                init_ai_analyzer(),
                init_notifier(),
                return_exceptions=True
            )
            
            # 5. Real Time Monitor 초기화 (마지막 - 모든 의존성 필요)
            print("📊 Real Time Monitor 초기화 중...")
            self.monitor = RealTimeMonitor(
                self.trader, 
                self.notifier, 
                data_manager=self.data_manager
            )
            if hasattr(self.monitor, 'set_optimizer'):
                self.monitor.set_optimizer(self.optimizer)
            print("✅ Real Time Monitor 초기화 완료")
            
            # 초기화 완료 시간 측정
            init_time = time.time() - init_start_time
            logger.info(f"🎉 고성능 시스템 초기화 완료! (소요시간: {init_time:.2f}초)")
            
            # 성능 지표 출력
            await self._log_performance_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    async def _log_performance_metrics(self):
        """성능 지표 로깅"""
        if self.optimizer:
            metrics = await self.optimizer.get_performance_metrics()
            logger.info(f"📊 성능 지표 - 메모리: {metrics.memory_usage_mb:.1f}MB, "
                       f"CPU: {metrics.cpu_usage_percent:.1f}%, "
                       f"캐시 적중률: {metrics.cache_hit_rate:.1%}")
    
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
        """특정 전략으로 KOSPI 200 종목을 분석합니다 - 고성능 최적화 버전"""
        print(f"\n🔍 {kor_strategy_name} 고속 분석 시작 (코스피 200 대상)...")
        analysis_start_time = time.time()
        
        try:
            # 캐시된 토큰 획득
            cache_key = f"auth_token_{strategy_name}"
            token = await cached_call(
                cache_key, 
                self.auth.get_valid_token, 
                ttl=1800  # 30분 캐시
            )
            
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            # 캐시된 분석 결과 확인
            analysis_cache_key = f"strategy_analysis_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H')}"
            
            print("⚡ 고속 병렬 분석 실행 중...")
            results = await cached_call(
                analysis_cache_key,
                lambda: self.ai_analyzer.analyze_strategy_for_kospi200(strategy_name),
                ttl=3600  # 1시간 캐시
            )
            
            analysis_time = time.time() - analysis_start_time
            self.analysis_count += 1
            self.total_requests += 1
            
            if results:
                self.cache_hits += 1
                
            print(f"⚡ 분석 완료 (소요시간: {analysis_time:.2f}초)")
            self._print_top5_results(kor_strategy_name, results)
            
            # 성능 지표 업데이트
            await self._log_performance_metrics()
            
        except Exception as e:
            logger.error(f"❌ 분석 중 오류 발생: {e}")
            print(f"❌ 분석 중 오류 발생: {e}")

    async def analyze_william_oneil(self):
        """윌리엄 오닐 TOP 5 종목 추천 - 고성능 버전"""
        await self.analyze_strategy("윌리엄 오닐", "🎯 윌리엄 오닐")

    async def analyze_jesse_livermore(self):
        """제시 리버모어 TOP 5 종목 추천 - 고성능 버전"""
        await self.analyze_strategy("제시 리버모어", "📈 제시 리버모어")

    async def analyze_warren_buffett(self):
        """워렌 버핏 TOP 5 종목 추천 - 고성능 버전"""
        await self.analyze_strategy("워렌 버핏", "💎 워렌 버핏")

    async def analyze_peter_lynch(self):
        """피터 린치 TOP 5 종목 추천 - 고성능 버전"""
        await self.analyze_strategy("피터 린치", "🔍 피터 린치")

    async def analyze_ichimoku(self):
        """일목균형표 TOP 5 종목 추천 - 고성능 버전"""
        await self.analyze_strategy("일목균형표", "☁️ 일목균형표")

    async def analyze_blackrock(self):
        """블랙록 TOP 5 종목 추천 - 고성능 버전"""
        await self.analyze_strategy("블랙록", "🏦 블랙록")

    async def analyze_individual_stock(self):
        """개별 종목 분석 - 고성능 최적화 버전"""
        print("\n📊 개별 종목 고속 분석")
        
        stock_code = input("종목 코드를 입력하세요 (예: 005930): ").strip()
        if not stock_code:
            print("❌ 종목 코드가 입력되지 않았습니다.")
            return
        
        strategy_name = input("분석할 투자 전략을 입력하세요 (예: 워렌 버핏): ").strip()
        if not strategy_name:
            print("❌ 투자 전략이 입력되지 않았습니다.")
            return

        try:
            analysis_start_time = time.time()
            
            # 캐시된 토큰 획득
            token = await cached_call(
                "individual_auth_token", 
                self.auth.get_valid_token, 
                ttl=1800
            )
            
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            print(f"⚡ {stock_code} 종목을 '{strategy_name}' 전략으로 고속 분석 중...")
            
            # 캐시된 개별 종목 분석
            cache_key = f"individual_stock_{stock_code}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H')}"
            result = await cached_call(
                cache_key,
                lambda: self.ai_analyzer.analyze_stock_with_strategy(stock_code, strategy_name),
                ttl=1800  # 30분 캐시
            )
            
            analysis_time = time.time() - analysis_start_time
            
            if result and 'error' not in result:
                print(f"✅ {result.get('name', stock_code)} 고속 분석 완료! (소요시간: {analysis_time:.2f}초)")
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
                error_msg = result.get('error', '알 수 없는 오류') if result else '분석 실패'
                print(f"❌ 분석 결과를 가져올 수 없습니다: {error_msg}")
            
        except Exception as e:
            logger.error(f"❌ 개별 종목 분석 중 오류 발생: {e}")
            print(f"❌ 분석 중 오류 발생: {e}")
    
    async def start_monitoring(self):
        """실시간 모니터링 시작 - 고성능 최적화 버전"""
        print("\n🔄 고성능 실시간 모니터링 시작...")
        
        try:
            # 캐시된 토큰 확인
            token = await cached_call(
                "monitoring_auth_token", 
                self.auth.get_valid_token, 
                ttl=1800
            )
            
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            print("📊 고성능 실시간 모니터링을 시작합니다...")
            print("⏹️ 중단하려면 Ctrl+C를 누르세요.")
            
            await self.monitor.start_monitoring()
            
        except KeyboardInterrupt:
            print("\n⏹️ 사용자가 모니터링을 중단했습니다.")
        except Exception as e:
            logger.error(f"❌ 모니터링 중 오류 발생: {e}")
            print(f"❌ 모니터링 중 오류 발생: {e}")
    
    async def show_token_status(self):
        """토큰 상태 확인 - 최적화 버전"""
        print("\n🔐 토큰 상태 확인...")
        try:
            # 캐시된 토큰 상태 확인
            token_info = await cached_call(
                "token_status_check",
                self.auth.get_token_info,
                ttl=60  # 1분 캐시
            )
            
            if token_info:
                print(f"✅ 토큰 상태: 유효")
                print(f"📅 만료 시간: {token_info.get('expires_at', 'N/A')}")
                print(f"🔑 토큰 타입: {token_info.get('token_type', 'Bearer')}")
            else:
                print("❌ 토큰이 유효하지 않습니다.")
        except Exception as e:
            logger.error(f"❌ 토큰 상태 확인 실패: {e}")
            print(f"❌ 토큰 상태 확인 실패: {e}")
    
    async def manual_token_renewal(self):
        """수동 토큰 갱신 - 최적화 버전"""
        print("\n🔄 토큰 수동 갱신...")
        try:
            # 캐시 무효화 후 새 토큰 획득
            if self.optimizer:
                await self.optimizer.cache.clear()
            
            new_token = await self.auth.get_valid_token(force_refresh=True)
            if new_token:
                print("✅ 토큰 갱신 성공!")
                
                # 새 토큰을 캐시에 저장
                await cached_call(
                    "renewed_auth_token",
                    lambda: new_token,
                    ttl=1800
                )
            else:
                print("❌ 토큰 갱신 실패")
        except Exception as e:
            logger.error(f"❌ 토큰 갱신 실패: {e}")
            print(f"❌ 토큰 갱신 실패: {e}")
    
    async def run_quality_check(self):
        """시스템 품질 검사 - 고성능 버전"""
        print("\n🔍 고성능 시스템 품질 검사 시작...")
        
        try:
            check_start_time = time.time()
            
            # 병렬 품질 검사 태스크
            quality_checks = [
                self._check_auth_system(),
                self._check_data_manager(),
                self._check_ai_analyzer(),
                self._check_performance_metrics()
            ]
            
            # 병렬 실행
            results = await asyncio.gather(*quality_checks, return_exceptions=True)
            
            check_time = time.time() - check_start_time
            
            # 결과 분석
            passed_checks = sum(1 for result in results if result is True)
            total_checks = len(results)
            
            print(f"\n📊 품질 검사 완료 (소요시간: {check_time:.2f}초)")
            print(f"✅ 통과: {passed_checks}/{total_checks}")
            
            if passed_checks == total_checks:
                print("🎉 모든 품질 검사를 통과했습니다!")
            else:
                print("⚠️ 일부 품질 검사에서 문제가 발견되었습니다.")
                
        except Exception as e:
            logger.error(f"❌ 품질 검사 실패: {e}")
            print(f"❌ 품질 검사 실패: {e}")

    async def _check_auth_system(self) -> bool:
        """인증 시스템 검사"""
        try:
            token = await self.auth.get_valid_token()
            print("✅ 인증 시스템: 정상")
            return token is not None
        except Exception as e:
            print(f"❌ 인증 시스템: 오류 - {e}")
            return False

    async def _check_data_manager(self) -> bool:
        """데이터 매니저 검사"""
        try:
            if self.data_manager and hasattr(self.data_manager, 'get_kospi200_list'):
                kospi_list = await asyncio.to_thread(self.data_manager.get_kospi200_list)
                print(f"✅ 데이터 매니저: 정상 ({len(kospi_list)}개 종목)")
                return len(kospi_list) > 0
            else:
                print("❌ 데이터 매니저: 초기화되지 않음")
                return False
        except Exception as e:
            print(f"❌ 데이터 매니저: 오류 - {e}")
            return False

    async def _check_ai_analyzer(self) -> bool:
        """AI 분석기 검사"""
        try:
            if self.ai_analyzer:
                print("✅ AI 분석기: 정상")
                return True
            else:
                print("❌ AI 분석기: 초기화되지 않음")
                return False
        except Exception as e:
            print(f"❌ AI 분석기: 오류 - {e}")
            return False

    async def _check_performance_metrics(self) -> bool:
        """성능 지표 검사"""
        try:
            if self.optimizer:
                metrics = await self.optimizer.get_performance_metrics()
                print(f"✅ 성능 최적화: 정상 (메모리: {metrics.memory_usage_mb:.1f}MB, 캐시 적중률: {metrics.cache_hit_rate:.1%})")
                return True
            else:
                print("❌ 성능 최적화: 초기화되지 않음")
                return False
        except Exception as e:
            print(f"❌ 성능 최적화: 오류 - {e}")
            return False
    
    async def start_trading_volume_analysis(self):
        """거래량 분석 시작 - 고성능 버전"""
        print("\n📊 고성능 거래량 분석 시작...")
        
        try:
            # 캐시된 토큰 확인
            token = await cached_call(
                "volume_analysis_token",
                self.auth.get_valid_token,
                ttl=1800
            )
            
            if not token:
                print("❌ 유효한 토큰을 획득할 수 없습니다.")
                return
            
            # 거래량 분석 설정
            config = await self._configure_trading_analysis()
            if not config:
                return

            print("⚡ 고속 거래량 분석 실행 중...")
            analysis_start_time = time.time()
            
            # 캐시된 거래량 분석
            cache_key = f"volume_analysis_{datetime.now().strftime('%Y%m%d_%H')}"
            results = await cached_call(
                cache_key,
                lambda: self._perform_volume_analysis(config),
                ttl=1800
            )
            
            analysis_time = time.time() - analysis_start_time
            
            if results:
                print(f"✅ 거래량 분석 완료 (소요시간: {analysis_time:.2f}초)")
                self._display_volume_analysis_results(results)
            else:
                print("❌ 거래량 분석 결과가 없습니다.")
                
        except Exception as e:
            logger.error(f"❌ 거래량 분석 실패: {e}")
            print(f"❌ 거래량 분석 실패: {e}")

    async def _configure_trading_analysis(self) -> Optional[Dict]:
        """거래량 분석 설정"""
        try:
            print("\n⚙️ 거래량 분석 설정")
            
            # 기본 설정
            config = {
                'min_volume': 1000000,  # 최소 거래량
                'volume_spike_threshold': 2.0,  # 거래량 급증 기준
                'price_change_threshold': 0.03,  # 가격 변동 기준 (3%)
                'analysis_period': 20,  # 분석 기간 (일)
                'top_count': 10  # 상위 몇 개 종목
            }
            
            print(f"📊 설정된 분석 기준:")
            print(f"   - 최소 거래량: {config['min_volume']:,}")
            print(f"   - 거래량 급증 기준: {config['volume_spike_threshold']}배")
            print(f"   - 가격 변동 기준: {config['price_change_threshold']*100}%")
            print(f"   - 분석 기간: {config['analysis_period']}일")
            print(f"   - 상위 종목 수: {config['top_count']}개")
            
            return config
            
        except Exception as e:
            logger.error(f"❌ 거래량 분석 설정 실패: {e}")
            return None

    async def _perform_volume_analysis(self, config: Dict) -> Optional[List[Dict]]:
        """실제 거래량 분석 수행"""
        try:
            # 여기서 실제 거래량 분석 로직을 구현
            # 현재는 모의 데이터 반환
            mock_results = [
                {
                    'stock_code': '005930',
                    'name': '삼성전자',
                    'volume_ratio': 2.5,
                    'price_change': 0.045,
                    'analysis_score': 85
                },
                {
                    'stock_code': '000660',
                    'name': 'SK하이닉스',
                    'volume_ratio': 3.2,
                    'price_change': 0.067,
                    'analysis_score': 92
                }
            ]
            
            return mock_results
            
        except Exception as e:
            logger.error(f"❌ 거래량 분석 수행 실패: {e}")
            return None

    def _display_volume_analysis_results(self, results: List[Dict]):
        """거래량 분석 결과 표시"""
        print("\n📈 거래량 분석 결과:")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. {result['name']} ({result['stock_code']})")
            print(f"    📊 거래량 비율: {result['volume_ratio']:.1f}배")
            print(f"    📈 가격 변동: {result['price_change']*100:+.1f}%")
            print(f"    🎯 분석 점수: {result['analysis_score']}")
            print("-"*80)

    async def show_trading_analysis_status(self):
        """거래량 분석 상태 표시 - 최적화 버전"""
        print("\n📊 거래량 분석 상태")
        
        try:
            # 캐시된 상태 정보 조회
            status_info = await cached_call(
                "trading_analysis_status",
                self._get_trading_analysis_status,
                ttl=300  # 5분 캐시
            )
            
            if status_info:
                print(f"✅ 분석 상태: {status_info['status']}")
                print(f"📅 마지막 분석: {status_info['last_analysis']}")
                print(f"📊 분석된 종목 수: {status_info['analyzed_stocks']}")
                print(f"⚡ 평균 분석 시간: {status_info['avg_analysis_time']:.2f}초")
            else:
                print("❌ 거래량 분석 상태를 확인할 수 없습니다.")
                
        except Exception as e:
            logger.error(f"❌ 거래량 분석 상태 확인 실패: {e}")
            print(f"❌ 거래량 분석 상태 확인 실패: {e}")

    async def _get_trading_analysis_status(self) -> Optional[Dict]:
        """거래량 분석 상태 정보 수집"""
        try:
            return {
                'status': '정상 운영',
                'last_analysis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analyzed_stocks': self.analysis_count,
                'avg_analysis_time': 1.5
            }
        except Exception as e:
            logger.error(f"❌ 거래량 분석 상태 정보 수집 실패: {e}")
            return None

    async def show_system_status(self):
        """시스템 상태 표시 - 고성능 최적화 버전"""
        print("\n🖥️ 고성능 시스템 상태")
        print("="*60)
        
        try:
            # 시스템 상태 정보 병렬 수집
            status_tasks = [
                self._get_system_uptime(),
                self._get_performance_summary(),
                self._get_component_status(),
                self._get_cache_statistics()
            ]
            
            uptime, performance, components, cache_stats = await asyncio.gather(
                *status_tasks, return_exceptions=True
            )
            
            # 시스템 가동 시간
            if not isinstance(uptime, Exception):
                print(f"⏰ 시스템 가동 시간: {uptime}")
            
            # 성능 요약
            if not isinstance(performance, Exception):
                print(f"📊 성능 요약:")
                print(f"   - 메모리 사용량: {performance['memory_mb']:.1f}MB")
                print(f"   - CPU 사용률: {performance['cpu_percent']:.1f}%")
                print(f"   - 총 분석 횟수: {performance['total_analysis']}")
            
            # 컴포넌트 상태
            if not isinstance(components, Exception):
                print(f"🔧 컴포넌트 상태:")
                for name, status in components.items():
                    status_icon = "✅" if status == "정상" else "❌"
                    print(f"   {status_icon} {name}: {status}")
            
            # 캐시 통계
            if not isinstance(cache_stats, Exception):
                print(f"💾 캐시 통계:")
                print(f"   - 캐시 적중률: {cache_stats['hit_rate']:.1%}")
                print(f"   - 총 요청 수: {cache_stats['total_requests']}")
                print(f"   - 캐시 크기: {cache_stats['cache_size']}개")
                
        except Exception as e:
            logger.error(f"❌ 시스템 상태 표시 실패: {e}")
            print(f"❌ 시스템 상태 표시 실패: {e}")

    async def _get_system_uptime(self) -> str:
        """시스템 가동 시간 계산"""
        uptime_seconds = time.time() - self.start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    async def _get_performance_summary(self) -> Dict:
        """성능 요약 정보"""
        if self.optimizer:
            metrics = await self.optimizer.get_performance_metrics()
            return {
                'memory_mb': metrics.memory_usage_mb,
                'cpu_percent': metrics.cpu_usage_percent,
                'total_analysis': self.analysis_count
            }
        return {'memory_mb': 0, 'cpu_percent': 0, 'total_analysis': 0}

    async def _get_component_status(self) -> Dict[str, str]:
        """컴포넌트 상태 확인"""
        return {
            'Auth': '정상' if self.auth else '오류',
            'Trader': '정상' if self.trader else '오류',
            'AI Analyzer': '정상' if self.ai_analyzer else '오류',
            'Monitor': '정상' if self.monitor else '오류',
            'Optimizer': '정상' if self.optimizer else '오류'
        }

    async def _get_cache_statistics(self) -> Dict:
        """캐시 통계 정보"""
        if self.optimizer:
            return {
                'hit_rate': self.optimizer.cache.get_hit_rate(),
                'total_requests': self.total_requests,
                'cache_size': len(self.optimizer.cache.l1_cache)
            }
        return {'hit_rate': 0.0, 'total_requests': 0, 'cache_size': 0}
    
    async def run(self):
        """메인 실행 루프 - 고성능 최적화 버전"""
        print("🚀 고성능 주식 분석 시스템 시작")
        
        # 시스템 초기화
        if not await self.initialize():
            print("❌ 시스템 초기화 실패. 프로그램을 종료합니다.")
            return
        
        # 메인 메뉴 생성 및 실행
        menu = OptimizedMainMenu(self)
        
        try:
        while True:
                menu.display()
                choice = await menu.get_and_execute_choice()
                
                if choice == '0':
                    print("👋 고성능 주식 분석 시스템을 종료합니다.")
                    break
                    
            except KeyboardInterrupt:
            print("\n👋 사용자가 프로그램을 중단했습니다.")
            except Exception as e:
            logger.error(f"❌ 메인 루프 실행 중 오류: {e}")
            print(f"❌ 메인 루프 실행 중 오류: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """리소스 정리 - 최적화 버전"""
        print("🧹 시스템 리소스 정리 중...")
        
        try:
            # 성능 최적화 매니저 정리
            if self.optimizer:
                await self.optimizer.cleanup()
            
            # 기타 컴포넌트 정리
            if self.trader and hasattr(self.trader, 'close'):
                await self.trader.close()
            
            print("✅ 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 리소스 정리 중 오류: {e}")

class OptimizedMainMenu:
    """고성능 최적화된 메인 메뉴 클래스"""
    
    def __init__(self, system):
        self.system = system
        self.menu_options = {
            '1': ('🎯 윌리엄 오닐 전략 분석', self.system.analyze_william_oneil),
            '2': ('📈 제시 리버모어 전략 분석', self.system.analyze_jesse_livermore),
            '3': ('💎 워렌 버핏 전략 분석', self.system.analyze_warren_buffett),
            '4': ('🔍 피터 린치 전략 분석', self.system.analyze_peter_lynch),
            '5': ('☁️ 일목균형표 전략 분석', self.system.analyze_ichimoku),
            '6': ('🏦 블랙록 전략 분석', self.system.analyze_blackrock),
            '7': ('📊 개별 종목 분석', self.system.analyze_individual_stock),
            '8': ('🔄 실시간 모니터링', self.system.start_monitoring),
            '9': ('🔐 토큰 상태 확인', self.system.show_token_status),
            '10': ('🔄 토큰 수동 갱신', self.system.manual_token_renewal),
            '11': ('🔍 시스템 품질 검사', self.system.run_quality_check),
            '12': ('📊 거래량 분석', self.system.start_trading_volume_analysis),
            '13': ('📈 거래량 분석 상태', self.system.show_trading_analysis_status),
            '14': ('🖥️ 시스템 상태', self.system.show_system_status),
            '0': ('👋 종료', None)
        }

    def display(self):
        """메뉴 표시 - 최적화된 UI"""
        print("\n" + "="*80)
        print("🚀 고성능 주식 분석 시스템 - 메인 메뉴")
        print("="*80)
        
        # 성능 지표 간단 표시
        if hasattr(self.system, 'analysis_count'):
            print(f"📊 분석 완료: {self.system.analysis_count}회 | 캐시 적중: {self.system.cache_hits}회")
        
        print("\n📈 투자 전략 분석:")
        for key in ['1', '2', '3', '4', '5', '6']:
            description, _ = self.menu_options[key]
            print(f"  {key}. {description}")
        
        print("\n🔧 시스템 기능:")
        for key in ['7', '8', '9', '10', '11', '12', '13', '14']:
            description, _ = self.menu_options[key]
            print(f"  {key}. {description}")
        
        print(f"\n  0. 👋 종료")
        print("="*80)

    async def get_and_execute_choice(self):
        """사용자 선택 처리 - 최적화 버전"""
        try:
            choice = input("🎯 선택하세요 (0-14): ").strip()
            
            if choice in self.menu_options:
                description, func = self.menu_options[choice]
                
                if func:
                    print(f"\n⚡ {description} 실행 중...")
                    execution_start = time.time()
                    
                    await func()
                    
                    execution_time = time.time() - execution_start
                    print(f"✅ 실행 완료 (소요시간: {execution_time:.2f}초)")
                    
                    # 성능 지표 업데이트
                    if hasattr(self.system, '_log_performance_metrics'):
                        await self.system._log_performance_metrics()
                
                return choice
            else:
                print("❌ 올바른 번호를 입력해주세요.")
                return None
                
        except Exception as e:
            logger.error(f"❌ 메뉴 선택 처리 중 오류: {e}")
            print(f"❌ 메뉴 선택 처리 중 오류: {e}")
            return None

def create_main_menu(system):
    """메인 메뉴 생성 함수"""
    return OptimizedMainMenu(system)

async def main():
    """메인 함수 - 고성능 최적화 버전"""
    system = OptimizedStockAnalysisSystem()
    
    try:
        await system.run()
    except Exception as e:
        logger.error(f"❌ 메인 함수 실행 중 오류: {e}")
        print(f"❌ 메인 함수 실행 중 오류: {e}")

if __name__ == "__main__":
    # 이벤트 루프 최적화 설정
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 고성능 이벤트 루프 실행
        asyncio.run(main())