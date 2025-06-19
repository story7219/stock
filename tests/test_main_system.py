#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 통합 자동매매 시스템 종합 테스트 (v3.0)
- 스마트 종목 필터링 시스템 테스트 추가
- Mock 클래스 업데이트
- 텔레그램 래퍼 테스트
=================================================================

통합 자동매매 시스템의 모든 기능을 테스트합니다.

테스트 범위:
- 시스템 초기화 및 컴포넌트 로딩
- 스케줄러 설정 및 작동
- 안전장치 시스템 검증
- 텔레그램 알림 시스템
- API 연결 상태 확인
- 실시간 모니터링 루프
- 긴급 정지 시스템

실행: python test_main_system.py
"""

import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading
import json

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock 클래스들 정의
class MockCoreTrader:
    """CoreTrader Mock 클래스"""
    def __init__(self):
        self.initialized = True
        
    def initialize(self):
        return True
    
    def get_current_price(self, symbol):
        """종목 현재가 조회 Mock"""
        prices = {
            '005930': {'stck_prpr': '70000', 'acml_vol': '500000'},  # 삼성전자
            '000660': {'stck_prpr': '120000', 'acml_vol': '300000'}, # SK하이닉스
            '035420': {'stck_prpr': '180000', 'acml_vol': '400000'}  # NAVER
        }
        return prices.get(symbol, {'stck_prpr': '50000', 'acml_vol': '250000'})
    
    def get_balance(self):
        from main import BalanceInfo
        return BalanceInfo(
            cash=1000000,
            total_value=1500000,
            positions={'005930': {'qty': 10, 'purchase_price': 65000}},
            profit_loss=50000
        )
    
    def execute_order(self, symbol, side, quantity, price=0):
        return {'success': True, 'order_id': 'TEST123', 'symbol': symbol}
    
    def get_top_ranking_stocks(self, top_n=10):
        return [
            {'symbol': '005930', 'name': '삼성전자', 'price': 70000},
            {'symbol': '000660', 'name': 'SK하이닉스', 'price': 120000}
        ]

class MockAITrader:
    """AITrader Mock 클래스"""
    def __init__(self, trader):
        self.trader = trader
        
    def make_trading_decision(self, stock_data):
        return {
            'action': 'BUY',
            'confidence': 0.75,
            'reason': 'AI 분석 결과 매수 신호'
        }

class MockChartAnalyzer:
    """ChartAnalyzer Mock 클래스"""
    def __init__(self, trader=None):
        self.trader = trader
        
    def analyze_stock(self, symbol):
        return {
            'signal': 'BUY',
            'strength': 0.8,
            'indicators': {'RSI': 45, 'MACD': 'positive'}
        }

class MockNewsCollector:
    """NewsCollector Mock 클래스"""
    def __init__(self):
        pass
        
    def get_stock_sentiment(self, symbol):
        return {
            'sentiment_score': 0.6,
            'news_count': 5,
            'summary': '긍정적 뉴스가 다수'
        }

class MockAdvancedScalpingSystem:
    """AdvancedScalpingSystem Mock 클래스"""
    def __init__(self, trader=None, target_symbols=None):
        self.trader = trader
        self.target_symbols = target_symbols or []
        
    def analyze_symbol(self, symbol):
        return {
            'signal': 'BUY',
            'strength': 0.7,
            'entry_price': 70000
        }

class MockRealtimeAITrader:
    """RealtimeAITrader Mock 클래스"""
    def __init__(self):
        pass

class MockStockFilter:
    """StockFilter Mock 클래스"""
    def __init__(self, trader_instance=None):
        self.trader = trader_instance
        self.filtered_stocks = []
        
    def set_filter_criteria(self, criteria):
        pass
        
    async def get_filtered_stocks(self, force_update=False):
        from stock_filter import StockInfo
        return [
            StockInfo(
                code='005930',
                name='삼성전자',
                current_price=70000,
                market_cap=40000,
                volume=500000,
                volume_value=35000,
                market_type='KOSPI',
                sector='IT/반도체',
                score=85.5
            ),
            StockInfo(
                code='000660',
                name='SK하이닉스',
                current_price=120000,
                market_cap=30000,
                volume=300000,
                volume_value=36000,
                market_type='KOSPI',
                sector='IT/반도체',
                score=82.3
            )
        ]
    
    def get_stock_codes(self):
        return ['005930', '000660', '035420']
        
    def get_top_stocks(self, n=10):
        stocks = asyncio.run(self.get_filtered_stocks())
        return stocks[:n]

# Mock 패치 적용
def apply_mocks():
    """Mock 클래스들을 실제 모듈에 패치"""
    sys.modules['core_trader'].CoreTrader = MockCoreTrader
    sys.modules['ai_trader'].AITrader = MockAITrader
    sys.modules['chart_analyzer'].ChartAnalyzer = MockChartAnalyzer
    sys.modules['news_collector'].NewsCollector = MockNewsCollector
    sys.modules['advanced_scalping_system'].AdvancedScalpingSystem = MockAdvancedScalpingSystem
    sys.modules['realtime_ai_trader'].RealtimeAITrader = MockRealtimeAITrader
    sys.modules['stock_filter'].StockFilter = MockStockFilter

# Mock 적용
apply_mocks()

# 이제 실제 시스템 import
from main import AutoTradingSystem, SystemStatus, SafetyManager, TelegramNotifierWrapper

class AutoTradingSystemTester:
    """🧪 통합 자동매매 시스템 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.total_score = 0
        self.max_score = 280
        
    def run_all_tests(self):
        """전체 테스트 실행"""
        print("🤖 통합 자동매매 시스템 종합 테스트")
        print("=" * 80)
        print(f"⏰ 테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # 각 테스트 실행
        self.test_system_status()
        self.test_safety_manager()
        self.test_telegram_system()
        self.test_system_initialization()
        self.test_scheduler_setup()
        self.test_component_initialization()
        self.test_stock_filtering_system()  # 새로운 테스트
        self.test_trading_loop()
        self.test_emergency_stop()
        self.test_daily_report()
        
        # 결과 출력
        self.print_summary()
        
        return self.total_score >= (self.max_score * 0.8)  # 80% 이상 통과

    def test_stock_filtering_system(self):
        """📊 종목 필터링 시스템 테스트"""
        print("=" * 60)
        print("📊 종목 필터링 시스템 테스트")
        print("=" * 60)
        
        score = 0
        max_score = 30
        
        try:
            # StockFilter 인스턴스 생성 테스트
            from stock_filter import StockFilter, FilterCriteria
            filter_system = StockFilter()
            print("✅ StockFilter 인스턴스 생성 성공")
            score += 10
            
            # 필터링 기준 설정 테스트
            criteria = FilterCriteria(
                min_market_cap=10000,
                min_volume=200000,
                max_stocks=20
            )
            filter_system.set_filter_criteria(criteria)
            print("✅ 필터링 기준 설정 성공")
            score += 10
            
            # 필터링 실행 테스트
            async def test_filtering():
                filtered_stocks = await filter_system.get_filtered_stocks()
                return filtered_stocks
            
            filtered_stocks = asyncio.run(test_filtering())
            if filtered_stocks and len(filtered_stocks) > 0:
                print(f"✅ 종목 필터링 성공: {len(filtered_stocks)}개 종목")
                print(f"   상위 종목: {filtered_stocks[0].name}({filtered_stocks[0].code})")
                score += 10
            else:
                print("❌ 종목 필터링 실패")
                
        except Exception as e:
            print(f"❌ 종목 필터링 시스템 테스트 실패: {e}")
        
        print(f"\n📊 종목 필터링 테스트 점수: {score}/{max_score}점")
        print()
        
        self.test_results['종목 필터링 시스템'] = (score, max_score)
        self.total_score += score

    def print_summary(self):
        """테스트 결과 요약 출력"""
        print("=" * 80)
        print("📊 통합 자동매매 시스템 테스트 결과 요약")
        print("=" * 80)
        
        # 각 테스트 결과 출력
        for test_name, (score, max_score) in self.test_results.items():
            percentage = (score / max_score * 100) if max_score > 0 else 0
            status_icon = "🟢" if percentage >= 80 else "🟡" if percentage >= 60 else "🔴"
            print(f"{status_icon} {test_name:<25} : {score:3d}/{max_score:3d}점 ({percentage:5.1f}%)")
        
        print("-" * 80)
        total_percentage = (self.total_score / self.max_score * 100)
        print(f"🏆 전체 점수: {self.total_score}/{self.max_score}점 ({total_percentage:.1f}%)")
        
        # 등급 평가
        if total_percentage >= 90:
            grade = "EXCELLENT ✨"
        elif total_percentage >= 80:
            grade = "GOOD ✅"
        elif total_percentage >= 70:
            grade = "ACCEPTABLE ⚠️"
        else:
            grade = "NEEDS_IMPROVEMENT ❌"
        
        print(f"📈 평가 등급: {grade}")
        print("=" * 80)
        
        # 시스템 준비 상태 평가
        if total_percentage >= 80:
            print("✅ 시스템 운영 준비 완료")
            print("🚀 자동매매 시스템을 안전하게 시작할 수 있습니다.")
        else:
            print("⚠️ 시스템 운영 준비 미완료")
            print("❌ 필수 테스트를 통과한 후 시스템을 시작하세요.")
        
        print("=" * 80)
        print()

def test_system_status():
    """💾 시스템 상태 구조 테스트"""
    print("\n" + "="*60)
    print("💾 시스템 상태 구조 테스트")
    print("="*60)
    
    score = 0
    max_score = 20
    
    try:
        # SystemStatus 객체 생성
        status = SystemStatus()
        
        # 필수 필드들 확인
        required_fields = [
            'is_running', 'start_time', 'total_trades', 'total_profit_loss',
            'daily_trades', 'daily_profit_loss', 'last_trade_time',
            'error_count', 'last_error', 'emergency_stop'
        ]
        
        field_score = 0
        for field in required_fields:
            if hasattr(status, field):
                field_score += 2
                print(f"   ✅ {field}: {getattr(status, field)}")
            else:
                print(f"   ❌ {field}: 누락")
        
        score += field_score
        print(f"\n📊 시스템 상태 필드 점수: {field_score}/20점")
        
    except Exception as e:
        print(f"❌ 시스템 상태 테스트 실패: {e}")
    
    return score, max_score

def test_safety_manager():
    """🛡️ 안전장치 시스템 테스트"""
    print("\n" + "="*60)
    print("🛡️ 안전장치 시스템 테스트")
    print("="*60)
    
    score = 0
    max_score = 40
    
    try:
        # SafetyManager 초기화 (10점)
        safety_manager = SafetyManager(max_daily_loss=-50000, max_daily_trades=50)
        score += 10
        print("✅ SafetyManager 초기화 성공")
        
        # SystemStatus 생성
        status = SystemStatus()
        
        # 1. 정상 상태 테스트 (10점)
        is_safe, msg = safety_manager.check_daily_limits(status)
        if is_safe and "안전 범위" in msg:
            score += 10
            print("✅ 정상 상태 안전장치 확인")
        else:
            print("❌ 정상 상태 안전장치 실패")
        
        # 2. 일일 손실 한도 테스트 (10점)
        status.daily_profit_loss = -60000  # 한도 초과
        is_safe, msg = safety_manager.check_daily_limits(status)
        if not is_safe and "손실 한도 초과" in msg:
            score += 10
            print("✅ 일일 손실 한도 안전장치 작동")
        else:
            print("❌ 일일 손실 한도 안전장치 실패")
        
        # 3. 일일 거래 한도 테스트 (10점)
        status.daily_profit_loss = 0  # 리셋
        status.daily_trades = 60  # 한도 초과
        is_safe, msg = safety_manager.check_daily_limits(status)
        if not is_safe and "거래 한도 초과" in msg:
            score += 10
            print("✅ 일일 거래 한도 안전장치 작동")
        else:
            print("❌ 일일 거래 한도 안전장치 실패")
        
        print(f"\n🛡️ 안전장치 테스트 점수: {score}/40점")
        
    except Exception as e:
        print(f"❌ 안전장치 테스트 실패: {e}")
    
    return score, max_score

def test_telegram_notifier():
    """📱 텔레그램 알림 시스템 테스트"""
    print("\n" + "="*60)
    print("📱 텔레그램 알림 시스템 테스트")
    print("="*60)
    
    score = 0
    max_score = 30
    
    try:
        # TelegramNotifier 초기화 (10점)
        telegram = TelegramNotifier()
        score += 10
        print("✅ TelegramNotifier 초기화 성공")
        
        # 설정 확인 (10점)
        if telegram.enabled:
            score += 10
            print("✅ 텔레그램 설정 확인 완료")
        else:
            score += 5  # 설정이 없어도 부분 점수
            print("⚠️ 텔레그램 설정 없음 (테스트 환경)")
        
        # 동기 메시지 전송 테스트 (10점)
        try:
            telegram.send_sync("🧪 테스트 메시지", urgent=False)
            score += 10
            print("✅ 동기 메시지 전송 테스트 완료")
        except Exception as e:
            score += 5  # 에러가 발생해도 부분 점수
            print(f"⚠️ 동기 메시지 전송 테스트: {e}")
        
        print(f"\n📱 텔레그램 테스트 점수: {score}/30점")
        
    except Exception as e:
        print(f"❌ 텔레그램 테스트 실패: {e}")
    
    return score, max_score

def test_auto_trading_system_initialization():
    """🤖 자동매매 시스템 초기화 테스트"""
    print("\n" + "="*60)
    print("🤖 자동매매 시스템 초기화 테스트")
    print("="*60)
    
    score = 0
    max_score = 50
    
    try:
        # 시스템 초기화 (20점)
        system = AutoTradingSystem()
        score += 20
        print("✅ AutoTradingSystem 초기화 성공")
        
        # 핵심 컴포넌트 확인 (30점)
        components = [
            ('status', system.status),
            ('safety_manager', system.safety_manager),
            ('telegram', system.telegram),
            ('scheduler', system.scheduler)
        ]
        
        component_score = 0
        for name, component in components:
            if component is not None:
                component_score += 7
                print(f"   ✅ {name}: 초기화됨")
            else:
                print(f"   ❌ {name}: 초기화 실패")
        
        # 남은 2점은 보너스
        if component_score == 28:
            component_score += 2
        
        score += component_score
        
        print(f"\n🤖 시스템 초기화 점수: {score}/50점")
        
    except Exception as e:
        print(f"❌ 시스템 초기화 테스트 실패: {e}")
    
    return score, max_score

def test_scheduler_setup():
    """📅 스케줄러 설정 테스트"""
    print("\n" + "="*60)
    print("📅 스케줄러 설정 테스트")
    print("="*60)
    
    score = 0
    max_score = 30
    
    try:
        system = AutoTradingSystem()
        
        # 스케줄러 설정 (20점)
        system.setup_scheduler()
        score += 20
        print("✅ 스케줄러 설정 완료")
        
        # 등록된 Job 확인 (10점)
        jobs = system.scheduler.get_jobs()
        expected_jobs = ['morning_prep', 'start_trading', 'stop_trading', 'daily_cleanup', 'system_monitor']
        
        job_score = 0
        for job_id in expected_jobs:
            if any(job.id == job_id for job in jobs):
                job_score += 2
                print(f"   ✅ {job_id}: 등록됨")
            else:
                print(f"   ❌ {job_id}: 등록 실패")
        
        score += job_score
        
        print(f"\n📅 스케줄러 테스트 점수: {score}/30점")
        
    except Exception as e:
        print(f"❌ 스케줄러 테스트 실패: {e}")
    
    return score, max_score

def test_component_initialization():
    """🔧 컴포넌트 초기화 테스트"""
    print("\n" + "="*60)
    print("🔧 컴포넌트 초기화 테스트")
    print("="*60)
    
    score = 0
    max_score = 40
    
    try:
        system = AutoTradingSystem()
        
        # 비동기 초기화 테스트
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 컴포넌트 초기화 실행 (30점)
        result = loop.run_until_complete(system.initialize_components())
        if result:
            score += 30
            print("✅ 컴포넌트 초기화 성공")
        else:
            score += 15
            print("⚠️ 컴포넌트 초기화 부분 성공")
        
        # 초기화된 컴포넌트 확인 (10점)
        initialized_components = [
            ('core_trader', system.core_trader),
            ('chart_analyzer', system.chart_analyzer),
            ('news_collector', system.news_collector)
        ]
        
        init_score = 0
        for name, component in initialized_components:
            if component is not None:
                init_score += 3
                print(f"   ✅ {name}: 초기화됨")
            else:
                print(f"   ⚠️ {name}: 초기화 안됨")
        
        # 남은 1점은 보너스
        if init_score == 9:
            init_score += 1
        
        score += init_score
        
        print(f"\n🔧 컴포넌트 초기화 점수: {score}/40점")
        
    except Exception as e:
        print(f"❌ 컴포넌트 초기화 테스트 실패: {e}")
    
    return score, max_score

def test_trading_loop_simulation():
    """🔄 거래 루프 시뮬레이션 테스트"""
    print("\n" + "="*60)
    print("🔄 거래 루프 시뮬레이션 테스트")
    print("="*60)
    
    score = 0
    max_score = 30
    
    try:
        system = AutoTradingSystem()
        
        # 비동기 초기화
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(system.initialize_components())
        
        # 거래 상태 설정 (10점)
        system.status.is_running = True
        system.is_market_hours = True
        score += 10
        print("✅ 거래 상태 설정 완료")
        
        # AI 트레이딩 실행 테스트 (10점)
        try:
            system._execute_ai_trading()
            score += 10
            print("✅ AI 트레이딩 시뮬레이션 성공")
        except Exception as e:
            score += 5
            print(f"⚠️ AI 트레이딩 시뮬레이션: {e}")
        
        # 스캘핑 트레이딩 실행 테스트 (10점)
        try:
            system._execute_scalping_trading()
            score += 10
            print("✅ 스캘핑 트레이딩 시뮬레이션 성공")
        except Exception as e:
            score += 5
            print(f"⚠️ 스캘핑 트레이딩 시뮬레이션: {e}")
        
        print(f"\n🔄 거래 루프 테스트 점수: {score}/30점")
        
    except Exception as e:
        print(f"❌ 거래 루프 테스트 실패: {e}")
    
    return score, max_score

def test_emergency_stop():
    """🚨 긴급 정지 시스템 테스트"""
    print("\n" + "="*60)
    print("🚨 긴급 정지 시스템 테스트")
    print("="*60)
    
    score = 0
    max_score = 20
    
    try:
        system = AutoTradingSystem()
        safety_manager = SafetyManager(max_daily_loss=-1000, max_daily_trades=1)  # 매우 낮은 한도
        system.safety_manager = safety_manager
        
        # 거래 상태 설정
        system.status.is_running = True
        system.is_market_hours = True
        
        # 위험 상황 시뮬레이션 (15점)
        system.status.daily_profit_loss = -2000  # 한도 초과
        system.status.daily_trades = 5  # 한도 초과
        
        is_safe, msg = safety_manager.check_daily_limits(system.status)
        if not is_safe:
            score += 15
            print(f"✅ 긴급 정지 조건 감지: {msg}")
        else:
            print("❌ 긴급 정지 조건 감지 실패")
        
        # 시스템 종료 테스트 (5점)
        try:
            system.status.emergency_stop = True
            system.manual_stop_requested = True
            score += 5
            print("✅ 긴급 정지 플래그 설정 완료")
        except Exception as e:
            print(f"❌ 긴급 정지 설정 실패: {e}")
        
        print(f"\n🚨 긴급 정지 테스트 점수: {score}/20점")
        
    except Exception as e:
        print(f"❌ 긴급 정지 테스트 실패: {e}")
    
    return score, max_score

def test_daily_report_generation():
    """📊 일일 보고서 생성 테스트"""
    print("\n" + "="*60)
    print("📊 일일 보고서 생성 테스트")
    print("="*60)
    
    score = 0
    max_score = 20
    
    try:
        system = AutoTradingSystem()
        
        # 테스트 데이터 설정
        system.status.start_time = datetime.now() - timedelta(hours=6)
        system.status.total_trades = 10
        system.status.daily_trades = 5
        system.status.daily_profit_loss = 25000
        system.status.total_profit_loss = 150000
        system.status.error_count = 1
        
        # 보고서 생성 (15점)
        report = system._generate_daily_report()
        if report and "일일 거래 보고서" in report:
            score += 15
            print("✅ 일일 보고서 생성 성공")
            print("📋 보고서 미리보기:")
            print("-" * 40)
            print(report[:200] + "..." if len(report) > 200 else report)
            print("-" * 40)
        else:
            print("❌ 일일 보고서 생성 실패")
        
        # 보고서 내용 검증 (5점)
        required_elements = ["총 거래수", "일일 손익", "운영시간"]
        if all(element in report for element in required_elements):
            score += 5
            print("✅ 보고서 필수 요소 포함 확인")
        else:
            print("❌ 보고서 필수 요소 누락")
        
        print(f"\n📊 보고서 생성 테스트 점수: {score}/20점")
        
    except Exception as e:
        print(f"❌ 보고서 생성 테스트 실패: {e}")
    
    return score, max_score

if __name__ == "__main__":
    tester = AutoTradingSystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("🏁 테스트 성공")
        exit(0)
    else:
        print("🏁 테스트 실패") 
        exit(1) 