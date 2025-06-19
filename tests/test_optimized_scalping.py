"""
🧪 최적화된 스캘핑 시스템 테스트
- 각 모듈별 단위 테스트
- 통합 시스템 테스트
- 성능 및 API 효율성 테스트
"""

import logging
import time
from datetime import datetime
from typing import Dict, List

from core_trader import CoreTrader
from advanced_scalping_system import AdvancedScalpingSystem
from scalping_modules.atr_analyzer import ATRAnalyzer
from scalping_modules.momentum_scorer import MomentumScorer
from scalping_modules.multi_timeframe_analyzer import MultiTimeframeAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scalping_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class ScalpingSystemTester:
    """스캘핑 시스템 테스터"""
    
    def __init__(self):
        """테스터 초기화"""
        self.trader = None
        self.scalping_system = None
        self.test_results = {}
        
        logger.info("🧪 스캘핑 시스템 테스터 초기화")
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        logger.info("🚀 전체 테스트 시작")
        start_time = time.time()
        
        test_results = {
            'start_time': datetime.now().isoformat(),
            'tests': {}
        }
        
        try:
            # 1. 기본 설정 테스트
            test_results['tests']['setup'] = self.test_basic_setup()
            
            # 2. 개별 모듈 테스트
            test_results['tests']['atr_analyzer'] = self.test_atr_analyzer()
            test_results['tests']['momentum_scorer'] = self.test_momentum_scorer()
            test_results['tests']['multi_timeframe'] = self.test_multi_timeframe_analyzer()
            
            # 3. 통합 시스템 테스트
            test_results['tests']['system_integration'] = self.test_system_integration()
            
            # 4. 성능 테스트
            test_results['tests']['performance'] = self.test_performance()
            
            # 5. API 효율성 테스트
            test_results['tests']['api_efficiency'] = self.test_api_efficiency()
            
            # 전체 결과 계산
            total_duration = time.time() - start_time
            passed_tests = sum(1 for test in test_results['tests'].values() if test.get('passed'))
            total_tests = len(test_results['tests'])
            
            test_results.update({
                'end_time': datetime.now().isoformat(),
                'total_duration': round(total_duration, 2),
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'success_rate': round(passed_tests / total_tests * 100, 1) if total_tests > 0 else 0
            })
            
            logger.info(f"✅ 전체 테스트 완료: {passed_tests}/{total_tests} 성공 "
                       f"({test_results['success_rate']}%)")
            
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 오류: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def test_basic_setup(self) -> Dict:
        """기본 설정 테스트"""
        logger.info("🔧 기본 설정 테스트")
        
        try:
            # CoreTrader 초기화 테스트
            self.trader = CoreTrader()
            if not self.trader:
                return {'passed': False, 'error': 'CoreTrader 초기화 실패'}
            
            # AdvancedScalpingSystem 초기화 테스트
            self.scalping_system = AdvancedScalpingSystem(self.trader)
            if not self.scalping_system:
                return {'passed': False, 'error': 'AdvancedScalpingSystem 초기화 실패'}
            
            # 시스템 상태 확인
            status = self.scalping_system.get_system_status()
            if 'error' in status:
                logger.warning(f"⚠️ 시스템 상태 경고: {status['error']}")
            
            return {
                'passed': True,
                'trader_mode': '모의투자' if self.trader.is_mock else '실전투자',
                'system_status': status
            }
            
        except Exception as e:
            logger.error(f"❌ 기본 설정 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_atr_analyzer(self) -> Dict:
        """ATR 분석기 테스트"""
        logger.info("📊 ATR 분석기 테스트")
        
        try:
            analyzer = ATRAnalyzer()
            
            # 테스트 데이터 생성
            test_prices = [
                10000, 10100, 10050, 10200, 10150,
                10300, 10250, 10400, 10350, 10500,
                10450, 10600, 10550, 10700, 10650
            ]
            
            # ATR 계산 테스트
            atr_result = analyzer.calculate_quick_atr(test_prices)
            if not atr_result:
                return {'passed': False, 'error': 'ATR 계산 실패'}
            
            # 결과 검증
            atr_percentage = atr_result.get('atr_percentage', 0)
            scalping_suitability = atr_result.get('scalping_suitability', 0)
            
            is_valid = (
                0 <= atr_percentage <= 100 and
                0 <= scalping_suitability <= 100
            )
            
            return {
                'passed': is_valid,
                'atr_percentage': atr_percentage,
                'scalping_suitability': scalping_suitability,
                'test_data_points': len(test_prices)
            }
            
        except Exception as e:
            logger.error(f"❌ ATR 분석기 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_momentum_scorer(self) -> Dict:
        """모멘텀 스코어러 테스트"""
        logger.info("⚡ 모멘텀 스코어러 테스트")
        
        try:
            scorer = MomentumScorer()
            
            # 상승 모멘텀 테스트 데이터
            rising_prices = [
                10000, 10050, 10100, 10150, 10200,
                10250, 10300, 10350, 10400, 10450
            ]
            rising_volumes = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
            
            # 하락 모멘텀 테스트 데이터
            falling_prices = [
                10500, 10450, 10400, 10350, 10300,
                10250, 10200, 10150, 10100, 10050
            ]
            falling_volumes = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
            
            # 상승 모멘텀 테스트
            rising_result = scorer.calculate_batch_momentum('TEST_RISING', rising_prices, rising_volumes)
            if not rising_result:
                return {'passed': False, 'error': '상승 모멘텀 계산 실패'}
            
            # 하락 모멘텀 테스트
            falling_result = scorer.calculate_batch_momentum('TEST_FALLING', falling_prices, falling_volumes)
            if not falling_result:
                return {'passed': False, 'error': '하락 모멘텀 계산 실패'}
            
            # 결과 검증
            rising_positive = rising_result.combined_score > 0
            falling_negative = falling_result.combined_score < 0
            
            return {
                'passed': rising_positive and falling_negative,
                'rising_momentum': {
                    'score': rising_result.combined_score,
                    'direction': rising_result.momentum_direction,
                    'strength': rising_result.momentum_strength
                },
                'falling_momentum': {
                    'score': falling_result.combined_score,
                    'direction': falling_result.momentum_direction,
                    'strength': falling_result.momentum_strength
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 모멘텀 스코어러 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_multi_timeframe_analyzer(self) -> Dict:
        """멀티타임프레임 분석기 테스트"""
        logger.info("⏰ 멀티타임프레임 분석기 테스트")
        
        try:
            analyzer = MultiTimeframeAnalyzer()
            
            # 테스트 데이터 (여러 타임프레임)
            timeframe_data = {
                '1m': [10000, 10020, 10010, 10030, 10025],
                '5m': [9900, 9950, 10000, 10050, 10100],
                '15m': [9800, 9850, 9900, 9950, 10000]
            }
            
            # 멀티타임프레임 분석 테스트
            analysis_result = analyzer.analyze_multiple_timeframes('TEST_SYMBOL', timeframe_data)
            
            if not analysis_result:
                return {'passed': False, 'error': '멀티타임프레임 분석 실패'}
            
            # 결과 검증
            has_timeframes = len(analysis_result.timeframe_data) > 0
            has_consensus = analysis_result.consensus_signal is not None
            
            return {
                'passed': has_timeframes and has_consensus,
                'timeframes_analyzed': len(analysis_result.timeframe_data),
                'consensus_signal': analysis_result.consensus_signal,
                'overall_score': analysis_result.overall_score
            }
            
        except Exception as e:
            logger.error(f"❌ 멀티타임프레임 분석기 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_system_integration(self) -> Dict:
        """시스템 통합 테스트"""
        logger.info("🔗 시스템 통합 테스트")
        
        if not self.scalping_system:
            return {'passed': False, 'error': '스캘핑 시스템이 초기화되지 않음'}
        
        try:
            # 빠른 시장 스캔 테스트
            scan_result = self.scalping_system.run_quick_scan()
            if 'error' in scan_result:
                logger.warning(f"⚠️ 스캔 경고: {scan_result['error']}")
            
            # 상위 기회 종목 조회 테스트
            opportunities = self.scalping_system.get_top_opportunities(5)
            
            # 시스템 상태 조회 테스트
            status = self.scalping_system.get_system_status()
            
            return {
                'passed': True,
                'scan_completed': 'error' not in scan_result,
                'opportunities_found': len(opportunities),
                'system_status': status.get('is_running', False),
                'scan_duration': scan_result.get('duration_seconds', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ 시스템 통합 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_performance(self) -> Dict:
        """성능 테스트"""
        logger.info("🚀 성능 테스트")
        
        try:
            performance_results = {}
            
            # 1. ATR 계산 성능
            atr_analyzer = ATRAnalyzer()
            test_prices = [10000 + i * 10 for i in range(100)]  # 100개 데이터
            
            start_time = time.time()
            for _ in range(100):  # 100회 반복
                atr_analyzer.calculate_quick_atr(test_prices)
            atr_duration = time.time() - start_time
            
            performance_results['atr_calculation'] = {
                'iterations': 100,
                'duration_seconds': round(atr_duration, 3),
                'avg_per_calculation': round(atr_duration / 100 * 1000, 2)  # ms
            }
            
            # 2. 모멘텀 계산 성능
            momentum_scorer = MomentumScorer()
            test_volumes = [1000 + i * 10 for i in range(100)]
            
            start_time = time.time()
            for _ in range(50):  # 50회 반복
                momentum_scorer.calculate_batch_momentum('PERF_TEST', test_prices, test_volumes)
            momentum_duration = time.time() - start_time
            
            performance_results['momentum_calculation'] = {
                'iterations': 50,
                'duration_seconds': round(momentum_duration, 3),
                'avg_per_calculation': round(momentum_duration / 50 * 1000, 2)  # ms
            }
            
            # 성능 기준 검증 (각 계산이 10ms 이하)
            atr_fast_enough = performance_results['atr_calculation']['avg_per_calculation'] < 10
            momentum_fast_enough = performance_results['momentum_calculation']['avg_per_calculation'] < 50
            
            return {
                'passed': atr_fast_enough and momentum_fast_enough,
                'results': performance_results,
                'performance_acceptable': atr_fast_enough and momentum_fast_enough
            }
            
        except Exception as e:
            logger.error(f"❌ 성능 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_api_efficiency(self) -> Dict:
        """API 효율성 테스트"""
        logger.info("📡 API 효율성 테스트")
        
        if not self.trader:
            return {'passed': False, 'error': 'Trader가 초기화되지 않음'}
        
        try:
            # API 호출 카운터 초기 상태
            initial_calls = self.trader.daily_counter.get_remaining_calls()
            
            # 여러 API 호출 시뮬레이션
            test_symbols = ['005930', '000660', '035420']  # 3개 종목
            api_test_results = []
            
            for symbol in test_symbols:
                start_time = time.time()
                
                # 현재가 조회 (API 호출)
                price_data = self.trader.get_current_price(symbol)
                
                call_duration = time.time() - start_time
                
                api_test_results.append({
                    'symbol': symbol,
                    'success': price_data is not None,
                    'duration_ms': round(call_duration * 1000, 2)
                })
                
                # API 호출 간격 (Rate Limiting 고려)
                time.sleep(0.1)
            
            # 최종 API 호출 수
            final_calls = self.trader.daily_counter.get_remaining_calls()
            
            # 결과 분석
            successful_calls = sum(1 for result in api_test_results if result['success'])
            avg_response_time = sum(result['duration_ms'] for result in api_test_results) / len(api_test_results)
            
            # 효율성 기준: 90% 이상 성공, 평균 응답시간 3초 이하
            efficiency_good = (
                successful_calls / len(api_test_results) >= 0.9 and
                avg_response_time <= 3000
            )
            
            return {
                'passed': efficiency_good,
                'api_calls_made': len(api_test_results),
                'successful_calls': successful_calls,
                'success_rate': round(successful_calls / len(api_test_results) * 100, 1),
                'avg_response_time_ms': round(avg_response_time, 1),
                'calls_remaining_change': initial_calls - final_calls if isinstance(initial_calls, int) and isinstance(final_calls, int) else 'N/A',
                'individual_results': api_test_results
            }
            
        except Exception as e:
            logger.error(f"❌ API 효율성 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}
    
    def test_individual_symbol_analysis(self, symbol: str = '005930') -> Dict:
        """개별 종목 분석 테스트"""
        logger.info(f"🔍 {symbol} 개별 분석 테스트")
        
        if not self.scalping_system:
            return {'passed': False, 'error': '스캘핑 시스템이 초기화되지 않음'}
        
        try:
            # 개별 종목 분석 수행
            analysis_result = self.scalping_system.analyze_symbol(symbol)
            
            if 'error' in analysis_result:
                return {'passed': False, 'error': analysis_result['error']}
            
            # 분석 결과 검증
            has_price = 'current_price' in analysis_result
            has_analysis = 'analysis' in analysis_result
            has_timestamp = 'timestamp' in analysis_result
            
            analysis_complete = has_price and has_analysis and has_timestamp
            
            return {
                'passed': analysis_complete,
                'symbol': symbol,
                'current_price': analysis_result.get('current_price'),
                'analysis_components': list(analysis_result.get('analysis', {}).keys()),
                'timestamp': analysis_result.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} 개별 분석 테스트 실패: {e}")
            return {'passed': False, 'error': str(e)}


def run_quick_test():
    """빠른 테스트 실행"""
    print("⚡ 빠른 스캘핑 시스템 테스트")
    print("=" * 50)
    
    tester = ScalpingSystemTester()
    
    # 기본 설정만 테스트
    setup_result = tester.test_basic_setup()
    print(f"기본 설정: {'✅ 성공' if setup_result.get('passed') else '❌ 실패'}")
    
    if setup_result.get('passed'):
        # 빠른 스캔 테스트
        try:
            scan_result = tester.scalping_system.run_quick_scan()
            print(f"시장 스캔: {'✅ 성공' if 'error' not in scan_result else '❌ 실패'}")
            
            if 'error' not in scan_result:
                print(f"  - 분석 시간: {scan_result.get('duration_seconds', 0):.1f}초")
                print(f"  - 기회 종목: {len(scan_result.get('top_opportunities', []))}개")
                print(f"  - 시장 분위기: {scan_result.get('market_analysis', {}).get('mood', 'Unknown')}")
            
        except Exception as e:
            print(f"시장 스캔: ❌ 실패 ({e})")
    
    print("=" * 50)
    print("🏁 빠른 테스트 완료")


def run_full_test():
    """전체 테스트 실행"""
    print("🧪 전체 스캘핑 시스템 테스트 시작")
    print("=" * 60)
    
    tester = ScalpingSystemTester()
    results = tester.run_all_tests()
    
    # 결과 출력
    print("\n📊 테스트 결과 요약:")
    print(f"전체 테스트: {results.get('total_tests', 0)}개")
    print(f"성공한 테스트: {results.get('passed_tests', 0)}개")
    print(f"성공률: {results.get('success_rate', 0)}%")
    print(f"총 소요 시간: {results.get('total_duration', 0)}초")
    
    print("\n📋 세부 테스트 결과:")
    for test_name, test_result in results.get('tests', {}).items():
        status = "✅" if test_result.get('passed') else "❌"
        print(f"  {status} {test_name}: {'성공' if test_result.get('passed') else '실패'}")
        
        if not test_result.get('passed') and 'error' in test_result:
            print(f"    오류: {test_result['error']}")
    
    print("=" * 60)
    print("🏁 전체 테스트 완료")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_test()
    else:
        run_full_test() 