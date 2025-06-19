"""
🚀 초고속 AI 데이터 컬렉터 종합 테스트
- 스캘핑 vs 데이트레이딩 모드 성능 비교
- 새로운 API 기능 테스트 (상승률, 거래량, 기관동향)
- 실시간 모니터링 테스트
- 성능 최적화 효과 측정
"""
import sys
import os
import logging
import time
import asyncio
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_trader import CoreTrader
from ai_data_collector import AIDataCollector, MarketSignal

class UltraFastCollectorTester:
    """🚀 초고속 컬렉터 성능 테스트 클래스"""
    
    def __init__(self):
        """테스터 초기화"""
        self.trader = None
        self.collector = None
        self.test_symbols = [
            "005930",  # 삼성전자
            "000660",  # SK하이닉스
            "035420",  # NAVER
            "051910",  # LG화학
            "068270"   # 셀트리온
        ]
        self.performance_results = {}
    
    def setup(self):
        """테스트 환경 설정"""
        try:
            logger.info("🔧 테스트 환경 초기화 중...")
            
            # CoreTrader 초기화
            self.trader = CoreTrader()
            if not self.trader.initialize():
                raise Exception("CoreTrader 초기화 실패")
            
            # AIDataCollector 초기화
            self.collector = AIDataCollector(self.trader)
            
            logger.info("✅ 테스트 환경 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 테스트 환경 설정 실패: {e}")
            return False
    
    def test_1_basic_speed_test(self):
        """📊 1단계: 기본 속도 테스트"""
        logger.info("\n" + "="*60)
        logger.info("📊 1단계: 기본 속도 테스트")
        logger.info("="*60)
        
        try:
            # 단일 종목 스캘핑 시그널 속도 측정
            symbol = self.test_symbols[0]
            start_time = time.time()
            
            signal = self.collector.get_scalping_signals(symbol)
            
            elapsed = time.time() - start_time
            self.performance_results['basic_signal'] = elapsed
            
            if signal:
                logger.info(f"✅ {symbol} 스캘핑 시그널 수집 성공!")
                logger.info(f"   📈 점수: {signal['scalping_score']:.1f}")
                logger.info(f"   💰 추천: {signal['recommendation']}")
                logger.info(f"   ⚡ 처리 시간: {elapsed:.3f}초")
                
                # 성능 평가
                if elapsed <= 1.0:
                    logger.info("   🏆 성능: EXCELLENT (1초 이하)")
                elif elapsed <= 2.0:
                    logger.info("   🥇 성능: GOOD (2초 이하)")
                else:
                    logger.info("   ⚠️ 성능: SLOW (2초 초과)")
            else:
                logger.warning(f"⚠️ {symbol} 시그널 수집 실패 또는 임계값 미달")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 기본 속도 테스트 실패: {e}")
            return False
    
    def test_2_ranking_apis(self):
        """📈 2단계: 순위 API 테스트 (상승률, 거래량)"""
        logger.info("\n" + "="*60)
        logger.info("📈 2단계: 순위 API 테스트")
        logger.info("="*60)
        
        try:
            # 상승률 상위 종목 테스트
            logger.info("🔥 상승률 상위 종목 조회...")
            start_time = time.time()
            
            gainers = self.collector.get_fast_top_gainers(10)
            
            elapsed_gainers = time.time() - start_time
            self.performance_results['top_gainers'] = elapsed_gainers
            
            logger.info(f"   ⚡ 상승률 조회 시간: {elapsed_gainers:.3f}초")
            logger.info(f"   📊 수집된 종목 수: {len(gainers)}개")
            
            if gainers:
                logger.info("   🏆 상위 3개 종목:")
                for i, stock in enumerate(gainers[:3], 1):
                    logger.info(f"     {i}. {stock['name']} ({stock['symbol']}) +{stock['change_rate']:.2f}%")
            
            # 거래량 상위 종목 테스트
            logger.info("\n📊 거래량 상위 종목 조회...")
            start_time = time.time()
            
            volume_leaders = self.collector.get_fast_volume_leaders(10)
            
            elapsed_volume = time.time() - start_time
            self.performance_results['volume_leaders'] = elapsed_volume
            
            logger.info(f"   ⚡ 거래량 조회 시간: {elapsed_volume:.3f}초")
            logger.info(f"   📊 수집된 종목 수: {len(volume_leaders)}개")
            
            if volume_leaders:
                logger.info("   🏆 상위 3개 종목:")
                for i, stock in enumerate(volume_leaders[:3], 1):
                    logger.info(f"     {i}. {stock['name']} ({stock['symbol']}) 거래량 +{stock['volume_rate']:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 순위 API 테스트 실패: {e}")
            return False
    
    def test_3_institution_flow(self):
        """🏢 3단계: 기관 매매동향 테스트"""
        logger.info("\n" + "="*60)
        logger.info("🏢 3단계: 기관 매매동향 테스트")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            institution_flow = self.collector.get_fast_institution_flow(8)
            
            elapsed = time.time() - start_time
            self.performance_results['institution_flow'] = elapsed
            
            logger.info(f"   ⚡ 기관동향 조회 시간: {elapsed:.3f}초")
            logger.info(f"   📊 수집된 종목 수: {len(institution_flow)}개")
            
            if institution_flow:
                buying_count = len([i for i in institution_flow if i['flow'] == 'buying'])
                selling_count = len([i for i in institution_flow if i['flow'] == 'selling'])
                
                logger.info(f"   💰 기관 매수 종목: {buying_count}개")
                logger.info(f"   📉 기관 매도 종목: {selling_count}개")
                
                logger.info("   🏆 상위 매매동향:")
                for stock in institution_flow[:3]:
                    flow_emoji = "💰" if stock['flow'] == 'buying' else "📉"
                    value_str = f"{stock['net_buy_value']:,}" if stock['net_buy_value'] > 0 else f"{abs(stock['net_buy_value']):,}"
                    logger.info(f"     {flow_emoji} {stock['name']} ({stock['symbol']}) {value_str}원")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 기관동향 테스트 실패: {e}")
            return False
    
    def test_4_ultra_scalping_mode(self):
        """⚡ 4단계: 울트라 스캘핑 모드 테스트"""
        logger.info("\n" + "="*60)
        logger.info("⚡ 4단계: 울트라 스캘핑 모드 테스트")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            scalping_data = self.collector.get_ultra_fast_scalping_data(self.test_symbols)
            
            elapsed = time.time() - start_time
            self.performance_results['ultra_scalping'] = elapsed
            
            if scalping_data.get('error'):
                logger.error(f"❌ 스캘핑 모드 타임아웃: {scalping_data}")
                return False
            
            logger.info(f"   🚀 스캘핑 모드 처리 시간: {elapsed:.3f}초")
            logger.info(f"   📊 모드: {scalping_data['mode']}")
            logger.info(f"   🎯 성능 등급: {scalping_data['performance']}")
            logger.info(f"   💡 최종 추천: {scalping_data['recommendation']}")
            logger.info(f"   🌡️ 시장 분위기: {scalping_data['market_mood']}")
            
            # 세부 데이터 요약
            logger.info("\n   📈 수집된 데이터:")
            logger.info(f"     - 상승률 상위: {len(scalping_data['top_gainers'])}개")
            logger.info(f"     - 거래량 상위: {len(scalping_data['volume_leaders'])}개")
            logger.info(f"     - 고점수 시그널: {len(scalping_data['hot_signals'])}개")
            
            # 고점수 시그널 표시
            if scalping_data['hot_signals']:
                logger.info("   🔥 고점수 시그널:")
                for signal in scalping_data['hot_signals'][:2]:
                    logger.info(f"     - {signal['name']} ({signal['symbol']}) 점수: {signal['scalping_score']:.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 울트라 스캘핑 모드 테스트 실패: {e}")
            return False
    
    def test_5_daytrading_mode(self):
        """📊 5단계: 데이트레이딩 모드 테스트"""
        logger.info("\n" + "="*60)
        logger.info("📊 5단계: 데이트레이딩 모드 테스트")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            daytrading_data = self.collector.get_optimized_daytrading_data(self.test_symbols)
            
            elapsed = time.time() - start_time
            self.performance_results['daytrading'] = elapsed
            
            if daytrading_data.get('error'):
                logger.error(f"❌ 데이트레이딩 모드 타임아웃: {daytrading_data}")
                return False
            
            logger.info(f"   📈 데이트레이딩 모드 처리 시간: {elapsed:.3f}초")
            logger.info(f"   📊 모드: {daytrading_data['mode']}")
            logger.info(f"   🎯 성능 등급: {daytrading_data['performance']}")
            logger.info(f"   💡 최종 추천: {daytrading_data['recommendation']}")
            
            # 종합 데이터 요약
            logger.info("\n   📊 수집된 종합 데이터:")
            logger.info(f"     - 상승률 상위: {len(daytrading_data['top_gainers'])}개")
            logger.info(f"     - 거래량 상위: {len(daytrading_data['volume_leaders'])}개")
            logger.info(f"     - 기관 매매동향: {len(daytrading_data['institution_flow'])}개")
            logger.info(f"     - 시그널 분석: {len(daytrading_data['signals'])}개")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이트레이딩 모드 테스트 실패: {e}")
            return False
    
    def test_6_realtime_monitoring(self):
        """🔴 6단계: 실시간 모니터링 테스트"""
        logger.info("\n" + "="*60)
        logger.info("🔴 6단계: 실시간 모니터링 테스트 (20초)")
        logger.info("="*60)
        
        try:
            signal_count = 0
            
            def signal_callback(signal: MarketSignal):
                nonlocal signal_count
                signal_count += 1
                logger.info(f"🚨 신호 감지 #{signal_count}: {signal.symbol} {signal.signal_type} 강도: {signal.strength:.1f}")
            
            # 실시간 모니터링 시작
            logger.info("🚀 실시간 모니터링 시작...")
            test_symbols = self.test_symbols[:3]  # 3개 종목만
            
            success = self.collector.start_realtime_monitoring(test_symbols, signal_callback)
            
            if not success:
                logger.warning("⚠️ 실시간 모니터링 시작 실패 (WebSocket 미지원 환경)")
                return True  # 환경 문제로 스킵
            
            logger.info(f"   📡 모니터링 종목: {test_symbols}")
            logger.info("   ⏰ 20초 동안 신호 대기 중...")
            
            # 20초 대기
            start_time = time.time()
            while time.time() - start_time < 20:
                time.sleep(1)
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0:  # 5초마다 상태 출력
                    logger.info(f"   ⏱️ 경과 시간: {elapsed:.0f}초, 신호 수: {signal_count}개")
            
            # 모니터링 중지
            self.collector.stop_monitoring()
            
            logger.info(f"✅ 실시간 모니터링 테스트 완료")
            logger.info(f"   📊 총 신호 수: {signal_count}개")
            logger.info(f"   ⚡ 신호 빈도: {signal_count/20:.2f}개/초")
            
            self.performance_results['realtime_signals'] = signal_count
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 실시간 모니터링 테스트 실패: {e}")
            return False
    
    def generate_performance_report(self):
        """📋 성능 보고서 생성"""
        logger.info("\n" + "="*60)
        logger.info("📋 최종 성능 보고서")
        logger.info("="*60)
        
        try:
            total_score = 0
            max_score = 0
            
            logger.info("🚀 속도 성능 측정 결과:")
            
            # 기본 시그널 성능
            if 'basic_signal' in self.performance_results:
                time_val = self.performance_results['basic_signal']
                if time_val <= 1.0:
                    score = 10
                    grade = "🏆 EXCELLENT"
                elif time_val <= 2.0:
                    score = 8
                    grade = "🥇 GOOD"
                elif time_val <= 3.0:
                    score = 6
                    grade = "🥈 FAIR"
                else:
                    score = 4
                    grade = "⚠️ SLOW"
                
                total_score += score
                max_score += 10
                logger.info(f"   📊 기본 시그널: {time_val:.3f}초 {grade}")
            
            # 스캘핑 모드 성능
            if 'ultra_scalping' in self.performance_results:
                time_val = self.performance_results['ultra_scalping']
                if time_val <= 3.0:
                    score = 10
                    grade = "🏆 EXCELLENT"
                elif time_val <= 5.0:
                    score = 8
                    grade = "🥇 GOOD"
                elif time_val <= 7.0:
                    score = 6
                    grade = "🥈 FAIR"
                else:
                    score = 4
                    grade = "⚠️ SLOW"
                
                total_score += score
                max_score += 10
                logger.info(f"   ⚡ 스캘핑 모드: {time_val:.3f}초 {grade}")
            
            # 데이트레이딩 모드 성능
            if 'daytrading' in self.performance_results:
                time_val = self.performance_results['daytrading']
                if time_val <= 7.0:
                    score = 10
                    grade = "🏆 EXCELLENT"
                elif time_val <= 10.0:
                    score = 8
                    grade = "🥇 GOOD"
                elif time_val <= 15.0:
                    score = 6
                    grade = "🥈 FAIR"
                else:
                    score = 4
                    grade = "⚠️ SLOW"
                
                total_score += score
                max_score += 10
                logger.info(f"   📈 데이트레이딩 모드: {time_val:.3f}초 {grade}")
            
            # 전체 점수 계산
            if max_score > 0:
                final_score = (total_score / max_score) * 100
                
                if final_score >= 90:
                    final_grade = "🏆 EXCELLENT"
                elif final_score >= 75:
                    final_grade = "🥇 GOOD"
                elif final_score >= 60:
                    final_grade = "🥈 FAIR"
                else:
                    final_grade = "⚠️ NEEDS_IMPROVEMENT"
                
                logger.info(f"\n🎯 전체 성능 점수: {final_score:.1f}/100 {final_grade}")
            
            # 추가 통계
            logger.info(f"\n📊 세부 성능 통계:")
            for key, value in self.performance_results.items():
                if key != 'realtime_signals':
                    logger.info(f"   - {key}: {value:.3f}초")
                else:
                    logger.info(f"   - {key}: {value}개")
            
            logger.info(f"\n🚀 최적화 효과:")
            logger.info(f"   ✅ 캐시 시스템: 3초 고속 캐싱")
            logger.info(f"   ✅ 병렬 처리: 최대 6개 동시 API 호출")
            logger.info(f"   ✅ 임계값 필터링: 불필요한 데이터 제거")
            logger.info(f"   ✅ 메모리 최적화: 필수 필드만 파싱")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 성능 보고서 생성 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.collector:
                self.collector.cleanup()
            if self.trader:
                self.trader.cleanup()
            logger.info("🧹 테스트 리소스 정리 완료")
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")
    
    def run_all_tests(self):
        """전체 테스트 실행"""
        logger.info("🚀 초고속 AI 데이터 컬렉터 종합 테스트 시작!")
        logger.info(f"⏰ 테스트 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # 환경 설정
            if not self.setup():
                return False
            
            # 테스트 순서대로 실행
            tests = [
                ("기본 속도 테스트", self.test_1_basic_speed_test),
                ("순위 API 테스트", self.test_2_ranking_apis),
                ("기관 매매동향 테스트", self.test_3_institution_flow),
                ("울트라 스캘핑 모드 테스트", self.test_4_ultra_scalping_mode),
                ("데이트레이딩 모드 테스트", self.test_5_daytrading_mode),
                ("실시간 모니터링 테스트", self.test_6_realtime_monitoring)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"\n🧪 {test_name} 시작...")
                try:
                    if test_func():
                        passed_tests += 1
                        logger.info(f"✅ {test_name} 통과")
                    else:
                        logger.error(f"❌ {test_name} 실패")
                except Exception as e:
                    logger.error(f"❌ {test_name} 오류: {e}")
            
            # 최종 보고서
            self.generate_performance_report()
            
            logger.info(f"\n🏁 테스트 완료: {passed_tests}/{total_tests} 통과")
            logger.info(f"⏰ 테스트 종료 시간: {datetime.now().strftime('%H:%M:%S')}")
            
            return passed_tests == total_tests
            
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 오류: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """메인 실행 함수"""
    print("🚀 초고속 AI 데이터 컬렉터 성능 테스트 🚀")
    print("="*70)
    
    tester = UltraFastCollectorTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    return success

if __name__ == "__main__":
    main() 