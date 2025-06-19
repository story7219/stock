"""
🚀 MarketDataCollector 종합 테스트 스크립트
- 한국투자증권 OpenAPI를 통한 실시간 시장 데이터 수집
- 스캘핑 vs 데이트레이딩 모드 비교
- WebSocket 실시간 모니터링 테스트
"""
import asyncio
import logging
from datetime import datetime
import time
from typing import List
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_trader import CoreTrader
from market_data_collector import MarketDataCollector, MarketSignal
from config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_collector_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketDataTester:
    """시장 데이터 수집기 테스터"""
    
    def __init__(self):
        """초기화"""
        self.config = Config()
        self.trader = None
        self.collector = None
        self.test_symbols = ["005930", "000660", "035420", "051910", "028260"]  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성물산
        
    def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("🔧 시장 데이터 수집기 초기화 중...")
            
            # CoreTrader 초기화
            self.trader = CoreTrader(self.config)
            
            if not self.trader.initialize():
                logger.error("❌ CoreTrader 초기화 실패")
                return False
            
            # MarketDataCollector 초기화
            self.collector = MarketDataCollector(self.trader)
            
            logger.info("✅ 시장 데이터 수집기 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    def test_basic_data_collection(self) -> None:
        """🥇 기본 데이터 수집 테스트"""
        logger.info("\n" + "="*50)
        logger.info("🔥 1단계: 기본 시장 데이터 수집 테스트")
        logger.info("="*50)
        
        try:
            # 상승률 상위 종목
            logger.info("\n📈 상승률 상위 종목 조회...")
            gainers = self.collector.get_top_gainers(10)
            if gainers:
                logger.info(f"✅ 상승률 상위 {len(gainers)}개 종목 수집 완료")
                for i, stock in enumerate(gainers[:3]):
                    logger.info(f"  {i+1}. {stock['name']} ({stock['symbol']}): {stock['change_rate']:.2f}%")
            else:
                logger.warning("⚠️ 상승률 데이터 수집 실패")
            
            time.sleep(1)
            
            # 거래량 상위 종목
            logger.info("\n📊 거래량 상위 종목 조회...")
            volume_leaders = self.collector.get_volume_leaders(10)
            if volume_leaders:
                logger.info(f"✅ 거래량 상위 {len(volume_leaders)}개 종목 수집 완료")
                for i, stock in enumerate(volume_leaders[:3]):
                    logger.info(f"  {i+1}. {stock['name']} ({stock['symbol']}): {stock['volume']:,}주")
            else:
                logger.warning("⚠️ 거래량 데이터 수집 실패")
            
            time.sleep(1)
            
            # 거래대금 상위 종목
            logger.info("\n💰 거래대금 상위 종목 조회...")
            value_leaders = self.collector.get_value_leaders(10)
            if value_leaders:
                logger.info(f"✅ 거래대금 상위 {len(value_leaders)}개 종목 수집 완료")
                for i, stock in enumerate(value_leaders[:3]):
                    logger.info(f"  {i+1}. {stock['name']} ({stock['symbol']}): {stock['value']:,}원")
            else:
                logger.warning("⚠️ 거래대금 데이터 수집 실패")
                
        except Exception as e:
            logger.error(f"❌ 기본 데이터 수집 테스트 실패: {e}")
    
    def test_investor_data_collection(self) -> None:
        """🥈 투자자별 매매 동향 테스트"""
        logger.info("\n" + "="*50)
        logger.info("🏢 2단계: 투자자별 매매 동향 테스트")
        logger.info("="*50)
        
        try:
            # 기관 순매수 순위
            logger.info("\n🏢 기관 순매수 순위 조회...")
            institution_buys = self.collector.get_institution_buys(10)
            if institution_buys:
                logger.info(f"✅ 기관 순매수 상위 {len(institution_buys)}개 종목 수집 완료")
                for i, stock in enumerate(institution_buys[:3]):
                    logger.info(f"  {i+1}. {stock['name']} ({stock['symbol']}): {stock['net_buy_value']:,}원")
            else:
                logger.warning("⚠️ 기관 매매동향 데이터 수집 실패")
            
            time.sleep(1)
            
            # 외국인 순매수 순위
            logger.info("\n🌍 외국인 순매수 순위 조회...")
            foreign_buys = self.collector.get_foreign_buys(10)
            if foreign_buys:
                logger.info(f"✅ 외국인 순매수 상위 {len(foreign_buys)}개 종목 수집 완료")
                for i, stock in enumerate(foreign_buys[:3]):
                    logger.info(f"  {i+1}. {stock['name']} ({stock['symbol']}): {stock['net_buy_value']:,}원")
            else:
                logger.warning("⚠️ 외국인 매매동향 데이터 수집 실패")
            
            time.sleep(1)
            
            # 섹터 로테이션 분석
            logger.info("\n🔄 섹터 로테이션 분석...")
            sector_data = self.collector.get_sector_rotation()
            if sector_data:
                logger.info("✅ 섹터 로테이션 분석 완료")
                if sector_data.get('hot_sectors'):
                    logger.info("🔥 상승 섹터:")
                    for sector in sector_data['hot_sectors'][:3]:
                        logger.info(f"  - {sector['name']}: {sector['change_rate']:.2f}%")
                if sector_data.get('cold_sectors'):
                    logger.info("❄️ 하락 섹터:")
                    for sector in sector_data['cold_sectors'][:3]:
                        logger.info(f"  - {sector['name']}: {sector['change_rate']:.2f}%")
            else:
                logger.warning("⚠️ 섹터 로테이션 데이터 수집 실패")
                
        except Exception as e:
            logger.error(f"❌ 투자자별 매매 동향 테스트 실패: {e}")
    
    def test_realtime_data(self) -> None:
        """🥉 실시간 데이터 테스트"""
        logger.info("\n" + "="*50)
        logger.info("📡 3단계: 실시간 데이터 테스트")
        logger.info("="*50)
        
        try:
            # 개별 종목 실시간 가격 조회
            for symbol in self.test_symbols[:3]:
                logger.info(f"\n💹 {symbol} 실시간 가격 조회...")
                price_data = self.collector.get_realtime_price(symbol)
                if price_data:
                    logger.info(f"✅ {price_data['name']}: {price_data['price']:,}원 (거래량: {price_data['volume']:,})")
                else:
                    logger.warning(f"⚠️ {symbol} 실시간 가격 조회 실패")
                
                time.sleep(0.5)
            
            # 호가창 조회
            test_symbol = self.test_symbols[0]  # 삼성전자
            logger.info(f"\n📋 {test_symbol} 호가창 조회...")
            orderbook = self.collector.get_orderbook(test_symbol)
            if orderbook:
                logger.info(f"✅ 호가창 조회 완료")
                logger.info(f"  매수총잔량: {orderbook.get('bid_total_volume', 0):,}주")
                logger.info(f"  매도총잔량: {orderbook.get('ask_total_volume', 0):,}주")
                logger.info(f"  시장심리: {orderbook.get('sentiment', 'neutral')}")
            else:
                logger.warning("⚠️ 호가창 조회 실패")
            
            # 체결강도 분석
            logger.info(f"\n💪 {test_symbol} 체결강도 분석...")
            strength_data = self.collector.analyze_trade_strength(test_symbol)
            if strength_data:
                logger.info(f"✅ 체결강도: {strength_data.get('strength', 0):.1f}% ({strength_data.get('signal', 'neutral')})")
            else:
                logger.warning("⚠️ 체결강도 분석 실패")
                
        except Exception as e:
            logger.error(f"❌ 실시간 데이터 테스트 실패: {e}")
    
    def test_scalping_mode(self) -> None:
        """⚡ 스캘핑 모드 테스트"""
        logger.info("\n" + "="*50)
        logger.info("⚡ 4단계: 스캘핑 모드 테스트 (최우선 + 고중요도)")
        logger.info("="*50)
        
        try:
            start_time = time.time()
            
            # 스캘핑용 핵심 데이터 수집
            scalping_data = self.collector.get_scalping_data(self.test_symbols)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"✅ 스캘핑 데이터 수집 완료 (응답시간: {response_time:.2f}초)")
            
            # 결과 분석
            if scalping_data:
                logger.info(f"📊 수집 모드: {scalping_data['mode']}")
                logger.info(f"⏰ 수집 시간: {scalping_data['timestamp']}")
                
                if scalping_data.get('top_gainers'):
                    logger.info(f"📈 상승률 상위: {len(scalping_data['top_gainers'])}개 종목")
                
                if scalping_data.get('volume_leaders'):
                    logger.info(f"📊 거래량 상위: {len(scalping_data['volume_leaders'])}개 종목")
                
                recommendation = scalping_data.get('recommendation', 'HOLD')
                logger.info(f"🎯 스캘핑 추천: {recommendation}")
                
                # 응답 시간 체크
                if response_time <= 5.0:
                    logger.info("🚀 우수한 응답속도 - 스캘핑에 적합!")
                else:
                    logger.warning("⚠️ 응답속도 개선 필요")
            
        except Exception as e:
            logger.error(f"❌ 스캘핑 모드 테스트 실패: {e}")
    
    def test_daytrading_mode(self) -> None:
        """📈 데이트레이딩 모드 테스트"""
        logger.info("\n" + "="*50)
        logger.info("📈 5단계: 데이트레이딩 모드 테스트 (전체 데이터)")
        logger.info("="*50)
        
        try:
            start_time = time.time()
            
            # 데이트레이딩용 종합 데이터 수집
            daytrading_data = self.collector.get_daytrading_data(self.test_symbols)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"✅ 데이트레이딩 데이터 수집 완료 (응답시간: {response_time:.2f}초)")
            
            # 결과 분석
            if daytrading_data:
                logger.info(f"📊 수집 모드: {daytrading_data['mode']}")
                logger.info(f"⏰ 수집 시간: {daytrading_data['timestamp']}")
                
                # 각 데이터 섹션 확인
                sections = [
                    ('top_gainers', '상승률 순위'),
                    ('volume_leaders', '거래량 순위'),
                    ('value_leaders', '거래대금 순위'),
                    ('institution_buys', '기관 순매수'),
                    ('foreign_buys', '외국인 순매수'),
                    ('sector_rotation', '섹터 로테이션')
                ]
                
                for section_key, section_name in sections:
                    section_data = daytrading_data.get(section_key)
                    if section_data:
                        if isinstance(section_data, list):
                            logger.info(f"📋 {section_name}: {len(section_data)}개 항목")
                        else:
                            logger.info(f"📋 {section_name}: 분석 완료")
                    else:
                        logger.warning(f"⚠️ {section_name}: 데이터 없음")
                
                # 종합 분석 결과
                analysis = daytrading_data.get('market_analysis', {})
                if analysis:
                    logger.info("\n🔍 종합 시장 분석:")
                    logger.info(f"  시장 심리: {analysis.get('market_sentiment', 'neutral')}")
                    logger.info(f"  섹터 동향: {analysis.get('sector_trend', 'mixed')}")
                    logger.info(f"  기관 자금: {analysis.get('institutional_flow', 'neutral')}")
                    logger.info(f"  외국인 자금: {analysis.get('foreign_flow', 'neutral')}")
                    logger.info(f"  종합 추천: {analysis.get('recommendation', 'HOLD')}")
                
                # 응답 시간 체크
                if response_time <= 10.0:
                    logger.info("📊 적절한 응답속도 - 데이트레이딩에 적합!")
                else:
                    logger.warning("⚠️ 응답속도 개선 필요")
            
        except Exception as e:
            logger.error(f"❌ 데이트레이딩 모드 테스트 실패: {e}")
    
    def test_realtime_monitoring(self) -> None:
        """📡 실시간 모니터링 테스트"""
        logger.info("\n" + "="*50)
        logger.info("📡 6단계: 실시간 모니터링 테스트 (WebSocket)")
        logger.info("="*50)
        
        signal_count = 0
        
        def signal_callback(signal: MarketSignal):
            """신호 감지 콜백"""
            nonlocal signal_count
            signal_count += 1
            logger.info(f"🚨 신호 #{signal_count}: {signal.signal_type} - {signal.name} ({signal.symbol})")
            logger.info(f"   강도: {signal.strength:.1f}/10, 시간: {signal.timestamp}")
        
        try:
            # 실시간 모니터링 시작
            logger.info(f"🚀 실시간 모니터링 시작: {len(self.test_symbols)}개 종목")
            success = self.collector.start_realtime_monitoring(
                self.test_symbols, 
                callback=signal_callback
            )
            
            if success:
                logger.info("✅ 실시간 모니터링 시작됨")
                
                # 30초간 모니터링
                logger.info("⏱️ 30초간 실시간 신호 감지 중...")
                time.sleep(30)
                
                # 모니터링 중지
                self.collector.stop_realtime_monitoring()
                logger.info("🛑 실시간 모니터링 중지됨")
                
                logger.info(f"📊 총 {signal_count}개 신호 감지됨")
                
            else:
                logger.warning("⚠️ 실시간 모니터링 시작 실패")
                
        except Exception as e:
            logger.error(f"❌ 실시간 모니터링 테스트 실패: {e}")
            # 에러 발생 시 모니터링 중지
            try:
                self.collector.stop_realtime_monitoring()
            except:
                pass
    
    def run_comprehensive_test(self) -> None:
        """종합 테스트 실행"""
        logger.info("🚀 MarketDataCollector 종합 테스트 시작!")
        logger.info(f"⏰ 테스트 시작 시간: {datetime.now()}")
        
        try:
            # 초기화
            if not self.initialize():
                logger.error("❌ 초기화 실패 - 테스트 중단")
                return
            
            # 1단계: 기본 데이터 수집
            self.test_basic_data_collection()
            
            # 2단계: 투자자별 매매 동향
            self.test_investor_data_collection()
            
            # 3단계: 실시간 데이터
            self.test_realtime_data()
            
            # 4단계: 스캘핑 모드
            self.test_scalping_mode()
            
            # 5단계: 데이트레이딩 모드
            self.test_daytrading_mode()
            
            # 6단계: 실시간 모니터링 (선택사항)
            user_input = input("\n📡 실시간 모니터링 테스트를 진행하시겠습니까? (y/n): ")
            if user_input.lower() == 'y':
                self.test_realtime_monitoring()
            
            logger.info("\n" + "="*50)
            logger.info("🎉 MarketDataCollector 종합 테스트 완료!")
            logger.info(f"⏰ 테스트 종료 시간: {datetime.now()}")
            logger.info("="*50)
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ 사용자에 의해 테스트 중단됨")
        except Exception as e:
            logger.error(f"❌ 테스트 실행 중 오류: {e}")
        finally:
            # 정리 작업
            if self.collector:
                try:
                    self.collector.stop_realtime_monitoring()
                except:
                    pass
            
            if self.trader:
                try:
                    self.trader.cleanup()
                except:
                    pass

def main():
    """메인 함수"""
    tester = MarketDataTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 