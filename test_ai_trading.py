"""
🚀 AI 투자 데이터 수집 테스트
- 효율적인 코드로 핵심 정보만 수집
- 스캘핑과 단기투자 시그널 확인
"""
import logging
from core_trader import CoreTrader
from ai_data_collector import AIDataCollector

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ai_data_collection():
    """AI 데이터 수집 테스트"""
    try:
        # 1. 트레이더 초기화
        logger.info("🔧 CoreTrader 초기화 중...")
        trader = CoreTrader()
        
        # 2. AI 데이터 수집기 초기화
        logger.info("🤖 AI 데이터 수집기 초기화 중...")
        ai_collector = AIDataCollector(trader)
        
        # 3. 개별 종목 스캘핑 시그널 테스트
        logger.info("📊 삼성전자 스캘핑 시그널 수집 중...")
        samsung_signals = ai_collector.get_scalping_signals('005930')
        
        if samsung_signals:
            logger.info("✅ 삼성전자 스캘핑 분석 결과:")
            logger.info(f"   종목명: {samsung_signals['name']}")
            logger.info(f"   현재가: {samsung_signals['price']:,}원")
            logger.info(f"   매수세 강도: {samsung_signals['bid_strength']}")
            logger.info(f"   변동성: {samsung_signals['volatility_level']}")
            logger.info(f"   거래량 급증: {samsung_signals['volume_surge']}")
            logger.info(f"   시장 분위기: {samsung_signals['market_sentiment']}")
            logger.info(f"   스캘핑 점수: {samsung_signals['scalping_score']}/10")
            logger.info(f"   추천: {samsung_signals['recommendation']}")
        
        # 4. 다중 종목 포트폴리오 분석 테스트
        logger.info("📈 포트폴리오 분석 중...")
        symbols = ['005930', '000660', '035420', '005380', '207940']  # 삼성전자, SK하이닉스, 네이버, 현대차, 삼성바이오
        
        portfolio_analysis = ai_collector.get_portfolio_analysis(symbols)
        
        if portfolio_analysis:
            logger.info("✅ 포트폴리오 분석 결과:")
            logger.info(f"   분석 종목 수: {portfolio_analysis['total_stocks']}개")
            logger.info(f"   시장 상황: {portfolio_analysis['market_summary']}")
            logger.info(f"   추천 액션: {portfolio_analysis['recommended_action']}")
            
            # 상위 추천 종목
            logger.info("🔥 상위 추천 종목:")
            for i, pick in enumerate(portfolio_analysis['top_picks'], 1):
                logger.info(f"   {i}. {pick['name']} ({pick['symbol']}) - 점수: {pick['scalping_score']}, 추천: {pick['recommendation']}")
        
        # 5. 실시간 모니터링 테스트 (10초간)
        logger.info("🔴 실시간 모니터링 테스트 시작...")
        
        def ai_signal_callback(signal_data):
            """AI 시그널 콜백 함수"""
            logger.info(f"🤖 실시간 AI 시그널: {signal_data['name']} "
                       f"점수 {signal_data['scalping_score']}/10 "
                       f"추천 {signal_data['recommendation']}")
        
        # 실시간 모니터링 시작
        success = ai_collector.start_realtime_monitoring(['005930'], ai_signal_callback)
        
        if success:
            logger.info("✅ 10초간 실시간 모니터링...")
            import time
            time.sleep(10)
            ai_collector.stop_monitoring()
            logger.info("🔴 실시간 모니터링 종료")
        
        logger.info("🎉 AI 데이터 수집 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_ai_data_collection() 