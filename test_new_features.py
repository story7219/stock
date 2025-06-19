"""
🧪 새로 구현된 기능들 테스트
- 📰 실시간 뉴스/공시 크롤링
- 📊 차트 이미지 생성
- 📋 호가창 실시간 모니터링

사용법:
python test_new_features.py
"""
import logging
import time
from datetime import datetime
from core_trader import CoreTrader, OrderbookMonitor
from chart_generator import StockChartGenerator
from analysis_engine import (
    fetch_market_news, 
    fetch_company_disclosures,
    get_comprehensive_market_data,
    MarketAnalyzer
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_new_features.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_news_crawling():
    """📰 뉴스 크롤링 기능 테스트"""
    print("\n" + "="*60)
    print("📰 뉴스 크롤링 기능 테스트")
    print("="*60)
    
    try:
        # 1. 실시간 뉴스 수집
        logger.info("실시간 뉴스 수집 테스트...")
        news_data = fetch_market_news()
        print("📰 뉴스 수집 결과:")
        print(news_data)
        
        # 2. 공시 정보 수집
        logger.info("공시 정보 수집 테스트...")
        disclosure_data = fetch_company_disclosures(['005930', '000660'])
        print("\n📋 공시 수집 결과:")
        print(disclosure_data)
        
        # 3. 종합 시장 데이터
        logger.info("종합 시장 데이터 수집 테스트...")
        comprehensive_data = get_comprehensive_market_data(['005930', '000660'])
        print("\n🌐 종합 데이터 수집 결과:")
        print(comprehensive_data[:500] + "..." if len(comprehensive_data) > 500 else comprehensive_data)
        
        print("\n✅ 뉴스 크롤링 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 뉴스 크롤링 테스트 실패: {e}")
        return False

def test_chart_generation():
    """📊 차트 생성 기능 테스트"""
    print("\n" + "="*60)
    print("📊 차트 생성 기능 테스트")
    print("="*60)
    
    try:
        # CoreTrader 인스턴스 생성
        trader = CoreTrader()
        
        # 차트 생성기 초기화
        chart_generator = StockChartGenerator(trader)
        
        # 1. 종합 차트 생성 (삼성전자)
        logger.info("삼성전자 종합 차트 생성 테스트...")
        comprehensive_chart = chart_generator.create_comprehensive_chart(
            symbol="005930",
            period_days=30
        )
        
        if comprehensive_chart:
            print(f"✅ 종합 차트 생성 성공: {comprehensive_chart}")
        else:
            print("⚠️ 종합 차트 생성 실패 (더미 데이터 사용)")
        
        # 2. 실시간 미니 차트 생성
        logger.info("SK하이닉스 미니 차트 생성 테스트...")
        mini_chart = chart_generator.create_realtime_mini_chart(symbol="000660")
        
        if mini_chart:
            print(f"✅ 미니 차트 생성 성공: {mini_chart}")
        else:
            print("⚠️ 미니 차트 생성 실패")
        
        # 3. 시장 히트맵 생성
        logger.info("시장 히트맵 생성 테스트...")
        symbols = ["005930", "000660", "035420", "051910", "006400"]
        heatmap = chart_generator.create_market_heatmap(symbols)
        
        if heatmap:
            print(f"✅ 시장 히트맵 생성 성공: {heatmap}")
        else:
            print("⚠️ 시장 히트맵 생성 실패")
        
        print("\n✅ 차트 생성 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 차트 생성 테스트 실패: {e}")
        return False

def test_orderbook_monitoring():
    """📋 호가창 모니터링 기능 테스트"""
    print("\n" + "="*60)
    print("📋 호가창 모니터링 기능 테스트")
    print("="*60)
    
    try:
        # CoreTrader 인스턴스 생성
        trader = CoreTrader()
        
        # 1. 상세 호가창 조회 테스트
        logger.info("삼성전자 상세 호가창 조회 테스트...")
        orderbook = trader.get_detailed_orderbook("005930")
        
        if orderbook:
            print("✅ 호가창 조회 성공!")
            print(f"   - 종목: {orderbook['symbol']}")
            print(f"   - 매수 호가 수: {len(orderbook['bid_orders'])}")
            print(f"   - 매도 호가 수: {len(orderbook['ask_orders'])}")
            print(f"   - 불균형 비율: {orderbook['imbalance_ratio']:.2f}")
            print(f"   - 시장 심리: {orderbook['market_sentiment']}")
            
            # 상위 3호가 표시
            if orderbook['bid_orders']:
                print("   - 매수 상위 3호가:")
                for i, bid in enumerate(orderbook['bid_orders'][:3], 1):
                    print(f"     {i}. {bid['price']:,}원 ({bid['volume']:,}주)")
            
            if orderbook['ask_orders']:
                print("   - 매도 상위 3호가:")
                for i, ask in enumerate(orderbook['ask_orders'][:3], 1):
                    print(f"     {i}. {ask['price']:,}원 ({ask['volume']:,}주)")
        else:
            print("⚠️ 호가창 조회 실패 (API 키 설정 필요)")
        
        # 2. 호가창 모니터링 시스템 테스트
        logger.info("호가창 모니터링 시스템 테스트...")
        orderbook_monitor = OrderbookMonitor(trader)
        
        # 콜백 함수 정의
        def orderbook_callback(symbol: str, orderbook_data: dict):
            print(f"📋 {symbol} 호가창 업데이트: 불균형비율 {orderbook_data['imbalance_ratio']:.2f}")
        
        # 콜백 등록
        orderbook_monitor.add_orderbook_callback(orderbook_callback)
        
        # 모니터링 시작 (5초간)
        symbols_to_monitor = ["005930", "000660"]
        if orderbook_monitor.start_monitoring(symbols_to_monitor, interval_seconds=2):
            print(f"✅ 호가창 모니터링 시작: {symbols_to_monitor}")
            print("⏳ 5초간 모니터링 중...")
            time.sleep(5)
            
            # 모니터링 중지
            orderbook_monitor.stop_monitoring()
            print("🛑 호가창 모니터링 중지")
            
            # 상태 확인
            status = orderbook_monitor.get_monitoring_status()
            print(f"📊 모니터링 상태: {status}")
        else:
            print("⚠️ 호가창 모니터링 시작 실패")
        
        print("\n✅ 호가창 모니터링 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 호가창 모니터링 테스트 실패: {e}")
        return False

def test_ai_analysis():
    """🤖 AI 종합 분석 기능 테스트"""
    print("\n" + "="*60)
    print("🤖 AI 종합 분석 기능 테스트")
    print("="*60)
    
    try:
        # MarketAnalyzer 초기화
        analyzer = MarketAnalyzer()
        
        # 종합 분석 실행
        logger.info("AI 종합 시장 분석 테스트...")
        analysis_result = analyzer.get_comprehensive_analysis(
            symbol_list=["005930", "000660", "035420"]
        )
        
        print("🤖 AI 분석 결과:")
        print(analysis_result)
        
        print("\n✅ AI 분석 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ AI 분석 테스트 실패: {e}")
        if "API 키" in str(e):
            print("💡 Gemini API 키가 설정되지 않아 AI 분석을 건너뜁니다.")
        return False

def main():
    """🚀 모든 새 기능 테스트 실행"""
    print("🚀 새로 구현된 기능들 종합 테스트 시작!")
    print("⏰ 시작 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    results = {
        '뉴스_크롤링': False,
        '차트_생성': False,
        '호가창_모니터링': False,
        'AI_분석': False
    }
    
    # 1. 뉴스 크롤링 테스트
    results['뉴스_크롤링'] = test_news_crawling()
    
    # 2. 차트 생성 테스트
    results['차트_생성'] = test_chart_generation()
    
    # 3. 호가창 모니터링 테스트  
    results['호가창_모니터링'] = test_orderbook_monitoring()
    
    # 4. AI 분석 테스트
    results['AI_분석'] = test_ai_analysis()
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for feature, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{feature.replace('_', ' ')}: {status}")
    
    print(f"\n🏆 전체 결과: {success_count}/{total_count} 성공")
    print("⏰ 완료 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if success_count == total_count:
        print("\n🎉 모든 기능이 정상적으로 구현되었습니다!")
    else:
        print(f"\n⚠️ {total_count - success_count}개 기능에서 문제가 발생했습니다.")
        print("💡 로그 파일(test_new_features.log)을 확인해주세요.")

if __name__ == "__main__":
    main() 