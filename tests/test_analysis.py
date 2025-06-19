"""
🎯 통합 분석 모듈 테스트
- 리팩토링된 데이터 수집 및 분석 파이프라인의 종단 간(E2E) 테스트.
- AIDataCollector가 DART, KIS, 분석 모듈을 올바르게 조립하여 작동하는지 검증합니다.
"""
import asyncio
import logging
import sys
import os
from typing import Dict, Any

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 리팩토링된 모듈 임포트
from core.core_trader import CoreTrader
from core.data_collector import AIDataCollector

def print_analysis_summary(symbol: str, data: Dict[str, Any]):
    """분석 결과 요약 출력"""
    print("\n" + "="*80)
    print(f"📊 종목 [{symbol}] 종합 분석 결과 요약")
    print("="*80)

    quality = data.get('data_quality', 'N/A')
    print(f"🔹 데이터 품질: {quality}")

    # DART 분석 결과
    dart_analysis = data.get('dart_analysis', {})
    if dart_analysis and not dart_analysis.get('error'):
        print("\n🏛️ DART 기반 펀더멘털 분석:")
        print(f"  - 기업명: {dart_analysis.get('corp_name', 'N/A')}")
        print(f"  - 펀더멘털 점수: {dart_analysis.get('fundamental_score', 0):.1f} / 100")
        print(f"  - 재무 건전성: {dart_analysis.get('financial_health', 'N/A')}")
        roe = dart_analysis.get('financial_ratios', {}).get('roe')
        if roe is not None:
            print(f"  - ROE: {roe:.2f}%")
        debt_ratio = dart_analysis.get('financial_ratios', {}).get('debt_ratio')
        if debt_ratio is not None:
            print(f"  - 부채비율: {debt_ratio:.2f}%")
        
    else:
        print(f"\n🏛️ DART 분석 실패: {dart_analysis.get('error', '알 수 없는 오류')}")

    # KIS 분석 결과
    kis_analysis = data.get('kis_analysis', {})
    if kis_analysis and not kis_analysis.get('error'):
        print("\n🚀 KIS 기반 기술적 분석:")
        price_info = kis_analysis.get('current_price_info', {})
        if price_info:
            price = float(price_info.get('stck_prpr', 0))
            change_rate = float(price_info.get('prdy_ctrt', 0))
            print(f"  - 현재가: {price:,.0f}원 ({change_rate:+.2f}%)")
        
        tech_indicators = kis_analysis.get('technical_indicators', {})
        if tech_indicators:
            print(f"  - RSI: {tech_indicators.get('rsi', 'N/A')}")
            print(f"  - 골든크로스: {'✅' if tech_indicators.get('is_golden_cross') else '❌'}")
    else:
        print(f"\n🚀 KIS 분석 실패: {kis_analysis.get('error', '알 수 없는 오류')}")
    
    print("="*80 + "\n")


async def main():
    """테스트 실행을 위한 메인 함수"""
    logger.info("🔥 통합 분석 파이프라인 테스트를 시작합니다...")
    trader = None
    try:
        trader = CoreTrader()
        await trader.async_initialize()

        if not trader.is_initialized():
            logger.error("❌ CoreTrader 초기화 실패. 테스트를 중단합니다.")
            return

        data_collector = AIDataCollector(trader)

        test_symbols = ['005930', '000660', '035720'] # 삼성전자, SK하이닉스, 카카오

        for symbol in test_symbols:
            logger.info(f"--- 종목 [{symbol}] 분석 중 ---")
            comprehensive_data = await data_collector.get_comprehensive_stock_data(symbol)
            
            if comprehensive_data:
                print_analysis_summary(symbol, comprehensive_data)
            else:
                logger.error(f"❌ 종목 [{symbol}] 분석 데이터 수집에 실패했습니다.")

    except Exception as e:
        logger.critical(f"💥 테스트 실행 중 심각한 오류 발생: {e}", exc_info=True)
    finally:
        if trader:
            await trader.close()
            logger.info("🔌 트레이더 연결을 종료했습니다.")
        logger.info("✅ 테스트가 종료되었습니다.")


if __name__ == "__main__":
    asyncio.run(main()) 