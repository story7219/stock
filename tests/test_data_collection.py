"""
🛠️ 데이터 수집 기능 검증 스크립트
=================================

AIDataCollector 모듈이 모든 데이터 소스(KIS, DART, FDR, Web Scraping)로부터
데이터를 정상적으로 수집하는지 독립적으로 테스트하기 위한 스크립트입니다.

실행: python test_data_collection.py
"""
import asyncio
import logging
from pprint import pprint
import traceback

# 설정 및 로거를 먼저 초기화합니다.
from utils.logger_config import setup_logging
setup_logging()

from core_trader import CoreTrader
from market_data_provider import AIDataCollector
import config

logger = logging.getLogger(__name__)

async def test_data_collection():
    """
    AIDataCollector의 모든 데이터 수집 기능이 정상적으로 작동하는지
    개별적으로 테스트하는 스크립트.
    """
    logger.info("="*60)
    logger.info("🕵️ 데이터 수집 기능 전체 검증 테스트를 시작합니다...")
    logger.info("="*60)

    trader = None
    try:
        # --- 1. 환경변수 검증 ---
        missing_configs, _ = config.validate_config()
        if missing_configs:
            # DART_API_KEY는 이 테스트의 핵심이므로 필수로 간주합니다.
            if 'DART_API_KEY' in missing_configs:
                logger.critical(f"❌ 필수 환경변수 'DART_API_KEY'가 설정되지 않았습니다. .env 파일을 확인해주세요.")
                return

        # --- 2. 의존성 객체 초기화 ---
        logger.info("🔧 [1/4] CoreTrader를 초기화합니다...")
        trader = CoreTrader(sheet_logger=None) # 시트 로거는 테스트에 불필요
        if not await trader.async_initialize():
            logger.error("❌ CoreTrader 초기화 실패. 테스트를 중단합니다.")
            return
        logger.info("✅ CoreTrader 초기화 완료.")

        logger.info("🔧 [2/4] AIDataCollector를 초기화합니다...")
        data_collector = AIDataCollector(trader)
        logger.info("✅ AIDataCollector 초기화 완료.")

        # --- 3. 테스트 대상 선정 ---
        test_symbol = "005930" # 삼성전자
        logger.info(f"🎯 [3/4] 테스트 대상 종목: 삼성전자 ({test_symbol})")

        # --- 4. 데이터 수집 실행 및 결과 확인 ---
        logger.info(f"🚀 [4/4] '{test_symbol}'에 대한 종합 데이터 수집을 시작합니다...")
        comprehensive_data = await data_collector.get_comprehensive_stock_data(test_symbol)

        logger.info("\n" + "="*60)
        logger.info("📊 수집된 전체 데이터 구조:")
        pprint(comprehensive_data)
        logger.info("="*60 + "\n")


        logger.info("🕵️ 항목별 데이터 수집 결과 요약:")
        logger.info("-"*40)

        success_count = 0
        total_count = 0
        if comprehensive_data:
            for key, value in comprehensive_data.items():
                total_count += 1
                # 데이터가 존재하고, 비어 있지 않은 경우 '성공'
                if value is not None and value != [] and value != {}:
                    logger.info(f"  [ ✅ 성공 ] '{key}'")
                    success_count += 1
                else:
                    # 참고: DART 공시는 해당일에 공시가 없으면 비어있는 것이 정상입니다.
                    logger.warning(f"  [ ⚠️  주의 ] '{key}' 데이터가 비어 있습니다 (정상일 수 있음).")
        else:
            logger.error("데이터 수집 결과가 없습니다.")

        logger.info("-"*40)
        logger.info(f"테스트 결과: 총 {total_count}개 항목 중 {success_count}개 데이터 수집 확인.")
        logger.info("="*60)
        logger.info("✅ 데이터 수집 기능 검증 테스트가 성공적으로 완료되었습니다.")


    except Exception as e:
        logger.error(f"💥 테스트 실행 중 심각한 오류 발생: {e}")
        logger.error(traceback.format_exc())

    finally:
        if trader and trader.http_client:
            await trader.http_client.aclose()
        logger.info("프로세스를 종료합니다.")


if __name__ == "__main__":
    # 필수 환경변수가 모두 설정되어 있는지 다시 한번 확인
    if not all([
        config.KIS_APP_KEY, config.KIS_APP_SECRET,
        config.GEMINI_API_KEY, config.DART_API_KEY
    ]):
         logger.critical("❌ 실행에 필요한 필수 API 키가 .env 파일에 설정되지 않았습니다.")
         logger.critical("검증에 필요한 키: KIS_APP_KEY, KIS_APP_SECRET, GEMINI_API_KEY, DART_API_KEY")
    else:
        asyncio.run(test_data_collection()) 