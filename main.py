import asyncio
import sys
import os
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# NumPy 2.0+ 호환성 패치: pandas-ta 라이브러리가 로드되기 전에 실행
import numpy as np
if not hasattr(np, 'NaN'):
    setattr(np, 'NaN', np.nan)

# 프로젝트 루트를 경로에 추가하여 다른 모듈들을 임포트할 수 있도록 함
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_collector.market_info_fetcher import get_and_save_all_stock_codes
from data_collector.dart_fetcher import fetch_and_save_financial_data
from data_collector.price_fetcher import fetch_and_save_price_data
from analysis.gemini_analyzer import analyze_stock_with_gemini
from config.settings import ensure_dir_exists, SEMAPHORE_LIMIT, DATA_PATH
from utils.logger import get_logger

# 로거 설정
logger = get_logger("main")

async def process_stock_data(ticker, name, semaphore):
    """단일 종목에 대한 데이터 수집 작업을 비동기적으로 처리합니다."""
    async with semaphore:
        try:
            # 두 가지 데이터 수집 작업을 동시에 진행
            await asyncio.gather(
                fetch_and_save_financial_data(ticker, name),
                fetch_and_save_price_data(ticker, name)
            )
        except Exception as e:
            logger.error(f"[{ticker}:{name}] 데이터 처리 중 오류 발생: {e}")

async def process_stock_analysis(ticker, name, semaphore):
    """단일 종목에 대한 AI 분석 작업을 비동기적으로 처리합니다."""
    async with semaphore:
        try:
            await analyze_stock_with_gemini(ticker, name)
        except Exception as e:
            logger.error(f"[{ticker}:{name}] AI 분석 처리 중 오류 발생: {e}")

async def main():
    """
    주식 분석 및 추천 시스템의 메인 실행 함수
    """
    logger.info("🚀 AI 주식 분석 및 추천 시스템을 시작합니다.")
    
    # 1. 필수 디렉토리 생성
    ensure_dir_exists()
    
    # 2. 전체 종목 코드 수집 및 로드
    stock_list_path = await get_and_save_all_stock_codes()
    if not stock_list_path or not os.path.exists(stock_list_path):
        logger.error("종목 코드 수집/로딩에 실패하여 프로그램을 종료합니다.")
        return
    
    df = pd.read_csv(stock_list_path)
    # 테스트를 위해 종목 수를 10개로 제한 (실제 운영 시에는 이 라인을 제거)
    df = df.head(10)
    logger.info(f"총 {len(df)}개 종목에 대한 데이터 수집을 시작합니다.")

    # 3. 각 종목에 대한 데이터 병렬 수집 (재무, 주가)
    data_semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    tasks = [process_stock_data(row['ticker'], row['name'], data_semaphore) for index, row in df.iterrows()]
    
    await tqdm_asyncio.gather(*tasks, desc="[1/2] 종합 데이터 수집")
    logger.info("✅ 모든 데이터 수집 프로세스가 완료되었습니다.")

    # 4. 각 종목에 대한 AI 분석 병렬 수행
    logger.info(f"총 {len(df)}개 종목에 대한 AI 분석을 시작합니다.")
    # Gemini API는 동시 요청 제한이 더 엄격할 수 있으므로 Semaphore 값을 조정 (예: 5)
    analysis_semaphore = asyncio.Semaphore(5) 
    analysis_tasks = [process_stock_analysis(row['ticker'], row['name'], analysis_semaphore) for index, row in df.iterrows()]

    await tqdm_asyncio.gather(*analysis_tasks, desc="[2/2] Gemini AI 리포트 생성")
    logger.info("✅ 모든 AI 분석 프로세스가 완료되었습니다.")
    logger.info(f"🏁 최종 분석 리포트는 'reports' 폴더에서 확인하실 수 있습니다.")


if __name__ == "__main__":
    asyncio.run(main()) 