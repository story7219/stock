import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
from pykrx import stock

from config.settings import DATA_PATH, ensure_dir_exists
from utils.logger import get_logger

# 로거 설정
logger = get_logger(__name__)

# pandas_ta 경고 메시지 비활성화 (필요 시)
# ta.logging.disable()

async def get_price_data_with_indicators(ticker: str, name: str, from_date: str, to_date: str):
    """
    특정 기간의 주가 데이터를 가져와 기술적 지표를 추가합니다.
    pykrx가 동기 라이브러리이므로 asyncio.to_thread를 사용합니다.
    
    :param ticker: 종목 티커
    :param name: 종목 이름
    :param from_date: 시작일 (YYYYMMDD)
    :param to_date: 종료일 (YYYYMMDD)
    :return: 기술적 지표가 추가된 DataFrame 또는 None
    """
    def fetch():
        try:
            df = stock.get_market_ohlcv(from_date, to_date, ticker)
            if df.empty:
                logger.warning(f"[{ticker}:{name}] pykrx에서 주가 데이터를 가져오지 못했습니다. (기간: {from_date}~{to_date})")
                return None
            
            # pandas-ta가 'close', 'high', 'low', 'open', 'volume' 컬럼을 기대하므로,
            # pykrx의 '종가', '고가', '저가', '시가', '거래량' 컬럼명을 변경해줍니다.
            df.rename(columns={
                '시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume'
            }, inplace=True)

            # 사용자 정의 전략 생성
            MyStrategy = ta.Strategy(
                name="Comprehensive_TA",
                description="SMA, EMA, BBands, RSI, MACD, Ichimoku",
                ta=[
                    {"kind": "sma", "length": 20},
                    {"kind": "sma", "length": 60},
                    {"kind": "sma", "length": 120},
                    {"kind": "ema", "length": 20},
                    {"kind": "bbands", "length": 20, "std": 2},
                    {"kind": "rsi", "length": 14},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "ichimoku"},
                ]
            )
            
            # 전략 적용
            df.ta.strategy(MyStrategy)
            
            return df
        except Exception as e:
            logger.error(f"[{ticker}:{name}] 주가 데이터/지표 계산 중 오류 발생: {e}")
            return None

    return await asyncio.to_thread(fetch)

async def fetch_and_save_price_data(ticker: str, name: str):
    """
    단일 종목에 대한 약 3년치 주가 데이터를 가져와 기술적 지표를 계산 후 CSV로 저장합니다.
    """
    try:
        to_date = datetime.now().strftime("%Y%m%d")
        from_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
        
        price_df = await get_price_data_with_indicators(ticker, name, from_date, to_date)
        
        if price_df is None or price_df.empty:
            # get_price_data_with_indicators 내부에서 이미 로깅했으므로 여기서는 pass
            return

        # 파일 저장 경로 및 이름 지정 (_price_data.csv)
        save_path = os.path.join(DATA_PATH, f"{ticker}_price_data.csv")
        
        price_df.to_csv(save_path, index=True, encoding='utf-8-sig') # 날짜 인덱스 저장
        logger.info(f"✅ [{ticker}:{name}] 주가 및 기술적 지표 저장 완료: {os.path.basename(save_path)}")

    except Exception as e:
        logger.error(f"❌ [{ticker}:{name}] 주가 데이터 처리 중 예상치 못한 오류 발생: {e}")

if __name__ == '__main__':
    async def test_fetcher():
        # 테스트를 위한 코드 (예: 삼성전자)
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from config.settings import ensure_dir_exists
        ensure_dir_exists()

        logger.info("--- price_fetcher.py 단독 테스트 시작 ---")
        await fetch_and_save_price_data("005930", "삼성전자")
        await fetch_and_save_price_data("000660", "SK하이닉스")
        # 거래정지 등으로 데이터가 없을 수 있는 종목 테스트
        await fetch_and_save_price_data("001820", "삼화콘덴서") 
        logger.info("--- price_fetcher.py 단독 테스트 종료 ---")

    asyncio.run(test_fetcher()) 