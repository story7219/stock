#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
바이낸스 선물옵션 데이터 수집 실행 스크립트
과거 최대치부터 현재까지의 데이터를 수집합니다.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from modules.collectors.binance_futures_collector import (
    BinanceFuturesCollector, 
    BinanceConfig, 
    CollectionConfig
)
from config.binance_config import get_binance_config, BinanceSettings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/binance_collector.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def collect_binance_futures_data():
    """바이낸스 선물 데이터 수집"""
    
    # .env 파일에서 바이낸스 설정 로드
    binance_settings = get_binance_config()
    
    # 설정
    config = BinanceConfig(
        api_key=binance_settings['api_key'],
        api_secret=binance_settings['api_secret'],
        testnet=binance_settings['testnet'],
        rate_limit=binance_settings['rate_limit']
    )
    
    # API 키 설정 확인
    if BinanceSettings.is_configured():
        logger.info("바이낸스 API 키가 설정되어 있습니다.")
    else:
        logger.warning("바이낸스 API 키가 설정되지 않았습니다. 공개 데이터만 수집합니다.")
    
    # 수집할 심볼들 (주요 선물 심볼)
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'BCHUSDT',
        'ETCUSDT', 'FILUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT'
    ]
    
    # 수집 설정
    collection_config = CollectionConfig(
        symbols=symbols,
        intervals=['1h', '4h', '1d'],  # 1시간, 4시간, 1일 캔들
        start_date=datetime.now(timezone.utc) - timedelta(days=365*2),  # 2년 전부터
        end_date=datetime.now(timezone.utc),
        save_format='parquet',
        compression='snappy'
    )
    
    # 출력 디렉토리
    output_dir = Path('data/binance_futures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("바이낸스 선물 데이터 수집 시작")
    logger.info(f"수집 심볼: {len(symbols)}개")
    logger.info(f"수집 기간: {collection_config.start_date} ~ {collection_config.end_date}")
    
    try:
        async with BinanceFuturesCollector(config) as collector:
            # 1. 과거 K라인 데이터 수집
            logger.info("과거 K라인 데이터 수집 중...")
            kline_results = await collector.collect_historical_data(collection_config, output_dir)
            
            # 2. 자금조달률 데이터 수집
            logger.info("자금조달률 데이터 수집 중...")
            funding_results = collector.collect_funding_rates(
                collection_config.symbols,
                collection_config.start_date,
                collection_config.end_date,
                output_dir / 'funding_rates'
            )
            
            # 3. 미결제약정 데이터 수집
            logger.info("미결제약정 데이터 수집 중...")
            for symbol in collection_config.symbols[:5]:  # 상위 5개만
                try:
                    interest_data = collector.get_open_interest_history(
                        symbol=symbol,
                        period='1h',
                        start_time=collection_config.start_date,
                        end_time=collection_config.end_date
                    )
                    
                    if interest_data:
                        import pandas as pd
                        df = pd.DataFrame(interest_data)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
                        df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')
                        
                        filename = f"{symbol}_open_interest.parquet"
                        filepath = output_dir / 'open_interest' / filename
                        filepath.parent.mkdir(exist_ok=True)
                        df.to_parquet(filepath, compression='snappy')
                        
                        logger.info(f"미결제약정 데이터 저장 완료: {symbol}")
                        
                except Exception as e:
                    logger.error(f"미결제약정 데이터 수집 실패 {symbol}: {e}")
            
            # 결과 출력
            print("\n" + "="*50)
            print("바이낸스 선물 데이터 수집 완료")
            print("="*50)
            print(f"수집 시간: {kline_results['duration']:.2f}초")
            print(f"수집된 심볼: {len(kline_results['collected_data'])}개")
            print(f"에러 개수: {len(kline_results['errors'])}개")
            
            if kline_results['errors']:
                print("\n에러 목록:")
                for error in kline_results['errors']:
                    print(f"  - {error}")
            
            print(f"\n데이터 저장 위치: {output_dir.absolute()}")
            print("="*50)
            
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}")
        raise

def main():
    """메인 함수"""
    try:
        asyncio.run(collect_binance_futures_data())
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 