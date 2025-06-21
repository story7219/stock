import asyncio
import os
from datetime import datetime
import pandas as pd
from pykrx import stock

# 설정 파일 및 유틸리티 임포트
# (주의: 상대 경로 임포트는 main.py에서 실행될 때를 기준으로 합니다.)
from config.settings import DATA_PATH, ensure_dir_exists

async def get_and_save_all_stock_codes():
    """
    코스피, 코스닥, 코스피200에 상장된 모든 종목의 티커와 이름을 가져와
    CSV 파일로 저장합니다.
    ETF, ETN 등은 제외하고 순수 주식 종목만 가져옵니다.
    """
    print("전체 상장 종목 코드 수집을 시작합니다...")
    
    # 데이터 디렉토리 확인 및 생성
    ensure_dir_exists()

    try:
        # 오늘 날짜 기준으로 데이터 가져오기
        today_str = datetime.now().strftime("%Y%m%d")

        # 시가총액 데이터 가져오기 (코스피, 코스닥)
        print("시가총액 데이터를 수집하여 1조원 이상 종목을 필터링합니다...")
        df_cap_kospi = stock.get_market_cap(today_str, market="KOSPI")
        df_cap_kosdaq = stock.get_market_cap(today_str, market="KOSDAQ")
        df_cap = pd.concat([df_cap_kospi, df_cap_kosdaq])

        # 시가총액 1조원 이상 필터링
        df_cap_filtered = df_cap[df_cap['시가총액'] >= 1_000_000_000_000]
        
        print(f"시가총액 1조원 이상 {len(df_cap_filtered)}개 종목을 대상으로 합니다.")

        all_tickers = set(df_cap_filtered.index)
        
        # 코스피, 코스닥, 코스피200 종목 리스트 원본 (market 정보 확인용)
        kospi_tickers = stock.get_market_ticker_list(today_str, market="KOSPI")
        kosdaq_tickers = stock.get_market_ticker_list(today_str, market="KOSDAQ")
        kospi200_tickers = stock.get_index_portfolio_deposit_file("1028")

        stock_list = []
        for ticker in all_tickers:
            # ETF, ETN 등은 제외 (일반적으로 종목코드가 6자리)
            if len(ticker) == 6:
                stock_name = stock.get_market_ticker_name(ticker)
                market = "KOSPI" if ticker in kospi_tickers else "KOSDAQ"
                is_kospi200 = "Yes" if ticker in kospi200_tickers else "No"
                stock_list.append({
                    "ticker": ticker,
                    "name": stock_name,
                    "market": market,
                    "is_kospi200": is_kospi200
                })
        
        df = pd.DataFrame(stock_list)
        
        # 파일 저장 경로 설정
        file_path = os.path.join(DATA_PATH, "stock_list.csv")
        
        # CSV 파일로 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"총 {len(df)}개의 필터링된 종목 정보를 '{file_path}'에 성공적으로 저장했습니다.")
        return file_path

    except Exception as e:
        print(f"종목 코드 수집 중 오류 발생: {e}")
        return None

if __name__ == '__main__':
    # 이 파일을 직접 실행할 경우를 위한 테스트 코드
    # (프로젝트 루트에서 실행해야 `config` 모듈을 찾을 수 있습니다.)
    asyncio.run(get_and_save_all_stock_codes()) 