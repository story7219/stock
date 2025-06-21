import asyncio
import os
import dart_fss as dart
from datetime import datetime
import pandas as pd

from config.settings import DART_API_KEY, DATA_PATH
from utils.logger import get_logger

# 로거 설정
logger = get_logger(__name__)

# DART API 키 설정
try:
    if not DART_API_KEY or DART_API_KEY == "YOUR_DART_API_KEY_HERE":
        raise ValueError("DART_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    dart.set_api_key(api_key=DART_API_KEY)
except ValueError as e:
    logger.error(e)
    # API 키가 없으면 DART 관련 기능 사용 불가
    _corp_list_cache = None
else:
    _corp_list_cache = None

# 기업 코드 목록 캐싱을 위한 전역 변수
_corp_list_cache = None

def get_corp_list_cached():
    """전체 기업 목록을 캐시하여 사용합니다. API 키가 없으면 빈 목록을 반환합니다."""
    global _corp_list_cache
    if DART_API_KEY and DART_API_KEY != "YOUR_DART_API_KEY_HERE":
        if _corp_list_cache is None:
            logger.info("DART 기업 목록을 처음 로드합니다...")
            try:
                _corp_list_cache = dart.get_corp_list()
                logger.info("DART 기업 목록 로드 완료.")
            except Exception as e:
                logger.error(f"DART 기업 목록 로드 실패: {e}")
                _corp_list_cache = [] # 오류 발생 시 빈 리스트로 초기화
        return _corp_list_cache
    return [] # API 키가 없는 경우

async def get_financial_statements(corp_code: str, year: int):
    """
    특정 기업의 연간 재무제표를 비동기적으로 가져옵니다.
    """
    def fetch():
        try:
            corp = dart.corp.Corp(corp_code=corp_code)
            # 연간 'consolidated' 재무제표를 우선적으로 가져옵니다.
            fs = corp.get_fs(bgn_de=f'{year}0101', end_de=f'{year}1231', fs_tp='consolidated')
            if fs:
                df = fs.show('df')
                # 연도 컬럼 추가
                df['year'] = year
                return df
            return None
        except Exception as e:
            logger.warning(f"[{corp_code}, {year}년] 재무제표 조회 중 오류 발생: {e}")
            return None

    return await asyncio.to_thread(fetch)

async def fetch_and_save_financial_data(ticker: str, name: str):
    """
    단일 종목에 대한 최근 5년치 연간 재무제표를 가져와 CSV 파일로 저장합니다.
    """
    try:
        corp_list = get_corp_list_cached()
        if not corp_list:
            logger.warning("DART 기업 목록이 없어 재무 데이터 수집을 건너뜁니다.")
            return

        corp_info = corp_list.find_by_stock_code(ticker)
        if not corp_info:
            logger.warning(f"[{ticker}:{name}] DART에서 기업 정보를 찾을 수 없습니다 (상장 폐지, 우선주 등).")
            return

        corp_code = corp_info.corp_code
        current_year = datetime.now().year
        
        tasks = [get_financial_statements(corp_code, year) for year in range(current_year - 5, current_year)]
        results = await asyncio.gather(*tasks)
        
        valid_results = [res for res in results if res is not None and not res.empty]
        if not valid_results:
            logger.warning(f"[{ticker}:{name}] 유효한 재무 데이터를 가져오지 못했습니다.")
            return

        # 모든 연도의 재무제표를 하나로 합침
        combined_df = pd.concat(valid_results)
        
        # 보기 좋게 정렬
        combined_df.sort_values(by=['year', 'account_nm'], inplace=True)
        
        # 파일 저장 경로 및 이름 지정 (_financials.csv)
        save_path = os.path.join(DATA_PATH, f"{ticker}_financials.csv")
        
        combined_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ [{ticker}:{name}] 재무제표 저장 완료: {os.path.basename(save_path)}")

    except Exception as e:
        logger.error(f"❌ [{ticker}:{name}] 재무제표 처리 중 예상치 못한 오류 발생: {e}")

if __name__ == '__main__':
    async def test_fetcher():
        # 테스트를 위한 코드 (예: 삼성전자)
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from config.settings import ensure_dir_exists
        ensure_dir_exists()
        
        logger.info("--- dart_fetcher.py 단독 테스트 시작 ---")
        await fetch_and_save_financial_data("005930", "삼성전자")
        await fetch_and_save_financial_data("000660", "SK하이닉스")
        # 오류가 예상되는 케이스
        await fetch_and_save_financial_data("999999", "없는회사")
        logger.info("--- dart_fetcher.py 단독 테스트 종료 ---")

    asyncio.run(test_fetcher()) 