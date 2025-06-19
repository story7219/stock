"""
📈 금융 데이터 시각화 모듈
-mplfinance 라이브러리를 사용하여 주가 데이터를 캔들스틱 차트로 생성하고,
 이동평균선(5, 20, 60일)과 거래량을 포함하여 이미지 파일로 저장합니다.
"""
import pandas as pd
import mplfinance as mpf
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def create_stock_chart(
    price_history: List[Dict], 
    symbol: str, 
    chart_dir: str = "charts"
) -> Optional[str]:
    """
    주어진 시세 기록을 사용하여 주식 차트 이미지를 생성하고 저장합니다.

    :param price_history: KIS API의 'inquire-daily-price' 결과 리스트
    :param symbol: 종목 코드
    :param chart_dir: 차트를 저장할 디렉토리
    :return: 저장된 차트 이미지의 파일 경로. 실패 시 None.
    """
    if not price_history:
        logger.warning(f"[{symbol}] 차트 생성을 위한 데이터가 없습니다.")
        return None

    try:
        # 1. 데이터프레임 생성 및 전처리
        df = pd.DataFrame(price_history)
        
        # 필요한 컬럼만 선택하고, 데이터 타입 변환
        df = df[[
            'stck_bsop_date', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol'
        ]]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
            
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df = df.set_index('Date')

        # API가 오래된 순으로 데이터를 주므로, 최신 날짜가 뒤로 가도록 정렬
        df = df.sort_index()

        # 2. 차트 스타일 및 설정
        # 한글 폰트 설정 (맑은 고딕)
        # 윈도우 환경에 맞는 폰트 경로를 지정해야 할 수 있습니다.
        try:
            mpf.rc('font', family='Malgun Gothic')
            mpf.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logger.warning(f"한글 폰트 'Malgun Gothic'을 찾을 수 없습니다. 기본 폰트로 차트를 생성합니다. 오류: {e}")

        style = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mpf.make_marketcolors(
                up='r', down='b', inherit=True
            ),
            gridstyle='--'
        )

        # 3. 차트 생성 및 저장
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)
            
        chart_path = os.path.join(chart_dir, f"{symbol}_chart.png")
        
        mpf.plot(
            df,
            type='candle',
            style=style,
            title=f'\n{symbol} Stock Chart',
            ylabel='Price (KRW)',
            volume=True,
            mav=(5, 20, 60),  # 5, 20, 60일 이동평균선
            ylabel_lower='Volume',
            figratio=(16, 9),
            savefig=dict(fname=chart_path, dpi=100, pad_inches=0.25)
        )
        
        logger.info(f"✅ [{symbol}] 차트 이미지를 성공적으로 생성했습니다: {chart_path}")
        return chart_path

    except Exception as e:
        logger.error(f"❌ [{symbol}] 차트 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # 테스트용 샘플 데이터 (실제 KIS API 응답 형식과 동일)
    sample_data = [
        {'stck_bsop_date': '20240501', 'stck_oprc': '10000', 'stck_hgpr': '10500', 'stck_lwpr': '9800', 'stck_clpr': '10200', 'acml_vol': '150000'},
        {'stck_bsop_date': '20240502', 'stck_oprc': '10200', 'stck_hgpr': '10800', 'stck_lwpr': '10100', 'stck_clpr': '10700', 'acml_vol': '200000'},
        {'stck_bsop_date': '20240503', 'stck_oprc': '10700', 'stck_hgpr': '11000', 'stck_lwpr': '10500', 'stck_clpr': '10900', 'acml_vol': '180000'},
        {'stck_bsop_date': '20240506', 'stck_oprc': '10900', 'stck_hgpr': '11200', 'stck_lwpr': '10800', 'stck_clpr': '11100', 'acml_vol': '220000'},
        {'stck_bsop_date': '20240507', 'stck_oprc': '11100', 'stck_hgpr': '11500', 'stck_lwpr': '10900', 'stck_clpr': '11300', 'acml_vol': '250000'},
    ]
    
    # 차트 생성 함수 테스트
    create_stock_chart(sample_data, "005930_test") 