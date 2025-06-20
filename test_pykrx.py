#!/usr/bin/env python3
"""
pykrx 데이터 조회 테스트 스크립트
"""

import sys
from datetime import datetime, timedelta

try:
    from pykrx import stock
    print("✅ pykrx 임포트 성공")
    
    # 1. 기본 날짜 조회
    try:
        today = stock.get_nearest_business_day_in_a_week()
        print(f"✅ 최근 영업일: {today}")
    except Exception as e:
        print(f"❌ 날짜 조회 실패: {e}")
        sys.exit(1)
    
    # 2. 종목 리스트 조회 (KOSPI 상위 5개)
    try:
        tickers = stock.get_market_ticker_list(today, market="KOSPI")[:5]
        print(f"✅ KOSPI 상위 5개 종목: {tickers}")
    except Exception as e:
        print(f"❌ 종목 리스트 조회 실패: {e}")
    
    # 3. 삼성전자 주가 데이터 조회 (최근 5일)
    try:
        samsung_code = "005930"
        start_date = (datetime.strptime(today, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
        
        price_data = stock.get_market_ohlcv_by_date(start_date, today, samsung_code)
        print(f"✅ 삼성전자 주가 데이터 조회 성공: {len(price_data)}일치")
        print(f"   최근 종가: {price_data['종가'].iloc[-1]:,}원")
    except Exception as e:
        print(f"❌ 삼성전자 주가 데이터 조회 실패: {e}")
    
    # 4. 펀더멘털 데이터 조회 (KOSPI 전체)
    try:
        fundamental_data = stock.get_market_fundamental(today, market="KOSPI")
        print(f"✅ KOSPI 펀더멘털 데이터 조회 성공: {len(fundamental_data)}개 종목")
    except Exception as e:
        print(f"❌ 펀더멘털 데이터 조회 실패: {e}")
    
    # 5. 시가총액 데이터 조회
    try:
        market_cap_data = stock.get_market_cap_by_ticker(today, market="KOSPI")
        print(f"✅ KOSPI 시가총액 데이터 조회 성공: {len(market_cap_data)}개 종목")
        
        # 상위 3개 종목 출력
        top_3 = market_cap_data.sort_values('시가총액', ascending=False).head(3)
        print("   시가총액 상위 3개 종목:")
        for idx, (code, row) in enumerate(top_3.iterrows(), 1):
            market_cap_trillion = row['시가총액'] / 1_0000_0000_0000
            print(f"   {idx}. {code}: {market_cap_trillion:.1f}조원")
            
    except Exception as e:
        print(f"❌ 시가총액 데이터 조회 실패: {e}")

except ImportError as e:
    print(f"❌ pykrx 임포트 실패: {e}")
    sys.exit(1)

print("\n🎯 pykrx 테스트 완료!") 