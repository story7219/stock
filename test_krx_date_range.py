#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_krx_date_range.py
목적: KRX에서 실제로 제공하는 데이터 날짜 범위 테스트
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from pathlib import Path

async def test_krx_date_availability():
    """KRX 데이터 가용성 테스트"""
    
    # 테스트할 날짜들
    test_dates = [
        '19970101',  # 1997년 1월 1일
        '19970501',  # 1997년 5월 1일
        '19980101',  # 1998년 1월 1일
        '19990101',  # 1999년 1월 1일
        '20000101',  # 2000년 1월 1일
        '20050101',  # 2005년 1월 1일
        '20100101',  # 2010년 1월 1일
        '20150101',  # 2015년 1월 1일
        '20200101',  # 2020년 1월 1일
        '20250101',  # 2025년 1월 1일 (현재)
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201',
        'Origin': 'http://data.krx.co.kr',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    results = {}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        for date_str in test_dates:
            print(f"🔍 {date_str} 테스트 중...")
            
            try:
                # 메인 페이지 방문
                async with session.get('http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201') as response:
                    if response.status != 200:
                        print(f"⚠️ 메인 페이지 접근 실패: {response.status}")
                
                await asyncio.sleep(1)  # 지연
                
                # KOSPI 데이터 요청
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': 'KOSPI',
                    'trdDd': date_str,
                    'share': '1',
                    'money': '1',
                    'csvxls_isNo': 'false'
                }
                
                async with session.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        try:
                            text = await response.text()
                            data = json.loads(text)
                            
                            # 데이터가 있는지 확인
                            if 'OutBlock_1' in data and data['OutBlock_1']:
                                print(f"✅ {date_str}: 데이터 있음 ({len(data['OutBlock_1'])}개 종목)")
                                results[date_str] = {
                                    'status': 'success',
                                    'count': len(data['OutBlock_1']),
                                    'sample': data['OutBlock_1'][0] if data['OutBlock_1'] else None
                                }
                            else:
                                print(f"❌ {date_str}: 데이터 없음")
                                results[date_str] = {
                                    'status': 'no_data',
                                    'count': 0
                                }
                                
                        except json.JSONDecodeError:
                            print(f"❌ {date_str}: JSON 파싱 실패")
                            results[date_str] = {
                                'status': 'json_error',
                                'count': 0
                            }
                    else:
                        print(f"❌ {date_str}: HTTP {response.status}")
                        results[date_str] = {
                            'status': f'http_{response.status}',
                            'count': 0
                        }
                        
            except Exception as e:
                print(f"❌ {date_str}: 예외 발생 - {e}")
                results[date_str] = {
                    'status': 'exception',
                    'error': str(e),
                    'count': 0
                }
            
            await asyncio.sleep(2)  # 요청 간 지연
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 KRX 데이터 가용성 테스트 결과")
    print("="*50)
    
    available_dates = []
    unavailable_dates = []
    
    for date_str, result in results.items():
        if result['status'] == 'success':
            available_dates.append(date_str)
            print(f"✅ {date_str}: 사용 가능 ({result['count']}개 종목)")
        else:
            unavailable_dates.append(date_str)
            print(f"❌ {date_str}: 사용 불가 ({result['status']})")
    
    print("\n" + "="*50)
    print("📈 요약")
    print("="*50)
    print(f"사용 가능한 날짜: {len(available_dates)}개")
    print(f"사용 불가능한 날짜: {len(unavailable_dates)}개")
    
    if available_dates:
        print(f"가장 오래된 데이터: {min(available_dates)}")
        print(f"가장 최신 데이터: {max(available_dates)}")
    
    # 결과 저장
    with open('krx_date_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 'krx_date_test_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(test_krx_date_availability()) 