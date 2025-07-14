#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_krx_current_date.py
목적: 현재 날짜와 다양한 시장 타입으로 KRX 데이터 테스트
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_krx_current_data():
    """현재 날짜 KRX 데이터 테스트"""
    
    current_date = datetime.now().strftime('%Y%m%d')
    print(f"📅 현재 날짜: {current_date}")
    
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
    
    # 테스트할 시장들
    markets = [
        ('STK', 'KOSPI', 'KOSPI 주식'),
        ('STK', 'KOSDAQ', 'KOSDAQ 주식'),
        ('ETF', 'ALL', 'ETF'),
        ('IDX', 'KOSPI', 'KOSPI 지수'),
        ('IDX', 'KOSDAQ', 'KOSDAQ 지수'),
        ('IDX', 'KOSPI200', 'KOSPI200 지수'),
    ]
    
    results = {}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        for data_type, market, description in markets:
            print(f"\n🔍 {description} 테스트 중...")
            
            try:
                # 메인 페이지 방문
                async with session.get('http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201') as response:
                    if response.status != 200:
                        print(f"⚠️ 메인 페이지 접근 실패: {response.status}")
                
                await asyncio.sleep(1)  # 지연
                
                # 데이터 요청
                params = {
                    'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
                    'mktId': market,
                    'trdDd': current_date,
                    'share': '1',
                    'money': '1',
                    'csvxls_isNo': 'false'
                }
                
                print(f"📡 요청 파라미터: {params}")
                
                async with session.post(
                    'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd',
                    data=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    print(f"📥 응답 상태: HTTP {response.status}")
                    
                    if response.status == 200:
                        try:
                            text = await response.text()
                            print(f"📄 응답 크기: {len(text)} 문자")
                            
                            # 응답 내용 확인
                            if len(text) < 1000:
                                print(f"📄 응답 내용: {text}")
                            
                            data = json.loads(text)
                            
                            # 데이터가 있는지 확인
                            if 'OutBlock_1' in data and data['OutBlock_1']:
                                print(f"✅ {description}: 데이터 있음 ({len(data['OutBlock_1'])}개)")
                                results[f"{data_type}_{market}"] = {
                                    'status': 'success',
                                    'count': len(data['OutBlock_1']),
                                    'description': description,
                                    'sample': data['OutBlock_1'][0] if data['OutBlock_1'] else None
                                }
                            else:
                                print(f"❌ {description}: 데이터 없음")
                                print(f"📄 응답 키: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                                results[f"{data_type}_{market}"] = {
                                    'status': 'no_data',
                                    'count': 0,
                                    'description': description,
                                    'response_keys': list(data.keys()) if isinstance(data, dict) else None
                                }
                                
                        except json.JSONDecodeError as e:
                            print(f"❌ {description}: JSON 파싱 실패 - {e}")
                            print(f"📄 응답 내용: {text[:500]}")
                            results[f"{data_type}_{market}"] = {
                                'status': 'json_error',
                                'count': 0,
                                'description': description,
                                'error': str(e)
                            }
                    else:
                        text = await response.text()
                        print(f"❌ {description}: HTTP {response.status}")
                        print(f"📄 오류 응답: {text[:500]}")
                        results[f"{data_type}_{market}"] = {
                            'status': f'http_{response.status}',
                            'count': 0,
                            'description': description,
                            'error_response': text[:500]
                        }
                        
            except Exception as e:
                print(f"❌ {description}: 예외 발생 - {e}")
                results[f"{data_type}_{market}"] = {
                    'status': 'exception',
                    'error': str(e),
                    'count': 0,
                    'description': description
                }
            
            await asyncio.sleep(2)  # 요청 간 지연
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 KRX 현재 날짜 데이터 테스트 결과")
    print("="*60)
    
    success_count = 0
    for key, result in results.items():
        if result['status'] == 'success':
            success_count += 1
            print(f"✅ {result['description']}: {result['count']}개")
        else:
            print(f"❌ {result['description']}: {result['status']}")
    
    print(f"\n📈 요약: {success_count}/{len(results)} 성공")
    
    # 결과 저장
    with open('krx_current_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 'krx_current_test_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(test_krx_current_data()) 