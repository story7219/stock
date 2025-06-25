#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 미국 파생상품 데이터 수집 테스트
=============================
다양한 API를 통한 미국 지수 선물/옵션 데이터 수집 테스트
"""

import asyncio
import logging
from datetime import datetime
from src.modules.us_realtime_derivatives import USRealTimeDerivatives, RealTimeConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/us_derivatives_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def test_us_derivatives():
    """미국 파생상품 데이터 수집 테스트"""
    print("🇺🇸 미국 파생상품 데이터 수집 테스트")
    print("=" * 60)
    
    # 설정 로드
    config = RealTimeConfig.from_env()
    
    print(f"📊 API 키 상태:")
    print(f"   - Polygon API: {'✅ 설정됨' if config.polygon_api_key else '❌ 미설정'}")
    print(f"   - Tradier API: {'✅ 설정됨' if config.tradier_token else '❌ 미설정'}")
    print(f"   - Finnhub API: {'✅ 설정됨' if config.finnhub_api_key else '❌ 미설정'}")
    print(f"   - Alpha Vantage: {'✅ 설정됨' if config.alpha_vantage_api_key else '❌ 미설정'}")
    
    # 미국 주요 지수 ETF 테스트
    test_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    async with USRealTimeDerivatives(config) as collector:
        
        # 데이터 콜백 함수
        def on_data_received(data):
            print(f"📈 실시간 데이터: {data.symbol} - ${data.current_price:.2f} "
                  f"({data.change_percent:+.2f}%)")
        
        collector.add_data_callback(on_data_received)
        
        print(f"\n1️⃣ 옵션 체인 데이터 테스트...")
        for symbol in test_symbols:
            try:
                print(f"\n📊 {symbol} 옵션 체인 조회 중...")
                options = await collector.get_options_chain_realtime(symbol)
                
                if options:
                    print(f"✅ {symbol} 옵션: {len(options)}개 발견")
                    
                    # 상위 5개 옵션 표시
                    for i, option in enumerate(options[:5]):
                        print(f"   {i+1}. {option.symbol}: ${option.current_price:.2f} "
                              f"(Strike: ${option.strike_price:.0f}, "
                              f"Type: {option.contract_type.upper()}, "
                              f"Vol: {option.volume:,}, "
                              f"IV: {option.implied_volatility:.2%})")
                else:
                    print(f"❌ {symbol} 옵션 데이터 없음")
                    
            except Exception as e:
                print(f"❌ {symbol} 옵션 조회 오류: {e}")
                continue
        
        print(f"\n2️⃣ 시장 현황 분석...")
        try:
            market_summary = await collector.get_market_summary()
            
            if market_summary:
                print(f"✅ 시장 현황:")
                print(f"   - 총 옵션 종목: {market_summary.get('total_options', 0):,}개")
                print(f"   - 고변동성 옵션: {len(market_summary.get('high_iv_options', []))}개")
                print(f"   - 대량거래 옵션: {len(market_summary.get('high_volume_options', []))}개")
                print(f"   - Put/Call 비율: {market_summary.get('put_call_ratio', 0):.2f}")
                
                # 고변동성 옵션 표시
                high_iv_options = market_summary.get('high_iv_options', [])[:3]
                if high_iv_options:
                    print(f"\n🔥 고변동성 옵션 Top 3:")
                    for i, option in enumerate(high_iv_options):
                        print(f"   {i+1}. {option['symbol']}: IV {option['implied_volatility']:.2%}")
                
                # 대량거래 옵션 표시
                high_vol_options = market_summary.get('high_volume_options', [])[:3]
                if high_vol_options:
                    print(f"\n📊 대량거래 옵션 Top 3:")
                    for i, option in enumerate(high_vol_options):
                        print(f"   {i+1}. {option['symbol']}: 거래량 {option['volume']:,}")
            else:
                print("❌ 시장 현황 데이터 없음")
                
        except Exception as e:
            print(f"❌ 시장 현황 분석 오류: {e}")
        
        print(f"\n3️⃣ 실시간 스트리밍 테스트 (10초간)...")
        try:
            # WebSocket 스트리밍 시작
            stream_symbols = ['SPY', 'QQQ']
            print(f"📡 {', '.join(stream_symbols)} 실시간 스트리밍 시작...")
            
            # 스트리밍 태스크 시작
            stream_task = asyncio.create_task(
                collector.start_websocket_stream(stream_symbols)
            )
            
            # 10초 대기
            await asyncio.sleep(10)
            
            # 스트리밍 중지
            await collector.stop_all_streams()
            stream_task.cancel()
            
            print("✅ 실시간 스트리밍 테스트 완료")
            
        except Exception as e:
            print(f"❌ 실시간 스트리밍 오류: {e}")
    
    print(f"\n🎯 테스트 결과 요약:")
    print(f"   - Yahoo Finance: 기본 옵션 데이터 수집 가능 (무료)")
    print(f"   - Polygon/Tradier: 실시간 고급 데이터 (API 키 필요)")
    print(f"   - WebSocket: 실시간 스트리밍 지원")
    print(f"\n💡 권장사항:")
    print(f"   1. Yahoo Finance로 기본 테스트 후")
    print(f"   2. Polygon/Tradier API 키 발급으로 고급 기능 활용")
    print(f"   3. KIS 파생상품 권한 승인 후 통합 시스템 구축")

if __name__ == "__main__":
    # 로그 디렉토리 생성
    import os
    os.makedirs('logs', exist_ok=True)
    
    asyncio.run(test_us_derivatives()) 