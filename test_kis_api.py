#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 한국투자증권 API 연동 테스트
============================
실시간 K200 파생상품 데이터 수집 테스트
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from src.modules.kis_derivatives_api import KISDerivativesAPI

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kis_api_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def test_kis_api():
    """한국투자증권 API 테스트"""
    print("🧪 한국투자증권 API 연동 테스트")
    print("=" * 50)
    
    # API 키 확인
    app_key = os.getenv('LIVE_KIS_APP_KEY')
    app_secret = os.getenv('LIVE_KIS_APP_SECRET')
    
    if not app_key or not app_secret:
        print("❌ 한국투자증권 API 키가 설정되지 않았습니다.")
        print("   .env 파일에 다음 키들을 설정해주세요:")
        print("   LIVE_KIS_APP_KEY=your_app_key")
        print("   LIVE_KIS_APP_SECRET=your_app_secret")
        return
    
    print(f"✅ API 키 확인: {app_key[:10]}...")
    
    try:
        async with KISDerivativesAPI() as api:
            print("\n1️⃣ 액세스 토큰 획득 테스트...")
            if api.access_token:
                print(f"✅ 토큰 획득 성공: {api.access_token[:20]}...")
            else:
                print("❌ 토큰 획득 실패")
                return
            
            print("\n2️⃣ KOSPI200 지수 조회 테스트...")
            kospi200 = await api.get_kospi200_index()
            if kospi200:
                print(f"✅ KOSPI200: {kospi200:,.2f}")
            else:
                print("❌ KOSPI200 지수 조회 실패")
            
            print("\n3️⃣ KOSPI200 선물 데이터 테스트...")
            futures = await api.get_kospi200_futures()
            if futures:
                print(f"✅ 선물 데이터 {len(futures)}개 수집:")
                for future in futures:
                    print(f"   📈 {future.name}: {future.current_price:,.0f} ({future.change_rate:+.2f}%)")
            else:
                print("⚠️ 선물 데이터 없음 (API 엔드포인트 확인 필요)")
            
            print("\n4️⃣ KOSPI200 옵션 데이터 테스트...")
            options = await api.get_kospi200_options()
            if options:
                print(f"✅ 옵션 데이터 {len(options)}개 수집:")
                for option in options[:3]:  # 상위 3개만
                    print(f"   📊 {option.name}: {option.current_price:,.0f} ({option.change_rate:+.2f}%)")
            else:
                print("⚠️ 옵션 데이터 없음 (API 엔드포인트 확인 필요)")
            
            print("\n5️⃣ 파생상품 종합 데이터 테스트...")
            summary = await api.get_derivatives_summary()
            if summary:
                print("✅ 종합 데이터:")
                print(f"   📊 KOSPI200 지수: {summary.get('kospi200_index', 0):,.2f}")
                print(f"   📈 총 파생상품: {summary.get('total_derivatives', 0)}개")
                print(f"   📋 Put/Call 비율: {summary.get('put_call_ratio', 0):.3f}")
                print(f"   📊 총 옵션 거래량: {summary.get('total_option_volume', 0):,}")
            else:
                print("❌ 종합 데이터 수집 실패")
            
            # WebSocket 테스트는 실제 종목코드가 필요하므로 스킵
            print("\n6️⃣ WebSocket 실시간 연결 테스트...")
            if futures:
                symbols = [future.symbol for future in futures[:2]]
                print(f"📡 테스트 심볼: {symbols}")
                print("⚠️ WebSocket 테스트는 실제 거래시간에만 가능합니다.")
            else:
                print("⚠️ WebSocket 테스트용 심볼 없음")
            
    except Exception as e:
        logger.error(f"API 테스트 오류: {e}")
        print(f"❌ 테스트 실패: {e}")
    
    print("\n✅ 테스트 완료!")

async def test_derivatives_monitor():
    """파생상품 모니터링 시스템 테스트"""
    print("\n🔄 파생상품 모니터링 시스템 테스트")
    print("=" * 50)
    
    try:
        from src.modules.derivatives_monitor import DerivativesMonitor
        
        async with DerivativesMonitor() as monitor:
            print("1️⃣ K200 파생상품 수집 테스트...")
            k200_derivatives = await monitor.collect_k200_derivatives()
            
            if k200_derivatives:
                print(f"✅ K200 파생상품 {len(k200_derivatives)}개 수집:")
                for deriv in k200_derivatives[:3]:  # 상위 3개
                    print(f"   📊 {deriv.symbol}: {deriv.current_price:,.0f} ({deriv.change_percent:+.2f}%)")
            else:
                print("❌ K200 파생상품 수집 실패")
            
            print("\n2️⃣ 시장 신호 분석 테스트...")
            if k200_derivatives:
                signals = await monitor.analyze_market_signals(k200_derivatives)
                if signals:
                    print(f"✅ 시장 신호 {len(signals)}개 감지:")
                    for signal in signals[:2]:  # 상위 2개
                        print(f"   🚨 {signal.signal_type}: {signal.underlying_asset} (위험도: {signal.risk_level})")
                else:
                    print("✅ 특별한 위험 신호 없음 (정상)")
            
    except Exception as e:
        logger.error(f"모니터링 테스트 오류: {e}")
        print(f"❌ 모니터링 테스트 실패: {e}")

async def main():
    """메인 테스트 함수"""
    print("🚀 한국투자증권 API + 파생상품 모니터링 통합 테스트")
    print("=" * 60)
    
    # 1. KIS API 테스트
    await test_kis_api()
    
    # 2. 파생상품 모니터링 테스트
    await test_derivatives_monitor()
    
    print("\n🎉 모든 테스트 완료!")
    print("=" * 60)
    print("📝 다음 단계:")
    print("   1. 실제 거래시간에 WebSocket 테스트")
    print("   2. 실시간 모니터링 실행: python main.py --monitor-derivatives")
    print("   3. 파생상품 전용 모니터링: python derivatives_crash_monitor.py")

if __name__ == "__main__":
    asyncio.run(main()) 