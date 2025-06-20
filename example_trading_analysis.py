#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 통합 실시간 모니터링 시스템 사용 예시

거래대금 TOP 20 종목 실시간 전략 매칭 분석 + 기본 모니터링 통합 시스템 데모
"""

import asyncio
from personal_blackrock.real_time_monitor import RealTimeMonitor
from personal_blackrock.stock_data_manager import DataManager
from personal_blackrock.notifier import Notifier


async def demo_integrated_monitoring():
    """통합 모니터링 시스템 데모"""
    print("🚀 통합 실시간 모니터링 시스템 데모 시작")
    print("="*60)
    
    try:
        # 시스템 구성 요소 초기화
        data_manager = DataManager()
        notifier = Notifier()
        
        # 통합 모니터링 시스템 초기화
        monitor = RealTimeMonitor(data_manager, notifier)
        
        print("✅ 통합 모니터링 시스템 초기화 완료")
        
        # 현재 설정 확인
        status = await monitor.get_current_analysis_status()
        print(f"\n📊 현재 설정:")
        print(f"  - 전략 분석 주기: {status['analysis_interval']}초")
        print(f"  - 기본 모니터링 주기: {status['monitoring_interval']}초")
        print(f"  - 최소 매칭 점수: {status['min_score_threshold']}점")
        print(f"  - 분석 전략: {', '.join(status['strategies'])}")
        
        # 설정 변경 예시
        print("\n⚙️ 설정 변경 예시...")
        await monitor.update_analysis_settings(
            interval=180,  # 3분마다 전략 분석
            min_score=75,  # 최소 75점
            monitoring_interval=20  # 20초마다 기본 모니터링
        )
        
        print("✅ 설정 변경 완료")
        
        # 변경된 설정 확인
        updated_status = await monitor.get_current_analysis_status()
        print(f"\n📊 변경된 설정:")
        print(f"  - 전략 분석 주기: {updated_status['analysis_interval']}초")
        print(f"  - 기본 모니터링 주기: {updated_status['monitoring_interval']}초")
        print(f"  - 최소 매칭 점수: {updated_status['min_score_threshold']}점")
        
        print("\n🔥 실시간 분석 시작!")
        print("⏹️ 중단하려면 Ctrl+C를 누르세요.")
        
        # 실시간 분석 시작 (데모용으로 짧은 시간만)
        await asyncio.wait_for(
            monitor.start_real_time_analysis(),
            timeout=60  # 1분 데모
        )
        
    except asyncio.TimeoutError:
        print("\n⏰ 데모 시간 종료 (1분)")
        await monitor.stop_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 데모를 중단했습니다.")
        await monitor.stop_monitoring()
    except Exception as e:
        print(f"❌ 데모 중 오류 발생: {e}")
    finally:
        await monitor.cleanup()
        print("✅ 데모 종료")


async def demo_settings_configuration():
    """설정 변경 데모"""
    print("\n🔧 설정 변경 데모")
    print("="*40)
    
    monitor = RealTimeMonitor()
    
    # 다양한 설정 변경 예시
    configurations = [
        {"interval": 120, "min_score": 80, "monitoring_interval": 15},  # 고성능 설정
        {"interval": 300, "min_score": 70, "monitoring_interval": 30},  # 균형 설정
        {"interval": 600, "min_score": 60, "monitoring_interval": 60},  # 저부하 설정
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"\n📝 설정 {i}: {config}")
        await monitor.update_analysis_settings(**config)
        
        status = await monitor.get_current_analysis_status()
        print(f"✅ 적용된 설정:")
        print(f"  - 전략 분석: {status['analysis_interval']}초")
        print(f"  - 기본 모니터링: {status['monitoring_interval']}초") 
        print(f"  - 최소 점수: {status['min_score_threshold']}점")


async def demo_monitoring_features():
    """모니터링 기능 데모"""
    print("\n📊 모니터링 기능 데모")
    print("="*40)
    
    monitor = RealTimeMonitor()
    
    # 거래대금 상위 종목 조회 데모
    print("1. 거래대금 상위 종목 조회...")
    top_stocks = await monitor._get_top_trading_value_stocks()
    
    if top_stocks:
        print(f"✅ {len(top_stocks)}개 종목 조회 완료")
        print("\n📈 상위 5개 종목:")
        for stock in top_stocks[:5]:
            print(f"  {stock.rank}. {stock.name} ({stock.code})")
            print(f"     현재가: {stock.current_price:,}원 ({stock.change_rate:+.2f}%)")
            print(f"     거래대금: {stock.trading_value//100000000:,}억원")
    else:
        print("⚠️ 샘플 데이터 사용")
    
    # 모니터링 요약 정보
    print("\n2. 모니터링 요약 정보...")
    summary = monitor.get_monitoring_summary()
    print(summary)


async def main():
    """메인 데모 실행"""
    print("🎯 통합 실시간 모니터링 시스템 종합 데모")
    print("="*60)
    
    demos = [
        ("통합 모니터링 시스템", demo_integrated_monitoring),
        ("설정 변경", demo_settings_configuration),
        ("모니터링 기능", demo_monitoring_features),
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n🚀 {demo_name} 데모 시작...")
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ {demo_name} 데모 중 오류: {e}")
        print(f"✅ {demo_name} 데모 완료\n")
        
        # 데모 간 간격
        await asyncio.sleep(2)
    
    print("🎉 모든 데모 완료!")


if __name__ == "__main__":
    print("""
🚀 통합 실시간 모니터링 시스템 사용 가이드

주요 기능:
1. 거래대금 TOP 20 종목 실시간 수집
2. 6가지 전략 자동 매칭 분석 (윌리엄 오닐, 제시 리버모어, 워렌 버핏, 피터 린치, 일목균형표, 블랙록)
3. 기본 모니터링 (가격 급변, 거래량 급증, RSI 과매수/과매도, 수급 급변)
4. 텔레그램 알림 연동
5. 중복 알림 방지
6. 실시간 설정 변경

사용법:
1. main.py 실행
2. 메뉴에서 "9. 거래대금 TOP 20 실시간 전략 매칭" 선택
3. 설정 변경 후 모니터링 시작
4. Ctrl+C로 중단

설정 가능 항목:
- 전략 분석 주기: 60초 이상 (기본 300초)
- 기본 모니터링 주기: 10초 이상 (기본 30초)
- 최소 매칭 점수: 50-100점 (기본 70점)

⚠️ 주의사항:
- 실제 거래 시간에만 정확한 데이터 수집 가능
- API 제한을 고려하여 적절한 주기 설정 필요
- 투자 결정은 추가 검토 후 신중하게 결정

데모를 시작하려면 Enter를 누르세요...
""")
    
    input()
    asyncio.run(main()) 