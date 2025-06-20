import asyncio
from main import StockAnalysisSystem

async def run_analysis_test():
    """
    시스템의 핵심 분석 기능을 테스트하기 위한 비동기 함수입니다.
    '윌리엄 오닐' 전략 분석을 실행하여 전체 파이프라인을 검증합니다.
    """
    print("🧪 시스템 분석 기능 테스트를 시작합니다...")
    
    system = StockAnalysisSystem()
    
    # 시스템 초기화
    initialized = await system.initialize()
    if not initialized:
        print("❌ 시스템 초기화에 실패하여 테스트를 중단합니다.")
        return

    print("\n✅ 시스템 초기화 완료. '윌리엄 오닐' 전략 분석을 시작합니다.")
    
    try:
        # 특정 전략 분석 기능 직접 호출
        await system.analyze_william_oneil()
        print("\n✅ '윌리엄 오닐' 전략 분석 테스트가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 심각한 오류 발생: {e}")
        
    finally:
        print("\n🧹 테스트 리소스를 정리합니다.")
        await system.cleanup()
        print("🧪 시스템 분석 기능 테스트를 종료합니다.")

if __name__ == "__main__":
    try:
        asyncio.run(run_analysis_test())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"💥 스크립트 실행 중 예외 발생: {e}") 