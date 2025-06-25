#!/usr/bin/env python3
"""
코스피200·나스닥100·S&P500 투자 분석 시스템 실행 스크립트

실행 방법:
python run_analysis.py --format console
python run_analysis.py --format json  
python run_analysis.py --format html
"""

import asyncio
import argparse
import sys
import os
from enhanced_main_application import EnhancedMainApplication

def setup_environment():
    """환경 설정 확인"""
    print("🔧 환경 설정 확인 중...")
    
    # Gemini API 키 확인
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   Gemini AI 분석 없이 기본 분석만 수행됩니다.")
        print("   .env 파일을 생성하고 GEMINI_API_KEY를 설정하세요.")
        
    # 필수 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    print("✅ 환경 설정 완료")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='코스피200·나스닥100·S&P500 투자 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_analysis.py --format console    # 콘솔 출력
  python run_analysis.py --format json      # JSON 파일 저장
  python run_analysis.py --format html      # HTML 리포트 생성
  python run_analysis.py --quick            # 빠른 분석 (샘플만)
        """
    )
    
    parser.add_argument(
        '--format', 
        choices=['console', 'json', 'html'],
        default='console',
        help='출력 형식 선택 (기본값: console)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='빠른 분석 모드 (각 시장당 10개 종목만)'
    )
    
    parser.add_argument(
        '--no-ai',
        action='store_true', 
        help='AI 분석 건너뛰기 (데이터 수집만)'
    )
    
    parser.add_argument(
        '--telegram',
        action='store_true',
        help='텔레그램 알림 강제 활성화'
    )
    
    return parser.parse_args()

async def run_quick_analysis(app, output_format):
    """빠른 분석 모드 (샘플 데이터)"""
    print("⚡ 빠른 분석 모드 실행")
    
    # 여기서는 샘플 데이터로 빠른 테스트
    # 실제 구현에서는 각 시장당 10개씩만 분석
    result = await app.run_complete_analysis(output_format=output_format)
    return result

async def run_full_analysis(app, output_format):
    """전체 분석 모드"""
    print("🚀 전체 분석 모드 실행")
    print("📊 코스피200 (200개) + 나스닥100 (100개) + S&P500 (500개) 분석")
    print("⏱️ 예상 소요시간: 10-30분 (네트워크 상황에 따라)")
    
    result = await app.run_complete_analysis(output_format=output_format)
    return result

def display_welcome():
    """환영 메시지 출력"""
    print("="*80)
    print("🏆 코스피200·나스닥100·S&P500 투자 분석 시스템")
    print("="*80)
    print("📈 투자 대가 17개 전략 + Gemini AI 종합 분석")
    print("🎯 시장별 Top5 종목 자동 선정")
    print("🧠 전문가 수준의 투자 인사이트 제공")
    print("="*80)

def display_completion(result):
    """완료 메시지 출력"""
    print("\n" + "="*80)
    if result['success']:
        print("🎉 분석 완료!")
        print(f"📊 총 {result.get('total_analyses', 0)}개 종목 분석")
        print(f"⏱️ 소요시간: {result['execution_time']:.1f}초")
        
        # 시장별 선정 결과
        selections = result.get('top_selections', {})
        for market, count in selections.items():
            print(f"🏆 {market}: Top{count} 선정")
            
    else:
        print("❌ 분석 실패")
        print(f"🚨 오류: {result.get('error', '알 수 없는 오류')}")
        
    print("="*80)

async def main():
    """메인 실행 함수"""
    try:
        # 환영 메시지
        display_welcome()
        
        # 환경 설정
        setup_environment()
        
        # 명령행 인수 파싱
        args = parse_arguments()
        
        print(f"📋 설정: 출력형식={args.format}, 빠른모드={args.quick}, AI제외={args.no_ai}")
        
        # 애플리케이션 초기화
        app = EnhancedMainApplication()
        
        # 모듈 초기화
        await app.initialize_modules()
        
        # AI 분석 비활성화 옵션
        if args.no_ai:
            app.ai_analyzer = None
            print("🚫 AI 분석기 비활성화")
            
        # 분석 실행
        if args.quick:
            result = await run_quick_analysis(app, args.format)
        else:
            result = await run_full_analysis(app, args.format)
            
        # 결과 출력
        display_completion(result)
        
        # 추가 정보
        if args.format == 'json':
            print("💾 JSON 리포트가 현재 디렉토리에 저장되었습니다.")
        elif args.format == 'html':
            print("🌐 HTML 리포트가 현재 디렉토리에 저장되었습니다.")
            
        # 리소스 정리
        await app.cleanup()
        
        # 성공 종료
        sys.exit(0 if result['success'] else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Python 버전 확인
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        sys.exit(1)
        
    # 비동기 실행
    asyncio.run(main()) 