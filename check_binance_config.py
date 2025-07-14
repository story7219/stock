#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
바이낸스 API 설정 확인 스크립트
.env 파일의 설정이 올바른지 확인합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

try:
    from config.binance_config import BinanceSettings, get_binance_config
except ImportError:
    print("❌ config.binance_config 모듈을 찾을 수 없습니다.")
    print("   config/binance_config.py 파일이 있는지 확인하세요.")
    sys.exit(1)

def check_config():
    """바이낸스 설정 확인"""
    print("🔍 바이낸스 API 설정 확인")
    print("=" * 50)
    
    # API 키 설정 확인
    api_key = BinanceSettings.get_api_key()
    api_secret = BinanceSettings.get_api_secret()
    
    if api_key and api_secret:
        print("✅ API 키가 설정되어 있습니다")
        print(f"   API 키: {api_key[:10]}...{api_key[-4:]}")
        print(f"   API 시크릿: {api_secret[:10]}...{api_secret[-4:]}")
    else:
        print("❌ API 키가 설정되지 않았습니다")
        print("   .env 파일을 확인하세요")
        print("   SETUP_BINANCE_API.md 파일을 참고하세요")
    
    # 기타 설정 확인
    print(f"   테스트넷: {BinanceSettings.get_testnet()}")
    print(f"   요청 제한: {BinanceSettings.get_rate_limit()}/분")
    
    # 설정된 권한 확인
    if BinanceSettings.is_configured():
        print("✅ 인증된 API로 더 높은 요청 제한 사용 가능")
        print("   - 분당 2,400회 요청 가능")
        print("   - 일일 100,000회 요청 가능")
    else:
        print("⚠️  공개 API로 제한된 요청 제한 사용")
        print("   - 분당 1,200회 요청 제한")
        print("   - 일일 50,000회 요청 제한")
    
    # .env 파일 존재 확인
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        print(f"✅ .env 파일이 존재합니다: {env_path}")
    else:
        print(f"❌ .env 파일이 없습니다: {env_path}")
        print("   SETUP_BINANCE_API.md 파일을 참고하여 .env 파일을 생성하세요")
    
    print("=" * 50)
    
    # 권장사항
    if not BinanceSettings.is_configured():
        print("\n📋 권장사항:")
        print("1. 바이낸스 계정에서 API 키를 생성하세요")
        print("2. .env 파일에 API 키를 설정하세요")
        print("3. API 키는 읽기 전용 권한만 설정하세요")
        print("4. IP 제한을 설정하여 보안을 강화하세요")
    
    return BinanceSettings.is_configured()

def test_api_connection():
    """API 연결 테스트"""
    if not BinanceSettings.is_configured():
        print("❌ API 키가 설정되지 않아 연결 테스트를 건너뜁니다.")
        return False
    
    try:
        from modules.collectors.binance_futures_collector import BinanceFuturesCollector, BinanceConfig
        
        config = BinanceConfig(
            api_key=BinanceSettings.get_api_key(),
            api_secret=BinanceSettings.get_api_secret(),
            testnet=BinanceSettings.get_testnet(),
            rate_limit=BinanceSettings.get_rate_limit()
        )
        
        print("\n🔗 API 연결 테스트 중...")
        collector = BinanceFuturesCollector(config)
        
        # 간단한 API 호출 테스트
        exchange_info = collector.get_exchange_info()
        print(f"✅ API 연결 성공!")
        print(f"   사용 가능한 심볼: {len(exchange_info['symbols'])}개")
        
        return True
        
    except Exception as e:
        print(f"❌ API 연결 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 바이낸스 API 설정 확인 도구")
    print()
    
    # 설정 확인
    config_ok = check_config()
    
    # API 연결 테스트
    if config_ok:
        connection_ok = test_api_connection()
        
        if connection_ok:
            print("\n🎉 모든 설정이 완료되었습니다!")
            print("   이제 데이터 수집을 시작할 수 있습니다:")
            print("   python run_binance_collector.py")
        else:
            print("\n⚠️  API 키는 설정되었지만 연결에 실패했습니다.")
            print("   API 키 권한과 IP 제한을 확인하세요.")
    else:
        print("\n📝 다음 단계:")
        print("1. SETUP_BINANCE_API.md 파일을 읽으세요")
        print("2. .env 파일에 API 키를 설정하세요")
        print("3. 이 스크립트를 다시 실행하세요")

if __name__ == "__main__":
    main() 