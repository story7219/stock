"""
.env 파일에서 환경 변수를 로드하고,
모의투자인지 실전투자인지에 따라 적절한 API 키와 계좌번호를 선택하여 제공합니다.
"""
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# --- 환경 설정 ---
# IS_MOCK이 'true' (대소문자 무관)이면 모의투자, 아니면 실전투자로 판단합니다.
IS_MOCK = os.getenv('IS_MOCK', 'true').lower() == 'true'

# --- KIS API 자격증명 ---
# 모의투자 여부에 따라 적절한 KIS API 정보를 선택합니다.
if IS_MOCK:
    KIS_APP_KEY = os.getenv('MOCK_KIS_APP_KEY')
    KIS_APP_SECRET = os.getenv('MOCK_KIS_APP_SECRET')
    KIS_ACCOUNT_NO = os.getenv('MOCK_KIS_ACCOUNT_NUMBER')
    KIS_BASE_URL = "https://openapivts.koreainvestment.com:29443"  # 모의투자 서버
else:
    KIS_APP_KEY = os.getenv('LIVE_KIS_APP_KEY')
    KIS_APP_SECRET = os.getenv('LIVE_KIS_APP_SECRET')
    KIS_ACCOUNT_NO = os.getenv('LIVE_KIS_ACCOUNT_NUMBER')
    KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"  # 실전투자 서버

# --- 외부 서비스 API 키 ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')
GOOGLE_SPREADSHEET_ID = os.getenv('GOOGLE_SPREADSHEET_ID')
GOOGLE_WORKSHEET_NAME = os.getenv('GOOGLE_WORKSHEET_NAME')

ZAPIER_NLA_API_KEY = os.getenv('ZAPIER_NLA_API_KEY')

# --- 시간 설정 ---
# 환경 변수에서 값을 읽어오되, 없으면 기본값(정수)을 사용합니다.
TOKEN_ISSUE_HOUR = int(os.getenv('TOKEN_ISSUE_HOUR', 6))
TOKEN_ISSUE_MINUTE = int(os.getenv('TOKEN_ISSUE_MINUTE', 0))

# --- 설정값 확인용 출력 ---
# 이 파일이 임포트될 때, 어떤 설정으로 실행되는지 명확히 보여줍니다.
print("=" * 50)
print("환경 변수 설정이 로드되었습니다.")
if IS_MOCK:
    print("✅ [모의투자] 모드로 실행됩니다.")
else:
    print("🔥 [실전투자] 모드로 실행됩니다.")
print(f"  - KIS APP KEY: ...{KIS_APP_KEY[-4:] if KIS_APP_KEY else ''}")
print(f"  - KIS ACCOUNT: {KIS_ACCOUNT_NO}")
print(f"  - KIS BASE URL: {KIS_BASE_URL}")
print("=" * 50)

# === 거래 안전 설정 ===
MINIMUM_CASH_RATIO = float(os.getenv('MINIMUM_CASH_RATIO', '0.1'))  # 최소 현금 비중 10%

# === 설정 검증 함수 ===
def validate_config():
    """필수 설정값들이 모두 설정되었는지 확인"""
    missing_configs = []
    
    # 한국투자증권 API 필수 설정
    if not KIS_APP_KEY:
        missing_configs.append(f"{'MOCK_KIS_APP_KEY' if IS_MOCK else 'LIVE_KIS_APP_KEY'}")
    if not KIS_APP_SECRET:
        missing_configs.append(f"{'MOCK_KIS_APP_SECRET' if IS_MOCK else 'LIVE_KIS_APP_SECRET'}")
    if not KIS_ACCOUNT_NO:
        missing_configs.append(f"{'MOCK_KIS_ACCOUNT_NUMBER' if IS_MOCK else 'LIVE_KIS_ACCOUNT_NUMBER'}")
    
    # 선택적 설정들 확인 (경고만 출력)
    optional_configs = []
    if not TELEGRAM_BOT_TOKEN:
        optional_configs.append('TELEGRAM_BOT_TOKEN')
    if not TELEGRAM_CHAT_ID:
        optional_configs.append('TELEGRAM_CHAT_ID')
    if not GEMINI_API_KEY:
        optional_configs.append('GEMINI_API_KEY')
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        optional_configs.append('GOOGLE_SERVICE_ACCOUNT_FILE')
    if not GOOGLE_SPREADSHEET_ID:
        optional_configs.append('GOOGLE_SPREADSHEET_ID')
    
    return missing_configs, optional_configs

# === 설정 출력 함수 ===
def print_config_status():
    """현재 설정 상태를 출력"""
    print("=" * 50)
    print(f"🔧 트레이딩 시스템 설정 상태")
    print("=" * 50)
    print(f"📊 거래 모드: {'모의투자' if IS_MOCK else '실전투자'}")
    print(f"🏢 서버 URL: {KIS_BASE_URL}")
    print()
    
    # 한국투자증권 API 설정 상태
    print("📈 한국투자증권 API:")
    print(f"   APP_KEY: {'✅ 설정됨' if KIS_APP_KEY else '❌ 누락'}")
    print(f"   APP_SECRET: {'✅ 설정됨' if KIS_APP_SECRET else '❌ 누락'}")
    print(f"   ACCOUNT_NUMBER: {'✅ 설정됨' if KIS_ACCOUNT_NO else '❌ 누락'}")
    
    # 기타 서비스 설정 상태
    print("\n🤖 기타 서비스:")
    print(f"   Telegram Bot: {'✅ 설정됨' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else '⚠️ 미설정'}")
    print(f"   Gemini AI: {'✅ 설정됨' if GEMINI_API_KEY else '⚠️ 미설정'}")
    print(f"   Google Sheets: {'✅ 설정됨' if GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_SPREADSHEET_ID else '⚠️ 미설정'}")
    print(f"   Zapier: {'✅ 설정됨' if ZAPIER_NLA_API_KEY else '⚠️ 미설정'}")
    
    print("=" * 50)

# === 초기화 시 설정 검증 ===
if __name__ == "__main__":
    missing, optional = validate_config()
    
    if missing:
        print("❌ 다음 필수 설정이 누락되었습니다:")
        for config in missing:
            print(f"   - {config}: *** 누락 ***")
        print("\n.env 파일을 확인해주세요.")
    else:
        print("✅ 모든 필수 설정이 완료되었습니다!")
        
    if optional:
        print("\n⚠️ 다음 선택적 설정이 누락되었습니다:")
        for config in optional:
            print(f"   - {config}: *** 누락 ***")
        print("필요에 따라 .env 파일에 추가해주세요.")
    
    print_config_status()
