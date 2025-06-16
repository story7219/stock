"""
환경 변수 중앙 관리 모듈 (GitHub Actions 호환)
"""
import os
import logging
import logging.handlers

# GitHub Actions 환경을 위한 dotenv 모듈 안전 로드
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    print("⚠️ python-dotenv 모듈이 설치되지 않음 - 환경변수만 사용")
    DOTENV_AVAILABLE = False

# .env 파일 로드 (UTF-8 강제)
def load_dotenv_utf8():
    """UTF-8 인코딩으로 .env 파일을 강제 로드 (dotenv 모듈 없어도 동작)"""
    import os
    from pathlib import Path
    
    env_file = Path('.env')
    if not env_file.exists():
        # GitHub Actions에서는 .env 파일이 없을 수 있음
        if not os.getenv('GITHUB_ACTIONS'):
            print("📄 .env 파일이 없습니다 - 환경변수 직접 사용")
        return
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ .env 파일 로드 완료")
    except Exception as e:
        print(f"⚠️ .env 파일 로드 실패: {e}")
        # 기본 dotenv 사용 (있다면)
        if DOTENV_AVAILABLE:
            try:
                load_dotenv()
            except:
                pass

# GitHub Actions 환경 감지 및 기본값 설정
def setup_github_actions_defaults():
    """GitHub Actions 환경에서 테스트용 기본값 설정"""
    if os.getenv('GITHUB_ACTIONS'):
        print("🔧 GitHub Actions 환경 감지 - 테스트용 기본값 설정")
        
        # 기본 환경변수 설정 (테스트용)
        defaults = {
            'IS_MOCK': 'true',
            'MOCK_KIS_APP_KEY': 'test_app_key_for_github_actions',
            'MOCK_KIS_APP_SECRET': 'test_app_secret_for_github_actions', 
            'MOCK_KIS_ACCOUNT_NUMBER': '12345678-01',
            'TELEGRAM_BOT_TOKEN': 'test_telegram_token',
            'TELEGRAM_CHAT_ID': 'test_chat_id',
            'GEMINI_API_KEY': 'test_gemini_key',
            'GOOGLE_SPREADSHEET_ID': 'test_spreadsheet_id',
            'LOG_LEVEL': 'INFO'
        }
        
        for key, value in defaults.items():
            if not os.getenv(key):
                os.environ[key] = value
                print(f"  ✅ {key}: 기본값 설정됨")

# 초기화 실행
setup_github_actions_defaults()
load_dotenv_utf8()

# 환경변수 값에서 주석을 제거하는 안전한 파싱 함수
def safe_parse_env_value(env_value, default_value, value_type=str):
    """환경변수 값에서 주석을 제거하고 타입 변환하는 안전한 함수"""
    if env_value is None:
        return value_type(default_value)
    
    # 주석 제거 (# 이후 모든 내용 제거)
    clean_value = env_value.split('#')[0].strip()
    
    try:
        return value_type(clean_value)
    except (ValueError, TypeError):
        return value_type(default_value)

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
TOKEN_ISSUE_HOUR = safe_parse_env_value(os.getenv('TOKEN_ISSUE_HOUR'), 6, int)
TOKEN_ISSUE_MINUTE = safe_parse_env_value(os.getenv('TOKEN_ISSUE_MINUTE'), 0, int)

# --- 거래 안전 설정 ---
MINIMUM_CASH_RATIO = safe_parse_env_value(os.getenv('MINIMUM_CASH_RATIO'), '0.1', float)  # 최소 현금 비중 10%

# --- KIS API 호출 제한 설정 (통합 관리) ---
# 한국투자증권 공식 문서 기준: 동일 TR 초당 2회, 전체 동시 TR 초당 10회
# 실전/모의투자 구분 없이 동일 적용 (향후 변경 시 환경변수만 수정)

# 개별 API 타입별 제한 (동일 TR 기준)
ORDER_API_CALLS_PER_SEC = safe_parse_env_value(os.getenv('ORDER_API_CALLS_PER_SEC'), '2', int)
MARKET_DATA_API_CALLS_PER_SEC = safe_parse_env_value(os.getenv('MARKET_DATA_API_CALLS_PER_SEC'), '2', int)
ACCOUNT_API_CALLS_PER_SEC = safe_parse_env_value(os.getenv('ACCOUNT_API_CALLS_PER_SEC'), '2', int)

# 전체 API 통합 제한 (전체 동시 TR 기준)
TOTAL_API_CALLS_PER_SEC = safe_parse_env_value(os.getenv('TOTAL_API_CALLS_PER_SEC'), '8', int)  # 안전 마진

# 일일 호출 제한 (환경별 조정 가능)
if IS_MOCK:
    DAILY_API_LIMIT = safe_parse_env_value(os.getenv('MOCK_DAILY_API_LIMIT'), '10000', int)
else:
    DAILY_API_LIMIT = safe_parse_env_value(os.getenv('LIVE_DAILY_API_LIMIT'), '20000', int)

# 기존 설정 (하위 호환성)
API_RATE_LIMIT_CALLS = MARKET_DATA_API_CALLS_PER_SEC
API_RATE_LIMIT_PERIOD = 1

# --- 로깅 설정 ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'trading_system.log')
LOG_MAX_SIZE = safe_parse_env_value(os.getenv('LOG_MAX_SIZE'), '10485760', int)  # 10MB
LOG_BACKUP_COUNT = safe_parse_env_value(os.getenv('LOG_BACKUP_COUNT'), '5', int)

# --- 재시도 설정 ---
MAX_RETRY_ATTEMPTS = safe_parse_env_value(os.getenv('MAX_RETRY_ATTEMPTS'), '3', int)
RETRY_DELAY_SECONDS = safe_parse_env_value(os.getenv('RETRY_DELAY_SECONDS'), '1.0', float)

# === 로깅 시스템 초기화 ===
def setup_logging():
    """구조화된 로깅 시스템 설정"""
    # 로그 포맷 설정
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (회전 로그)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH, 
            maxBytes=LOG_MAX_SIZE, 
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"✅ 로그 파일 설정 완료: {LOG_FILE_PATH}")
    except Exception as e:
        logging.warning(f"⚠️ 로그 파일 설정 실패: {e}")
    
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    
    return root_logger

# 로깅 시스템 초기화
setup_logging()

# --- 설정값 확인용 출력 (GitHub Actions에서는 간소화) ---
def print_startup_info():
    """시작 시 설정 정보 출력"""
    print("=" * 50)
    print("환경 변수 설정이 로드되었습니다.")
    if IS_MOCK:
        print("✅ [모의투자] 모드로 실행됩니다.")
    else:
        print("🔥 [실전투자] 모드로 실행됩니다.")
    print(f"  - KIS APP KEY: ...{KIS_APP_KEY[-4:] if KIS_APP_KEY else '없음'}")
    print(f"  - KIS ACCOUNT: {KIS_ACCOUNT_NO}")
    print(f"  - KIS BASE URL: {KIS_BASE_URL}")
    print("=" * 50)

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
    
    # API 호출 제한 정보
    print(f"\n⚡ API 호출 제한 (통합 설정):")
    print(f"   주문 API: 초당 {ORDER_API_CALLS_PER_SEC}회")
    print(f"   시세 API: 초당 {MARKET_DATA_API_CALLS_PER_SEC}회") 
    print(f"   계좌 API: 초당 {ACCOUNT_API_CALLS_PER_SEC}회")
    print(f"   전체 제한: 초당 {TOTAL_API_CALLS_PER_SEC}회")
    print(f"   일일 제한: {DAILY_API_LIMIT}회")
    print(f"\n💡 제한 변경 시 환경변수만 수정:")
    print(f"   ORDER_API_CALLS_PER_SEC={ORDER_API_CALLS_PER_SEC}")
    print(f"   MARKET_DATA_API_CALLS_PER_SEC={MARKET_DATA_API_CALLS_PER_SEC}")
    print(f"   ACCOUNT_API_CALLS_PER_SEC={ACCOUNT_API_CALLS_PER_SEC}")
    print(f"   TOTAL_API_CALLS_PER_SEC={TOTAL_API_CALLS_PER_SEC}")
    
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

# 모듈 로드 시 자동 실행
print_startup_info()
