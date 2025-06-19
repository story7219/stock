import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- API Keys ---
# 아래 주석 처리된 부분에 실제 API 키를 입력하거나 .env 파일에 추가하세요.
# 예: DART_API_KEY="your_dart_api_key"
DART_API_KEY = os.getenv("DART_API_KEY", "YOUR_DART_API_KEY_HERE")
KIS_API_KEY = os.getenv("KIS_API_KEY", "YOUR_KIS_API_KEY_HERE")
KIS_SECRET_KEY = os.getenv("KIS_SECRET_KEY", "YOUR_KIS_SECRET_KEY_HERE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")


# --- File Paths ---
# 데이터와 결과물이 저장될 기본 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
REPORT_PATH = os.path.join(BASE_DIR, "report")
LOG_PATH = os.path.join(BASE_DIR, "logs")

# --- Data Collection Settings ---
# 비동기 요청 시 동시 실행 개수 제한
SEMAPHORE_LIMIT = 20
# 외국인/기관 매매 동향 조회 기간 (개월)
KIS_MONTHS_TO_FETCH = 3
# 뉴스 조회 기간 (일)
NEWS_DAYS_TO_FETCH = 30
# 공시 조회 기간 (개월)
DART_MONTHS_TO_FETCH = 6

# --- Logging Settings ---
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOG_PATH, "stock_analyzer.log")

# --- Utility Functions ---
def ensure_dir_exists():
    """프로젝트에 필요한 디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(REPORT_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True) 