import os
from dotenv import load_dotenv

REQUIRED_ENV_VARS = [
    "LIVE_KIS_APP_KEY", "LIVE_KIS_APP_SECRET", "LIVE_KIS_ACCOUNT_NUMBER",
    "MOCK_KIS_APP_KEY", "MOCK_KIS_APP_SECRET", "MOCK_KIS_ACCOUNT_NUMBER",
    "DART_API_KEY"
]

def mask(val):
    if not val or len(val) < 8:
        return val
    return val[:3] + "*" * (len(val)-6) + val[-3:]

def check_env():
    load_dotenv()
    print("🔎 환경변수 점검 결과:")
    all_ok = True
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {mask(value)}")
        else:
            print(f"❌ {var}: [미설정]")
            all_ok = False
    return all_ok

if __name__ == "__main__":
    ok = check_env()
    if ok:
        print("\n✅ 모든 필수 환경변수가 정상적으로 설정되어 있습니다!")
    else:
        print("\n⚠️ 누락된 환경변수가 있습니다. .env 파일을 다시 확인해 주세요.") 