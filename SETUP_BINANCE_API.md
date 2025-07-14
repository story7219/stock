# 바이낸스 API 키 설정 가이드

## 🔑 API 키 설정 방법

### 1. 바이낸스 계정에서 API 키 생성

1. **바이낸스 로그인**: https://www.binance.com
2. **API 관리**: 프로필 → API 관리
3. **새 API 키 생성**: 
   - API 키 이름 입력 (예: "Data Collector")
   - 권한 설정: **읽기 전용** (Read Only)
   - IP 제한: 선택사항 (보안 강화용)

### 2. .env 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가:

```bash
# 바이낸스 API 설정
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_api_secret_here
BINANCE_TESTNET=false
BINANCE_RATE_LIMIT=1200

# 데이터베이스 설정
DATABASE_URL=sqlite:///data/trading_data.db

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=logs/trading_system.log

# 데이터 수집 설정
DATA_COLLECTION_INTERVAL=3600
MAX_RETRIES=3
TIMEOUT=30
```

### 3. API 키 보안 주의사항

⚠️ **중요**: API 키는 절대 공개하지 마세요!

- `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 업로드되지 않습니다
- API 키는 **읽기 전용** 권한만 설정하세요
- IP 제한을 설정하여 보안을 강화하세요

### 4. 설정 확인

```bash
# 설정이 올바른지 확인
python -c "
from config.binance_config import BinanceSettings
print(f'API 키 설정됨: {BinanceSettings.is_configured()}')
print(f'테스트넷: {BinanceSettings.get_testnet()}')
print(f'요청 제한: {BinanceSettings.get_rate_limit()}/분')
"
```

### 5. 테스트 실행

```bash
# 데이터 수집 테스트
python run_binance_collector.py
```

## 🔧 고급 설정

### 테스트넷 사용 (개발용)
```bash
# .env 파일에서
BINANCE_TESTNET=true
```

### 요청 제한 조정
```bash
# .env 파일에서
BINANCE_RATE_LIMIT=2400  # 인증된 API 키 사용 시
```

### IP 제한 설정
바이낸스 API 관리 페이지에서:
- 특정 IP 주소만 허용
- 서브넷 범위 설정
- 지역별 제한

## 🚨 문제 해결

### 1. API 키 오류
```
BinanceAPIException: APIError(code=-2015): Invalid API-key
```
**해결**: API 키와 시크릿을 다시 확인

### 2. 권한 오류
```
BinanceAPIException: APIError(code=-2014): API-key format invalid
```
**해결**: API 키 권한을 읽기 전용으로 설정

### 3. IP 제한 오류
```
BinanceAPIException: APIError(code=-2015): Invalid API-key
```
**해결**: 바이낸스에서 현재 IP를 허용 목록에 추가

### 4. 요청 제한 오류
```
BinanceAPIException: APIError(code=-429): Too many requests
```
**해결**: BINANCE_RATE_LIMIT 값을 낮춰서 재시도

## 📊 API 키별 제한

| API 키 타입 | 분당 요청 제한 | 일일 요청 제한 |
|------------|---------------|---------------|
| 공개 API | 1,200 | 50,000 |
| 읽기 전용 | 2,400 | 100,000 |
| 거래 권한 | 2,400 | 100,000 |

## 🔍 설정 확인 스크립트

```python
#!/usr/bin/env python3
# check_binance_config.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.binance_config import BinanceSettings, get_binance_config

def check_config():
    """바이낸스 설정 확인"""
    print("🔍 바이낸스 API 설정 확인")
    print("=" * 40)
    
    # API 키 설정 확인
    api_key = BinanceSettings.get_api_key()
    api_secret = BinanceSettings.get_api_secret()
    
    if api_key and api_secret:
        print("✅ API 키가 설정되어 있습니다")
        print(f"   API 키: {api_key[:10]}...")
        print(f"   API 시크릿: {api_secret[:10]}...")
    else:
        print("❌ API 키가 설정되지 않았습니다")
        print("   .env 파일을 확인하세요")
    
    # 기타 설정 확인
    print(f"   테스트넷: {BinanceSettings.get_testnet()}")
    print(f"   요청 제한: {BinanceSettings.get_rate_limit()}/분")
    
    # 설정된 권한 확인
    if BinanceSettings.is_configured():
        print("✅ 인증된 API로 더 높은 요청 제한 사용 가능")
    else:
        print("⚠️  공개 API로 제한된 요청 제한 사용")
    
    print("=" * 40)

if __name__ == "__main__":
    check_config()
```

## 📞 지원

API 키 설정에 문제가 있으면:
1. 바이낸스 API 관리 페이지 확인
2. IP 제한 설정 확인
3. API 키 권한 확인
4. .env 파일 형식 확인

---

**보안 우선**: API 키는 항상 안전하게 보관하세요! 