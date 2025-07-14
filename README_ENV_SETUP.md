# 🔧 환경변수 설정 가이드

## 📋 개요

이 문서는 투자 시스템에 필요한 모든 환경변수를 설정하는 방법을 설명합니다.

## 🚀 빠른 시작

### 1. 환경변수 파일 복사
```bash
# Windows
copy env_complete.txt .env

# Linux/Mac
cp env_complete.txt .env
```

### 2. API 키 설정
`.env` 파일을 열고 다음 항목들을 실제 값으로 변경하세요:

## 🔑 필수 API 키 설정

### 한국투자증권 API
```bash
# 실거래 API (실제 거래용)
LIVE_KIS_APP_KEY=your_actual_live_app_key
LIVE_KIS_APP_SECRET=your_actual_live_app_secret
LIVE_KIS_ACCOUNT_NUMBER=your_actual_account_number

# 모의투자 API (테스트용)
PAPER_KIS_APP_KEY=your_actual_paper_app_key
PAPER_KIS_APP_SECRET=your_actual_paper_app_secret
PAPER_KIS_ACCOUNT_NUMBER=your_actual_paper_account
```

### 데이터 수집 API
```bash
# Alpha Vantage (해외 주식)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# DART (금융감독원)
DART_API_KEY=your_dart_api_key

# BOK (한국은행)
BOK_API_KEY=your_bok_api_key
```

### 알림 설정
```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## 📊 환경변수 카테고리별 설명

### 🔐 보안 관련
- `ENCRYPTION_KEY`: 32자리 암호화 키
- `JWT_SECRET_KEY`: JWT 토큰 암호화 키
- `SESSION_SECRET_KEY`: 세션 암호화 키

### 💾 데이터베이스 설정
- `DB_HOST`, `DB_PORT`, `DB_NAME`: PostgreSQL 설정
- `REDIS_HOST`, `REDIS_PORT`: Redis 캐시 설정
- `SQLITE_DB_PATH`: SQLite 데이터베이스 경로

### ☁️ 클라우드 스토리지
- `NAVER_CLOUD_ACCESS_KEY`: 네이버 클라우드 액세스 키
- `AWS_ACCESS_KEY_ID`: AWS S3 액세스 키
- `GITHUB_TOKEN`: GitHub 백업용 토큰

### 🤖 AI/ML 설정
- `GEMINI_API_KEY`: Google Gemini AI API 키
- `TF_GPU_MEMORY_LIMIT`: TensorFlow GPU 메모리 제한
- `TORCH_DEVICE`: PyTorch 디바이스 설정

### 📈 성능 최적화
- `MAX_WORKERS`: 최대 워커 프로세스 수
- `ASYNC_CONCURRENCY_LIMIT`: 비동기 동시 처리 제한
- `CACHE_TTL`: 캐시 유효 시간

## 🔧 설정 단계별 가이드

### 1단계: 기본 API 키 설정
```bash
# 1. 한국투자증권 개발자센터에서 API 키 발급
# https://securities.koreainvestment.com/main/index.jsp

# 2. Alpha Vantage API 키 발급
# https://www.alphavantage.co/support/#api-key

# 3. DART API 키 발급
# https://opendart.fss.or.kr/guide/main.do
```

### 2단계: 데이터베이스 설정
```bash
# PostgreSQL 설치 및 설정
# Redis 설치 및 설정
# 또는 SQLite 사용 (개발용)
```

### 3단계: 알림 설정
```bash
# Telegram Bot 생성
# 1. @BotFather에서 봇 생성
# 2. 봇 토큰 획득
# 3. 채팅 ID 확인
```

### 4단계: 클라우드 스토리지 설정
```bash
# Naver Cloud Platform 또는 AWS S3 설정
# 백업용 스토리지 구성
```

## 🛡️ 보안 주의사항

### ✅ 해야 할 것
- `.env` 파일을 `.gitignore`에 포함
- API 키를 정기적으로 갱신
- 프로덕션에서는 강력한 암호화 사용
- 백업 파일을 안전한 곳에 보관

### ❌ 하지 말아야 할 것
- API 키를 Git에 커밋하지 않기
- `.env` 파일을 공개 저장소에 업로드하지 않기
- API 키를 로그에 출력하지 않기
- 개발용 키를 프로덕션에서 사용하지 않기

## 🔍 환경변수 검증

### Python 스크립트로 검증
```python
import os
from dotenv import load_dotenv

load_dotenv()

# 필수 API 키 확인
required_keys = [
    'LIVE_KIS_APP_KEY',
    'ALPHA_VANTAGE_API_KEY',
    'DART_API_KEY'
]

missing_keys = []
for key in required_keys:
    if not os.getenv(key):
        missing_keys.append(key)

if missing_keys:
    print(f"❌ 누락된 API 키: {missing_keys}")
else:
    print("✅ 모든 필수 API 키가 설정되었습니다.")
```

### 환경변수 테스트
```bash
# Windows
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('KIS_APP_KEY:', os.getenv('LIVE_KIS_APP_KEY')[:10] + '...' if os.getenv('LIVE_KIS_APP_KEY') else 'Not set')"

# Linux/Mac
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('KIS_APP_KEY:', os.getenv('LIVE_KIS_APP_KEY')[:10] + '...' if os.getenv('LIVE_KIS_APP_KEY') else 'Not set')"
```

## 🚨 문제 해결

### 일반적인 문제들

#### 1. API 키 인증 실패
```bash
# 해결 방법:
# 1. API 키가 올바른지 확인
# 2. API 키 권한이 올바른지 확인
# 3. 계정 상태가 정상인지 확인
```

#### 2. 환경변수가 로드되지 않음
```python
# 해결 방법:
import os
from dotenv import load_dotenv

# .env 파일 경로 명시
load_dotenv('.env')

# 환경변수 확인
print(os.getenv('LIVE_KIS_APP_KEY'))
```

#### 3. 데이터베이스 연결 실패
```bash
# 해결 방법:
# 1. 데이터베이스 서비스가 실행 중인지 확인
# 2. 연결 정보가 올바른지 확인
# 3. 방화벽 설정 확인
```

## 📝 환경별 설정

### 개발 환경
```bash
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=DEBUG
SIMULATE_TRADING=True
```

### 스테이징 환경
```bash
ENVIRONMENT=staging
DEBUG=False
LOG_LEVEL=INFO
SIMULATE_TRADING=True
```

### 프로덕션 환경
```bash
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=WARNING
SIMULATE_TRADING=False
```

## 🔄 자동화 스크립트

### 환경변수 설정 스크립트
```bash
#!/bin/bash
# setup_env.sh

echo "🔧 환경변수 설정 시작..."

# .env 파일 복사
cp env_complete.txt .env

echo "📝 .env 파일이 생성되었습니다."
echo "🔑 API 키를 설정해주세요."
echo "📖 README_ENV_SETUP.md를 참고하세요."
```

### 환경변수 검증 스크립트
```python
#!/usr/bin/env python3
# verify_env.py

import os
from dotenv import load_dotenv

def verify_environment():
    """환경변수 검증"""
    load_dotenv()
    
    # 필수 API 키 목록
    required_keys = {
        'LIVE_KIS_APP_KEY': '한국투자증권 API 키',
        'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage API 키',
        'DART_API_KEY': 'DART API 키',
        'TELEGRAM_BOT_TOKEN': 'Telegram 봇 토큰'
    }
    
    missing_keys = []
    valid_keys = []
    
    for key, description in required_keys.items():
        value = os.getenv(key)
        if value and value != f'your_{key.lower()}_here':
            valid_keys.append(f"✅ {description}")
        else:
            missing_keys.append(f"❌ {description}")
    
    print("🔍 환경변수 검증 결과:")
    print("\n".join(valid_keys))
    if missing_keys:
        print("\n⚠️  설정이 필요한 항목:")
        print("\n".join(missing_keys))
    else:
        print("\n🎉 모든 필수 환경변수가 설정되었습니다!")

if __name__ == "__main__":
    verify_environment()
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **API 키 발급**: 각 서비스의 개발자 센터에서 API 키를 발급받았는지 확인
2. **권한 설정**: API 키에 필요한 권한이 부여되었는지 확인
3. **계정 상태**: 각 서비스의 계정이 정상 상태인지 확인
4. **네트워크**: 인터넷 연결이 정상인지 확인

---

**⚠️ 주의**: 이 파일에는 민감한 정보가 포함되어 있으므로 절대 공개 저장소에 업로드하지 마세요! 