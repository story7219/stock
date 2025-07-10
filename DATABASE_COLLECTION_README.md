# 🗄️ 데이터베이스 기반 고성능 실시간 데이터 수집 시스템

## 📋 개요

이 시스템은 KIS API를 통해 최대한 많은 실시간 주식 데이터를 수집하여 PostgreSQL 데이터베이스에 저장하는 고성능 시스템입니다.

## ✨ 주요 기능

### 🚀 고성능 데이터 수집
- **실시간 수집**: 1초 간격으로 실시간 데이터 수집
- **배치 처리**: 대용량 데이터를 효율적으로 처리
- **동시 처리**: 최대 50개 동시 요청으로 성능 최적화
- **캐싱**: Redis를 통한 중복 요청 방지

### 📊 데이터 관리
- **자동 파티셔닝**: 월별 파티셔닝으로 성능 향상
- **데이터 보관 정책**: 자동으로 오래된 데이터 정리
- **백업 시스템**: 24시간마다 자동 백업
- **인덱스 최적화**: 빠른 쿼리를 위한 인덱스 자동 생성

### 🔧 시스템 모니터링
- **실시간 통계**: 수집 성공률, 처리량 모니터링
- **성능 지표**: 데이터베이스 쓰기, 캐시 히트율 추적
- **오류 처리**: 자동 재시도 및 오류 로깅

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   KIS API       │    │   Redis Cache   │    │   PostgreSQL    │
│                 │    │                 │    │                 │
│ • 실시간 데이터  │───▶│ • 중복 방지     │───▶│ • 주식 가격     │
│ • 호가 데이터    │    │ • 성능 향상     │    │ • 호가 데이터    │
│ • 거래량 데이터  │    │ • TTL 관리      │    │ • 시장 데이터    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   배치 처리      │
                       │                 │
                       │ • 1000개씩 처리  │
                       │ • 자동 저장      │
                       │ • 오류 복구      │
                       └─────────────────┘
```

## 📦 설치 및 설정

### 1. 필수 요구사항

```bash
# Python 3.11+
python --version

# PostgreSQL 14+
# Redis 6+
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
```
sqlalchemy>=2.0.0
asyncpg>=0.28.0
redis>=4.5.0
aiohttp>=3.8.0
pandas>=1.5.0
numpy>=1.24.0
pykis>=0.7.0
python-dotenv>=1.0.0
```

### 3. 환경 설정

`env_example.txt`를 참고하여 `.env` 파일을 생성하세요:

```bash
cp env_example.txt .env
```

필수 환경변수:
```env
# KIS API 설정
LIVE_KIS_APP_KEY=your_live_kis_app_key_here
LIVE_KIS_APP_SECRET=your_live_kis_app_secret_here
LIVE_KIS_ACCOUNT_NUMBER=your_account_number_here

# PostgreSQL 설정
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

# Redis 설정
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 4. 데이터베이스 설정

```bash
# 데이터베이스 초기 설정
python run_data_collection.py --mode setup
```

## 🚀 사용법

### 1. 테스트 모드 (소량 데이터)

```bash
python run_data_collection.py --mode test
```

- 5개 종목만 수집
- 5초 간격으로 수집
- 빠른 테스트용

### 2. 실시간 모드 (기본)

```bash
python run_data_collection.py --mode realtime
```

- 모든 주요 종목 수집
- 1초 간격으로 수집
- 일반적인 사용

### 3. 프로덕션 모드 (고성능)

```bash
python run_data_collection.py --mode production
```

- 최대 성능으로 수집
- 모든 종목 대상
- 서버 환경용

### 4. 데이터베이스 설정만

```bash
python run_data_collection.py --mode setup
```

## 📊 데이터 구조

### 1. 주식 가격 테이블 (stock_prices)

```sql
CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,           -- 종목코드
    timestamp TIMESTAMP NOT NULL,          -- 시간
    current_price DECIMAL(15,2) NOT NULL,  -- 현재가
    open_price DECIMAL(15,2),              -- 시가
    high_price DECIMAL(15,2),              -- 고가
    low_price DECIMAL(15,2),               -- 저가
    prev_close DECIMAL(15,2),              -- 전일종가
    change_rate DECIMAL(10,4),             -- 등락률
    volume BIGINT,                         -- 거래량
    category VARCHAR(20),                  -- 카테고리 (kospi/kosdaq/futures)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. 호가 테이블 (order_books)

```sql
CREATE TABLE order_books (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,           -- 종목코드
    timestamp TIMESTAMP NOT NULL,          -- 시간
    bid_prices JSONB,                      -- 매수 호가
    ask_prices JSONB,                      -- 매도 호가
    bid_volumes JSONB,                     -- 매수 수량
    ask_volumes JSONB,                     -- 매도 수량
    category VARCHAR(20),                  -- 카테고리
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. 시장 데이터 테이블 (market_data)

```sql
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,           -- 종목코드
    timestamp TIMESTAMP NOT NULL,          -- 시간
    data_type VARCHAR(50),                 -- 데이터 타입
    data JSONB,                            -- JSON 데이터
    category VARCHAR(20),                  -- 카테고리
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 성능 최적화

### 1. 인덱스 최적화

```sql
-- 복합 인덱스 (종목 + 시간)
CREATE INDEX idx_stock_prices_symbol_timestamp 
ON stock_prices (symbol, timestamp DESC);

-- 시간 인덱스
CREATE INDEX idx_stock_prices_timestamp 
ON stock_prices (timestamp DESC);

-- 카테고리 인덱스
CREATE INDEX idx_stock_prices_category 
ON stock_prices (category);
```

### 2. 파티셔닝

```sql
-- 월별 파티셔닝
CREATE TABLE stock_prices_partitioned (
    LIKE stock_prices INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- 파티션 생성
CREATE TABLE stock_prices_2025_01
PARTITION OF stock_prices_partitioned
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

### 3. 캐싱 전략

- **Redis TTL**: 1초 캐시로 중복 요청 방지
- **배치 처리**: 1000개씩 묶어서 처리
- **연결 풀**: PostgreSQL 연결 풀 최적화

## 📈 모니터링 및 통계

### 1. 실시간 통계

```
📊 시스템 상태:
   실행 시간: 2:30:15
   총 요청: 125,430
   성공률: 98.5%
   데이터베이스 쓰기: 375,890
   캐시 히트율: 85.2%
   버퍼 크기: 가격=156, 호가=89, 시장=23
```

### 2. 성능 지표

- **처리량**: 초당 1,000+ 데이터 포인트
- **성공률**: 98% 이상
- **지연시간**: 평균 50ms 이하
- **메모리 사용량**: 2GB 이하

## 🔄 데이터 보관 정책

### 1. 자동 정리

```python
# 가격 데이터: 1년 보관
price_data_retention_days = 365

# 호가 데이터: 30일 보관
orderbook_data_retention_days = 30

# 시장 데이터: 90일 보관
market_data_retention_days = 90
```

### 2. 백업 정책

```python
# 24시간마다 백업
backup_interval_hours = 24

# 백업 30일 보관
backup_retention_days = 30
```

## 🛠️ 문제 해결

### 1. 일반적인 문제

**문제**: PostgreSQL 연결 실패
```bash
# PostgreSQL 서비스 확인
sudo systemctl status postgresql

# 연결 테스트
psql -h localhost -U postgres -d trading_data
```

**문제**: Redis 연결 실패
```bash
# Redis 서비스 확인
sudo systemctl status redis

# 연결 테스트
redis-cli ping
```

**문제**: KIS API 인증 실패
```bash
# 환경변수 확인
echo $LIVE_KIS_APP_KEY
echo $LIVE_KIS_APP_SECRET
```

### 2. 성능 문제

**문제**: 느린 데이터 수집
```python
# 동시 요청 수 증가
MAX_CONCURRENT_REQUESTS = 100

# 배치 크기 증가
BATCH_SIZE = 2000
```

**문제**: 메모리 사용량 높음
```python
# 배치 크기 감소
BATCH_SIZE = 500

# 수집 간격 증가
COLLECTION_INTERVAL = 2.0
```

## 📝 로그 분석

### 1. 로그 파일 위치

```
collection.log          # 메인 로그
database_collection.log # 데이터베이스 로그
max_data_collection.log # 파일 기반 수집 로그
```

### 2. 로그 레벨

```python
# 개발 환경
LOG_LEVEL = DEBUG

# 프로덕션 환경
LOG_LEVEL = INFO
```

## 🔮 향후 계획

### 1. 기능 개선

- [ ] 실시간 알림 시스템
- [ ] 웹 대시보드
- [ ] API 엔드포인트
- [ ] 머신러닝 통합

### 2. 성능 개선

- [ ] 클러스터링 지원
- [ ] 스트리밍 처리
- [ ] 압축 저장
- [ ] CDN 연동

### 3. 확장성

- [ ] 마이크로서비스 아키텍처
- [ ] 컨테이너화 (Docker)
- [ ] 쿠버네티스 배포
- [ ] 클라우드 네이티브

## 📞 지원

문제가 발생하거나 질문이 있으시면:

1. 로그 파일 확인
2. 환경변수 설정 확인
3. 데이터베이스 연결 상태 확인
4. KIS API 키 유효성 확인

## 📄 라이선스

MIT License

---

**🎯 이 시스템으로 최대한 많은 실시간 데이터를 효율적으로 수집하고 관리할 수 있습니다!** 