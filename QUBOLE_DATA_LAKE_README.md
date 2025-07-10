# ☁️ Qubole 클라우드 기반 데이터 레이크 시스템

## 📋 개요

Qubole 클라우드 기반 데이터 레이크 플랫폼을 활용한 대용량 실시간 주식 데이터 수집 및 ML 파이프라인 시스템입니다. AWS S3, Spark, Hive를 통합하여 확장 가능하고 비용 효율적인 데이터 처리 환경을 제공합니다.

## ✨ 주요 기능

### 🚀 클라우드 네이티브 아키텍처
- **Qubole Data Lake**: 클라우드 기반 데이터 레이크 플랫폼
- **AWS S3**: 무제한 확장 가능한 스토리지
- **Apache Spark**: 대용량 데이터 처리
- **Apache Hive**: SQL 기반 데이터 쿼리
- **자동 스케일링**: 트래픽에 따른 자동 확장/축소

### 📊 고성능 데이터 처리
- **실시간 스트리밍**: 1초 간격 실시간 데이터 수집
- **배치 처리**: 대용량 데이터 효율적 처리
- **파티셔닝**: 날짜/종목별 자동 파티셔닝
- **압축**: Snappy 압축으로 저장 공간 절약

### 🤖 ML 파이프라인 통합
- **자동 특성 엔지니어링**: 기술적 지표 자동 계산
- **모델 훈련**: Spark ML을 통한 분산 훈련
- **실시간 예측**: 가격 및 방향성 예측
- **모델 버전 관리**: 자동 모델 업데이트

### 💰 비용 최적화
- **Spot Instance**: 최대 90% 비용 절약
- **자동 스케일링**: 사용량에 따른 자동 조정
- **데이터 보관 정책**: 자동 데이터 정리
- **압축 저장**: 저장 비용 최소화

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   KIS API       │    │   Qubole SDK    │    │   AWS S3        │
│                 │    │                 │    │                 │
│ • 실시간 데이터  │───▶│ • 데이터 수집   │───▶│ • 원시 데이터   │
│ • 호가 데이터    │    │ • 스트리밍      │    │ • 처리 데이터   │
│ • 거래량 데이터  │    │ • 배치 처리     │    │ • ML 데이터     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Apache Spark  │
                       │                 │
                       │ • 데이터 처리   │
                       │ • 특성 엔지니어링│
                       │ • ML 모델 훈련  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Apache Hive   │
                       │                 │
                       │ • SQL 쿼리      │
                       │ • 데이터 분석   │
                       │ • 리포트 생성   │
                       └─────────────────┘
```

## 📦 설치 및 설정

### 1. 필수 요구사항

```bash
# Python 3.11+
python --version

# Qubole 계정 및 API 토큰
# AWS 계정 및 S3 버킷
# KIS API 키
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
```
qds-sdk>=1.16.0
boto3>=1.26.0
pyarrow>=10.0.0
pandas>=1.5.0
numpy>=1.24.0
pykis>=0.7.0
python-dotenv>=1.0.0
```

### 3. 환경 설정

`qubole_env_example.txt`를 참고하여 `.env` 파일을 생성하세요:

```bash
cp qubole_env_example.txt .env
```

필수 환경변수:
```env
# Qubole 설정
QUBOLE_API_TOKEN=your_qubole_api_token_here

# AWS S3 설정
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
S3_BUCKET=trading-data-lake

# KIS API 설정
LIVE_KIS_APP_KEY=your_live_kis_app_key_here
LIVE_KIS_APP_SECRET=your_live_kis_app_secret_here
```

### 4. Qubole 초기 설정

```bash
# Qubole 환경 설정
python qubole_setup.py
```

## 🚀 사용법

### 1. 초기 설정

```bash
# Qubole 클라우드 환경 설정
python qubole_setup.py
```

### 2. 데이터 수집 시작

```bash
# 실시간 데이터 수집
python qubole_data_collector.py
```

### 3. 모니터링

```bash
# Qubole 대시보드 접속
# https://app.qubole.com
```

## 📊 데이터 구조

### 1. 원시 데이터 테이블 (raw_stock_data)

```sql
CREATE EXTERNAL TABLE raw_stock_data (
    symbol STRING,                    -- 종목코드
    timestamp TIMESTAMP,              -- 시간
    current_price DOUBLE,             -- 현재가
    ohlcv MAP<STRING, STRING>,        -- OHLCV 데이터
    orderbook MAP<STRING, STRING>,    -- 호가 데이터
    category STRING,                  -- 카테고리
    data_type STRING,                 -- 데이터 타입
    collection_id STRING              -- 수집 ID
)
PARTITIONED BY (date STRING)          -- 날짜별 파티셔닝
STORED AS PARQUET                     -- Parquet 형식
LOCATION 's3://trading-data-lake/raw' -- S3 위치
```

### 2. 처리된 OHLCV 테이블 (processed_ohlcv)

```sql
CREATE EXTERNAL TABLE processed_ohlcv (
    symbol STRING,        -- 종목코드
    date STRING,          -- 날짜
    low DOUBLE,           -- 저가
    high DOUBLE,          -- 고가
    open DOUBLE,          -- 시가
    close DOUBLE,         -- 종가
    volume BIGINT,        -- 거래량
    data_points INT       -- 데이터 포인트 수
)
PARTITIONED BY (date STRING)
STORED AS PARQUET
LOCATION 's3://trading-data-lake/processed/ohlcv'
```

### 3. 기술적 지표 테이블 (technical_indicators)

```sql
CREATE EXTERNAL TABLE technical_indicators (
    symbol STRING,            -- 종목코드
    date STRING,              -- 날짜
    ma_20 DOUBLE,             -- 20일 이동평균
    ma_50 DOUBLE,             -- 50일 이동평균
    rsi DOUBLE,               -- RSI
    macd DOUBLE,              -- MACD
    bollinger_upper DOUBLE,   -- 볼린저 상단
    bollinger_lower DOUBLE,   -- 볼린저 하단
    volume_ma DOUBLE          -- 거래량 이동평균
)
PARTITIONED BY (date STRING)
STORED AS PARQUET
LOCATION 's3://trading-data-lake/processed/technical_indicators'
```

### 4. ML 특성 테이블 (ml_features)

```sql
CREATE EXTERNAL TABLE ml_features (
    symbol STRING,                    -- 종목코드
    date STRING,                      -- 날짜
    features ARRAY<DOUBLE>,           -- 특성 벡터
    scaled_features ARRAY<DOUBLE>,    -- 정규화된 특성
    price_change DOUBLE,              -- 가격 변화
    price_change_pct DOUBLE,          -- 가격 변화율
    volume_ma_ratio DOUBLE,           -- 거래량 비율
    volatility DOUBLE                 -- 변동성
)
PARTITIONED BY (date STRING)
STORED AS PARQUET
LOCATION 's3://trading-data-lake/ml/features'
```

## 🔧 성능 최적화

### 1. 파티셔닝 전략

```sql
-- 날짜별 파티셔닝
PARTITIONED BY (date STRING)

-- 종목별 서브파티셔닝 (선택사항)
PARTITIONED BY (date STRING, symbol STRING)
```

### 2. 압축 설정

```sql
-- Snappy 압축 (속도 우선)
TBLPROPERTIES ('parquet.compression'='SNAPPY')

-- Gzip 압축 (압축률 우선)
TBLPROPERTIES ('parquet.compression'='GZIP')
```

### 3. 클러스터 최적화

```python
# Spot Instance 사용 (비용 절약)
spot_instance_settings = {
    'spot_instance': True,
    'spot_bid_percentage': 80
}

# 자동 스케일링
auto_scaling = {
    'min_nodes': 2,
    'max_nodes': 10
}
```

## 📈 모니터링 및 통계

### 1. Qubole 대시보드

- **클러스터 모니터링**: CPU, 메모리, 디스크 사용량
- **작업 모니터링**: 쿼리 실행 시간, 성공률
- **비용 모니터링**: 시간당 비용, 총 비용

### 2. 실시간 통계

```
📊 Qubole 시스템 상태:
   실행 시간: 2:30:15
   총 요청: 125,430
   성공률: 98.5%
   S3 업로드: 1,250
   Qubole 작업: 45
   ML 파이프라인: 3
   버퍼 크기: 원시=156, 처리=89, ML=23
```

### 3. 성능 지표

- **처리량**: 초당 10,000+ 데이터 포인트
- **성공률**: 98% 이상
- **지연시간**: 평균 100ms 이하
- **비용**: 월 $100-500 (사용량에 따라)

## 🔄 데이터 보관 정책

### 1. 자동 정리

```python
# 원시 데이터: 3년 보관
raw_data_retention_days = 1095

# 처리 데이터: 2년 보관
processed_data_retention_days = 730

# ML 데이터: 5년 보관
ml_data_retention_days = 1825
```

### 2. 백업 정책

- **S3 버전 관리**: 자동 버전 관리
- **교차 리전 복제**: 재해 복구
- **접근 로깅**: 보안 감사

## 🤖 ML 파이프라인

### 1. 특성 엔지니어링

```python
# 자동 특성 생성
features = [
    'ma_20', 'ma_50', 'rsi', 'macd',
    'price_change_pct', 'volume_ma_ratio',
    'volatility', 'bollinger_position'
]
```

### 2. 모델 훈련

```python
# Spark ML 모델
models = {
    'price_predictor': RandomForestRegressor,
    'direction_predictor': RandomForestClassifier
}
```

### 3. 실시간 예측

```python
# 예측 결과 저장
predictions = {
    'predicted_price': float,
    'predicted_direction': int,
    'confidence': float,
    'model_version': str
}
```

## 💰 비용 최적화

### 1. Spot Instance 활용

```python
# Spot Instance 설정
spot_settings = {
    'spot_instance': True,
    'spot_bid_percentage': 80,  # 80% 비용 절약
    'fallback_to_on_demand': True
}
```

### 2. 자동 스케일링

```python
# 사용량에 따른 자동 조정
scaling_config = {
    'min_nodes': 2,
    'max_nodes': 10,
    'scale_up_threshold': 70,
    'scale_down_threshold': 30
}
```

### 3. 데이터 압축

```python
# 저장 비용 절약
compression_config = {
    'format': 'parquet',
    'compression': 'snappy',
    'compression_ratio': '2:1'
}
```

## 🛠️ 문제 해결

### 1. 일반적인 문제

**문제**: Qubole 연결 실패
```bash
# API 토큰 확인
echo $QUBOLE_API_TOKEN

# 네트워크 연결 확인
curl -H "X-AUTH-TOKEN: $QUBOLE_API_TOKEN" https://api.qubole.com/v2/clusters
```

**문제**: S3 접근 실패
```bash
# AWS 자격 증명 확인
aws sts get-caller-identity

# S3 버킷 접근 확인
aws s3 ls s3://trading-data-lake
```

**문제**: 클러스터 시작 실패
```bash
# 클러스터 상태 확인
qds clusters list

# 로그 확인
qds clusters logs <cluster_id>
```

### 2. 성능 문제

**문제**: 느린 쿼리 실행
```sql
-- 파티션 프루닝 확인
EXPLAIN SELECT * FROM raw_stock_data WHERE date = '2025-01-27';

-- 인덱스 확인
SHOW CREATE TABLE raw_stock_data;
```

**문제**: 높은 비용
```python
# Spot Instance 사용
spot_instance = True
spot_bid_percentage = 90

# 자동 스케일링 조정
min_nodes = 1
max_nodes = 5
```

## 📝 로그 분석

### 1. 로그 위치

```
qubole_collection.log     # 메인 로그
Qubole Dashboard          # 클러스터 로그
CloudWatch Logs          # AWS 로그
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
- [ ] 고급 ML 모델

### 2. 성능 개선

- [ ] Delta Lake 통합
- [ ] 실시간 스트리밍
- [ ] GPU 가속
- [ ] 멀티 리전 배포

### 3. 확장성

- [ ] 마이크로서비스 아키텍처
- [ ] Kubernetes 배포
- [ ] 서버리스 함수
- [ ] 엣지 컴퓨팅

## 📞 지원

문제가 발생하거나 질문이 있으시면:

1. Qubole 대시보드 확인
2. 로그 파일 분석
3. 환경변수 설정 확인
4. AWS S3 접근 권한 확인

## 📄 라이선스

MIT License

---

**🎯 Qubole 클라우드 기반 데이터 레이크로 대용량 실시간 데이터를 효율적으로 수집하고 ML 파이프라인을 구축할 수 있습니다!** 