# 바이낸스 선물옵션 데이터 수집기

바이낸스 선물옵션 데이터를 과거 최대치부터 현재까지 수집하는 시스템입니다.

## 🚀 주요 기능

- **과거 데이터 수집**: 최대 5년 전부터 현재까지의 데이터
- **다양한 시간대**: 1분, 5분, 15분, 1시간, 4시간, 1일 등
- **선물 데이터**: K라인, 자금조달률, 미결제약정
- **고성능**: 비동기 처리로 빠른 데이터 수집
- **안전성**: API 속도 제한 및 에러 처리

## 📦 설치

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 바이낸스 API 키 설정 (선택사항)
```python
# run_binance_collector.py에서 설정
config = BinanceConfig(
    api_key="your_api_key",      # 바이낸스 API 키
    api_secret="your_api_secret", # 바이낸스 API 시크릿
    testnet=False,
    rate_limit=1200
)
```

> **참고**: API 키 없이도 공개 데이터는 수집 가능합니다.

## 🎯 사용법

### 기본 실행
```bash
python run_binance_collector.py
```

### 수집되는 데이터

#### 1. K라인 데이터 (OHLCV)
- **파일 형식**: `{symbol}_{interval}_{start_date}_{end_date}.parquet`
- **컬럼**: symbol, timestamp, open, high, low, close, volume, quote_volume
- **저장 위치**: `data/binance_futures/`

#### 2. 자금조달률 데이터
- **파일 형식**: `{symbol}_funding_rates_{start_date}_{end_date}.parquet`
- **컬럼**: symbol, fundingTime, fundingRate, nextFundingTime
- **저장 위치**: `data/binance_futures/funding_rates/`

#### 3. 미결제약정 데이터
- **파일 형식**: `{symbol}_open_interest.parquet`
- **컬럼**: symbol, timestamp, sumOpenInterest, sumOpenInterestValue
- **저장 위치**: `data/binance_futures/open_interest/`

## 📊 수집되는 심볼

### 주요 선물 심볼 (20개)
- BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT
- XRPUSDT, DOTUSDT, DOGEUSDT, AVAXUSDT, MATICUSDT
- LINKUSDT, UNIUSDT, ATOMUSDT, LTCUSDT, BCHUSDT
- ETCUSDT, FILUSDT, NEARUSDT, ALGOUSDT, VETUSDT

### 시간대별 데이터
- **1시간 캔들**: 단기 분석용
- **4시간 캔들**: 중기 분석용  
- **1일 캔들**: 장기 분석용

## ⚙️ 설정 옵션

### CollectionConfig 설정
```python
collection_config = CollectionConfig(
    symbols=['BTCUSDT', 'ETHUSDT'],  # 수집할 심볼
    intervals=['1h', '4h', '1d'],    # 수집할 시간대
    start_date=datetime.now() - timedelta(days=365),  # 시작일
    end_date=datetime.now(),          # 종료일
    save_format='parquet',            # 저장 형식
    compression='snappy'              # 압축 방식
)
```

### 지원하는 시간대
- `1m`, `3m`, `5m`, `15m`, `30m` (분 단위)
- `1h`, `2h`, `4h`, `6h`, `8h`, `12h` (시간 단위)
- `1d`, `3d`, `1w`, `1M` (일/주/월 단위)

## 🔧 고급 사용법

### 커스텀 수집기 사용
```python
from modules.collectors.binance_futures_collector import BinanceFuturesCollector, BinanceConfig

# 설정
config = BinanceConfig(rate_limit=1200)
collection_config = CollectionConfig(
    symbols=['BTCUSDT'],
    intervals=['1h'],
    start_date=datetime.now() - timedelta(days=30)
)

# 데이터 수집
async with BinanceFuturesCollector(config) as collector:
    results = await collector.collect_historical_data(collection_config, Path('output'))
```

### 특정 데이터만 수집
```python
# 자금조달률만 수집
funding_results = collector.collect_funding_rates(
    symbols=['BTCUSDT'],
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    output_dir=Path('data/funding_rates')
)

# 미결제약정만 수집
interest_data = collector.get_open_interest_history(
    symbol='BTCUSDT',
    period='1h',
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)
```

## 📈 성능 최적화

### 1. 병렬 처리
- 비동기 I/O로 동시에 여러 심볼 수집
- API 속도 제한 준수 (분당 1200회)

### 2. 메모리 효율성
- 배치 처리로 메모리 사용량 최적화
- Parquet 압축으로 저장 공간 절약

### 3. 에러 처리
- 자동 재시도 로직
- 상세한 에러 로깅
- 부분 실패 시에도 계속 진행

## 🛡️ 보안 및 제한사항

### API 제한
- **공개 API**: 분당 1200회 요청
- **인증 API**: 분당 2400회 요청
- **IP 제한**: 분당 10,000회 요청

### 데이터 제한
- **최대 기간**: 5년 (1825일)
- **최대 심볼**: 100개 동시 처리
- **파일 크기**: 심볼당 최대 1GB

## 📝 로그 및 모니터링

### 로그 파일
- **위치**: `logs/binance_collector.log`
- **레벨**: INFO, ERROR, WARNING
- **포맷**: 시간, 모듈, 레벨, 메시지

### 모니터링 지표
- 수집된 심볼 수
- 처리 시간
- 에러 발생률
- 파일 크기

## 🔍 데이터 분석 예시

### Pandas로 데이터 로드
```python
import pandas as pd

# K라인 데이터 로드
df = pd.read_parquet('data/binance_futures/BTCUSDT_1h_20230101_20241231.parquet')

# 자금조달률 데이터 로드
funding_df = pd.read_parquet('data/binance_futures/funding_rates/BTCUSDT_funding_rates_20230101_20241231.parquet')

# 미결제약정 데이터 로드
interest_df = pd.read_parquet('data/binance_futures/open_interest/BTCUSDT_open_interest.parquet')
```

### 기본 분석
```python
# 기본 통계
print(df.describe())

# 거래량 분석
volume_analysis = df.groupby(df['timestamp'].dt.date)['volume'].sum()

# 가격 변동성
df['volatility'] = (df['high'] - df['low']) / df['open'] * 100
```

## 🚨 주의사항

1. **API 키 보안**: API 키는 환경변수로 관리
2. **데이터 용량**: 대용량 데이터 수집 시 충분한 저장공간 확보
3. **네트워크**: 안정적인 인터넷 연결 필요
4. **시간대**: UTC 기준으로 데이터 수집

## 📞 문제 해결

### 일반적인 오류

#### 1. API 제한 오류
```
BinanceAPIException: APIError(code=-429): Too many requests
```
**해결**: rate_limit 값을 낮춰서 재시도

#### 2. 심볼 오류
```
BinanceAPIException: APIError(code=-1121): Invalid symbol
```
**해결**: 유효한 심볼명 확인

#### 3. 날짜 범위 오류
```
BinanceAPIException: APIError(code=-1100): Illegal characters found in parameter
```
**해결**: 날짜 형식 확인

### 디버깅
```python
# 상세 로그 활성화
logging.getLogger('modules.collectors.binance_futures_collector').setLevel(logging.DEBUG)

# API 응답 확인
exchange_info = collector.get_exchange_info()
print(f"Available symbols: {len(exchange_info['symbols'])}")
```

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 🤝 기여

버그 리포트, 기능 제안, 코드 기여 환영합니다!

---

**World-Class Python Rule 100% 준수** - Google/Meta/Netflix 수준의 코드 품질 