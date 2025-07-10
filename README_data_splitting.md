# 데이터 분리 전략 가이드

## 📋 개요

이 문서는 트레이딩 AI 시스템에서 사용할 수 있는 다양한 데이터 분리 전략을 설명합니다. 일반적인 머신러닝과 달리, 금융 데이터는 시계열 특성과 look-ahead bias 방지가 중요합니다.

## 🎯 주요 특징

### 1. 일반 데이터 분리 (`data_split_strategies.py`)
- **무작위 분리**: 기본적인 데이터 분리
- **계층적 분리**: 클래스 불균형 처리
- **시계열 분리**: 시간 순서 고려
- **교차 검증**: K-Fold, Stratified K-Fold, TimeSeries

### 2. 트레이딩 전용 분리 (`trading_data_splitter.py`)
- **시장 체제 감지**: Bull/Bear/Sideways 시장 구분
- **변동성 기반 분리**: 낮은/중간/높은 변동성 구간 분리
- **계절성 고려**: 연도별 분리
- **Look-ahead bias 방지**: 엄격한 시간 순서 준수

## 📊 데이터 분리 비율

### 표준 비율
```python
# 가장 일반적인 비율
train:val:test = 6:2:2  # 60% 훈련, 20% 검증, 20% 테스트

# 검증 세트 없는 경우
train:test = 7:3        # 70% 훈련, 30% 테스트
train:test = 8:2        # 80% 훈련, 20% 테스트
```

### 대규모 데이터셋
```python
# 수십만~수백만 개 데이터
train:val:test = 98:1:1  # 98% 훈련, 1% 검증, 1% 테스트
```

## 🚀 사용 방법

### 1. 기본 데이터 분리

```python
from data_split_strategies import SplitConfig, DataSplitter

# 설정
config = SplitConfig(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    stratified=True,  # 계층적 분리
    random_state=42
)

# 분리기 생성
splitter = DataSplitter(config)

# 데이터 분리
result = splitter.split_data(X, y)

# 결과 확인
print(f"훈련 크기: {result.train_size}")
print(f"검증 크기: {result.val_size}")
print(f"테스트 크기: {result.test_size}")
```

### 2. 트레이딩 데이터 분리

```python
from trading_data_splitter import TradingSplitConfig, TradingDataSplitter

# 설정
config = TradingSplitConfig(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    detect_market_regimes=True,  # 시장 체제 감지
    volatility_based_split=False,
    min_train_days=252,  # 최소 1년
    random_state=42
)

# 분리기 생성
splitter = TradingDataSplitter(config)

# 데이터 분리
result = splitter.split_trading_data(
    data=df,
    target_column='target',
    feature_columns=['close', 'volume', 'rsi', 'ma_20']
)

# 품질 분석
analysis = splitter.analyze_split_quality(result)
print(f"훈련 기간: {result.train_period}")
print(f"시장 체제 수: {len(result.market_regimes)}")
```

## 📈 분리 전략별 특징

### 1. 무작위 분리 (Random Split)
```python
# 장점
- 구현이 간단
- 빠른 처리 속도
- 일반적인 머신러닝에 적합

# 단점
- 시계열 데이터에 부적합
- Look-ahead bias 발생 가능
- 시장 체제 변화 무시
```

### 2. 시계열 분리 (Time Series Split)
```python
# 장점
- 시간 순서 보장
- Look-ahead bias 방지
- 실제 트레이딩 환경과 유사

# 단점
- 최신 데이터가 테스트에만 포함
- 과거 데이터의 패턴 변화 무시
```

### 3. 시장 체제 기반 분리
```python
# 장점
- 다양한 시장 환경에서의 성능 평가
- 체제별 모델 최적화 가능
- 실제 시장 변화 반영

# 단점
- 복잡한 구현
- 체제 감지 정확도에 의존
```

### 4. 변동성 기반 분리
```python
# 장점
- 변동성 구간별 성능 평가
- 리스크 관리에 유용
- 극단적 시장 상황 대비

# 단점
- 변동성 계산 방법에 의존
- 균형잡힌 데이터 분포 어려움
```

## 🔧 고급 설정

### 1. 교차 검증 설정
```python
config = SplitConfig(
    cv_folds=5,
    cv_strategy='timeseries',  # 'kfold', 'stratified_kfold', 'timeseries'
    random_state=42
)

# 교차 검증 실행
cv_results = splitter.cross_validate(X, y, estimator, scoring='accuracy')
print(f"평균 점수: {cv_results['mean_score']:.4f}")
print(f"표준편차: {cv_results['std_score']:.4f}")
```

### 2. 시장 체제 감지 설정
```python
config = TradingSplitConfig(
    detect_market_regimes=True,
    n_regimes=3,           # 체제 개수
    regime_window=60,      # 60일 이동평균
    random_state=42
)
```

### 3. 변동성 기반 분리 설정
```python
config = TradingSplitConfig(
    volatility_based_split=True,
    vol_window=20,         # 20일 변동성
    random_state=42
)
```

## 📊 시각화 및 분석

### 1. 분리 결과 시각화
```python
# 기본 시각화
splitter.visualize_split(result, "split_visualization.png")

# 트레이딩 분석 시각화
splitter.visualize_split_analysis(result, analysis, "trading_analysis.png")
```

### 2. 품질 분석
```python
# 분리 품질 분석
analysis = splitter.analyze_split_quality(result)

# 수익률 통계
print("훈련 수익률 통계:")
print(f"  평균: {analysis['return_stats']['train']['mean']:.6f}")
print(f"  표준편차: {analysis['return_stats']['train']['std']:.6f}")
print(f"  왜도: {analysis['return_stats']['train']['skew']:.4f}")
print(f"  첨도: {analysis['return_stats']['train']['kurtosis']:.4f}")
```

## ⚠️ 주의사항

### 1. Look-ahead Bias 방지
```python
# ❌ 잘못된 방법
future_data = data[data.index > current_date]  # 미래 데이터 사용

# ✅ 올바른 방법
past_data = data[data.index <= current_date]   # 과거 데이터만 사용
```

### 2. 데이터 누수 방지
```python
# ❌ 잘못된 방법
scaler.fit(data)  # 전체 데이터로 스케일링

# ✅ 올바른 방법
scaler.fit(X_train)  # 훈련 데이터로만 스케일링
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### 3. 최소 데이터 요구사항
```python
# 트레이딩 데이터 최소 요구사항
min_train_days = 252    # 최소 1년
min_val_days = 63       # 최소 3개월
min_test_days = 63      # 최소 3개월
```

## 🎯 사용 시나리오

### 1. 단기 트레이딩 (일봉)
```python
config = TradingSplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    detect_market_regimes=True,
    min_train_days=252,  # 1년
    random_state=42
)
```

### 2. 중기 트레이딩 (주봉)
```python
config = TradingSplitConfig(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seasonal_split=True,  # 연도별 분리
    min_train_days=104,   # 2년
    random_state=42
)
```

### 3. 장기 투자 (월봉)
```python
config = TradingSplitConfig(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    volatility_based_split=True,
    min_train_days=60,    # 5년
    random_state=42
)
```

## 📝 예시 코드

### 완전한 예시
```python
import pandas as pd
from trading_data_splitter import TradingSplitConfig, TradingDataSplitter, create_sample_trading_data

# 1. 샘플 데이터 생성
data = create_sample_trading_data(n_days=1000, start_date='2020-01-01')

# 2. 설정
config = TradingSplitConfig(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    detect_market_regimes=True,
    random_state=42
)

# 3. 분리기 생성 및 실행
splitter = TradingDataSplitter(config)
result = splitter.split_trading_data(
    data=data,
    target_column='target',
    feature_columns=['close', 'volume', 'ma_5', 'ma_20', 'rsi', 'returns']
)

# 4. 품질 분석
analysis = splitter.analyze_split_quality(result)

# 5. 결과 출력
print("=== 데이터 분리 결과 ===")
print(f"훈련 기간: {result.train_period[0].date()} ~ {result.train_period[1].date()}")
print(f"검증 기간: {result.val_period[0].date()} ~ {result.val_period[1].date()}")
print(f"테스트 기간: {result.test_period[0].date()} ~ {result.test_period[1].date()}")
print(f"시장 체제 수: {len(result.market_regimes)}")

# 6. 시각화
splitter.visualize_split_analysis(result, analysis, "trading_analysis.png")
```

## 🔍 성능 최적화

### 1. 메모리 효율성
```python
# 대용량 데이터 처리
def process_large_data(data_path: str, chunk_size: int = 10000):
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        # 청크별 처리
        yield process_chunk(chunk)
```

### 2. 병렬 처리
```python
# 교차 검증 병렬 처리
cv_results = splitter.cross_validate(
    X, y, estimator, 
    scoring='accuracy',
    n_jobs=-1  # 모든 CPU 코어 사용
)
```

### 3. 캐싱
```python
# 분리 결과 캐싱
import pickle

# 결과 저장
with open('split_result.pkl', 'wb') as f:
    pickle.dump(result, f)

# 결과 로드
with open('split_result.pkl', 'rb') as f:
    result = pickle.load(f)
```

## 📚 추가 자료

### 관련 문서
- [Scikit-learn 데이터 분리 가이드](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Pandas 시계열 처리](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [금융 머신러닝 모범 사례](https://www.quantopian.com/posts/quantopian-lecture-series-machine-learning)

### 참고 논문
- "Time Series Cross-Validation Methods for Forecasting"
- "Machine Learning for Market Regime Detection"
- "Look-Ahead Bias in Financial Machine Learning"

---

**⚠️ 중요**: 트레이딩 AI 시스템에서는 데이터 분리의 정확성이 모델 성능보다 더 중요할 수 있습니다. 잘못된 분리는 과적합과 실제 성능 저하를 초래할 수 있으므로 신중하게 선택하세요. 