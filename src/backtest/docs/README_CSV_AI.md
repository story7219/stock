# 📊 CSV 기반 주식 트레이딩 AI 파이프라인

실제 CSV 파일에서 주식 데이터를 불러와 AI 모델을 훈련하고 예측하는 완전한 파이프라인입니다.

## 🚀 주요 기능

### 📈 데이터 처리
- **CSV 직접 로드**: 시가, 고가, 저가, 종가, 거래량 등 다양한 주가 정보
- **기술적 지표 지원**: RSI, MACD, 이동평균선, 일목균형표, 볼린저밴드 등
- **자동 컬럼 감지**: 가격, 거래량, 기술적 지표 컬럼 자동 분류
- **데이터 품질 관리**: 결측치, 이상치, 중복 데이터 자동 처리

### 🤖 AI 모델
- **LSTM (Keras)**: 시계열 예측에 최적화된 딥러닝 모델
- **PyTorch 지원**: 확장 가능한 딥러닝 프레임워크
- **자동 하이퍼파라미터**: 최적의 모델 구조 자동 설정

### 📊 성능 평가
- **정확도, 정밀도, 재현율, F1 점수**: 종합적인 성능 평가
- **훈련/검증/테스트 분할**: 60:20:20 비율로 안정적인 평가
- **결과 저장**: 성능 지표와 예측 결과 자동 저장

## 📋 CSV 파일 형식

### 필수 컬럼
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,50000,51000,49500,50500,1000000
2023-01-02,50500,51500,50000,51000,1200000
...
```

### 기술적 지표 컬럼 (선택사항)
```csv
Date,Open,High,Low,Close,Volume,RSI,MACD,MA5,MA20,BB_Upper,BB_Lower
2023-01-01,50000,51000,49500,50500,1000000,45.2,150.5,50200,49800,52000,48000
...
```

### 지원되는 기술적 지표
- **RSI**: 상대강도지수
- **MACD**: 이동평균수렴확산지수
- **MA5, MA20**: 5일, 20일 이동평균
- **BB_Upper, BB_Lower**: 볼린저밴드 상단/하단
- **Stoch_K, Stoch_D**: 스토캐스틱
- **Ichimoku_***: 일목균형표 지표들

## 🛠️ 설치 및 실행

### 1. 필요한 라이브러리 설치
```bash
pip install pandas numpy scikit-learn tensorflow torch
```

### 2. 기본 실행 (ML.NET 스타일)
```bash
python csv_trading_ai.py --dataset "stock_data.csv"
```

### 3. 고급 옵션 사용
```bash
python csv_trading_ai.py \
    --dataset "C:\data\stock_data.csv" \
    --output "C:\results" \
    --model-dir "C:\models" \
    --target "Close" \
    --sequence-length 30 \
    --model-type keras \
    --epochs 100
```

## 📝 사용 예시

### 예시 1: 기본 실행
```python
from csv_trading_ai import TradingAIPipeline

# 파이프라인 생성
pipeline = TradingAIPipeline("stock_data.csv")

# 파이프라인 실행
results = pipeline.run_pipeline(
    target_col="Close",
    sequence_length=30,
    model_type="keras",
    epochs=50
)

print(f"정확도: {results['accuracy']:.4f}")
```

### 예시 2: 커스텀 설정
```python
# 데이터 로더 직접 사용
from csv_trading_ai import CSVDataLoader, DataSplitter, ScalerManager

# CSV 로드
loader = CSVDataLoader("stock_data.csv")
df = loader.load_csv()
df_clean = loader.preprocess_data()

# 컬럼 감지
columns = loader.detect_columns()
print(f"가격 컬럼: {columns['price_cols']}")
print(f"기술적 지표: {columns['technical_cols']}")

# 데이터 분할
splitter = DataSplitter()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.split_sequences(
    df_clean, 
    feature_cols=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI'],
    target_col='Close',
    sequence_length=20
)
```

## 🔧 명령행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dataset` | CSV 데이터 파일 경로 | `C:\data\stock_data.csv` |
| `--output` | 결과 저장 디렉토리 | `C:\results` |
| `--model-dir` | 모델 저장 디렉토리 | `C:\models` |
| `--target` | 타겟 컬럼명 (자동 감지 시 None) | `None` |
| `--sequence-length` | 시계열 시퀀스 길이 | `30` |
| `--model-type` | 모델 타입 (keras/pytorch) | `keras` |
| `--epochs` | 훈련 에포크 수 | `100` |

## 📊 출력 파일

### 1. 성능 결과 (`model_performance.csv`)
```csv
accuracy,precision,recall,f1_score,model_type,sequence_length,feature_columns,target_column
0.8234,0.8156,0.7892,0.8023,keras,30,"['Open', 'High', 'Low', 'Close', 'Volume']",Close
```

### 2. 예측 결과 (`predictions.csv`)
```csv
y_true,y_pred,y_pred_prob
1,1,0.8234
0,0,0.1567
1,1,0.9123
...
```

### 3. 모델 파일
- `best_keras_model.h5`: 최고 성능 Keras 모델
- `scaler.pkl`: 정규화 스케일러 (추후 구현)

## 🎯 실제 사용 시나리오

### 시나리오 1: KRX 데이터 사용
```bash
# KRX에서 다운로드한 CSV 파일 사용
python csv_trading_ai.py --dataset "krx_stock_data.csv" --target "종가"
```

### 시나리오 2: 기술적 지표 포함
```bash
# RSI, MACD 등 기술적 지표가 포함된 CSV
python csv_trading_ai.py \
    --dataset "technical_indicators.csv" \
    --sequence-length 50 \
    --epochs 200
```

### 시나리오 3: 실시간 예측
```python
# 훈련된 모델로 실시간 예측
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model("C:/models/best_keras_model.h5")

# 새로운 데이터로 예측
new_data = preprocess_new_data("latest_data.csv")
prediction = model.predict(new_data)
print(f"다음 거래일 예측: {'상승' if prediction > 0.5 else '하락'}")
```

## 🔍 데이터 품질 체크

파이프라인은 자동으로 다음 데이터 품질 문제를 해결합니다:

### 1. 결측치 처리
- **가격 데이터**: 이전 값으로 채우기 (forward fill)
- **거래량**: 0으로 채우기
- **기술적 지표**: 이전 값으로 채우기

### 2. 이상치 처리
- **IQR 방법**: 1.5 * IQR 범위를 벗어나는 값 클리핑
- **가격 데이터**: 상한/하한 값으로 제한

### 3. 중복 제거
- **완전 중복**: 동일한 모든 컬럼 값 제거
- **날짜 정렬**: 시간순으로 정렬

## ⚡ 성능 최적화

### 1. 빠른 훈련을 위한 설정
```bash
python csv_trading_ai.py \
    --sequence-length 10 \
    --epochs 20 \
    --model-type keras
```

### 2. 고정밀 예측을 위한 설정
```bash
python csv_trading_ai.py \
    --sequence-length 60 \
    --epochs 500 \
    --model-type keras
```

## 🚨 주의사항

1. **데이터 크기**: 대용량 CSV 파일은 메모리 사용량에 주의
2. **날짜 형식**: CSV의 날짜 컬럼이 표준 형식이어야 함
3. **GPU 사용**: TensorFlow/PyTorch GPU 버전 설치 권장
4. **결과 해석**: 과적합 방지를 위해 검증 성능도 함께 확인

## 📞 문제 해결

### 일반적인 오류

1. **CSV 로드 실패**
   ```bash
   # 인코딩 문제 해결
   python csv_trading_ai.py --dataset "stock_data.csv"
   ```

2. **메모리 부족**
   ```bash
   # 시퀀스 길이 줄이기
   python csv_trading_ai.py --sequence-length 10
   ```

3. **모델 훈련 실패**
   ```bash
   # 에포크 수 줄이기
   python csv_trading_ai.py --epochs 20
   ```

## 🔮 향후 개선 사항

- [ ] PyTorch 모델 완전 구현
- [ ] 앙상블 모델 지원
- [ ] 실시간 데이터 스트리밍
- [ ] 웹 대시보드 연동
- [ ] 자동 하이퍼파라미터 튜닝
- [ ] 백테스팅 기능 추가

---

**🎉 이제 CSV 파일만 있으면 바로 AI 트레이딩 모델을 훈련할 수 있습니다!** 