# 고급 자동매매 봇 시스템

## 📋 개요

ML+DL+AI 알고리즘과 고급 기술적 분석을 결합한 데이트레이딩/스윙 매매 특화 자동매매 봇입니다. 요청하신 3가지 매수 타점을 포함한 종합적인 트레이딩 시스템을 제공합니다.

## 🎯 주요 특징

### 1. 복합 알고리즘 시스템
- **ML (Machine Learning)**: Random Forest, Gradient Boosting 앙상블
- **DL (Deep Learning)**: LSTM 기반 시계열 예측
- **AI (Artificial Intelligence)**: 지능형 신호 결합 및 최적화
- **기술적 분석**: 일목균형표, 피보나치, 거래량 분석

### 2. 요청 매수 타점 구현
1. **추세 전환 매수**: 일목균형표 기반 추세 전환 감지
2. **피보나치 눌림목 매수**: 38.2%, 61.8% 레벨에서 지지 확인
3. **거래량 돌파 매수**: 평균 거래량 2배 이상 + 전고점 돌파

### 3. 고급 분석 모듈
- **선물/옵션 데이터**: KIS API 연동 준비
- **일목균형표 시간론**: 9, 17, 26, 33, 42, 65일 주기 분석
- **대등수치**: 가격, 거래량, 시간, 모멘텀 대등수치
- **3역호전**: 추세전환 → 조정 → 재추세 확립 패턴

## 🚀 시스템 구성

### 1. 핵심 모듈
```
advanced_trading_bot.py          # 메인 봇 시스템
advanced_analysis_modules.py     # 고급 분석 모듈
data_split_strategies.py         # 데이터 분리 전략
trading_data_splitter.py         # 트레이딩 데이터 분리
```

### 2. 주요 클래스
- `AdvancedTradingBot`: 메인 봇 클래스
- `AdvancedTechnicalAnalyzer`: 기술적 분석기
- `MLPredictor`: 머신러닝 예측기
- `DLPredictor`: 딥러닝 예측기
- `FuturesOptionsDataCollector`: 선물/옵션 데이터 수집
- `IchimokuTimeAnalyzer`: 일목균형표 시간론 분석
- `EquivalentValueAnalyzer`: 대등수치 분석
- `ThreePhaseReversalAnalyzer`: 3역호전 패턴 분석

## 📊 매수 신호 상세 분석

### 1. 추세 전환 매수 신호
```python
def detect_trend_reversal(self, data, ichimoku):
    # 1. 전환선과 기준선 교차
    tenkan_cross_up = (
        ichimoku.tenkan_sen.iloc[-1] > ichimoku.kijun_sen.iloc[-1] and
        ichimoku.tenkan_sen.iloc[-2] <= ichimoku.kijun_sen.iloc[-2]
    )
    
    # 2. 구름대 돌파
    cloud_breakout = (
        current['close'] > ichimoku.cloud_high.iloc[-1] and
        prev['close'] <= ichimoku.cloud_high.iloc[-2]
    )
    
    # 3. 지지선 돌파
    support_breakout = (
        current['close'] > current['low'] * 1.02 and
        current['volume'] > data['volume'].rolling(20).mean().iloc[-1]
    )
    
    return tenkan_cross_up or cloud_breakout or support_breakout
```

### 2. 피보나치 눌림목 매수 신호
```python
def detect_fibonacci_pullback(self, data, lookback_period=20):
    # 최근 고점과 저점 찾기
    recent_high = data['high'].rolling(lookback_period).max().iloc[-1]
    recent_low = data['low'].rolling(lookback_period).min().iloc[-1]
    
    # 피보나치 레벨 계산
    fib_levels = self.calculate_fibonacci_levels(recent_high, recent_low)
    
    # 38.2% 또는 61.8% 레벨에서 지지 확인
    support_382 = (
        abs(current_price - fib_levels.level_382) / fib_levels.level_382 < 0.01 and
        data['volume'].iloc[-1] > data['volume'].rolling(10).mean().iloc[-1]
    )
    
    support_618 = (
        abs(current_price - fib_levels.level_618) / fib_levels.level_618 < 0.01 and
        data['volume'].iloc[-1] > data['volume'].rolling(10).mean().iloc[-1]
    )
    
    return support_382 or support_618
```

### 3. 거래량 돌파 매수 신호
```python
def detect_volume_breakout(self, data, volume_multiplier=2.0):
    current_volume = data['volume'].iloc[-1]
    avg_volume = data['volume'].rolling(20).mean().iloc[-1]
    
    # 거래량이 평균의 2배 이상
    volume_surge = current_volume > avg_volume * volume_multiplier
    
    # 가격도 함께 상승
    price_rise = data['close'].iloc[-1] > data['close'].iloc[-2]
    
    # 전고점 돌파 확인
    recent_high = data['high'].rolling(20).max().iloc[-2]
    breakout = data['close'].iloc[-1] > recent_high
    
    return volume_surge and price_rise and breakout
```

## 🔧 사용 방법

### 1. 기본 사용법
```python
from advanced_trading_bot import AdvancedTradingBot, TradingMode
from advanced_analysis_modules import MarketType

# 봇 초기화
bot = AdvancedTradingBot(
    trading_mode=TradingMode.DAY_TRADING,  # 또는 SWING_TRADING
    risk_per_trade=0.02  # 거래당 2% 리스크
)

# 백테스트 실행
results = bot.run_backtest(data, initial_capital=100000)

# 결과 확인
print(f"총 수익률: {results['performance']['total_return']:.2%}")
print(f"승률: {results['performance']['win_rate']:.2%}")
print(f"최대 낙폭: {results['performance']['max_drawdown']:.2%}")
```

### 2. 고급 분석 사용법
```python
from advanced_analysis_modules import AdvancedAnalysisDashboard

# 대시보드 초기화
dashboard = AdvancedAnalysisDashboard()

# 종합 분석 실행
analysis_results = dashboard.run_comprehensive_analysis(
    data=data,
    market_type=MarketType.STOCK  # 또는 FUTURES, OPTIONS
)

# 결과 확인
if 'equivalent_values' in analysis_results:
    eq = analysis_results['equivalent_values']
    print(f"가격 대등수치: {eq.price_equivalent:.4f}")
    print(f"거래량 대등수치: {eq.volume_equivalent:.4f}")

if 'reversal_pattern' in analysis_results and analysis_results['reversal_pattern']:
    pattern = analysis_results['reversal_pattern']
    print(f"3역호전 패턴 감지: {pattern.pattern_confidence:.2%} 신뢰도")
```

### 3. 선물/옵션 데이터 수집
```python
from advanced_analysis_modules import FuturesOptionsDataCollector

# 데이터 수집기 초기화
collector = FuturesOptionsDataCollector(api_key="your_api_key")

# 선물 데이터 수집
futures_data = collector.get_futures_data(
    symbol="ES",  # E-mini S&P 500
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d"
)

# 옵션 데이터 수집
options_data = collector.get_options_data(
    symbol="AAPL",
    expiry_date="2024-12-20",
    option_type="call"
)
```

## 📈 성능 최적화

### 1. 모델 훈련 최적화
```python
# ML 모델 설정
ml_predictor = MLPredictor(model_type='ensemble')

# DL 모델 설정
dl_predictor = DLPredictor(sequence_length=60)

# 조기 종료 및 학습률 조정
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5)
]
```

### 2. 실시간 처리 최적화
```python
# 데이터 캐싱
@lru_cache(maxsize=1000)
def calculate_technical_indicators(data_hash):
    # 기술적 지표 계산
    pass

# 병렬 처리
from concurrent.futures import ThreadPoolExecutor

def parallel_analysis(data_chunks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(analyze_chunk, data_chunks))
    return results
```

## 🛡️ 리스크 관리

### 1. 포지션 사이징
```python
def calculate_position_size(self, signal, capital):
    # 켈리 공식 기반 포지션 사이징
    win_rate = 0.6  # 예상 승률
    avg_win = 0.02  # 평균 수익
    avg_loss = 0.01  # 평균 손실
    
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    # 최대 2% 리스크 제한
    max_risk = capital * self.risk_per_trade
    position_size = min(kelly_fraction * capital, max_risk)
    
    return position_size
```

### 2. 손절매 전략
```python
def check_stop_loss(self, current_price, entry_price):
    # 고정 손절매
    fixed_stop = entry_price * (1 - self.risk_per_trade)
    
    # 이동 손절매 (ATR 기반)
    atr = self.calculate_atr(self.data)
    trailing_stop = current_price - (atr * 2)
    
    return min(fixed_stop, trailing_stop)
```

## 📊 백테스트 결과 분석

### 1. 성과 지표
- **총 수익률**: 전체 기간 수익률
- **승률**: 수익 거래 비율
- **최대 낙폭**: 최대 손실 구간
- **샤프 비율**: 위험 대비 수익률
- **칼마 비율**: 최대 낙폭 대비 수익률

### 2. 신호별 성과 분석
```python
def analyze_signal_performance(self, results):
    signal_performance = {}
    
    for signal_type in ['trend_reversal', 'fibonacci_pullback', 'volume_breakout']:
        signal_trades = [t for t in results['trades'] if t['source'] == signal_type]
        
        if signal_trades:
            profits = [t['profit'] for t in signal_trades if t['profit'] > 0]
            losses = [t['profit'] for t in signal_trades if t['profit'] < 0]
            
            signal_performance[signal_type] = {
                'win_rate': len(profits) / len(signal_trades),
                'avg_profit': np.mean(profits) if profits else 0,
                'avg_loss': np.mean(losses) if losses else 0,
                'profit_factor': abs(sum(profits) / sum(losses)) if losses else float('inf')
            }
    
    return signal_performance
```

## 🔍 시각화 및 모니터링

### 1. 실시간 대시보드
```python
def create_realtime_dashboard(self, data, signals):
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Price & Signals', 'Volume Analysis',
            'Technical Indicators', 'ML/DL Predictions',
            'Risk Metrics', 'Performance Summary',
            'Market Regimes', 'Signal Distribution'
        )
    )
    
    # 가격 차트와 신호
    fig.add_trace(
        go.Scatter(x=data.index, y=data['close'], name='Price'),
        row=1, col=1
    )
    
    # 매수 신호 표시
    buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
    if buy_signals:
        fig.add_trace(
            go.Scatter(
                x=[s.timestamp for s in buy_signals],
                y=[s.price for s in buy_signals],
                mode='markers',
                name='Buy Signals',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    fig.show()
```

### 2. 성과 리포트
```python
def generate_performance_report(self, results):
    report = {
        'summary': {
            'total_return': results['performance']['total_return'],
            'win_rate': results['performance']['win_rate'],
            'max_drawdown': results['performance']['max_drawdown'],
            'sharpe_ratio': results['performance']['sharpe_ratio']
        },
        'monthly_returns': self._calculate_monthly_returns(results),
        'signal_analysis': self.analyze_signal_performance(results),
        'risk_metrics': self._calculate_risk_metrics(results)
    }
    
    return report
```

## ⚙️ 설정 및 커스터마이징

### 1. 트레이딩 모드 설정
```python
# 데이트레이딩 설정
day_trading_config = {
    'risk_per_trade': 0.01,      # 1% 리스크
    'max_positions': 3,          # 최대 3개 포지션
    'stop_loss': 0.005,          # 0.5% 손절매
    'take_profit': 0.015,        # 1.5% 익절매
    'time_filter': {
        'start_time': '09:30',
        'end_time': '15:30'
    }
}

# 스윙 매매 설정
swing_trading_config = {
    'risk_per_trade': 0.02,      # 2% 리스크
    'max_positions': 5,          # 최대 5개 포지션
    'stop_loss': 0.03,           # 3% 손절매
    'take_profit': 0.06,         # 6% 익절매
    'holding_period': '1-5 days'
}
```

### 2. 신호 가중치 조정
```python
signal_weights = {
    'trend_reversal': 0.3,       # 추세 전환 30%
    'fibonacci_pullback': 0.25,  # 피보나치 25%
    'volume_breakout': 0.25,     # 거래량 돌파 25%
    'ml_dl_combined': 0.2        # ML/DL 결합 20%
}

# 신호 결합
combined_confidence = sum(
    signal.confidence * signal_weights[signal.signal_source]
    for signal in signals
)
```

## 🚨 주의사항

### 1. 백테스팅 한계
- **Look-ahead bias**: 미래 정보 사용 방지
- **Overfitting**: 과적합 방지를 위한 교차 검증
- **Transaction costs**: 거래 비용 고려
- **Slippage**: 슬리피지 반영

### 2. 실전 적용 시 고려사항
- **시장 상황 변화**: 모델 재훈련 필요
- **유동성**: 거래량 부족 시 신호 무시
- **뉴스 이벤트**: 급격한 시장 변화 시 거래 중단
- **시스템 장애**: 백업 시스템 구축

## 📚 추가 자료

### 관련 문서
- [일목균형표 완전 가이드](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)
- [피보나치 되돌림 분석](https://www.investopedia.com/terms/f/fibonacciretracement.asp)
- [거래량 분석 기법](https://www.investopedia.com/terms/v/volume.asp)

### 참고 논문
- "Machine Learning for Financial Time Series Prediction"
- "Deep Learning in Quantitative Finance"
- "Technical Analysis and Market Efficiency"

---

**⚠️ 중요**: 이 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 거래에 사용하기 전에 충분한 테스트와 검증이 필요합니다. 투자 손실에 대한 책임은 사용자에게 있습니다. 