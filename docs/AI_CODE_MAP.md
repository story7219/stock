# 🗺️ AI 코드 맵 - 즉시 찾기 가이드

## 🎯 **핵심 코드 위치 (AI 전용)**

### 📊 **1. 투자 전략 수정 포인트**

#### 🏆 **워런 버핏 전략**
```python
# 파일: src/strategies.py
# 라인: 50-90
class BuffettStrategy(BaseStrategy):
    def apply_strategy(self, stocks: List[StockInfo]) -> List[StockAnalysisResult]:
        # 🔧 수정 포인트: 여기에 새로운 조건 추가
        filtered_stocks = [
            stock for stock in stocks if
            stock.roe > 10 and  # ← ROE 조건 수정
            stock.debt_ratio < 0.5 and  # ← 부채비율 조건 수정
            stock.pe_ratio > 0  # ← PER 조건 수정
        ]
```

#### 🚀 **피터 린치 전략**
```python
# 파일: src/strategies.py  
# 라인: 120-160
class LynchStrategy(BaseStrategy):
    def apply_strategy(self, stocks: List[StockInfo]) -> List[StockAnalysisResult]:
        # 🔧 수정 포인트: 성장률 조건 수정
        filtered_stocks = [
            stock for stock in stocks if
            stock.growth_rate > 15 and  # ← 성장률 조건 수정
            stock.peg_ratio < 1.5  # ← PEG 비율 수정
        ]
```

### 📈 **2. 데이터 수집 수정 포인트**

#### 🇰🇷 **코스피200 데이터**
```python
# 파일: src/data_collector.py
# 라인: 116-180
async def collect_kospi_data(self) -> List[StockInfo]:
    # 🔧 수정 포인트: 새로운 데이터 항목 추가
    stock_info = StockInfo(
        symbol=symbol,
        name=name,
        price=price,
        # ← 여기에 새 항목 추가 (PBR, 배당수익률 등)
    )
```

#### 🇺🇸 **나스닥100 데이터**
```python
# 파일: src/data_collector.py
# 라인: 523-580
async def collect_nasdaq_data(self) -> List[StockInfo]:
    # 🔧 수정 포인트: API 소스 변경 또는 항목 추가
    for symbol in self.nasdaq100_symbols:
        # ← 데이터 수집 로직 수정
```

### 🤖 **3. AI 분석 수정 포인트**

#### 💬 **Gemini 프롬프트**
```python
# 파일: src/gemini_analyzer.py
# 라인: 180-220
def _create_analysis_prompt(self, strategy_results, market_data):
    prompt = f"""
    당신은 세계 최고 수준의 AI 투자 분석가입니다.
    
    # 🔧 수정 포인트: 여기서 프롬프트 개선
    분석 요구사항:
    1. 기술적 분석 강화
    2. 리스크 평가 추가  
    3. 시장 상황 반영
    # ← 새로운 분석 항목 추가
    """
```

#### 📊 **AI 응답 파싱**
```python
# 파일: src/gemini_analyzer.py
# 라인: 250-300
def _parse_ai_response(self, response_text: str):
    # 🔧 수정 포인트: 새로운 응답 항목 파싱
    patterns = {
        'top5_selections': r'TOP5.*?(?=\n\n|\Z)',
        'market_outlook': r'시황분석.*?(?=\n\n|\Z)',
        # ← 새로운 파싱 패턴 추가
    }
```

### 🧹 **4. 데이터 정제 수정 포인트**

#### 🔍 **이상치 탐지**
```python
# 파일: src/data_cleaner.py
# 라인: 400-450
def detect_outliers(self, data):
    # 🔧 수정 포인트: 새로운 이상치 탐지 알고리즘
    outliers = []
    if abs(data.pe_ratio) > 100:  # ← 임계값 수정
        outliers.append('pe_ratio')
    # ← 새로운 이상치 조건 추가
```

#### 🔄 **결측치 보정**
```python
# 파일: src/data_cleaner.py  
# 라인: 150-200
def impute_missing_values(self, stocks):
    # 🔧 수정 포인트: 보정 알고리즘 개선
    for stock in stocks:
        if stock.pe_ratio is None:
            # ← 보정 로직 수정
            stock.pe_ratio = self.calculate_median_pe()
```

---

## 🔧 **즉시 수정 가능한 코드 블록**

### 💡 **1. 새로운 투자 전략 추가**

#### 📍 **위치**: `src/strategies.py` 끝부분 (라인 440-450)
```python
# 🔧 여기에 새 전략 클래스 추가
class NewStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("새로운전략", "설명")
    
    def apply_strategy(self, stocks: List[StockInfo]) -> List[StockAnalysisResult]:
        # 새 전략 로직 구현
        pass
```

#### 📍 **연동 위치**: `src/strategies.py` 라인 420-430
```python
class StrategyManager:
    def __init__(self):
        self.strategies = {
            # 🔧 여기에 새 전략 추가
            "new_strategy": NewStrategy(),
        }
```

### 💡 **2. 새로운 시장 추가**

#### 📍 **위치**: `src/data_collector.py` 끝부분
```python
# 🔧 새 시장 수집 함수 추가
async def collect_new_market_data(self) -> List[StockInfo]:
    """새로운 시장 데이터 수집"""
    # 구현 코드
    pass
```

#### 📍 **연동 위치**: `main.py` 라인 24-27
```python
MARKET_LIST = [
    "한국주식(코스피200)", 
    "미국주식(나스닥100)", 
    "미국주식(S&P500)",
    # 🔧 여기에 새 시장 추가
    "새로운시장"
]
```

### 💡 **3. 새로운 데이터 항목 추가**

#### 📍 **위치**: `src/data_collector.py` 라인 1-50
```python
@dataclass
class StockInfo:
    symbol: str
    name: str
    price: float
    # 🔧 여기에 새 항목 추가
    # pbr: Optional[float] = None
    # dividend_yield: Optional[float] = None
```

### 💡 **4. 기술적 지표 추가**

#### 📍 **위치**: `src/technical_analyzer.py` 라인 100-150
```python
class TechnicalAnalyzer:
    # 🔧 새 지표 함수 추가
    def calculate_new_indicator(self, data):
        """새로운 기술적 지표 계산"""
        # 구현 코드
        pass
```

---

## 🎯 **자주 사용되는 수정 패턴**

### 🔄 **조건 수정 패턴**
```python
# Before (기존)
if stock.pe_ratio > 0:

# After (수정)  
if stock.pe_ratio > 0 and stock.pe_ratio < 20:
```

### 📊 **새 항목 추가 패턴**
```python
# Before (기존)
result = StockAnalysisResult(
    stock=stock,
    score=score,
    reasoning=reasoning
)

# After (수정)
result = StockAnalysisResult(
    stock=stock,
    score=score,
    reasoning=reasoning,
    new_field=new_value  # ← 새 항목
)
```

### 🤖 **프롬프트 개선 패턴**
```python
# Before (기존)
prompt = "분석해주세요"

# After (수정)
prompt = """
상세한 분석을 해주세요:
1. 기술적 분석
2. 재무 분석  
3. 리스크 평가
"""
```

---

## ⚡ **AI 즉시 실행 명령어**

### 🔍 **코드 검색**
```bash
# 특정 클래스 찾기
grep -rn "class.*Strategy" src/

# 특정 함수 찾기
grep -rn "def collect.*data" src/

# 특정 변수 찾기
grep -rn "MARKET_LIST" .

# 특정 문자열 찾기
grep -rn "Gemini" src/
```

### 🧪 **기능 테스트**
```bash
# 전략 테스트
python -c "from src.strategies import StrategyManager; sm=StrategyManager(); print(list(sm.strategies.keys()))"

# 데이터 수집 테스트
python -c "from src.data_collector import DataCollector; dc=DataCollector(); print('OK')"

# AI 분석기 테스트
python -c "from src.gemini_analyzer import GeminiAnalyzer; print('OK')"
```

### 📝 **빠른 수정 확인**
```bash
# 문법 오류 확인
python -m py_compile main.py

# 임포트 오류 확인
python -c "import src.strategies; import src.data_collector; import src.gemini_analyzer; print('All imports OK')"
```

---

## 🎪 **AI 요청 → 즉시 수정 플로우**

### 1️⃣ **요청 접수**
```
사용자: "워런 버핏 전략에 ROE 15% 조건 추가해줘"
```

### 2️⃣ **위치 식별**
```
AI: src/strategies.py 라인 50-90 BuffettStrategy.apply_strategy() 확인
```

### 3️⃣ **코드 수정**
```python
# 수정 전
stock.roe > 10

# 수정 후  
stock.roe > 15
```

### 4️⃣ **테스트 실행**
```bash
python -c "from src.strategies import BuffettStrategy; print('Strategy updated')"
```

### 5️⃣ **완료 보고**
```
AI: ✅ 워런 버핏 전략 ROE 조건을 15%로 상향 조정 완료
```

---

*🤖 AI가 즉시 찾아서 수정할 수 있도록 최적화된 코드 맵*  
*⚡ 모든 수정 포인트가 라인 번호와 함께 명시됨* 