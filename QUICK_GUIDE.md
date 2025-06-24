# 🤖 AI 작업 요청 빠른 가이드

## 🎯 **AI에게 쉽게 요청하는 방법**

### 📋 **1. 전략 관련 요청**
```
예시: "워런 버핏 전략에 ROE 15% 이상 조건 추가해줘"
```
- **찾을 위치**: `src/strategies.py` 라인 50-150
- **관련 함수**: `BuffettStrategy.apply_strategy()`
- **수정 방법**: 필터 조건에 `stock.roe >= 15` 추가

### 📋 **2. 시장 데이터 요청**
```
예시: "코스닥 종목도 추가해줘"
```
- **찾을 위치**: `src/data_collector.py` 라인 700-800
- **관련 함수**: `collect_kosdaq_data()` 새로 생성
- **연동 위치**: `main.py` 라인 120-140 (시장 선택 부분)

### 📋 **3. AI 분석 개선 요청**
```
예시: "Gemini AI가 더 자세한 분석 결과 줬으면 좋겠어"
```
- **찾을 위치**: `src/gemini_analyzer.py` 라인 150-200
- **관련 함수**: `analyze_candidates()`
- **프롬프트 위치**: 라인 180-190

### 📋 **4. 기술적 지표 요청**
```
예시: "MACD 지표 추가해줘"
```
- **찾을 위치**: `src/technical_analyzer.py` 라인 100-200
- **관련 함수**: `calculate_macd()` 새로 생성
- **연동 위치**: `src/data_collector.py` 라인 50-80

---

## 🔧 **자주 요청되는 작업별 위치**

### 🎯 **투자 전략 수정**
| 요청 내용 | 파일 위치 | 함수명 | 라인 범위 |
|-----------|-----------|--------|-----------|
| 새 전략 추가 | `src/strategies.py` | `StrategyManager` | 420-450 |
| 기존 전략 수정 | `src/strategies.py` | 각 전략 클래스 | 50-400 |
| 점수 가중치 변경 | `src/strategies.py` | `calculate_score()` | 각 전략별 |

### 📊 **데이터 수집 개선**
| 요청 내용 | 파일 위치 | 함수명 | 라인 범위 |
|-----------|-----------|--------|-----------|
| 새 시장 추가 | `src/data_collector.py` | 새 함수 생성 | 마지막 부분 |
| 데이터 항목 추가 | `src/data_collector.py` | `StockInfo` 클래스 | 1-50 |
| API 소스 변경 | `src/data_collector.py` | 각 수집 함수 | 해당 시장별 |

### 🤖 **AI 분석 고도화**
| 요청 내용 | 파일 위치 | 함수명 | 라인 범위 |
|-----------|-----------|--------|-----------|
| 프롬프트 개선 | `src/gemini_analyzer.py` | `_create_analysis_prompt()` | 180-220 |
| 응답 파싱 개선 | `src/gemini_analyzer.py` | `_parse_ai_response()` | 250-300 |
| 새 분석 항목 | `src/gemini_analyzer.py` | `GeminiAnalysisResult` | 20-50 |

### 🧹 **데이터 정제 개선**
| 요청 내용 | 파일 위치 | 함수명 | 라인 범위 |
|-----------|-----------|--------|-----------|
| 새 보정 알고리즘 | `src/data_cleaner.py` | `AdvancedDataCleaner` | 150-300 |
| 이상치 탐지 개선 | `src/data_cleaner.py` | `detect_outliers()` | 400-450 |
| 품질 검증 강화 | `src/data_cleaner.py` | `validate_data_quality()` | 500-550 |

---

## 🚀 **AI 요청 템플릿**

### 📝 **효과적인 요청 방법**

#### ✅ **좋은 요청 예시**
```
"나스닥100에서 PER 15배 이하인 종목만 골라서 
피터 린치 전략으로 분석해줘. 
성장률 20% 이상 조건도 추가해줘."
```

#### ❌ **피해야 할 요청**
```
"더 좋게 해줘"  (구체적이지 않음)
"버그 고쳐줘"   (어떤 버그인지 명시 안함)
```

### 🎯 **구체적 요청 가이드**

#### **1. 전략 요청**
```
템플릿: "[전략명]에서 [조건] 추가해줘"
예시: "워런 버핏 전략에서 ROE 15% 이상, 부채비율 30% 이하 조건 추가해줘"
```

#### **2. 데이터 요청**
```
템플릿: "[시장명] 데이터에 [항목] 추가해줘"
예시: "코스피200 데이터에 PBR, 배당수익률 항목 추가해줘"
```

#### **3. AI 분석 요청**
```
템플릿: "Gemini AI가 [분석 항목]도 포함해서 분석하게 해줘"
예시: "Gemini AI가 ESG 점수, 기업 지배구조도 포함해서 분석하게 해줘"
```

#### **4. 기능 추가 요청**
```
템플릿: "[기능명] 기능 추가해줘. [구체적 동작]"
예시: "텔레그램 알림 기능 추가해줘. Top5 선정되면 메시지 보내게"
```

---

## 📂 **빠른 파일 찾기**

### 🔍 **코드 검색 키워드**

#### **전략 관련**
- `class BuffettStrategy` - 워런 버핏 전략
- `class LynchStrategy` - 피터 린치 전략  
- `apply_strategy` - 전략 적용 함수
- `calculate_score` - 점수 계산

#### **데이터 관련**
- `collect_kospi_data` - 코스피 데이터 수집
- `collect_nasdaq_data` - 나스닥 데이터 수집
- `StockInfo` - 주식 정보 클래스
- `clean_stock_data` - 데이터 정제

#### **AI 관련**
- `GeminiAnalyzer` - AI 분석기
- `analyze_candidates` - AI 분석 실행
- `_create_analysis_prompt` - AI 프롬프트 생성
- `GeminiAnalysisResult` - AI 결과 클래스

---

## ⚡ **즉시 실행 명령어**

### 🛠️ **개발 도구**
```bash
# 전체 테스트
python main.py

# 특정 모듈 테스트
python -c "from src.strategies import StrategyManager; print('OK')"

# 코드 검색 (Git Bash)
grep -r "class.*Strategy" src/

# 라인 번호와 함께 검색
grep -rn "def apply_strategy" src/
```

### 📊 **성능 확인**
```bash
# 메모리 사용량 확인
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# 실행 시간 측정
python -c "import time; start=time.time(); exec(open('main.py').read()); print(f'Time: {time.time()-start:.2f}s')"
```

---

## 🎯 **AI가 바로 이해할 수 있는 요청 방법**

### 📝 **요청 시 포함할 정보**

1. **무엇을** - 구체적인 기능/수정사항
2. **어디에** - 파일명 또는 기능 영역  
3. **어떻게** - 동작 방식이나 조건
4. **왜** - 목적이나 이유 (선택사항)

### 예시:
```
"src/strategies.py의 워런 버핏 전략에서 
ROE 15% 이상 조건을 추가해줘.
현재 너무 많은 종목이 선택되어서 더 엄격하게 필터링하고 싶어."
```

### 🚀 **즉시 처리 가능한 요청 유형**

✅ **전략 조건 수정**  
✅ **새로운 지표 추가**  
✅ **AI 프롬프트 개선**  
✅ **데이터 항목 추가**  
✅ **알림 기능 추가**  
✅ **리포트 형식 변경**  

---

*🤖 AI가 효율적으로 작업할 수 있도록 설계된 가이드*  
*📅 업데이트: 2024년 1월* 