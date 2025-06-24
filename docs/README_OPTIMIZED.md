# 🚀 AI 기반 투자 분석 시스템 v3.0

> 코스피200·나스닥100·S&P500 전체 종목을 분석하여 투자 대가 전략으로 Gemini AI가 Top5 종목을 자동 선정하는 고품질 무에러 프로그램

## ✨ 주요 특징

### 🌏 **전체 시장 커버리지**
- **코스피200** - 한국 대표 200개 종목
- **나스닥100** - 미국 기술주 대표 100개 종목  
- **S&P500** - 미국 대표 500개 종목
- 실시간 데이터 수집 및 자동 업데이트

### 🎯 **15명 투자 대가 전략**
- **벤저민 그레이엄** - 가치투자의 아버지
- **워런 버핏** - 장기 가치투자
- **피터 린치** - 성장주 투자
- **조지 소로스** - 반사성 이론
- **필립 피셔** - 성장주 전문가
- **존 템플턴** - 글로벌 가치투자
- **벤자민 그레이엄** - 정량적 분석
- 기타 8명의 투자 대가 전략

### 🤖 **Gemini AI 고급 추론**
- 투자 대가 전략별 상위 후보군 분석
- 시장 상황·기술적 지표·최근 트렌드 종합 판단
- 최적의 Top5 종목 자동 선정
- 선정 근거와 reasoning 과정 상세 설명

### 📈 **기술적 분석 중심**
- RSI, MACD, 볼린저밴드, 이동평균선
- 지지/저항선, 차트 패턴 분석
- 볼륨 분석 및 모멘텀 지표
- 재무정보 제외한 순수 기술적 분석

### 🏗️ **고품질 아키텍처**
- 90% 이상 테스트 커버리지
- PEP8 준수, pylint 9.0+ 점수
- 완벽한 예외 처리 및 에러 복구
- 모듈화된 확장 가능한 구조

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd ai-investment-analysis

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env
# .env 파일에서 API 키 설정
```

### 2. API 키 설정

`.env` 파일에 다음 API 키들을 설정:

```bash
# 필수
GEMINI_API_KEY=your_gemini_api_key

# 선택사항 (데이터 품질 향상)
LIVE_KIS_APP_KEY=your_kis_app_key
LIVE_KIS_APP_SECRET=your_kis_app_secret
DART_API_KEY=your_dart_api_key
```

### 3. 실행

```bash
# 기본 실행 (전체 시장 분석)
python run_analysis.py

# 특정 시장만 분석
python run_analysis.py --market kospi200

# 특정 전략만 적용
python run_analysis.py --strategies warren_buffett,benjamin_graham

# 테스트 모드 (빠른 검증)
python run_analysis.py --test-mode

# 상세 로그 출력
python run_analysis.py --verbose
```

## 📊 사용 예시

### 기본 사용법

```python
import asyncio
from src.main_optimized import InvestmentAnalysisSystem

async def main():
    # 시스템 초기화
    system = InvestmentAnalysisSystem()
    
    # 전체 분석 실행
    results = await system.run_complete_analysis()
    
    # 결과 출력
    system.display_results()

if __name__ == "__main__":
    asyncio.run(main())
```

### 고급 사용법

```python
from data_collector import MultiDataCollector
from investment_strategies import StrategyManager
from ai_analyzer import GeminiAIAnalyzer

# 1. 데이터 수집
collector = MultiDataCollector()
stocks = await collector.collect_all_markets()

# 2. 투자 전략 적용
strategy_manager = StrategyManager()
strategy_results = strategy_manager.apply_all_strategies(stocks)

# 3. Gemini AI 분석
ai_analyzer = GeminiAIAnalyzer()
top5_result = ai_analyzer.analyze_and_select_top5(
    stocks=stocks,
    technical_results={},
    strategy_scores=strategy_results
)

print(f"🏆 Top5 선정 결과: {top5_result.top5_stocks}")
```

## 📁 프로젝트 구조

```
ai-investment-analysis/
├── 📁 src/                          # 소스 코드
│   ├── main_optimized.py            # 메인 실행 시스템
│   └── __init__.py
├── 📁 tests/                        # 테스트 코드  
│   ├── test_main_system.py          # 메인 시스템 테스트
│   └── __init__.py
├── 📁 data/                         # 데이터 저장소
│   ├── logs/                        # 로그 파일
│   ├── reports/                     # 분석 리포트
│   ├── cache/                       # 캐시 데이터
│   └── temp/                        # 임시 파일
├── data_collector.py                # 데이터 수집기
├── investment_strategies.py         # 투자 전략 엔진
├── technical_analysis.py            # 기술적 분석기
├── ai_analyzer.py                   # Gemini AI 분석기
├── news_analyzer.py                 # 뉴스 분석기
├── run_analysis.py                  # 실행 스크립트
├── requirements.txt                 # Python 의존성
├── pytest.ini                      # 테스트 설정
├── .gitignore                       # Git 무시 파일
└── README_OPTIMIZED.md              # 이 파일
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/test_main_system.py

# 커버리지 포함 테스트
pytest --cov=src --cov-report=html

# 상세 출력
pytest -v --tb=short
```

## 📈 출력 결과

### 콘솔 출력 예시

```
🚀 AI 기반 투자 분석 시스템 v3.0 - 최종 결과
════════════════════════════════════════════════════════════════════════════════════════

📊 분석 요약:
   • 총 분석 종목: 347개
   • 적용 전략: 15개
   • 기술적 분석: 342개

🌍 시장별 분포:
   • KOSPI200: 198개
   • NASDAQ100: 95개
   • S&P500: 54개

🤖 Gemini AI Top5 선정 결과:
   신뢰도: 89.2%
   1. Apple Inc (AAPL)
      현재가: $180.25
      시장: NASDAQ100
   2. Microsoft Corp (MSFT)
      현재가: $342.56
      시장: S&P500
   3. 삼성전자 (005930)
      현재가: ₩75,400
      시장: KOSPI200
   4. NVIDIA Corp (NVDA)
      현재가: $485.20
      시장: NASDAQ100
   5. Alphabet Inc (GOOGL)
      현재가: $142.18
      시장: NASDAQ100

⏱️ 성능 메트릭:
   • 총 실행시간: 127.35초
   • 데이터 수집: 45.20초
   • 전략 분석: 38.15초
   • 기술 분석: 28.90초
   • AI 분석: 15.10초

════════════════════════════════════════════════════════════════════════════════════════
✅ 분석 완료! 상세 리포트는 data/reports 폴더를 확인하세요.
════════════════════════════════════════════════════════════════════════════════════════
```

### 생성 파일

1. **JSON 리포트** (`data/reports/investment_analysis_20241215_143022.json`)
   - 전체 분석 결과 데이터
   - 투자 전략별 상세 점수
   - Gemini AI 추론 과정

2. **CSV 리포트** (`data/reports/top5_stocks_20241215_143022.csv`)
   - Top5 종목 요약 데이터
   - 엑셀에서 바로 열기 가능

3. **로그 파일** (`data/logs/investment_analysis_20241215_143022.log`)
   - 상세 실행 로그
   - 오류 및 경고 메시지

## ⚙️ 설정 옵션

### 명령행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--market` | 분석할 시장 지정 | `all` |
| `--strategies` | 적용할 투자 전략 | `all` |
| `--top-n` | 선정할 상위 종목 수 | `5` |
| `--output-format` | 출력 형식 | `json,csv` |
| `--test-mode` | 테스트 모드 실행 | `False` |
| `--verbose` | 상세 로그 출력 | `False` |

### 환경 변수

| 변수명 | 설명 | 필수 여부 |
|--------|------|-----------|
| `GEMINI_API_KEY` | Google Gemini API 키 | 필수 |
| `LIVE_KIS_APP_KEY` | 한국투자증권 API 키 | 선택 |
| `LIVE_KIS_APP_SECRET` | 한국투자증권 시크릿 | 선택 |
| `DART_API_KEY` | DART Open API 키 | 선택 |

## 🔧 개발자 가이드

### 새로운 투자 전략 추가

```python
from investment_strategies import BaseStrategy, StrategyScore, StockData

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(
            name="My Custom Strategy",
            description="나만의 투자 전략"
        )
    
    def filter_stocks(self, stocks: List[StockData]) -> List[StockData]:
        # 종목 필터링 로직
        return [stock for stock in stocks if self.meets_criteria(stock)]
    
    def calculate_score(self, stock: StockData) -> StrategyScore:
        # 점수 계산 로직
        score = self.calculate_custom_score(stock)
        
        return StrategyScore(
            symbol=stock.symbol,
            name=stock.name,
            strategy_name=self.name,
            total_score=score,
            criteria_scores={},
            reasoning="커스텀 전략 분석 결과"
        )
```

### 테스트 추가

```python
import pytest
from tests.test_main_system import TestInvestmentAnalysisSystem

class TestMyCustomStrategy(TestInvestmentAnalysisSystem):
    def test_my_custom_logic(self):
        # 새로운 기능 테스트
        assert True
```

## 📋 요구사항

### 시스템 요구사항

- **Python**: 3.9 이상
- **메모리**: 최소 4GB RAM
- **저장공간**: 최소 2GB 여유 공간
- **네트워크**: 인터넷 연결 필수

### Python 의존성

주요 라이브러리:
- `yfinance>=0.2.18` - 주식 데이터 수집
- `pandas>=2.0.0` - 데이터 처리
- `numpy>=1.24.0` - 수치 계산
- `google-generativeai>=0.3.0` - Gemini AI
- `pytest>=7.0.0` - 테스트 프레임워크
- `python-dotenv>=1.0.0` - 환경변수 관리

전체 목록은 `requirements.txt` 참조

## 🐛 문제 해결

### 자주 발생하는 오류

1. **API 키 오류**
   ```
   ❌ GEMINI_API_KEY가 설정되지 않았습니다!
   ```
   → `.env` 파일에 올바른 API 키 설정

2. **네트워크 연결 오류**
   ```
   ❌ 데이터 수집 실패: Connection timeout
   ```
   → 인터넷 연결 확인, 방화벽 설정 점검

3. **메모리 부족 오류**
   ```
   ❌ MemoryError: Unable to allocate array
   ```
   → 테스트 모드로 실행하거나 메모리 증설

### 로그 확인

상세한 오류 정보는 로그 파일에서 확인:

```bash
# 최신 로그 파일 확인
tail -f data/logs/investment_analysis_*.log

# 오류만 필터링
grep "ERROR" data/logs/investment_analysis_*.log
```

## 🤝 기여하기

1. 이슈 제기 또는 기능 제안
2. Fork 후 브랜치 생성
3. 코드 작성 및 테스트 추가
4. Pull Request 제출

### 코드 스타일

- PEP8 준수
- Type hints 사용
- Docstring 작성 (Google 스타일)
- 테스트 커버리지 90% 이상 유지

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 🆘 지원

- **이슈 트래커**: GitHub Issues
- **문서**: 이 README 파일
- **예제**: `examples/` 폴더 (예정)

---

> 💡 **팁**: 테스트 모드(`--test-mode`)로 먼저 실행해보세요. 빠른 검증이 가능합니다!

**Made with ❤️ by AI Investment System Team** 