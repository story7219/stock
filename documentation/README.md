# 🚀 Investing TOP5

> **투자 대가들의 전략으로 찾는 최고의 주식 TOP5**  
> 워렌 버핏, 피터 린치, 조엘 그린블라트의 투자 전략을 AI와 결합한 종합 주식 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI](https://img.shields.io/badge/AI-Gemini-orange.svg)](https://ai.google.dev/)

## 📋 목차

- [🎯 프로젝트 소개](#-프로젝트-소개)
- [✨ 주요 기능](#-주요-기능)
- [🏗️ 시스템 구조](#️-시스템-구조)
- [🚀 빠른 시작](#-빠른-시작)
- [📊 투자 전략](#-투자-전략)
- [🤖 AI 분석](#-ai-분석)
- [📈 사용법](#-사용법)
- [⚙️ 설정](#️-설정)
- [🔧 개발](#-개발)
- [📄 라이선스](#-라이선스)

## 🎯 프로젝트 소개

**Investing TOP5**는 세계적인 투자 대가들의 검증된 투자 전략을 현대적인 AI 기술과 결합하여, 한국과 미국 주식 시장에서 최고의 투자 기회를 찾아주는 종합 분석 시스템입니다.

### 🌟 핵심 가치

- **검증된 전략**: 워렌 버핏, 피터 린치, 조엘 그린블라트의 실제 투자 철학 구현
- **AI 강화**: Google Gemini AI를 활용한 심층 분석 및 통찰
- **실시간 데이터**: Yahoo Finance, KRX 등 신뢰할 수 있는 데이터 소스
- **사용자 친화적**: 직관적인 인터페이스와 명확한 분석 결과

## ✨ 주요 기능

### 📊 투자 전략 분석
- **워렌 버핏 전략**: 가치투자, 내재가치 분석, 장기 투자 관점
- **피터 린치 전략**: 성장투자, PEG 비율, 모멘텀 분석
- **조엘 그린블라트 전략**: 마법공식, 수익률과 밸류에이션 균형

### 🤖 AI 통합 분석
- 실시간 시장 상황 분석
- 종목별 심층 분석 및 투자 등급 부여
- 포트폴리오 최적화 제안
- 리스크 평가 및 관리

### 🌍 다중 시장 지원
- **한국 시장**: KOSPI, KOSDAQ 전 종목 분석
- **미국 시장**: NYSE, NASDAQ 주요 종목 분석
- **글로벌 포트폴리오**: 국가간 분산투자 전략

### 📈 고급 분석 기능
- TOP5 종목 자동 추천
- 사용자 맞춤 포트폴리오 분석
- 실시간 시장 동향 분석
- 성과 추적 및 백테스팅

## 🏗️ 시스템 구조

```
investing_top5/
├── 📁 data/                      # 데이터 저장소
│   ├── 📁 raw/                  # 원시 데이터
│   ├── 📁 processed/            # 가공된 데이터
│   └── 📁 external/             # 외부 API 데이터
├── 📁 strategies/               # 투자 전략 구현
│   ├── 📄 buffett.py           # 워렌 버핏 전략
│   ├── 📄 lynch.py             # 피터 린치 전략
│   ├── 📄 greenblatt.py        # 조엘 그린블라트 전략
│   └── 📄 common.py            # 공통 지표 계산
├── 📁 recommenders/             # 추천 시스템
│   ├── 📄 recommender.py       # 통합 추천 엔진
│   └── 📄 scorer.py            # 점수화 로직
├── 📁 ai_integration/           # AI 연동
│   ├── 📄 gemini_client.py     # Gemini AI 클라이언트
│   └── 📄 ai_preprocessor.py   # 데이터 전처리
├── 📁 utils/                    # 유틸리티
│   ├── 📄 data_manager.py      # 데이터 관리
│   ├── 📄 news_processor.py    # 뉴스 분석
│   └── 📄 backtest.py          # 백테스팅
├── 📁 configs/                  # 설정 파일
│   ├── 📄 config.py            # 기본 설정
│   └── 📄 settings.py          # 상세 설정
├── 📁 notebooks/                # 분석 노트북
├── 📄 main.py                   # 메인 실행 파일
├── 📄 requirements.txt          # 패키지 의존성
└── 📄 README.md                # 프로젝트 문서
```

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/investing_top5.git
cd investing_top5

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ API 키 설정

`.env` 파일을 생성하고 필요한 API 키를 설정하세요:

```env
# Google Gemini AI API 키 (필수)
GEMINI_API_KEY=your_gemini_api_key_here

# 선택적 설정
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 3️⃣ 실행

```bash
# 메인 프로그램 실행
python main.py
```

## 📊 투자 전략

### 🏛️ 워렌 버핏 전략 (가치투자)

> "가격은 당신이 지불하는 것이고, 가치는 당신이 얻는 것이다."

**핵심 원칙:**
- 내재가치 대비 저평가된 우량기업 선호
- 높은 ROE(15% 이상)와 안정적인 수익성
- 낮은 부채비율(30% 이하)과 강한 재무구조
- 지속가능한 경쟁우위(Economic Moat)
- 장기 보유 관점의 투자

**평가 지표:**
- PER < 15, PBR < 1.5
- ROE > 15%, ROA > 10%
- 부채비율 < 30%
- 배당수익률 > 2%

### 🚀 피터 린치 전략 (성장투자)

> "당신이 이해할 수 있는 회사에 투자하라."

**핵심 원칙:**
- PEG 비율 1.0 이하의 성장주 선호
- 15-30% 성장률의 적정 성장 기업
- 이해하기 쉬운 비즈니스 모델
- 업계 내 경쟁우위와 시장 점유율 확대
- 모멘텀과 기술적 요인 고려

**평가 지표:**
- PEG < 1.0
- 매출성장률 15-30%
- 순이익성장률 > 20%
- 시장점유율 확대 추세

### ⚡ 조엘 그린블라트 전략 (마법공식)

> "좋은 회사를 싼 가격에 사는 것이 핵심이다."

**핵심 원칙:**
- 높은 자본수익률(ROIC) 기업 선호
- 저평가된 기업(높은 Earnings Yield)
- 수익률과 밸류에이션의 균형
- 정량적 지표 기반 체계적 접근
- 시장 비효율성 활용

**평가 지표:**
- ROIC > 20%
- Earnings Yield > 10%
- 마법공식 순위 상위 25%

## 🤖 AI 분석

### 🧠 Gemini AI 통합

**분석 영역:**
- **정량 분석**: 재무제표, 밸류에이션, 기술적 지표
- **정성 분석**: 뉴스, 시장 동향, 업계 전망
- **리스크 분석**: 변동성, 상관관계, 시나리오 분석
- **포트폴리오 최적화**: 분산효과, 리밸런싱 제안

**AI 등급 시스템:**
- **A+**: 90점 이상 - 최우수 투자 종목
- **A**: 80-89점 - 우수 투자 종목
- **B+**: 70-79점 - 양호한 투자 종목
- **B**: 60-69점 - 보통 투자 종목
- **C+**: 50-59점 - 주의 필요 종목
- **C**: 40-49점 - 위험 종목
- **D**: 40점 미만 - 투자 부적합 종목

## 📈 사용법

### 🎯 TOP5 종목 추천

```python
# 한국 시장 TOP5 추천
results = await system.get_top5_recommendations("KR")

# 미국 시장 TOP5 추천  
results = await system.get_top5_recommendations("US")
```

### 📊 사용자 포트폴리오 분석

```python
# 포트폴리오 분석
symbols = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER
analysis = await system.analyze_custom_portfolio(symbols, "KR")
```

### 📈 시장 분석

```python
# 시장 전체 분석
market_analysis = await system._get_market_analysis("KR")
```

## ⚙️ 설정

### 📝 주요 설정 파일

- `configs/config.py`: 기본 설정
- `configs/settings.py`: 상세 시스템 설정
- `.env`: 환경변수 및 API 키

### 🎛️ 설정 가능한 항목

```python
# 전략 가중치 조정
STRATEGY_WEIGHTS = {
    'buffett': 0.35,
    'lynch': 0.35, 
    'greenblatt': 0.30
}

# 점수 계산 가중치
SCORING_WEIGHTS = {
    'financial_health': 0.25,
    'profitability': 0.25,
    'growth': 0.20,
    'valuation': 0.20,
    'momentum': 0.10
}
```

## 🔧 개발

### 🧪 테스트 실행

```bash
# 전체 테스트
python -m pytest tests/

# 특정 테스트
python tests/test_integrated.py
```

### 📊 성능 모니터링

```bash
# 시스템 성능 통계
python -c "from main import InvestingTOP5; system = InvestingTOP5(); print(system.ai_client.get_performance_stats())"
```

### 🐛 디버깅

로그 파일 위치: `logs/investing_top5.log`

```bash
# 실시간 로그 모니터링
tail -f logs/investing_top5.log
```

## 📊 성과 예시

### 🏆 실제 분석 결과 (예시)

| 순위 | 종목명 | 등급 | 점수 | 전략 | 추천 이유 |
|------|--------|------|------|------|-----------|
| 1위 | 삼성전자 | A+ | 92.5 | 버핏+린치 | 강력한 경쟁우위, 안정적 성장 |
| 2위 | NAVER | A | 88.3 | 린치+그린블라트 | 높은 성장률, 저평가 |
| 3위 | LG화학 | A | 85.7 | 버핏+그린블라트 | 우수한 재무구조, 미래 성장 |
| 4위 | 카카오 | B+ | 78.9 | 린치 | 플랫폼 경쟁력, 성장 잠재력 |
| 5위 | 셀트리온 | B+ | 76.2 | 그린블라트 | 바이오 혁신, 글로벌 진출 |

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원 및 문의

- **이슈 리포트**: [GitHub Issues](https://github.com/your-username/investing_top5/issues)
- **기능 제안**: [GitHub Discussions](https://github.com/your-username/investing_top5/discussions)
- **이메일**: your-email@example.com

## ⚠️ 면책조항

이 프로그램은 교육 및 연구 목적으로 제작되었습니다. 실제 투자 결정은 사용자의 책임이며, 투자로 인한 손실에 대해 개발자는 책임지지 않습니다. 투자 전 전문가와 상담하시기 바랍니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

<div align="center">

**🚀 Investing TOP5로 스마트한 투자를 시작하세요!**

[![Star](https://img.shields.io/github/stars/your-username/investing_top5?style=social)](https://github.com/your-username/investing_top5/stargazers)
[![Fork](https://img.shields.io/github/forks/your-username/investing_top5?style=social)](https://github.com/your-username/investing_top5/network/members)

</div>
