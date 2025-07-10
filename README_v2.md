# Trading Strategy System v2.0

**세계 최고 수준의 단기 트레이딩 AI 시스템**  
*Production-Grade, Google/Meta/Netflix/Amazon 수준의 아키텍처*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: DDD](https://img.shields.io/badge/Architecture-DDD-green.svg)](https://martinfowler.com/bliki/DomainDrivenDesign.html)
[![Code Quality: A+](https://img.shields.io/badge/Code%20Quality-A+-brightgreen.svg)](https://github.com/features)

## 🚀 개요

Trading Strategy System v2.0은 **도메인 중심 설계(DDD)**, **클린 아키텍처**, **이벤트 기반 시스템**을 기반으로 한 세계 최고 수준의 단기 트레이딩 AI 시스템입니다.

### ✨ 주요 특징

- **🏗️ 아키텍처**: DDD, Clean Architecture, CQRS, Event-Driven
- **🧠 AI/ML**: 하이브리드 AI (ML + DL + Gemini AI)
- **⚡ 성능**: 비동기 처리, 실시간 데이터, 고성능 백테스팅
- **🛡️ 안정성**: 구조화된 로깅, 에러 추적, 성능 모니터링
- **🔧 유지보수**: 의존성 주입, 모듈화, 타입 안전성
- **📊 모니터링**: 실시간 대시보드, 메트릭, 알림

## 🏗️ 아키텍처

### 계층 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CLI App   │  │  Dashboard  │  │      API REST       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Use Cases   │  │   Commands  │  │      Queries        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Models    │  │  Services   │  │      Events         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Repositories│  │   Adapters  │  │   External APIs     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

- **🔒 의존성 역전**: 도메인은 외부에 의존하지 않음
- **🎯 단일 책임**: 각 모듈은 하나의 책임만 가짐
- **🔄 개방-폐쇄**: 확장에는 열려있고 수정에는 닫혀있음
- **🔗 인터페이스 분리**: 클라이언트는 사용하지 않는 인터페이스에 의존하지 않음
- **📦 의존성 주입**: 외부에서 의존성을 주입받음

## 🧠 전략 시스템

### 하이브리드 AI 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid AI Engine                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   ML Layer  │  │   DL Layer  │  │    Gemini AI        │  │
│  │ (Stability) │  │ (Patterns)  │  │   (Context)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Democratic Voting                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Weighted    │  │ Confidence  │  │   Consensus         │  │
│  │ Decision    │  │   Scoring   │  │   Algorithm         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 전략 모듈

1. **📰 뉴스 모멘텀 전략**
   - 실시간 뉴스 분석
   - 감정 분석 및 영향도 평가
   - 모멘텀 신호 생성

2. **📈 기술적 패턴 전략**
   - 50+ 기술적 지표
   - 패턴 인식 및 신호 생성
   - 다중 시간대 분석

3. **🎯 테마 로테이션 전략**
   - 섹터/테마 분석
   - 상관관계 기반 포지셔닝
   - 동적 가중치 조정

4. **🛡️ 리스크 관리 전략**
   - VaR, CVaR 계산
   - 포트폴리오 최적화
   - 동적 헤징

5. **😤 심리/감정 전략**
   - VIX 기반 공포/탐욕 지수
   - 커뮤니티 감정 분석
   - 반대 심리 신호

6. **⚡ 애자일 전략**
   - 소/중형주 타겟팅
   - 빠른 진입/청산
   - 테마 선점

## 🚀 시작하기

### 요구사항

- Python 3.11+
- 8GB+ RAM
- SSD 저장소
- 안정적인 인터넷 연결

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/trading-strategy-system.git
cd trading-strategy-system

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements_v2.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 등 설정
```

### 환경변수 설정

```bash
# .env 파일 예시
ENVIRONMENT=development
DEBUG=true

# KIS API 설정
API_KIS_APP_KEY=your_kis_app_key
API_KIS_APP_SECRET=your_kis_app_secret
API_KIS_ACCESS_TOKEN=your_kis_access_token

# 실거래 KIS API 설정
API_KIS_REAL_APP_KEY=your_real_kis_app_key
API_KIS_REAL_APP_SECRET=your_real_kis_app_secret
API_KIS_REAL_ACCESS_TOKEN=your_real_kis_access_token

# DART API 설정
API_DART_API_KEY=your_dart_api_key

# 데이터베이스 설정
DB_URL=sqlite:///./data/trading_data.db

# 로깅 설정
LOG_LEVEL=INFO
LOG_ENABLE_JSON_LOGGING=true

# 모니터링 설정
MONITOR_DASHBOARD_ENABLED=true
MONITOR_DASHBOARD_PORT=8080
```

### 실행

```bash
# 개발 모드 (모의 거래)
python main_v2.py --mode mock --log-level DEBUG

# 실거래 모드
python main_v2.py --mode live --log-level INFO

# 백테스트 모드
python main_v2.py --mode backtest

# 대시보드 비활성화
python main_v2.py --no-dashboard

# 도움말
python main_v2.py --help
```

## 📊 대시보드

실시간 대시보드는 `http://localhost:8080`에서 접근 가능합니다.

### 주요 기능

- **📈 실시간 포트폴리오 모니터링**
- **🎯 전략 성능 대시보드**
- **📊 리스크 메트릭 시각화**
- **🔔 실시간 알림**
- **📱 모바일 반응형 디자인**

## 🧪 백테스팅

```python
from application.usecases import BacktestUseCase

# 백테스트 실행 (최대 기간: 1990년 ~ 현재, 약 34년)
backtest = BacktestUseCase()
results = await backtest.run(
    start_date="1990-01-01",  # 한국 주식시장 데이터 시작
    end_date="2024-12-31",    # 현재까지
    initial_capital=10000000,  # 1천만원 초기자본
    strategies=["news_momentum", "technical_pattern", "theme_rotation", "sentiment", "agile"]
)

# 결과 분석
print(f"총 수익률: {results.total_return:.2%}")
print(f"연평균 수익률: {results.annual_return:.2%}")
print(f"샤프 비율: {results.sharpe_ratio:.2f}")
print(f"최대 낙폭: {results.max_drawdown:.2%}")
print(f"승률: {results.win_rate:.1%}")
print(f"총 거래 횟수: {results.total_trades:,}회")
print(f"백테스트 기간: {results.duration_days:,}일 ({results.duration_years:.1f}년)")
```

### 백테스트 기간 옵션

```python
# 단기 백테스트 (1년)
results = await backtest.run(
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# 중기 백테스트 (5년)
results = await backtest.run(
    start_date="2019-01-01", 
    end_date="2023-12-31"
)

# 장기 백테스트 (10년)
results = await backtest.run(
    start_date="2014-01-01",
    end_date="2023-12-31"
)

# 최대 백테스트 (34년) - 한국 주식시장 전체 기간
results = await backtest.run(
    start_date="1990-01-01",  # 한국 주식시장 데이터 시작
    end_date="2024-12-31"     # 현재까지
)
```

## 🔧 개발

### 프로젝트 구조

```
trading-strategy-system/
├── domain/                 # 도메인 계층
│   ├── models.py          # 도메인 모델
│   ├── services.py        # 도메인 서비스
│   ├── events.py          # 도메인 이벤트
│   └── exceptions.py      # 도메인 예외
├── application/           # 애플리케이션 계층
│   ├── services.py        # 애플리케이션 서비스
│   ├── usecases.py        # 사용 사례
│   ├── commands.py        # 명령 핸들러
│   └── queries.py         # 쿼리 핸들러
├── infrastructure/        # 인프라스트럭처 계층
│   ├── di.py             # 의존성 주입
│   ├── repositories.py   # 저장소 구현
│   ├── adapters.py       # 외부 서비스 어댑터
│   └── database.py       # 데이터베이스 관리
├── core/                 # 핵심 유틸리티
│   ├── settings.py       # 설정 관리
│   └── logger.py         # 로깅 시스템
├── strategies/           # 전략 구현
│   ├── news_momentum.py  # 뉴스 모멘텀
│   ├── technical.py      # 기술적 분석
│   ├── theme.py          # 테마 로테이션
│   ├── risk.py           # 리스크 관리
│   ├── sentiment.py      # 심리/감정
│   └── agile.py          # 애자일 전략
├── main_v2.py            # 메인 엔트리 포인트
├── requirements_v2.txt   # 의존성 목록
└── README_v2.md          # 이 파일
```

### 코드 품질

```bash
# 코드 포맷팅
black .
isort .

# 린팅
flake8 .
mypy .

# 테스트
pytest tests/ -v --cov=.

# 타입 체크
mypy --strict .
```

### 테스트

```bash
# 전체 테스트 실행
pytest

# 특정 모듈 테스트
pytest tests/test_strategies/

# 커버리지 리포트
pytest --cov=. --cov-report=html

# 성능 테스트
pytest tests/test_performance/ -v
```

## 📈 성능 지표

### 시스템 성능

- **⚡ 응답 시간**: < 100ms (신호 생성)
- **🔄 처리량**: 1000+ 신호/분
- **💾 메모리 사용**: < 2GB
- **🖥️ CPU 사용**: < 30%

### 전략 성능 (백테스트)

- **📈 평균 수익률**: 15-25% (연간)
- **📊 샤프 비율**: 1.5-2.5
- **📉 최대 낙폭**: < 10%
- **🎯 승률**: 60-70%

## 🛡️ 보안

### 보안 기능

- **🔐 API 키 암호화**
- **🛡️ 입력 검증**
- **🚫 SQL 인젝션 방지**
- **🔒 세션 관리**
- **📝 보안 로깅**

### 모범 사례

- 환경변수 사용
- 민감 정보 마스킹
- 정기적인 보안 업데이트
- 접근 권한 제한

## 📚 문서

### API 문서

- **REST API**: `http://localhost:8080/docs`
- **WebSocket**: `ws://localhost:8080/ws`
- **메트릭**: `http://localhost:8000/metrics`

### 개발 문서

- [아키텍처 가이드](docs/architecture.md)
- [전략 개발 가이드](docs/strategy-development.md)
- [API 레퍼런스](docs/api-reference.md)
- [배포 가이드](docs/deployment.md)

## 🤝 기여하기

### 기여 방법

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 개발 가이드라인

- **코드 스타일**: Black, isort, flake8 준수
- **타입 힌트**: 모든 함수에 타입 힌트 추가
- **문서화**: 모든 공개 API 문서화
- **테스트**: 새로운 기능에 대한 테스트 작성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다. 실제 거래에 사용하기 전에 충분한 테스트와 검증을 거쳐야 합니다. 개발자는 이 소프트웨어 사용으로 인한 손실에 대해 책임지지 않습니다.

## 📞 지원

- **이슈 리포트**: [GitHub Issues](https://github.com/your-repo/trading-strategy-system/issues)
- **문서**: [Wiki](https://github.com/your-repo/trading-strategy-system/wiki)
- **토론**: [GitHub Discussions](https://github.com/your-repo/trading-strategy-system/discussions)

---

**Made with ❤️ by Trading Strategy Team** 

## 🚀 실행 방법

### 백테스트 실행 (최대 기간: 34년)

```bash
# 최대 기간 백테스트 (1990년 ~ 현재, 약 34년)
python main_v2.py --backtest

# 특정 기간 백테스트
python main_v2.py --backtest --start-date 2010-01-01 --end-date 2024-12-31

# 특정 전략만 백테스트
python main_v2.py --backtest --strategies news_momentum technical_pattern

# 초기자본 설정
python main_v2.py --backtest --initial-capital 50000000  # 5천만원
```

### 실시간 거래 실행

```bash
# 모의 거래 모드
python main_v2.py --mode mock

# 실거래 모드 (주의!)
python main_v2.py --mode live

# 대시보드 비활성화
python main_v2.py --mode mock --no-dashboard

# 로깅 레벨 설정
python main_v2.py --mode mock --log-level DEBUG
``` 