"""
🏗️ AI 기반 투자 분석 시스템 - 표준 프로젝트 구조

## 📁 프로젝트 구조 (개발 정석 기준)

```
test_stock/                              # 프로젝트 루트
├── 📂 src/                              # 메인 소스코드
│   ├── 📂 investment_analyzer/          # 메인 패키지 (도메인명)
│   │   ├── 📂 core/                     # 핵심 비즈니스 로직
│   │   │   ├── __init__.py
│   │   │   ├── models.py                # 데이터 모델 (StockData 등)
│   │   │   ├── base_strategy.py         # 기본 전략 클래스
│   │   │   ├── exceptions.py            # 커스텀 예외
│   │   │   └── constants.py             # 상수 정의
│   │   │
│   │   ├── 📂 data/                     # 데이터 계층
│   │   │   ├── __init__.py
│   │   │   ├── collectors/              # 데이터 수집기들
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_collector.py
│   │   │   │   ├── kospi_collector.py
│   │   │   │   ├── nasdaq_collector.py
│   │   │   │   └── sp500_collector.py
│   │   │   ├── processors/              # 데이터 처리
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cleaner.py
│   │   │   │   ├── validator.py
│   │   │   │   └── enricher.py
│   │   │   └── repositories/            # 데이터 저장소
│   │   │       ├── __init__.py
│   │   │       ├── cache_repository.py
│   │   │       └── file_repository.py
│   │   │
│   │   ├── 📂 analysis/                 # 분석 엔진
│   │   │   ├── __init__.py
│   │   │   ├── technical/               # 기술적 분석
│   │   │   │   ├── __init__.py
│   │   │   │   ├── indicators.py        # 기술적 지표
│   │   │   │   ├── patterns.py          # 차트 패턴
│   │   │   │   └── signals.py           # 매매 신호
│   │   │   ├── fundamental/             # 기본 분석
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ratios.py
│   │   │   │   └── scoring.py
│   │   │   └── quantitative/            # 퀀트 분석
│   │   │       ├── __init__.py
│   │   │       ├── factors.py
│   │   │       └── models.py
│   │   │
│   │   ├── 📂 strategies/               # 투자 전략
│   │   │   ├── __init__.py
│   │   │   ├── masters/                 # 투자 대가들
│   │   │   │   ├── __init__.py
│   │   │   │   ├── buffett.py
│   │   │   │   ├── lynch.py
│   │   │   │   ├── graham.py
│   │   │   │   └── ... (기타 13명)
│   │   │   ├── factory.py               # 전략 팩토리
│   │   │   └── manager.py               # 전략 매니저
│   │   │
│   │   ├── 📂 ai/                       # AI 모듈
│   │   │   ├── __init__.py
│   │   │   ├── engines/                 # AI 엔진들
│   │   │   │   ├── __init__.py
│   │   │   │   ├── gemini_engine.py
│   │   │   │   ├── openai_engine.py
│   │   │   │   └── anthropic_engine.py
│   │   │   ├── prompts/                 # 프롬프트 관리
│   │   │   │   ├── __init__.py
│   │   │   │   ├── strategy_prompts.py
│   │   │   │   └── analysis_prompts.py
│   │   │   └── reasoning/               # 추론 로직
│   │   │       ├── __init__.py
│   │   │       ├── selector.py
│   │   │       └── optimizer.py
│   │   │
│   │   ├── 📂 reporting/                # 리포트 생성
│   │   │   ├── __init__.py
│   │   │   ├── generators/              # 리포트 생성기
│   │   │   │   ├── __init__.py
│   │   │   │   ├── pdf_generator.py
│   │   │   │   ├── html_generator.py
│   │   │   │   └── excel_generator.py
│   │   │   ├── templates/               # 템플릿
│   │   │   │   ├── pdf_templates/
│   │   │   │   ├── html_templates/
│   │   │   │   └── email_templates/
│   │   │   └── visualizations/          # 차트/그래프
│   │   │       ├── __init__.py
│   │   │       ├── charts.py
│   │   │       └── dashboard.py
│   │   │
│   │   ├── 📂 integrations/             # 외부 연동
│   │   │   ├── __init__.py
│   │   │   ├── apis/                    # API 클라이언트
│   │   │   │   ├── __init__.py
│   │   │   │   ├── yahoo_finance.py
│   │   │   │   ├── naver_finance.py
│   │   │   │   └── kis_api.py
│   │   │   ├── notifications/           # 알림 서비스
│   │   │   │   ├── __init__.py
│   │   │   │   ├── email_notifier.py
│   │   │   │   ├── telegram_notifier.py
│   │   │   │   └── slack_notifier.py
│   │   │   └── storage/                 # 외부 저장소
│   │   │       ├── __init__.py
│   │   │       ├── google_sheets.py
│   │   │       └── database.py
│   │   │
│   │   ├── 📂 utils/                    # 유틸리티
│   │   │   ├── __init__.py
│   │   │   ├── helpers.py               # 헬퍼 함수
│   │   │   ├── decorators.py            # 데코레이터
│   │   │   ├── validators.py            # 검증기
│   │   │   └── formatters.py            # 포맷터
│   │   │
│   │   ├── 📂 config/                   # 설정 관리
│   │   │   ├── __init__.py
│   │   │   ├── settings.py              # 설정 클래스
│   │   │   ├── environments/            # 환경별 설정
│   │   │   │   ├── development.py
│   │   │   │   ├── production.py
│   │   │   │   └── testing.py
│   │   │   └── validators.py            # 설정 검증
│   │   │
│   │   └── __init__.py                  # 메인 패키지 초기화
│   │
│   └── 📂 cli/                          # 명령행 인터페이스
│       ├── __init__.py
│       ├── main.py                      # CLI 메인
│       ├── commands/                    # CLI 명령어들
│       │   ├── __init__.py
│       │   ├── analyze.py
│       │   ├── collect.py
│       │   └── report.py
│       └── interfaces/                  # 사용자 인터페이스
│           ├── __init__.py
│           ├── menus.py
│           └── progress.py
│
├── 📂 tests/                            # 테스트 코드
│   ├── 📂 unit/                         # 단위 테스트
│   │   ├── test_data/
│   │   ├── test_analysis/
│   │   ├── test_strategies/
│   │   └── test_ai/
│   ├── 📂 integration/                  # 통합 테스트
│   │   ├── test_workflows/
│   │   └── test_apis/
│   ├── 📂 fixtures/                     # 테스트 데이터
│   │   ├── sample_stocks.json
│   │   └── mock_responses.json
│   ├── conftest.py                      # pytest 설정
│   └── __init__.py
│
├── 📂 docs/                             # 문서
│   ├── 📂 api/                          # API 문서
│   ├── 📂 user_guide/                   # 사용자 가이드
│   ├── 📂 developer_guide/              # 개발자 가이드
│   ├── 📂 architecture/                 # 아키텍처 문서
│   └── 📂 strategies/                   # 전략 설명서
│
├── 📂 data/                             # 데이터 디렉토리
│   ├── 📂 cache/                        # 캐시 데이터
│   ├── 📂 reports/                      # 생성된 리포트
│   ├── 📂 logs/                         # 로그 파일
│   └── 📂 temp/                         # 임시 파일
│
├── 📂 scripts/                          # 스크립트
│   ├── setup_env.py                     # 환경 설정
│   ├── deploy.py                        # 배포 스크립트
│   └── maintenance.py                   # 유지보수
│
├── 📂 assets/                           # 정적 자원
│   ├── 📂 images/                       # 이미지
│   ├── 📂 fonts/                        # 폰트
│   └── 📂 icons/                        # 아이콘
│
├── 📂 docker/                           # Docker 설정
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── 📄 main.py                           # 메인 진입점
├── 📄 requirements.txt                  # 의존성
├── 📄 requirements-dev.txt              # 개발 의존성
├── 📄 pyproject.toml                    # 프로젝트 설정
├── 📄 setup.py                          # 설치 스크립트
├── 📄 Makefile                          # 빌드 설정
├── 📄 .env.example                      # 환경변수 예시
├── 📄 .gitignore                        # Git 무시 파일
├── 📄 README.md                         # 프로젝트 설명
├── 📄 CHANGELOG.md                      # 변경 이력
├── 📄 LICENSE                           # 라이선스
└── 📄 CONTRIBUTING.md                   # 기여 가이드
```

## 🎯 **폴더 구조 설계 원칙**

### 1. **도메인 중심 설계 (DDD)**
- `investment_analyzer` 메인 패키지로 도메인 명확화
- 각 모듈이 비즈니스 도메인을 반영

### 2. **관심사 분리 (SoC)**
- 데이터, 분석, 전략, AI, 리포팅 계층 분리
- 각 계층은 독립적으로 발전 가능

### 3. **의존성 방향성**
- 상위 계층이 하위 계층에만 의존
- core → data → analysis → strategies → ai → reporting

### 4. **확장성 고려**
- 새로운 전략, AI 엔진, 데이터 소스 쉽게 추가
- 플러그인 아키텍처 지원

### 5. **테스트 용이성**
- 각 모듈별 단위 테스트 가능
- Mock 객체 활용 용이한 구조

## 🚀 **주요 개선사항**

1. **패키지명 표준화**: snake_case 적용
2. **계층 구조 명확화**: 비즈니스 로직과 인프라 분리  
3. **설정 관리 체계화**: 환경별 설정 분리
4. **테스트 구조 개선**: 단위/통합 테스트 분리
5. **문서화 체계**: 사용자/개발자 가이드 분리
6. **배포 지원**: Docker, 스크립트 추가

## 📋 **마이그레이션 순서**

1. 새 폴더 구조 생성
2. 기존 파일들을 새 위치로 이동
3. import 경로 수정
4. 테스트 코드 업데이트  
5. 문서 업데이트
6. CI/CD 파이프라인 조정
""" 