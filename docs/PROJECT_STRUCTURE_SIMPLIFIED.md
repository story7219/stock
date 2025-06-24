"""
🏗️ AI 기반 투자 분석 시스템 - 효율적 프로젝트 구조

## 📁 단순화된 실용적 구조

```
test_stock/                              # 프로젝트 루트
├── 📂 src/                              # 메인 소스코드
│   ├── 📂 core/                         # 핵심 모델 & 기본 클래스
│   │   ├── __init__.py
│   │   ├── models.py                    # StockData, AnalysisResult 등
│   │   ├── base_strategy.py             # 기본 전략 클래스
│   │   └── exceptions.py                # 커스텀 예외
│   │
│   ├── 📂 data/                         # 데이터 수집 & 처리
│   │   ├── __init__.py
│   │   ├── collectors.py                # 데이터 수집기 (코스피, 나스닥, S&P500)
│   │   ├── cleaner.py                   # 데이터 정제
│   │   └── multi_collector.py           # 통합 수집기
│   │
│   ├── 📂 analysis/                     # 분석 엔진
│   │   ├── __init__.py
│   │   ├── technical_analyzer.py        # 기술적 분석 (차트패턴, 지표)
│   │   └── indicators.py                # 기술적 지표들
│   │
│   ├── 📂 strategies/                   # 투자 전략
│   │   ├── __init__.py
│   │   ├── investment_masters.py        # 15명 투자 대가 전략
│   │   └── strategy_manager.py          # 전략 관리자
│   │
│   ├── 📂 ai/                           # AI 분석
│   │   ├── __init__.py
│   │   ├── gemini_analyzer.py           # Gemini AI 분석기
│   │   └── prompts.py                   # AI 프롬프트 템플릿
│   │
│   ├── 📂 reports/                      # 리포트 생성
│   │   ├── __init__.py
│   │   ├── generator.py                 # 리포트 생성기
│   │   └── templates/                   # 리포트 템플릿
│   │       ├── html_template.py
│   │       └── pdf_template.py
│   │
│   ├── 📂 utils/                        # 유틸리티
│   │   ├── __init__.py
│   │   ├── helpers.py                   # 헬퍼 함수
│   │   ├── config.py                    # 설정 관리
│   │   └── logger.py                    # 로깅 설정
│   │
│   └── __init__.py                      # 메인 패키지 초기화
│
├── 📂 tests/                            # 테스트
│   ├── test_data.py
│   ├── test_strategies.py
│   ├── test_ai.py
│   └── fixtures/                        # 테스트 데이터
│
├── 📂 data/                             # 데이터 저장소
│   ├── cache/                           # 캐시
│   ├── reports/                         # 생성된 리포트
│   └── logs/                            # 로그 파일
│
├── 📂 docs/                             # 문서
│   ├── README.md
│   ├── API.md
│   └── strategies/                      # 전략 설명서
│
├── 📄 main.py                           # 메인 진입점
├── 📄 requirements.txt                  # 의존성
├── 📄 requirements-dev.txt              # 개발 의존성
├── 📄 .env.example                      # 환경변수 예시
├── 📄 setup.py                          # 설치 스크립트
├── 📄 pytest.ini                        # 테스트 설정
├── 📄 .gitignore                        # Git 무시
└── 📄 README.md                         # 프로젝트 설명
```

## 🎯 **핵심 설계 원칙**

### 1. **단순성 우선**
- 과도한 폴더 분리 제거
- 실제 개발에 필요한 구조만 유지

### 2. **기능별 모듈화**
- core: 기본 모델과 클래스
- data: 데이터 관련 모든 기능
- analysis: 분석 엔진
- strategies: 투자 전략
- ai: AI 분석
- reports: 리포트 생성
- utils: 공통 유틸리티

### 3. **확장성 보장**
- 새로운 전략, AI 엔진 쉽게 추가
- 모듈간 느슨한 결합

### 4. **실용성 극대화**
- 개발자가 쉽게 찾을 수 있는 구조
- IDE 지원 최적화

## 🚀 **마이그레이션 계획**

1. **기존 파일 정리**: 중복 폴더 제거
2. **새 구조 생성**: 단순화된 폴더 생성
3. **파일 이동**: 기존 파일들을 새 위치로 이동
4. **import 수정**: 경로 업데이트
5. **테스트 업데이트**: 테스트 코드 수정

## 🔥 **개선 효과**

- **개발 속도 향상**: 파일 찾기 쉬워짐
- **유지보수 편의**: 구조가 직관적
- **팀 협업 개선**: 누구나 이해하기 쉬운 구조
- **확장성 보장**: 새 기능 추가 용이
""" 