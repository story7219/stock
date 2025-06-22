# 🚀 고성능 HTS (Home Trading System)

**비동기 처리 | 멀티레벨 캐싱 | 성능 최적화**

투자 대가들의 전략을 기반으로 한 AI 기반 주식 분석 및 추천 시스템입니다.

## ✨ 주요 특징

### 🎯 투자 대가별 전략 분석
- **Warren Buffett**: 가치투자 전략 (안정성, 장기 추세 분석)
- **Peter Lynch**: 성장투자 전략 (모멘텀, 섹터 분석)
- **William O'Neil**: CAN SLIM 전략 (종합적 기술 분석)

### 🚀 고성능 아키텍처
- **비동기 처리**: asyncio 기반 병렬 데이터 처리
- **멀티레벨 캐싱**: 메모리 → Redis → 디스크 계층화
- **커넥션 풀링**: 데이터베이스 연결 최적화
- **메모리 최적화**: 약한 참조, __slots__ 사용

### 📊 실시간 모니터링
- CPU, 메모리, 디스크, 네트워크 사용률 모니터링
- 성능 메트릭 실시간 수집 및 분석
- 자동 메모리 정리 및 최적화

### 🎨 현대적 UI
- Tkinter 기반 반응형 GUI
- 실시간 차트 및 데이터 시각화
- 지연 로딩 및 가상화된 컴포넌트

## 📦 설치 및 실행

### 필수 요구사항
- Python 3.10 이상
- Windows 10/11 (현재 버전)

### 1. 저장소 클론
```bash
git clone <repository-url>
cd test_stock
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 실행
```bash
# Python으로 직접 실행
python main.py

# 또는 배치 파일 사용 (Windows)
run_hts.bat
```

## 🏗️ 프로젝트 구조

```
test_stock/
├── config/                 # 설정 관리
│   └── settings.py         # 환경변수 및 앱 설정
├── core/                   # 핵심 시스템
│   ├── cache_manager.py    # 멀티레벨 캐싱 시스템
│   ├── database_manager.py # 비동기 DB 관리
│   └── performance_monitor.py # 성능 모니터링
├── ui_interfaces/          # 사용자 인터페이스
│   ├── optimized_hts_gui.py # 메인 GUI 시스템
│   ├── chart_manager.py    # 차트 렌더링
│   ├── data_manager.py     # 데이터 처리
│   └── ai_manager.py       # AI 분석 엔진
├── tests/                  # 테스트 코드
│   └── test_hts_system.py  # 통합 테스트
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 의존성 패키지
├── run_hts.bat            # Windows 실행 스크립트
└── README.md              # 프로젝트 문서
```

## 🔧 주요 구성 요소

### CacheManager (core/cache_manager.py)
- 3단계 캐시 시스템: 메모리 → Redis → 디스크
- TTL 기반 자동 만료
- 캐시 히트율 및 성능 통계 수집

### DatabaseManager (core/database_manager.py)
- SQLAlchemy 기반 비동기 ORM
- 커넥션 풀링으로 성능 최적화
- 대량 데이터 배치 처리

### PerformanceMonitor (core/performance_monitor.py)
- 실시간 시스템 리소스 모니터링
- 메모리 누수 감지 및 자동 정리
- 성능 메트릭 수집 및 분석

### DataManager (ui_interfaces/data_manager.py)
- 주식 데이터 수집 및 처리
- 기술적 지표 계산 (RSI, MA, MACD, 볼린저 밴드)
- 실시간 데이터 시뮬레이션

### AIManager (ui_interfaces/ai_manager.py)
- 투자 대가별 전략 구현
- 종목 스크리닝 및 분석
- 시장 심리 분석

## 🎯 사용법

### 1. 기본 실행
프로그램을 실행하면 메인 GUI가 나타납니다:
- 좌측: 주식 목록 및 필터링
- 중앙: 차트 및 기술적 지표
- 우측: AI 분석 결과 및 추천

### 2. 투자 전략 선택
AI 분석 탭에서 원하는 투자 대가의 전략을 선택:
- Warren Buffett: 안정적인 가치투자
- Peter Lynch: 성장주 발굴
- William O'Neil: 기술적 분석 기반

### 3. 종목 분석
- 종목 코드 입력 또는 목록에서 선택
- AI 분석 버튼 클릭
- 상세한 분석 결과 및 추천 이유 확인

### 4. 시장 모니터링
- 실시간 성능 메트릭 확인
- 시장 심리 분석 결과 모니터링
- 섹터별 분석 및 비교

## 🔬 테스트

### 단위 테스트 실행
```bash
python -m pytest tests/ -v
```

### 통합 테스트 실행
```bash
python tests/test_hts_system.py
```

### 성능 테스트
```bash
python -m pytest tests/test_hts_system.py::TestHTSPerformance -v
```

## ⚙️ 설정

### 환경변수 설정
`.env` 파일을 생성하여 다음 설정을 추가할 수 있습니다:

```env
# 데이터베이스 설정
DATABASE_URL=sqlite:///./hts_data.db
DATABASE_POOL_SIZE=20

# Redis 설정 (선택사항)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=

# 캐시 설정
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=10000

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=hts_application.log

# 성능 설정
PERFORMANCE_MONITOR_INTERVAL=1.0
MAX_MEMORY_USAGE_PERCENT=80
```

### 성능 튜닝
- `CACHE_MAX_SIZE`: 메모리 캐시 최대 크기
- `DATABASE_POOL_SIZE`: DB 커넥션 풀 크기
- `PERFORMANCE_MONITOR_INTERVAL`: 모니터링 주기
- `MAX_MEMORY_USAGE_PERCENT`: 메모리 사용률 임계값

## 🚨 문제 해결

### 일반적인 문제들

#### 1. 패키지 설치 오류
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 클리어 후 재설치
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

#### 2. 메모리 부족 오류
- `MAX_MEMORY_USAGE_PERCENT` 값을 낮춤 (예: 70)
- `CACHE_MAX_SIZE` 값을 줄임
- 가상 메모리 설정 확인

#### 3. 데이터베이스 연결 오류
- `DATABASE_URL` 설정 확인
- 데이터베이스 파일 권한 확인
- `DATABASE_POOL_SIZE` 값 조정

#### 4. GUI 응답 없음
- 메인 스레드 블로킹 확인
- 비동기 작업 상태 확인
- 로그 파일에서 오류 메시지 확인

### 로그 확인
```bash
# 애플리케이션 로그
tail -f hts_application.log

# 에러 로그만 필터링
grep ERROR hts_application.log
```

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🔮 향후 계획

### v1.1 (예정)
- [ ] 실제 주식 API 연동 (yfinance, Alpha Vantage)
- [ ] 백테스팅 기능 추가
- [ ] 포트폴리오 관리 기능

### v1.2 (예정)
- [ ] 웹 인터페이스 추가 (FastAPI + React)
- [ ] 모바일 앱 지원
- [ ] 클라우드 배포 지원

### v2.0 (장기)
- [ ] 머신러닝 모델 통합
- [ ] 실시간 뉴스 분석
- [ ] 소셜 미디어 감성 분석

## 📞 지원

문제가 발생하거나 질문이 있으시면:
- GitHub Issues에 등록
- 이메일: [your-email@example.com]
- 문서: [프로젝트 위키](wiki-url)

---

**⚠️ 투자 주의사항**

이 시스템은 교육 및 연구 목적으로 개발되었습니다. 실제 투자 결정은 본인의 판단과 책임 하에 이루어져야 하며, 과거 성과가 미래 수익을 보장하지 않습니다. 투자 전 충분한 검토와 전문가 상담을 권장합니다. 