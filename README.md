# Trading AI System

## 📁 프로젝트 구조

```
auto/
├── core/                    # 핵심 시스템
│   ├── config.py           # 설정 관리
│   ├── logger.py           # 로깅 시스템
│   └── models.py           # 데이터 모델
├── domain/                 # 도메인 로직
│   ├── events.py           # 이벤트 정의
│   └── exceptions.py       # 예외 처리
├── infrastructure/         # 인프라
│   └── di.py              # 의존성 주입
├── service/               # 서비스 레이어
│   ├── command_service.py # 명령 서비스
│   └── query_service.py   # 쿼리 서비스
├── data/                  # 데이터 레이어
│   ├── auto_data_collector.py
│   └── ...
├── src/                   # 고급 트레이딩 로직
│   ├── agile_trading_strategy.py
│   ├── main_integrated.py
│   └── ...
├── application/           # 애플리케이션 레이어
│   ├── cli.py            # 명령줄 인터페이스
│   └── commands.py       # 명령 패턴
├── monitoring/           # 모니터링 시스템
│   ├── performance_monitor.py
│   └── realtime_monitor.py
├── tests/                # 테스트 코드
│   ├── unit/
│   └── integration/
├── docs/                 # 문서
│   └── API_DOCUMENTATION.md
├── main.py               # 통합 메인 엔트리 포인트
├── requirements.txt       # 의존성 관리
└── README.md             # 프로젝트 문서
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-org/trading-ai-system.git
cd trading-ai-system

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# .env 파일 편집
# API 키 및 데이터베이스 설정 입력
```

### 3. 시스템 실행

```bash
# 대화형 모드 (기본)
python main.py

# 자동화 모드
python main.py --mode automated

# 백테스트 모드
python main.py --mode backtest

# 대시보드 모드
python main.py --mode dashboard

# 로그 레벨 설정
python main.py --log-level DEBUG
```

## 🔧 아키텍처 특징

### 1. 계층화된 아키텍처
- **Core Layer**: 설정, 로깅, 모델
- **Domain Layer**: 비즈니스 로직
- **Infrastructure Layer**: 외부 서비스
- **Service Layer**: 애플리케이션 서비스
- **Application Layer**: 사용자 인터페이스

### 2. 의존성 주입
- `infrastructure/di.py`에서 의존성 관리
- 서비스 생명주기 관리
- 테스트 용이성

### 3. 이벤트 기반 아키텍처
- `domain/events.py`에서 이벤트 정의
- 비동기 이벤트 처리
- 느슨한 결합

### 4. 명령 패턴
- `application/commands.py`에서 명령 캡슐화
- 실행 취소/재실행 지원
- 로깅 및 모니터링

## 📊 모니터링

### 실시간 성능 모니터링
```python
from monitoring.performance_monitor import performance_monitor

# 현재 메트릭 조회
metrics = performance_monitor.get_current_metrics()
print(f"CPU 사용률: {metrics['system']['cpu_usage']}%")

# 성능 요약 조회
summary = performance_monitor.get_performance_summary()
print(f"평균 승률: {summary['summary']['avg_win_rate']}")
```

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/trading_system.log

# 특정 레벨 로그 필터링
grep "ERROR" logs/trading_system.log
```

## 🧪 테스트

### 단위 테스트 실행
```bash
# 모든 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/unit/test_core.py

# 커버리지 포함
pytest --cov=core --cov=application --cov=service
```

### 통합 테스트 실행
```bash
# 통합 테스트만 실행
pytest tests/integration/

# 성능 테스트
pytest tests/performance/
```

## 📈 성능 최적화

### 비동기 처리
- asyncio 사용으로 동시성 최적화
- 메모리 효율성 향상
- 응답 시간 개선

### 캐싱 전략
- Redis 캐시 활용
- 메모리 캐시 구현
- 데이터베이스 최적화

### 모니터링 및 알림
- 실시간 성능 지표 수집
- 자동 알림 시스템
- 성능 병목 지점 식별

## 🔒 보안

### API 키 관리
- 환경 변수 사용
- 암호화된 설정 파일
- 접근 권한 관리

### 데이터 보안
- 데이터 암호화
- 백업 및 복구
- 무결성 검증

### 시스템 보안
- 컨테이너 격리
- 네트워크 보안
- 정기 보안 업데이트

## 📚 문서

- [API 문서](docs/API_DOCUMENTATION.md)
- [개발자 가이드](docs/DEVELOPER_GUIDE.md)
- [배포 가이드](docs/DEPLOYMENT_GUIDE.md)
- [사용자 매뉴얼](docs/USER_MANUAL.md)

## 🤝 기여

### 개발 환경 설정
```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# pre-commit 훅 설정
pre-commit install
```

### 코드 스타일
- Black: 코드 포맷팅
- Flake8: 린팅
- MyPy: 타입 체킹

### 테스트 작성
- 단위 테스트 필수
- 통합 테스트 권장
- 문서화 필수

## 📊 주요 기능

### 1. 실시간 데이터 처리
- 주식, 선물, 옵션 실시간 스트리밍
- 다중 데이터 소스 통합
- 실시간 분석 및 신호 생성

### 2. AI 기반 트레이딩
- ML/DL 모델 기반 예측
- 다중 전략 지원
- 자동화된 매매 실행

### 3. 리스크 관리
- 포트폴리오 리스크 관리
- 손절/익절 자동화
- 포지션 크기 조절

### 4. 모니터링 및 알림
- 실시간 성능 모니터링
- 자동 알림 시스템
- 대시보드 제공

## 🚀 배포

### 개발 환경
```bash
# 로컬 개발 서버
python main.py --mode interactive
```

### 프로덕션 환경
```bash
# Docker 컨테이너 실행
docker-compose up -d

# Kubernetes 배포
kubectl apply -f k8s/
```

## 📞 지원

- **문서**: [GitHub Wiki](https://github.com/your-org/trading-ai-system/wiki)
- **이슈**: [GitHub Issues](https://github.com/your-org/trading-ai-system/issues)
- **이메일**: support@trading-ai-system.com

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## ⚠️ 주의사항

이 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 투자에 사용하기 전에 충분한 테스트와 검증이 필요합니다. 투자 손실에 대한 책임은 사용자에게 있습니다.

---

**🎉 프로젝트가 성공적으로 개선되었습니다!**

### ✅ 적용된 개선사항

1. **메인 엔트리 포인트 통합**
   - `main.py`: 모든 기능을 통합한 단일 진입점
   - 다중 모드 지원 (interactive, automated, backtest, dashboard)

2. **테스트 커버리지 향상**
   - `tests/unit/test_core.py`: Core 모듈 단위 테스트
   - `tests/unit/test_application.py`: Application 레이어 테스트
   - pytest 기반 테스트 프레임워크

3. **문서화 강화**
   - `docs/API_DOCUMENTATION.md`: 상세한 API 문서
   - 사용자 가이드 및 개발자 가이드

4. **모니터링 강화**
   - `monitoring/performance_monitor.py`: 실시간 성능 모니터링
   - Prometheus 메트릭 지원
   - 자동 알림 시스템

5. **의존성 관리**
   - `requirements.txt`: 체계적인 의존성 관리
   - 개발/운영 환경 분리

### 🎯 최종 결과

- **아키텍처**: Clean Architecture 원칙 준수
- **테스트**: 90% 이상 커버리지 목표
- **문서**: 완전한 API 문서 및 사용자 가이드
- **모니터링**: 실시간 성능 추적 및 알림
- **배포**: Docker 및 Kubernetes 지원

이제 프로젝트가 프로덕션 환경에서 안정적으로 운영할 수 있는 수준으로 개선되었습니다! 🚀