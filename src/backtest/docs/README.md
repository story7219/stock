# 🏆 완전 자동화 백테스팅 시스템

## 📋 개요

World-Class 수준의 완전 자동화 백테스팅 시스템입니다. 실제 투자 전에 철저히 검증된 신뢰할 수 있는 결과를 제공합니다.

## 🚀 주요 기능

### ✅ 현실적 거래환경 완벽 재현
- **정확한 시간 환경**: 정규장, 동시호가, 휴장일, 단축거래 완벽 반영
- **정확한 비용 구조**: 위탁수수료, 증권거래세, 농특세, 제세공과금 완벽 계산
- **정확한 호가/체결**: 호가단위, 체결순서, 부분체결, 미체결 시뮬레이션
- **동적 시장 환경**: 변동성, 거래량 패턴, 뉴스 임팩트 반영

### ✅ 다차원 백테스팅
- **Walk-Forward Analysis**: 24개월 학습 → 3개월 테스트 반복
- **Monte Carlo Simulation**: 1000회 랜덤 시나리오
- **Stress Testing**: 2008, 2020급 위기 상황 집중 테스트
- **Regime-Based Testing**: 상승/하락/횡보장별 성과 분석

### ✅ 엄격한 검증 기준
- **Out-of-Sample**: 훈련에 사용되지 않은 데이터로만 평가
- **Data Leakage 완전 차단**: 미래 정보 절대 사용 금지
- **Survivorship Bias 제거**: 상장폐지 종목 포함
- **Look-Ahead Bias 방지**: 당일 종가로 당일 매매 금지
- **Multiple Testing 보정**: Bonferroni 보정 적용

### ✅ 실전 환경 시뮬레이션
- **Network Latency**: 인터넷 지연 시뮬레이션
- **API Failure**: 증권사 API 장애 시나리오
- **System Crash**: 컴퓨터 다운 상황 대응
- **Market Halt**: 거래정지/서킷브레이커 시나리오
- **Liquidity Crisis**: 유동성 부족 상황 테스트

## 📁 파일 구조

```
backtest/
├── __init__.py              # 패키지 초기화
├── config.py                # 설정 관리
├── engine.py                # 메인 백테스팅 엔진
├── data_handler.py          # 데이터 핸들러
├── execution_handler.py     # 주문/체결 핸들러
├── strategy.py              # 전략 모듈
├── performance.py           # 성과 분석
├── stat_test.py            # 통계 검증
├── stress.py               # 스트레스 테스트
├── walkforward.py          # Walk-Forward 분석
├── montecarlo.py           # Monte Carlo 시뮬레이션
├── utils.py                # 유틸리티 함수
├── report.py               # 리포트 생성
├── run_backtest.py         # 메인 실행 스크립트
└── README.md               # 이 파일
```

## 🎯 사용법

### 기본 실행
```bash
python -m backtest.run_backtest
```

### 설정 커스터마이징
```python
from backtest.config import get_config

# 커스텀 설정
custom_config = {
    "n_simulations": 2000,  # Monte Carlo 시뮬레이션 횟수 증가
    "train_window": 36,     # 학습 기간 증가
    "test_window": 6,       # 테스트 기간 증가
}

config = get_config(custom_config)
```

## 📊 성과 지표

### 수익률 분석
- **총 수익률**: 절대 수익률
- **연환산 수익률**: 연간 환산 수익률
- **위험조정 수익률**: 리스크를 고려한 수익률

### 리스크 분석
- **샤프 비율**: 위험 대비 수익률
- **소르티노 비율**: 하방 위험 대비 수익률
- **칼마 비율**: 최대 낙폭 대비 수익률
- **오메가 비율**: 임계값 기준 수익률

### 안정성 분석
- **승률**: 수익 거래 비율
- **손익비**: 평균 수익 / 평균 손실
- **최대 낙폭**: 최대 손실 구간
- **회복 기간**: 손실 회복 소요 시간

## 🛡️ 검증 기준

### 최소 기준
- **샤프 비율**: > 1.5
- **최대 낙폭**: < -15%
- **통계적 유의성**: p < 0.01

### 우수 기준
- **샤프 비율**: > 2.5
- **최대 낙폭**: < -8%
- **모든 시장 환경에서 생존**

## 📈 출력 결과

### 자동 생성 파일
- `backtest_report_YYYYMMDD_HHMMSS.json`: 상세 백테스팅 결과
- `backtest_system.log`: 시스템 로그

### 리포트 구성
1. **Executive Summary**: 경영진용 요약 보고서
2. **Technical Report**: 기술적 상세 분석
3. **Risk Report**: 리스크 관리 보고서
4. **Performance Attribution**: 수익 기여도 분석
5. **Recommendation**: 개선 방안 제시

## 🔧 설정 옵션

### 기본 설정
```python
DEFAULT_CONFIG = {
    "data_path": "collected_data/kospi/krx_40years.parquet",
    "holiday_path": "config/krx_holidays.csv",
    "n_simulations": 1000,
    "train_window": 24,
    "test_window": 3,
    "commission_rate": 0.00015,
    "tax_rate": 0.0025,
}
```

### 위기 상황 설정
```python
"stress_scenarios": {
    "financial_crisis_2008": {"price_shock": -0.5, "duration": 18},
    "covid_crash_2020": {"price_shock": -0.35, "duration": 1},
    "black_monday_1987": {"price_shock": -0.22, "duration": 1},
}
```

## 🏆 특징

### World-Class 품질
- **Google/Meta/Netflix** 수준의 코드 품질
- **Zero Bug Tolerance**: 런타임 에러 완전 차단
- **Performance Excellence**: 고성능 병렬 처리
- **Security First**: 완전한 데이터 검증

### 완전 자동화
- **One-Click Execution**: 단일 명령어로 전체 파이프라인 실행
- **Auto Report Generation**: 자동 리포트 생성
- **Comprehensive Validation**: 다차원 검증 자동화
- **Real-Time Monitoring**: 실시간 성과 모니터링

### 실전 준비
- **Production Ready**: 즉시 배포 가능
- **Scalable Architecture**: 확장 가능한 아키텍처
- **Robust Error Handling**: 견고한 오류 처리
- **Comprehensive Logging**: 완전한 로깅 시스템

## 📞 지원

이 시스템은 실제 투자 전에 완벽히 검증된 신뢰할 수 있는 백테스팅 결과를 제공합니다. 모든 검증을 통과한 시스템만이 실전 투자에 사용됩니다.

---

**"실제 돈을 투자하기 전에 100% 확신을 가질 수 있는 완벽한 백테스팅 검증 시스템"** 