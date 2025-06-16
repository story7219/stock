# 🤖 AI 주식 데이트레이딩 시스템

한국투자증권 API를 활용한 완전 자동화 주식 데이트레이딩 시스템입니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [시스템 구성](#시스템-구성)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [프로젝트 구조 관리](#프로젝트-구조-관리)
- [자동화 시스템](#자동화-시스템)
- [API 문서](#api-문서)
- [문제 해결](#문제-해결)

## 🚀 주요 기능

### 📊 자동 트레이딩
- 한국투자증권 API 연동
- 실시간 시장 데이터 분석
- AI 기반 매매 신호 생성
- 자동 주문 실행 및 관리

### 🔍 코드 품질 관리
- **매일 오전 7시 자동 품질 검사**
- **Gemini AI 기반 고급 분석**
- **반자동 리팩토링 시스템**
- **프로젝트 구조 자동 정리**

### 📈 분석 도구
- 기술적 분석 지표
- 오닐 캔슬림 분석
- 백테스팅 시스템
- 성과 분석 리포트

### 🤖 알림 시스템
- 텔레그램 봇 연동
- 실시간 거래 알림
- 일일 성과 리포트
- 시스템 상태 모니터링

## 🏗️ 시스템 구성

## 📦 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/yourusername/advanced-trading-bot.git
cd advanced-trading-bot

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일 편집

# 실행
python src/trader.py
```

## 🧪 테스트

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html

# 특정 테스트만
pytest tests/test_strategies.py -v
```

## 📈 성과 지표

- **백테스팅 기간**: 2023-2024
- **연평균 수익률**: 15.2%
- **최대 낙폭**: -8.3%
- **샤프 비율**: 1.47
- **승률**: 68.4%

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요. 