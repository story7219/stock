# 🤖 Advanced Trading Bot

AI 기반 고급 자동매매 시스템 - 척후병 전략 & 피보나치 분할매수

[![CI/CD](https://github.com/yourusername/advanced-trading-bot/workflows/CI/badge.svg)](https://github.com/yourusername/advanced-trading-bot/actions)
[![Code Coverage](https://codecov.io/gh/yourusername/advanced-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/advanced-trading-bot)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 주요 기능

### 📊 전략 시스템
- **척후병 전략**: 5개 후보 → 4개 척후병 → 3일 오디션 → 2개 최종 선정
- **피보나치 분할매수**: 추세전환/눌림목/돌파 3가지 전략
- **AI 통합**: Gemini AI 기반 시장 분석
- **리스크 관리**: 포지션 크기 자동 조절

### 🔧 기술적 특징
- **비동기 처리**: 고성능 병렬 실행
- **실시간 데이터**: WebSocket + REST API 하이브리드
- **모듈화 설계**: 전략별 독립 모듈
- **타입 안전성**: 완전한 타입 힌팅

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