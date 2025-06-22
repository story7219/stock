# 🧠 제미나이 투자천재 시스템 (Gemini Investment Genius)

## 📋 시스템 개요

제미나이 AI를 활용한 완전 자동화 투자 분석 시스템입니다. 퀀트 전략을 기반으로 한국/미국 주식을 분석하고 텔레그램으로 결과를 알려줍니다.

## 🚀 5단계 구성

### 1단계: 고정된 데이터 포맷
```csv
Date,Ticker,Market,Close,PER,ROE,ROIC,Earnings,MarketCap,3M_Return,6M_Return,Volatility,Sector,Revenue_Growth,Debt_Ratio
2024-06-01,AAPL,US,192.30,22.1,35.2,27.3,5.21,2.3e12,0.11,0.23,0.15,Technology,0.08,0.25
```

### 2단계: 퀀트 전략 정의
- **마법공식**: PER↓ + ROIC↑ (조엘 그린블라트)
- **퀄리티+모멘텀**: ROE↑ + 6M수익률↑ + 변동성↓

### 3단계: 제미나이 프롬프트
완성형 프롬프트로 AI에게 명령하여 일관된 분석 결과를 얻습니다.

### 4단계: 연속 분석 학습
이전 분석과 동일한 방식으로 반복 분석하여 일관성을 유지합니다.

### 5단계: 텔레그램 알림
분석 결과를 자동으로 텔레그램으로 전송합니다.

## 🛠️ 설치 및 실행

### 1. 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
`data/stock_data.csv` 파일을 준비하거나 샘플 데이터를 사용하세요.

### 3. 시스템 실행
```bash
python gemini_investment_genius.py
```

## 📊 사용법

### 기본 분석
```python
from gemini_investment_genius import GeminiInvestmentGenius

# 시스템 초기화
genius = GeminiInvestmentGenius()

# 통합 전략 분석
result = genius.run_analysis("combined")
print(result)
```

### 텔레그램 설정
1. 봇파더에서 텔레그램 봇 생성
2. 봇 토큰과 채팅 ID 확인
3. 시스템에서 설정

```python
genius.setup_telegram("YOUR_BOT_TOKEN", "YOUR_CHAT_ID")
```

## 🎯 전략 상세

### 마법공식 (Magic Formula)
- PER 낮은 종목 우선 선택
- ROIC 높은 종목 우선 선택
- 순위 합산으로 최종 점수 계산

### 퀄리티+모멘텀
- ROE: 수익성 지표 (40%)
- 6개월 수익률: 모멘텀 지표 (40%)
- 변동성: 리스크 지표 (20%, 역산)

### 통합 전략
- 마법공식 60% + 퀄리티모멘텀 40%
- 가중 평균으로 최종 점수 산출

## 📈 결과 해석

### 출력 항목
- **Ticker**: 종목 코드
- **Market**: 시장 구분 (US/KR)
- **Close**: 현재가
- **PER**: 주가수익비율
- **ROIC**: 투하자본수익률
- **ROE**: 자기자본수익률
- **6M_Return**: 6개월 수익률
- **Volatility**: 변동성
- **Sector**: 섹터
- **Score**: 종합 점수 (0-100점)

### 점수 기준
- **90점 이상**: 매우 우수한 투자 기회
- **80-90점**: 우수한 투자 기회
- **70-80점**: 양호한 투자 기회
- **60-70점**: 보통 수준
- **60점 미만**: 투자 주의

## 🔧 설정 파일

`config/gemini_config.json`에서 다음 항목을 조정할 수 있습니다:

```json
{
  "telegram": {
    "bot_token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID",
    "enabled": true
  },
  "strategies": {
    "magic_formula_weight": 0.6,
    "quality_momentum_weight": 0.4,
    "top_n_stocks": 10
  },
  "analysis": {
    "min_market_cap": 1e10,
    "max_per": 50,
    "min_roe": 5
  }
}
```

## 🤖 제미나이 프롬프트 예시

```
제미나이야, stock_data.csv 파일을 기반으로 다음 조건에 맞춰 투자 유망 종목을 추출해줘.

✅ 전략 조건:
- PER 낮고 ROIC 높은 종목 우선 (마법공식) - 가중치 0.6
- ROE 높고, 6개월 수익률 높고, 변동성 낮은 종목도 추가 점수 부여 - 가중치 0.4
- 총 점수로 상위 10개 종목을 선정

📊 필터링 조건:
- 시가총액 100억 이상
- PER 50 이하
- ROE 5% 이상

📈 출력 항목:
- Ticker, Market, Close, PER, ROIC, ROE, 6M_Return, Volatility, Sector, Score
- 결과는 top_10_stocks.csv로 저장하는 Pandas 코드를 포함해줘
```

## 📱 텔레그램 알림 예시

```
🧠 제미나이 투자천재 분석 결과
📅 2024-06-01 15:30

🏆 TOP 10 추천 종목:
1. 🇺🇸 AAPL
   💰 192 | 📊 87.5점
   PER: 22.1 | ROE: 35.2%

2. 🇰🇷 005930
   💰 72,500 | 📊 85.2점
   PER: 8.3 | ROE: 14.1%

💡 투자는 본인 책임하에 신중히 결정하세요!
```

## 🔄 자동화 설정

### 1. 스케줄러 설정 (Windows)
```batch
# 매일 오후 3시에 실행
schtasks /create /tn "GeminiGenius" /tr "python C:\path\to\gemini_investment_genius.py" /sc daily /st 15:00
```

### 2. 크론탭 설정 (Linux/Mac)
```bash
# 매일 오후 3시에 실행
0 15 * * * cd /path/to/project && python gemini_investment_genius.py
```

## 🚨 주의사항

1. **투자 책임**: 모든 투자 결정은 본인 책임입니다.
2. **데이터 품질**: 정확한 데이터 입력이 중요합니다.
3. **시장 변동성**: 급격한 시장 변화 시 추가 검토가 필요합니다.
4. **분산 투자**: 한 종목에 집중 투자하지 마세요.

## 📞 문의 및 지원

- 시스템 오류 시 로그 파일 확인
- 텔레그램 봇 설정 문제 시 봇파더 재설정
- 데이터 형식 오류 시 CSV 파일 점검

---

**💡 제미나이 투자천재 시스템으로 더 스마트한 투자를 시작하세요!** 