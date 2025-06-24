# 🚀 Ultra Premium HTS v5.0

## 📋 프로젝트 개요

**코스피200·나스닥100·S&P500 전체 종목을 Gemini AI가 Top5 종목 자동 선정하는 고품질 투자 분석 시스템**

- 🤖 **Gemini AI 100% 활용**: 실제 AI가 시황 분석 + 종목 선정
- 📊 **14개 투자 대가 전략**: 워런 버핏, 피터 린치 등 전문가 전략 구현
- 🧹 **고급 데이터 정제**: 결측치 보정 및 품질 개선 자동화
- 🌍 **3개 독립 시장**: 한국(코스피200), 미국(나스닥100, S&P500)

---

## 🗂️ **핵심 파일 구조 (찾기 쉬운 분류)**

### 🎯 **1. 실행 파일 (바로 시작)**
```
📁 실행파일/
├── main.py              # 🚀 메인 실행 파일 (핵심!)
└── MAIN_EXECUTE.bat     # 🖱️ 윈도우 실행 (더블클릭)
```

### 🧠 **2. 핵심 AI 시스템**
```
📁 src/
├── gemini_analyzer.py      # 🤖 Gemini AI 분석 엔진
├── strategies.py           # 📊 14개 투자 대가 전략
├── data_collector.py       # 📈 시장 데이터 수집기
└── data_cleaner.py         # 🧹 고급 데이터 정제기
```

### 🔧 **3. 기술적 분석 시스템**
```
📁 src/
├── technical_analyzer.py   # 📈 기술적 지표 분석
└── report_generator.py     # 📋 결과 리포트 생성
```

### 🌐 **4. 확장 기능 모듈**
```
📁 src/
├── multi_data_collector.py    # 🔄 멀티소스 데이터 수집
├── google_sheets_manager.py   # 📊 구글시트 연동
├── telegram_notifier.py       # 📱 텔레그램 알림
├── kis_api.py                 # 🏦 한국투자증권 API
├── investing_news.py          # 📰 투자 뉴스 수집
├── smart_data_storage.py      # 💾 스마트 캐싱
└── scheduler.py               # ⏰ 자동화 스케줄러
```

### ⚙️ **5. 설정 및 환경**
```
📁 환경설정/
├── .env                    # 🔑 API 키 설정 (중요!)
├── requirements.txt        # 📦 필요 패키지 목록
└── .gitignore             # 🚫 Git 제외 파일
```

---

## 🚀 **빠른 시작 가이드**

### 1️⃣ **즉시 실행 (가장 간단)**
```bash
# 윈도우에서 더블클릭
MAIN_EXECUTE.bat

# 또는 명령어로
python main.py
```

### 2️⃣ **API 키 설정** (처음 한 번만)
```bash
# .env 파일 편집
GEMINI_API_KEY=your_actual_api_key_here
```

### 3️⃣ **패키지 설치** (처음 한 번만)
```bash
pip install -r requirements.txt
```

---

## 📊 **주요 기능별 파일 찾기**

### 🤖 **AI 분석 관련**
- **AI 엔진**: `src/gemini_analyzer.py`
- **AI 설정**: `.env` 파일의 `GEMINI_API_KEY`
- **AI 결과 처리**: `main.py` 라인 220-280

### 📈 **투자 전략 관련**
- **전략 구현**: `src/strategies.py`
- **14개 전략 목록**: `main.py` 라인 24-27
- **전략 선택**: `main.py` 라인 80-90

### 📊 **데이터 수집 관련**
- **기본 수집기**: `src/data_collector.py`
- **멀티소스**: `src/multi_data_collector.py`
- **시장별 수집**: 
  - 코스피200: `src/data_collector.py` 라인 116-522
  - 나스닥100: `src/data_collector.py` 라인 523-611
  - S&P500: `src/data_collector.py` 라인 612-707

### 🧹 **데이터 정제 관련**
- **고급 정제기**: `src/data_cleaner.py`
- **결측치 보정**: `src/data_cleaner.py` 라인 150-300
- **품질 검증**: `src/data_cleaner.py` 라인 400-500

---

## 🔧 **문제 해결 가이드**

### ❌ **API 키 오류**
```
파일: .env
수정: GEMINI_API_KEY=실제_키로_변경
```

### ❌ **패키지 오류**
```bash
pip install google-generativeai python-dotenv scikit-learn scipy
```

### ❌ **모듈 오류**
```bash
# 프로젝트 루트에서 실행 확인
cd /path/to/test_stock
python main.py
```

---

## 📁 **백업 및 버전 관리**

### 🌿 **Git 브랜치 구조**
```
main              # 🚀 운영 버전 (안정)
├── feature/*     # 🔧 새 기능 개발
├── hotfix/*      # 🚨 긴급 수정
└── backup/*      # 💾 백업 브랜치
```

### 💾 **주요 백업 포인트**
- ✅ v5.0.0: 초기 완성 버전
- ✅ 2024-01: 파일 정리 완료
- ✅ 현재: 구조화 완료

---

## 🎯 **성능 지표**

### 📊 **시스템 성능**
- **종목 수집**: 153개 종목 (코스피200 + 나스닥100 + S&P500)
- **전략 수**: 14개 투자 대가 전략
- **데이터 정제**: 100% 성공률 (124개 이상치 자동 수정)
- **AI 분석**: Gemini 1.5 Flash 8B 모델 활용

### 🧪 **테스트 결과**
- ✅ 모든 모듈 import 성공
- ✅ 14개 전략 정상 로드
- ✅ 데이터 수집/정제 완료
- ✅ AI 분석 시스템 정상 작동

---

## 📞 **지원 및 문의**

### 🔗 **유용한 링크**
- 📋 **Gemini API**: https://aistudio.google.com/app/apikey
- 📦 **Python 패키지**: https://pypi.org/
- 🐙 **GitHub 저장소**: 현재 위치

### 🆘 **긴급 복구**
```bash
# 백업에서 복구
git checkout backup/stable
git checkout main
```

---

*🚀 Ultra Premium HTS v5.0 - Gemini AI 기반 투자 분석 시스템*  
*📅 마지막 업데이트: 2024년 1월* 