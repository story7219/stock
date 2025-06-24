# 📁 프로젝트 구조 v2.0 - 효율적 파일 관리 시스템

## 🎯 **개요**
체계적이고 효율적인 파일 관리를 위해 완전히 재구성된 프로젝트 구조입니다.

```
test_stock/
├── 🚀 MASTER_LAUNCHER.py          # 마스터 런처 (메인 진입점)
├── 📁 file_manager.py             # 파일 관리 시스템
├── 📦 requirements.txt            # 의존성 패키지
│
├── 📁 core/                       # 핵심 실행 파일들
│   ├── __init__.py
│   ├── app.py                     # 메인 애플리케이션
│   ├── launcher.py                # 투자분석 런처
│   └── run_analysis.py            # 빠른 분석 실행기
│
├── 📁 modules/                    # 분석 모듈들  
│   ├── __init__.py
│   ├── ai_analyzer.py             # AI 분석기
│   ├── data_collector.py          # 데이터 수집기
│   ├── investment_strategies.py   # 투자 전략
│   ├── news_analyzer.py           # 뉴스 분석기
│   └── technical_analysis.py      # 기술적 분석
│
├── 📁 config/                     # 설정 파일들
│   ├── env_example.txt            # 환경변수 예시
│   └── SERVICE_ACCOUNT_JSON       # 서비스 계정 JSON
│
├── 📁 docs/                       # 문서 파일들
│   ├── README.md
│   ├── README_OPTIMIZED.md
│   ├── PROJECT_STRUCTURE.md
│   ├── PROJECT_STRUCTURE_SIMPLIFIED.md
│   ├── PROJECT_STRUCTURE_v2.md    # 새 구조 문서 (현재)
│   ├── EASY_REQUESTS.md
│   ├── AI_CODE_MAP.md
│   ├── QUICK_GUIDE.md
│   └── CHANGELOG.md
│
├── 📁 scripts/                    # 스크립트 파일들
│   ├── check_system_specs.py      # 시스템 점검
│   ├── MAIN_EXECUTE.bat           # 메인 실행 배치
│   ├── QUICK_START.bat            # 빠른 시작 배치
│   └── Makefile                   # 빌드 스크립트
│
├── 📁 logs/                       # 로그 파일들
│   └── (런타임에 생성됨)
│
├── 📁 reports/                    # 리포트 파일들
│   └── (분석 결과 저장됨)
│
├── 📁 tests/                      # 테스트 파일들
│   └── (테스트 코드)
│
├── 📁 src/                        # 기존 소스 파일들
│   ├── ml_engine.py
│   ├── scheduler.py
│   └── system_monitor.py
│
└── 📁 backup_old_files/           # 백업된 구 파일들
    ├── automated_master.py
    ├── main.py
    ├── main_new.py
    ├── simple_run.py
    └── requirements_backup.txt
```

## 🎯 **핵심 기능별 파일 위치**

### 1. **실행 파일들**
- `MASTER_LAUNCHER.py` - 🚀 **메인 진입점** (모든 기능 통합)
- `core/launcher.py` - 투자 분석 전용 런처
- `core/app.py` - 완전 자동화 시스템
- `core/run_analysis.py` - 빠른 분석 실행

### 2. **분석 모듈들**
- `modules/ai_analyzer.py` - Gemini AI 분석
- `modules/data_collector.py` - 데이터 수집
- `modules/investment_strategies.py` - 투자 전략
- `modules/technical_analysis.py` - 기술적 분석
- `modules/news_analyzer.py` - 뉴스 분석

### 3. **관리 시스템**
- `file_manager.py` - 📁 **파일 관리 마스터 시스템**
- `scripts/check_system_specs.py` - 시스템 점검
- `src/system_monitor.py` - 시스템 모니터링

## 🚀 **사용 방법**

### 1. **기본 실행**
```bash
# Windows
scripts\MAIN_EXECUTE.bat

# 또는 직접 실행
python MASTER_LAUNCHER.py
```

### 2. **빠른 시작**
```bash
scripts\QUICK_START.bat
```

### 3. **파일 관리**
```bash
python file_manager.py
```

## 📋 **마스터 런처 메뉴 구조**

```
🚀 마스터 런처 v2.0
├── 1. 🎯 투자 분석 시스템 실행
├── 2. 📁 파일 관리 시스템 실행
├── 3. ⚡ 빠른 투자 분석
├── 4. 🔧 시스템 점검
├── 5. 💾 프로젝트 백업
├── 6. 📊 프로젝트 상태 확인
├── 7. ℹ️ 프로젝트 정보
├── 8. 📂 탐색기에서 열기
└── 0. 🚪 종료
```

## 📁 **파일 관리 시스템 기능**

```
📁 파일 관리 마스터 시스템
├── 1. 📊 전체 파일 스캔
├── 2. 🗂️ 자동 파일 정리 (시뮬레이션)
├── 3. 🗂️ 자동 파일 정리 (실제 실행)
├── 4. 🔍 중복 파일 검색
├── 5. 📋 분석 리포트 생성
├── 6. 💾 프로젝트 백업
├── 7. 🧹 임시 파일 정리
├── 8. 💾 분석 결과 저장
├── 9. 📊 현재 구조 보기
└── 0. 🚪 종료
```

## 🔧 **설정 관리**

### **환경변수**
- `config/env_example.txt` - 환경변수 설정 예시
- `.env` - 실제 환경변수 파일 (사용자가 생성)

### **의존성 관리**
- `requirements.txt` - Python 패키지 의존성 (UTF-8, 71개 패키지)

## 📊 **자동 분류 시스템**

파일 관리 시스템이 자동으로 파일을 분류합니다:

- **core/** - 핵심 실행 파일들
- **modules/** - `.py` 분석 모듈들
- **config/** - `.env`, `.json`, `.yaml` 등 설정 파일들
- **docs/** - `.md`, `.txt`, `.pdf` 등 문서 파일들
- **scripts/** - `.bat`, `.sh`, `Makefile` 등 스크립트들
- **logs/** - `.log` 파일들
- **reports/** - 분석 결과 파일들
- **backup_old_files/** - 백업된 구 파일들

## 🎯 **장점**

1. **🗂️ 체계적 구조** - 기능별로 명확히 분리된 디렉토리
2. **🚀 통합 런처** - 하나의 인터페이스로 모든 기능 접근
3. **📁 자동 관리** - 파일 자동 분류 및 중복 제거
4. **💾 백업 시스템** - 자동 백업 및 복원 기능
5. **📊 모니터링** - 실시간 프로젝트 상태 확인
6. **🔧 유지보수** - 간편한 관리 및 확장성

## 📝 **업데이트 히스토리**

### v2.0 (현재)
- 완전한 파일 구조 재정리
- 마스터 런처 시스템 도입
- 파일 관리 자동화 시스템 구축
- 통합 백업 및 복원 시스템

### v1.0 (이전)
- 기본 투자 분석 시스템
- 개별 모듈들 분산 관리

---
**🎯 이제 모든 기능을 MASTER_LAUNCHER.py 하나로 관리할 수 있습니다!**