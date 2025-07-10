# DART API 과거 데이터 수집기

## 📋 개요

DART API를 활용하여 한국 상장기업의 과거 데이터를 종합적으로 수집하고 CSV 형태로 저장하는 시스템입니다.

## 🚀 주요 기능

### 📊 수집 데이터 유형
- **기업 개황 정보**: 기업 기본 정보, 업종, 대표자 등
- **공시 정보**: 최근 5년간 공시 내역
- **재무제표**: 최근 3년간 분기/연간 재무제표
- **임원 정보**: 임원 현황 및 이력
- **배당 정보**: 최근 5년간 배당 내역
- **감사 정보**: 감사보고서 및 감사인 정보

### 🔧 시스템 특징
- **비동기 처리**: 고성능 데이터 수집
- **에러 처리**: 안정적인 오류 복구
- **진행률 모니터링**: 실시간 수집 현황 확인
- **결과 통계**: 수집 결과 상세 분석
- **설정 가능**: JSON 설정 파일 지원

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
pip install pandas aiohttp dart-fss OpenDartReader
```

### 2. DART API 키 발급

1. [DART 오픈API](https://opendart.fss.or.kr/) 접속
2. 회원가입 및 로그인
3. '오픈API 신청' 메뉴에서 API 키 발급
4. 발급받은 API 키를 환경변수로 설정

### 3. 환경변수 설정

**Windows PowerShell:**
```powershell
$env:DART_API_KEY="your_api_key_here"
```

**Windows Command Prompt:**
```cmd
set DART_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export DART_API_KEY=your_api_key_here
```

## 🎯 사용 방법

### 기본 실행

```bash
python run_dart_collector.py
```

### 설정 파일 사용

```python
from dart_historical_data_collector import DARTHistoricalCollector, CollectionConfig
import json

# 설정 파일 로드
with open('dart_collector_config.json', 'r', encoding='utf-8') as f:
    config_data = json.load(f)

# 설정 적용
config = CollectionConfig(
    api_key=os.environ.get('DART_API_KEY'),
    output_dir=Path(config_data['output']['base_dir']),
    start_year=config_data['collection']['start_year'],
    end_year=config_data['collection']['end_year'],
    include_disclosures=config_data['collection']['include_disclosures'],
    include_financials=config_data['collection']['include_financials'],
    include_executives=config_data['collection']['include_executives'],
    include_dividends=config_data['collection']['include_dividends'],
    include_auditors=config_data['collection']['include_auditors'],
    include_corp_info=config_data['collection']['include_corp_info']
)

# 수집기 실행
async with DARTHistoricalCollector(config) as collector:
    await collector.collect_all_historical_data()
```

## 📁 출력 구조

```
dart_historical_data/
├── corp_list.csv                    # 전체 기업 목록
├── collection_results.csv           # 수집 결과 통계
├── corp_info/                      # 기업 개황 정보
│   ├── 00126380/                   # 삼성전자
│   │   └── corp_info.csv
│   └── ...
├── disclosures/                     # 공시 정보
│   ├── 00126380/
│   │   └── disclosures.csv
│   └── ...
├── financials/                      # 재무제표
│   ├── 00126380/
│   │   └── financial_statements.csv
│   └── ...
├── executives/                      # 임원 정보
│   ├── 00126380/
│   │   └── executives.csv
│   └── ...
├── dividends/                       # 배당 정보
│   ├── 00126380/
│   │   └── dividends.csv
│   └── ...
└── auditors/                        # 감사 정보
    ├── 00126380/
    │   └── auditors.csv
    └── ...
```

## ⚙️ 설정 옵션

### CollectionConfig 주요 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `api_key` | str | - | DART API 키 |
| `output_dir` | Path | dart_historical_data | 출력 디렉토리 |
| `start_year` | int | 2015 | 수집 시작 연도 |
| `end_year` | int | 현재연도 | 수집 종료 연도 |
| `request_delay` | float | 0.1 | API 호출 간격 (초) |
| `max_retries` | int | 3 | 재시도 횟수 |
| `include_disclosures` | bool | True | 공시 정보 수집 여부 |
| `include_financials` | bool | True | 재무제표 수집 여부 |
| `include_executives` | bool | True | 임원 정보 수집 여부 |
| `include_dividends` | bool | True | 배당 정보 수집 여부 |
| `include_auditors` | bool | True | 감사 정보 수집 여부 |
| `include_corp_info` | bool | True | 기업 개황 수집 여부 |

## 📊 수집 결과

### collection_results.csv 구조

| 컬럼 | 설명 |
|------|------|
| corp_code | 기업 코드 |
| corp_name | 기업명 |
| data_type | 데이터 유형 |
| success | 수집 성공 여부 |
| record_count | 수집된 레코드 수 |
| error_message | 오류 메시지 |
| timestamp | 수집 시간 |

### 통계 정보

수집 완료 후 다음과 같은 통계 정보를 확인할 수 있습니다:

- **총 기업 수**: 수집 대상 기업 수
- **성공 건수**: 성공적으로 수집된 데이터 건수
- **총 레코드 수**: 수집된 총 데이터 레코드 수

## 🔍 로그 확인

### 로그 파일 위치
- `dart_collector_YYYYMMDD_HHMMSS.log`: 실행별 로그 파일
- `dart_collector.log`: 기본 로그 파일

### 주요 로그 메시지

```
🚀 DART 과거 데이터 수집 시작
✅ DART API 초기화 완료
📋 기업 목록 수집 완료: 2,500개
📊 기업 데이터 수집 진행률: 1/2500 - 삼성전자
✅ 공시 정보 수집 완료: 150건
✅ 재무제표 수집 완료: 12건
📊 수집 결과 통계:
  - 총 기업 수: 2,500
  - 성공 건수: 12,500
  - 총 레코드 수: 125,000
✅ DART 과거 데이터 수집 완료
```

## ⚠️ 주의사항

### API 호출 제한
- DART API는 일일 호출 제한이 있습니다
- `request_delay` 설정으로 호출 간격 조절
- 대량 수집 시 시간이 오래 걸릴 수 있습니다

### 메모리 사용량
- 대용량 데이터 수집 시 메모리 사용량 증가
- 필요시 `batch_size` 조정

### 디스크 공간
- 수집된 데이터는 상당한 디스크 공간 필요
- 충분한 여유 공간 확보 필요

## 🛠️ 고급 설정

### 특정 기업만 수집

```python
# 특정 기업 코드 리스트
target_corps = ['00126380', '00164779']  # 삼성전자, SK하이닉스

# 기업 목록 필터링
corp_list = [corp for corp in dart.get_corp_list() 
             if corp.corp_code in target_corps]
```

### 특정 기간만 수집

```python
config = CollectionConfig(
    start_year=2020,  # 2020년부터
    end_year=2023,    # 2023년까지
    # ... 기타 설정
)
```

### 특정 데이터만 수집

```python
config = CollectionConfig(
    include_disclosures=True,   # 공시만 수집
    include_financials=False,   # 재무제표 제외
    include_executives=False,   # 임원정보 제외
    include_dividends=False,    # 배당정보 제외
    include_auditors=False,     # 감사정보 제외
    include_corp_info=False,    # 기업개황 제외
    # ... 기타 설정
)
```

## 🔧 문제 해결

### 일반적인 오류

1. **API 키 오류**
   ```
   ❌ DART_API_KEY 환경변수가 설정되지 않았습니다.
   ```
   - 환경변수 설정 확인
   - API 키 유효성 검증

2. **네트워크 오류**
   ```
   ❌ DART 데이터 수집 중 오류 발생: Connection timeout
   ```
   - 네트워크 연결 확인
   - `request_delay` 증가

3. **메모리 부족**
   ```
   ❌ MemoryError: Unable to allocate array
   ```
   - `batch_size` 감소
   - 시스템 메모리 확인

### 성능 최적화

1. **병렬 처리 증가**
   ```python
   config = CollectionConfig(
       batch_size=100,  # 배치 크기 증가
       request_delay=0.05,  # 호출 간격 감소
       # ... 기타 설정
   )
   ```

2. **메모리 최적화**
   ```python
   config = CollectionConfig(
       batch_size=25,  # 배치 크기 감소
       # ... 기타 설정
   )
   ```

## 📈 성능 지표

### 예상 처리 시간
- **기업 1개당**: 약 2-5초
- **전체 상장사 (2,500개)**: 약 2-3시간
- **데이터 용량**: 약 500MB-1GB

### 시스템 요구사항
- **메모리**: 최소 2GB RAM
- **디스크**: 최소 2GB 여유 공간
- **네트워크**: 안정적인 인터넷 연결

## 📞 지원

문제가 발생하거나 개선 사항이 있으시면 이슈를 등록해 주세요.

---

**© 2025 Trading AI System. All rights reserved.** 