# 🚀 Ultra AI 주식 분석 시스템 - 협업 가이드

## 📋 프로젝트 개요

이 프로젝트는 **한국투자증권 HTS 수준**의 전문적인 주식 분석 플랫폼입니다. AI 기반 분석, 실시간 차트, 재무제표 분석 등을 제공하는 종합적인 투자 도구입니다.

## 🏗️ 프로젝트 구조

```
test_stock/
├── 📁 ai_integration/          # AI 분석 엔진
│   ├── ai_preprocessor.py      # 데이터 전처리
│   └── ultra_ai_analyzer.py    # 메인 AI 분석기
├── 📁 ui_interfaces/           # 사용자 인터페이스
│   └── comprehensive_hts_gui.py # 메인 HTS GUI
├── 📁 strategies/              # 투자 전략
│   ├── mid_term.py             # 중기 투자 전략
│   └── value/                  # 가치투자 전략
├── 📁 utils/                   # 유틸리티 함수들
├── 📁 data/                    # 데이터 관리
├── 📁 config/                  # 설정 파일들
├── 📁 tests/                   # 테스트 코드
└── 📁 documentation/           # 문서화
```

## 🤝 협업 방법

### 1. 이슈 생성 및 관리
- 새로운 기능 요청: `enhancement` 라벨 사용
- 버그 리포트: `bug` 라벨 사용
- 문서화: `documentation` 라벨 사용

### 2. 브랜치 전략
```bash
# 새로운 기능 개발
git checkout -b feature/새기능명

# 버그 수정
git checkout -b bugfix/버그설명

# 핫픽스
git checkout -b hotfix/긴급수정
```

### 3. 커밋 메시지 규칙 (Conventional Commits)
```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서화 작업
style: 코드 스타일 변경
refactor: 코드 리팩토링
test: 테스트 코드 추가/수정
chore: 빌드 프로세스 또는 보조 도구 변경
```

### 4. Pull Request 가이드라인
1. **명확한 제목**: 변경사항을 한 줄로 요약
2. **상세한 설명**: 변경 이유와 방법 설명
3. **테스트 결과**: 테스트 통과 여부 확인
4. **스크린샷**: UI 변경 시 Before/After 이미지 첨부

## 🛠️ 개발 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/story7219/stock.git
cd stock
```

### 2. 가상환경 설정
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
```bash
# .env 파일 생성 (env_example.txt 참고)
cp env_example.txt .env
# 필요한 API 키 등 설정
```

## 📝 코딩 컨벤션

### Python 스타일 가이드
- **PEP 8** 준수
- **타입 힌트** 사용 권장
- **Docstring** 필수 작성
- **변수명**: snake_case
- **클래스명**: PascalCase
- **상수명**: UPPER_CASE

### 예시 코드
```python
from typing import List, Optional, Dict, Any

class StockAnalyzer:
    """주식 분석기 클래스"""
    
    def __init__(self, api_key: str) -> None:
        """
        주식 분석기를 초기화합니다.
        
        Args:
            api_key: API 인증 키
        """
        self.api_key = api_key
    
    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """
        주식을 분석합니다.
        
        Args:
            symbol: 주식 심볼 (예: "005930")
            
        Returns:
            분석 결과 딕셔너리
        """
        # 구현 코드
        pass
```

## 🧪 테스트 가이드

### 1. 테스트 실행
```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 테스트 파일 실행
python -m pytest tests/test_analyzer.py

# 커버리지 확인
python -m pytest --cov=. tests/
```

### 2. 테스트 작성 규칙
- 모든 새로운 함수/클래스에 테스트 작성
- `test_` 접두사 사용
- `assert` 구문으로 검증
- 모킹(Mocking) 적극 활용

## 🚀 배포 프로세스

### 1. 개발 환경 테스트
```bash
python ui_interfaces/comprehensive_hts_gui.py
```

### 2. 프로덕션 빌드
```bash
# 실행 파일 생성 (선택사항)
pyinstaller --onefile --windowed ui_interfaces/comprehensive_hts_gui.py
```

## 📊 성능 최적화 가이드

### 1. 코드 프로파일링
```python
import cProfile
import pstats

# 성능 측정
cProfile.run('your_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### 2. 메모리 사용량 모니터링
```python
import tracemalloc

tracemalloc.start()
# 코드 실행
current, peak = tracemalloc.get_traced_memory()
print(f"현재 메모리: {current / 1024 / 1024:.1f} MB")
print(f"최대 메모리: {peak / 1024 / 1024:.1f} MB")
```

## 🔧 문제 해결

### 자주 발생하는 문제들

1. **모듈 import 오류**
   ```bash
   # PYTHONPATH 설정
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **GUI 실행 오류**
   ```bash
   # tkinter 설치 확인
   python -c "import tkinter; print('tkinter 정상')"
   ```

3. **API 인증 오류**
   - `.env` 파일에 올바른 API 키 설정 확인
   - API 사용량 제한 확인

## 📞 연락처 및 지원

- **이슈 리포트**: [GitHub Issues](https://github.com/story7219/stock/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/story7219/stock/discussions)
- **보안 취약점**: 비공개로 maintainer에게 연락

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

## 🎯 기여 방법

1. **Fork** 저장소를 본인 계정으로 포크
2. **Branch** 새로운 기능 브랜치 생성
3. **Commit** 변경사항 커밋
4. **Push** 본인 저장소에 푸시
5. **Pull Request** 메인 저장소에 PR 생성

### 기여자 인정
모든 기여자는 `CONTRIBUTORS.md` 파일에 기록되며, 프로젝트에 대한 공로를 인정받습니다.

**함께 만들어가는 세계 최고 수준의 주식 분석 플랫폼! 🚀** 