# 🚀 Ultra Premium Stock Analysis System Makefile v2.0
# 코스피200·나스닥100·S&P500 투자 대가 전략 AI 분석 시스템
# Makefile Tools 최대 활용 버전

# 변수 정의
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
PYLINT := pylint
MYPY := mypy
BANDIT := bandit

# 색상 정의 (Makefile Tools 활용)
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
BOLD := \033[1m
NC := \033[0m # No Color

# 프로젝트 정보
PROJECT_NAME := stock-analysis-system
VERSION := $(shell python -c "import src; print(getattr(src, '__version__', '5.0.0'))")
BUILD_DATE := $(shell date +%Y%m%d_%H%M%S)
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# 디렉토리 설정
SRC_DIR := src
TEST_DIR := tests
LOGS_DIR := logs
DATA_DIR := data
REPORTS_DIR := reports
BACKUP_DIR := backups
VENV_DIR := venv

# 파일 패턴
PYTHON_FILES := $(shell find $(SRC_DIR) $(TEST_DIR) -name "*.py")
LOG_FILES := $(shell find $(LOGS_DIR) -name "*.log" 2>/dev/null || echo "")

.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check security clean run check schedule docs docker build release

# 🎯 메인 타겟들
.DEFAULT_GOAL := help

help: ## 📚 모든 사용 가능한 명령어 표시
	@echo "$(BOLD)$(CYAN)🚀 Stock Analysis System v$(VERSION) - Make Commands$(NC)"
	@echo "$(BOLD)$(CYAN)================================================================$(NC)"
	@echo "$(YELLOW)빌드 정보:$(NC)"
	@echo "  버전: $(VERSION)"
	@echo "  빌드 날짜: $(BUILD_DATE)"
	@echo "  Git 커밋: $(GIT_COMMIT)"
	@echo ""
	@echo "$(YELLOW)사용 가능한 명령어:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)빠른 시작:$(NC)"
	@echo "  $(GREEN)make dev-setup$(NC)     - 개발 환경 전체 설정"
	@echo "  $(GREEN)make run$(NC)           - 시스템 실행"
	@echo "  $(GREEN)make test$(NC)          - 전체 테스트"
	@echo "  $(GREEN)make quality$(NC)       - 코드 품질 검사"

# 📦 설치 관련
install: ## 📦 프로덕션 패키지 설치
	@echo "$(BLUE)📦 프로덕션 패키지 설치 중...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✅ 프로덕션 패키지 설치 완료$(NC)"

install-dev: ## 🛠️ 개발용 패키지 설치
	@echo "$(BLUE)🛠️ 개발용 패키지 설치 중...$(NC)"
	$(PIP) install -e .[dev]
	$(PIP) install pre-commit pytest-xdist pytest-benchmark
	pre-commit install
	@echo "$(GREEN)✅ 개발용 패키지 설치 완료$(NC)"

install-all: install-dev ## 📦 모든 패키지 설치 (개발 + 프로덕션)
	@echo "$(GREEN)✅ 모든 패키지 설치 완료$(NC)"

upgrade-deps: ## ⬆️ 의존성 패키지 업그레이드
	@echo "$(BLUE)⬆️ 의존성 패키지 업그레이드 중...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✅ 의존성 업그레이드 완료$(NC)"

# 🚀 실행 관련
run: ## 🚀 전체 분석 실행
	@echo "$(BLUE)🚀 전체 분석 시스템 시작...$(NC)"
	$(PYTHON) main.py analyze

run-debug: ## 🐛 디버그 모드로 실행
	@echo "$(YELLOW)🐛 디버그 모드 실행...$(NC)"
	$(PYTHON) -u main.py analyze --debug

check: ## 🔍 빠른 상태 점검
	@echo "$(BLUE)🔍 시스템 상태 점검 중...$(NC)"
	$(PYTHON) main.py check

schedule: ## ⏰ 자동 스케줄러 모드
	@echo "$(BLUE)⏰ 자동 스케줄러 시작...$(NC)"
	$(PYTHON) main.py schedule

run-launcher: ## 🎮 런처 스크립트 실행
	@echo "$(BLUE)🎮 런처 실행...$(NC)"
	$(PYTHON) run_system.py

demo: ## 🎬 데모 실행 (샘플 데이터)
	@echo "$(PURPLE)🎬 데모 모드 실행...$(NC)"
	$(PYTHON) -c "from src.multi_data_collector import MultiDataCollector; import asyncio; asyncio.run(MultiDataCollector().demo_run())"

# 🧪 테스트 관련
test: ## 🧪 모든 테스트 실행
	@echo "$(BLUE)🧪 전체 테스트 실행 중...$(NC)"
	$(PYTEST) -v --tb=short

test-unit: ## 🔬 단위 테스트만 실행
	@echo "$(BLUE)🔬 단위 테스트 실행 중...$(NC)"
	$(PYTEST) -m unit -v

test-integration: ## 🔗 통합 테스트만 실행
	@echo "$(BLUE)🔗 통합 테스트 실행 중...$(NC)"
	$(PYTEST) -m integration -v

test-cov: ## 📊 커버리지 포함 테스트
	@echo "$(BLUE)📊 커버리지 테스트 실행 중...$(NC)"
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "$(GREEN)📊 커버리지 리포트: htmlcov/index.html$(NC)"

test-fast: ## ⚡ 빠른 테스트 (병렬 실행)
	@echo "$(BLUE)⚡ 병렬 테스트 실행 중...$(NC)"
	$(PYTEST) -n auto --tb=short

test-verbose: ## 📝 상세 테스트 출력
	@echo "$(BLUE)📝 상세 테스트 실행 중...$(NC)"
	$(PYTEST) -v -s --tb=long

test-benchmark: ## ⏱️ 성능 벤치마크 테스트
	@echo "$(BLUE)⏱️ 성능 벤치마크 테스트...$(NC)"
	$(PYTEST) --benchmark-only --benchmark-sort=mean

test-watch: ## 👀 파일 변경시 자동 테스트
	@echo "$(BLUE)👀 파일 감시 모드 (Ctrl+C로 종료)...$(NC)"
	$(PYTEST) -f

# 🎨 코드 품질 관련
lint: ## 🔍 코드 린팅 (pylint)
	@echo "$(BLUE)🔍 코드 린팅 실행 중...$(NC)"
	$(PYLINT) $(SRC_DIR)/ --score=y --output-format=colorized

lint-json: ## 📋 JSON 형식으로 린팅 결과 출력
	@echo "$(BLUE)📋 JSON 린팅 실행 중...$(NC)"
	$(PYLINT) $(SRC_DIR)/ --output-format=json > reports/pylint-report.json
	@echo "$(GREEN)📋 린팅 결과: reports/pylint-report.json$(NC)"

format: ## 🎨 코드 포맷팅 (black + isort)
	@echo "$(BLUE)🎨 코드 포맷팅 실행 중...$(NC)"
	$(BLACK) $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ 코드 포맷팅 완료$(NC)"

format-check: ## 🔍 포맷팅 검사만 수행
	@echo "$(BLUE)🔍 포맷팅 검사 중...$(NC)"
	$(BLACK) --check $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) --check-only $(SRC_DIR)/ $(TEST_DIR)/

format-diff: ## 📋 포맷팅 차이점 표시
	@echo "$(BLUE)📋 포맷팅 차이점 확인...$(NC)"
	$(BLACK) --diff $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) --diff $(SRC_DIR)/ $(TEST_DIR)/

type-check: ## 🔍 타입 체크 (mypy)
	@echo "$(BLUE)🔍 타입 체크 실행 중...$(NC)"
	$(MYPY) $(SRC_DIR)/ --html-report reports/mypy-report
	@echo "$(GREEN)🔍 타입 체크 리포트: reports/mypy-report/index.html$(NC)"

security: ## 🔒 보안 검사 (bandit)
	@echo "$(BLUE)🔒 보안 검사 실행 중...$(NC)"
	$(BANDIT) -r $(SRC_DIR)/ -f json -o reports/bandit-report.json
	$(BANDIT) -r $(SRC_DIR)/ -f txt
	@echo "$(GREEN)🔒 보안 리포트: reports/bandit-report.json$(NC)"

quality: ## 🏆 전체 코드 품질 검사
	@echo "$(BOLD)$(BLUE)🏆 전체 코드 품질 검사 시작...$(NC)"
	@make format-check
	@make lint
	@make type-check
	@make security
	@make test-fast
	@echo "$(BOLD)$(GREEN)✅ 코드 품질 검사 완료$(NC)"

quality-report: ## 📊 품질 리포트 생성
	@echo "$(BLUE)📊 품질 리포트 생성 중...$(NC)"
	@mkdir -p $(REPORTS_DIR)
	@make lint-json
	@make type-check
	@make security
	@make test-cov
	@echo "$(GREEN)📊 모든 품질 리포트가 $(REPORTS_DIR)/ 에 생성되었습니다$(NC)"

fix: ## 🔧 자동 수정 가능한 이슈 수정
	@echo "$(BLUE)🔧 자동 수정 실행 중...$(NC)"
	$(BLACK) $(SRC_DIR)/ $(TEST_DIR)/
	$(ISORT) $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ 자동 수정 완료$(NC)"

# 🧹 정리 관련
clean: ## 🧹 임시 파일 정리
	@echo "$(BLUE)🧹 임시 파일 정리 중...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage .pytest_cache/ .mypy_cache/
	@echo "$(GREEN)✅ 임시 파일 정리 완료$(NC)"

clean-logs: ## 📋 로그 파일 정리
	@echo "$(BLUE)📋 로그 파일 정리 중...$(NC)"
	find $(LOGS_DIR) -name "*.log" -type f -delete 2>/dev/null || true
	@echo "$(GREEN)✅ 로그 파일 정리 완료$(NC)"

clean-data: ## 🗂️ 데이터 파일 정리
	@echo "$(BLUE)🗂️ 데이터 파일 정리 중...$(NC)"
	rm -rf $(DATA_DIR)/* $(REPORTS_DIR)/* $(BACKUP_DIR)/* 2>/dev/null || true
	@echo "$(GREEN)✅ 데이터 파일 정리 완료$(NC)"

clean-all: clean clean-logs clean-data ## 🧹 모든 파일 정리
	@echo "$(GREEN)✅ 전체 정리 완료$(NC)"

clean-deep: clean-all ## 🔥 딥 클리닝 (가상환경 포함)
	@echo "$(RED)🔥 딥 클리닝 실행 중... (가상환경 삭제)$(NC)"
	rm -rf $(VENV_DIR)/ .tox/ .coverage.* junit.xml
	@echo "$(GREEN)✅ 딥 클리닝 완료$(NC)"

# 📚 문서 관련
docs: ## 📚 문서 생성
	@echo "$(BLUE)📚 문서 생성 중...$(NC)"
	@mkdir -p docs
	$(PYTHON) -c "import pydoc; pydoc.writedoc('src')"
	@echo "$(GREEN)📚 문서가 docs/ 디렉토리에 생성되었습니다$(NC)"

docs-serve: ## 🌐 문서 서버 실행
	@echo "$(BLUE)🌐 문서 서버 시작... (http://localhost:8000)$(NC)"
	cd docs && $(PYTHON) -m http.server 8000

# 🛠️ 개발 환경 관련
setup-dev: ## 🛠️ 개발 환경 설정
	@echo "$(BLUE)🛠️ 개발 환경 설정 중...$(NC)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(YELLOW)가상환경 생성 완료. 다음 명령어로 활성화하세요:$(NC)"
	@echo "$(GREEN)Windows: $(VENV_DIR)\\Scripts\\activate$(NC)"
	@echo "$(GREEN)macOS/Linux: source $(VENV_DIR)/bin/activate$(NC)"

create-dirs: ## 📁 필요한 디렉토리 생성
	@echo "$(BLUE)📁 디렉토리 구조 생성 중...$(NC)"
	@mkdir -p $(LOGS_DIR) $(DATA_DIR) $(REPORTS_DIR) $(BACKUP_DIR) docs
	@echo "$(GREEN)✅ 디렉토리 구조 생성 완료$(NC)"

env-check: ## 🔧 환경 변수 확인
	@echo "$(BLUE)🔧 환경 변수 확인:$(NC)"
	@echo "$(CYAN)GEMINI_API_KEY:$(NC) $${GEMINI_API_KEY:+$(GREEN)✅ 설정됨$(NC)}$${GEMINI_API_KEY:-$(RED)❌ 미설정$(NC)}"
	@echo "$(CYAN)TELEGRAM_BOT_TOKEN:$(NC) $${TELEGRAM_BOT_TOKEN:+$(GREEN)✅ 설정됨$(NC)}$${TELEGRAM_BOT_TOKEN:-$(RED)❌ 미설정$(NC)}"
	@echo "$(CYAN)TELEGRAM_CHAT_ID:$(NC) $${TELEGRAM_CHAT_ID:+$(GREEN)✅ 설정됨$(NC)}$${TELEGRAM_CHAT_ID:-$(RED)❌ 미설정$(NC)}"
	@echo "$(CYAN)GOOGLE_SHEETS_CREDENTIALS:$(NC) $${GOOGLE_SHEETS_CREDENTIALS_PATH:+$(GREEN)✅ 설정됨$(NC)}$${GOOGLE_SHEETS_CREDENTIALS_PATH:-$(RED)❌ 미설정$(NC)}"
	@echo "$(CYAN)GOOGLE_SHEETS_ID:$(NC) $${GOOGLE_SHEETS_SPREADSHEET_ID:+$(GREEN)✅ 설정됨$(NC)}$${GOOGLE_SHEETS_SPREADSHEET_ID:-$(RED)❌ 미설정$(NC)}"

env-template: ## 📝 .env 템플릿 생성
	@echo "$(BLUE)📝 .env 템플릿 생성 중...$(NC)"
	@echo "# 🚀 Stock Analysis System 환경 변수" > .env.template
	@echo "# Gemini AI 설정" >> .env.template
	@echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env.template
	@echo "GEMINI_MODEL=gemini-1.5-flash" >> .env.template
	@echo "" >> .env.template
	@echo "# 텔레그램 봇 설정" >> .env.template
	@echo "TELEGRAM_BOT_TOKEN=your_bot_token_here" >> .env.template
	@echo "TELEGRAM_CHAT_ID=your_chat_id_here" >> .env.template
	@echo "" >> .env.template
	@echo "# 구글 시트 설정" >> .env.template
	@echo "GOOGLE_SHEETS_CREDENTIALS_PATH=credentials.json" >> .env.template
	@echo "GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id_here" >> .env.template
	@echo "$(GREEN)✅ .env.template 파일이 생성되었습니다$(NC)"

requirements: ## 📋 requirements.txt 업데이트
	@echo "$(BLUE)📋 requirements.txt 업데이트 중...$(NC)"
	$(PIP) freeze > requirements-freeze.txt
	@echo "$(GREEN)✅ requirements-freeze.txt 생성 완료$(NC)"

# 🏗️ 빌드 및 배포 관련
build: clean ## 🏗️ 패키지 빌드
	@echo "$(BLUE)🏗️ 패키지 빌드 중...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✅ 빌드 완료: dist/$(NC)"

build-wheel: ## ⚙️ wheel 패키지만 빌드
	@echo "$(BLUE)⚙️ wheel 패키지 빌드 중...$(NC)"
	$(PYTHON) -m build --wheel

build-sdist: ## 📦 source distribution 빌드
	@echo "$(BLUE)📦 source distribution 빌드 중...$(NC)"
	$(PYTHON) -m build --sdist

release-test: build ## 🧪 테스트 PyPI에 업로드
	@echo "$(BLUE)🧪 테스트 PyPI 업로드 중...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*

release: build quality ## 🚀 PyPI에 업로드 (실제 배포)
	@echo "$(RED)🚀 실제 PyPI 배포 중... (주의!)$(NC)"
	$(PYTHON) -m twine upload dist/*

# 🐳 Docker 관련
docker-build: ## 🐳 Docker 이미지 빌드
	@echo "$(BLUE)🐳 Docker 이미지 빌드 중...$(NC)"
	docker build -t $(PROJECT_NAME):$(VERSION) -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✅ Docker 이미지 빌드 완료$(NC)"

docker-run: ## 🏃 Docker 컨테이너 실행
	@echo "$(BLUE)🏃 Docker 컨테이너 실행 중...$(NC)"
	docker run -it --rm --name $(PROJECT_NAME) $(PROJECT_NAME):latest

docker-run-bg: ## 🔄 Docker 컨테이너 백그라운드 실행
	@echo "$(BLUE)🔄 Docker 컨테이너 백그라운드 실행...$(NC)"
	docker run -d --name $(PROJECT_NAME)-daemon $(PROJECT_NAME):latest

docker-stop: ## ⏹️ Docker 컨테이너 중지
	@echo "$(BLUE)⏹️ Docker 컨테이너 중지...$(NC)"
	docker stop $(PROJECT_NAME)-daemon 2>/dev/null || true
	docker rm $(PROJECT_NAME)-daemon 2>/dev/null || true

docker-logs: ## 📋 Docker 컨테이너 로그 확인
	@echo "$(BLUE)📋 Docker 컨테이너 로그:$(NC)"
	docker logs $(PROJECT_NAME)-daemon

docker-shell: ## 🐚 Docker 컨테이너 쉘 접속
	@echo "$(BLUE)🐚 Docker 컨테이너 쉘 접속...$(NC)"
	docker exec -it $(PROJECT_NAME)-daemon /bin/bash

docker-compose-up: ## 🚀 Docker Compose로 실행
	@echo "$(BLUE)🚀 Docker Compose 실행...$(NC)"
	docker-compose up -d

docker-compose-down: ## ⏹️ Docker Compose 중지
	@echo "$(BLUE)⏹️ Docker Compose 중지...$(NC)"
	docker-compose down

docker-clean: ## 🧹 Docker 이미지/컨테이너 정리
	@echo "$(BLUE)🧹 Docker 정리 중...$(NC)"
	docker system prune -f
	docker image prune -f

# 🖥️ 시스템 관련
system-info: ## 🖥️ 시스템 정보 표시
	@echo "$(BOLD)$(BLUE)🖥️ 시스템 정보:$(NC)"
	@echo "$(CYAN)Python:$(NC) $(shell $(PYTHON) --version)"
	@echo "$(CYAN)Pip:$(NC) $(shell $(PIP) --version)"
	@echo "$(CYAN)OS:$(NC) $(shell uname -s 2>/dev/null || echo Windows)"
	@echo "$(CYAN)아키텍처:$(NC) $(shell uname -m 2>/dev/null || echo unknown)"
	@echo "$(CYAN)작업 디렉토리:$(NC) $(shell pwd)"
	@echo "$(CYAN)Git 브랜치:$(NC) $(shell git branch --show-current 2>/dev/null || echo unknown)"
	@echo "$(CYAN)Git 커밋:$(NC) $(GIT_COMMIT)"

health-check: ## 🏥 시스템 헬스 체크
	@echo "$(BLUE)🏥 시스템 헬스 체크 실행 중...$(NC)"
	@$(PYTHON) -c "import sys; print('✅ Python:', sys.version.split()[0])"
	@$(PYTHON) -c "import src.multi_data_collector; print('✅ 데이터 수집기 모듈 OK')" 2>/dev/null || echo "❌ 데이터 수집기 모듈 오류"
	@$(PYTHON) -c "import src.gemini_analyzer; print('✅ Gemini 분석기 모듈 OK')" 2>/dev/null || echo "❌ Gemini 분석기 모듈 오류"
	@$(PYTHON) -c "import src.telegram_notifier; print('✅ 텔레그램 알림 모듈 OK')" 2>/dev/null || echo "❌ 텔레그램 알림 모듈 오류"
	@$(PYTHON) -c "import src.google_sheets_manager; print('✅ 구글시트 관리 모듈 OK')" 2>/dev/null || echo "❌ 구글시트 관리 모듈 오류"
	@echo "$(GREEN)🏥 헬스 체크 완료$(NC)"

performance-test: ## ⚡ 성능 테스트
	@echo "$(BLUE)⚡ 성능 테스트 실행 중...$(NC)"
	@$(PYTHON) -m timeit -n 1 -r 1 "import src.multi_data_collector"
	@echo "$(GREEN)⚡ 성능 테스트 완료$(NC)"

# 📊 모니터링 관련
logs: ## 📋 로그 파일 보기
	@echo "$(BLUE)📋 최근 로그 (50줄):$(NC)"
	@tail -n 50 $(LOGS_DIR)/main.log 2>/dev/null || echo "$(YELLOW)로그 파일이 없습니다.$(NC)"

logs-error: ## ❌ 에러 로그만 보기
	@echo "$(RED)❌ 에러 로그:$(NC)"
	@grep -i -n error $(LOGS_DIR)/*.log 2>/dev/null || echo "$(GREEN)에러 로그가 없습니다.$(NC)"

logs-tail: ## 👀 실시간 로그 모니터링
	@echo "$(BLUE)👀 실시간 로그 모니터링 (Ctrl+C로 종료):$(NC)"
	@tail -f $(LOGS_DIR)/main.log 2>/dev/null || echo "$(YELLOW)로그 파일이 없습니다.$(NC)"

stats: ## 📊 프로젝트 통계
	@echo "$(BOLD)$(BLUE)📊 프로젝트 통계:$(NC)"
	@echo "$(CYAN)소스 코드 라인 수:$(NC) $(shell find $(SRC_DIR) -name '*.py' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $$1}' || echo 0)"
	@echo "$(CYAN)테스트 코드 라인 수:$(NC) $(shell find $(TEST_DIR) -name '*.py' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $$1}' || echo 0)"
	@echo "$(CYAN)총 Python 파일 수:$(NC) $(shell find . -name '*.py' | wc -l)"
	@echo "$(CYAN)총 함수 수:$(NC) $(shell grep -r "^def " $(SRC_DIR) | wc -l 2>/dev/null || echo 0)"
	@echo "$(CYAN)총 클래스 수:$(NC) $(shell grep -r "^class " $(SRC_DIR) | wc -l 2>/dev/null || echo 0)"
	@echo "$(CYAN)TODO 항목:$(NC) $(shell grep -r "TODO\|FIXME\|XXX" $(SRC_DIR) | wc -l 2>/dev/null || echo 0)"

monitor: ## 📈 시스템 모니터링 (CPU, 메모리)
	@echo "$(BLUE)📈 시스템 리소스 모니터링:$(NC)"
	@$(PYTHON) -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%'); print(f'메모리: {psutil.virtual_memory().percent}%'); print(f'디스크: {psutil.disk_usage(".").percent}%')" 2>/dev/null || echo "$(YELLOW)psutil 패키지가 필요합니다$(NC)"

# 🔄 개발 워크플로우
dev-setup: setup-dev install-dev create-dirs env-template ## 🛠️ 개발 환경 전체 설정
	@echo "$(BOLD)$(GREEN)✅ 개발 환경 설정 완료$(NC)"
	@echo "$(YELLOW)다음 단계:$(NC)"
	@echo "1. 가상환경 활성화"
	@echo "2. .env.template을 .env로 복사하고 설정 값 입력"
	@echo "3. $(GREEN)make run$(NC) 으로 시스템 실행"

dev-test: format lint type-check test-fast ## 🧪 개발용 전체 테스트
	@echo "$(BOLD)$(GREEN)✅ 개발 테스트 완료$(NC)"

pre-commit: format lint type-check test-fast ## 📤 커밋 전 검사
	@echo "$(BOLD)$(GREEN)✅ 커밋 준비 완료$(NC)"

ci: format-check lint type-check security test-cov ## 🔄 CI 파이프라인
	@echo "$(BOLD)$(GREEN)✅ CI 검사 완료$(NC)"

deploy-check: quality health-check env-check ## 🚀 배포 전 검사
	@echo "$(BOLD)$(GREEN)✅ 배포 준비 완료$(NC)"

# 🎯 프로덕션 관련
prod-setup: install create-dirs ## 🏭 프로덕션 환경 설정
	@echo "$(BLUE)🏭 프로덕션 환경 설정 중...$(NC)"
	@echo "$(GREEN)✅ 프로덕션 환경 설정 완료$(NC)"

prod-check: ## 🏭 프로덕션 준비 상태 확인
	@echo "$(BOLD)$(BLUE)🏭 프로덕션 준비 상태 확인:$(NC)"
	@make system-info
	@make env-check
	@make health-check
	@echo "$(BOLD)$(GREEN)✅ 프로덕션 준비 완료$(NC)"

backup: ## 💾 데이터 백업
	@echo "$(BLUE)💾 데이터 백업 중...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@tar -czf $(BACKUP_DIR)/backup_$(BUILD_DATE).tar.gz $(DATA_DIR) $(LOGS_DIR) 2>/dev/null || echo "$(YELLOW)백업할 데이터가 없습니다$(NC)"
	@echo "$(GREEN)✅ 백업 완료: $(BACKUP_DIR)/backup_$(BUILD_DATE).tar.gz$(NC)"

restore: ## 🔄 최신 백업 복원
	@echo "$(BLUE)🔄 최신 백업 복원 중...$(NC)"
	@latest_backup=$$(ls -t $(BACKUP_DIR)/backup_*.tar.gz 2>/dev/null | head -1); \
	if [ -n "$$latest_backup" ]; then \
		tar -xzf "$$latest_backup"; \
		echo "$(GREEN)✅ 복원 완료: $$latest_backup$(NC)"; \
	else \
		echo "$(YELLOW)복원할 백업 파일이 없습니다$(NC)"; \
	fi

# 🔧 유틸리티
update-version: ## 🔢 버전 업데이트
	@echo "$(BLUE)🔢 현재 버전: $(VERSION)$(NC)"
	@read -p "새 버전을 입력하세요: " new_version; \
	sed -i "s/__version__ = .*/__version__ = '$$new_version'/" src/__init__.py; \
	echo "$(GREEN)✅ 버전이 $$new_version 으로 업데이트되었습니다$(NC)"

git-status: ## 📋 Git 상태 확인
	@echo "$(BLUE)📋 Git 상태:$(NC)"
	@git status --porcelain 2>/dev/null || echo "$(YELLOW)Git 저장소가 아닙니다$(NC)"

git-commit: ## 📤 Git 커밋 (자동 메시지)
	@echo "$(BLUE)📤 Git 커밋 준비...$(NC)"
	@make pre-commit
	@git add .
	@git commit -m "🚀 Auto commit - $(BUILD_DATE)" 2>/dev/null || echo "$(YELLOW)커밋할 변경사항이 없습니다$(NC)"
	@echo "$(GREEN)✅ Git 커밋 완료$(NC)"

todo: ## 📝 TODO 항목 표시
	@echo "$(BLUE)📝 TODO 항목:$(NC)"
	@grep -rn "TODO\|FIXME\|XXX" $(SRC_DIR) $(TEST_DIR) 2>/dev/null || echo "$(GREEN)TODO 항목이 없습니다$(NC)"

tree: ## 🌳 프로젝트 구조 표시
	@echo "$(BLUE)🌳 프로젝트 구조:$(NC)"
	@tree -I '__pycache__|*.pyc|.git|venv|node_modules' 2>/dev/null || find . -type d -name '__pycache__' -prune -o -type f -print | head -20

# 🎮 인터랙티브 메뉴
menu: ## 🎮 인터랙티브 메뉴 실행
	@echo "$(BOLD)$(CYAN)🎮 Stock Analysis System - 인터랙티브 메뉴$(NC)"
	@echo "$(CYAN)================================================$(NC)"
	@echo "$(GREEN)1)$(NC) 전체 분석 실행 (run)"
	@echo "$(GREEN)2)$(NC) 상태 점검 (check)"
	@echo "$(GREEN)3)$(NC) 테스트 실행 (test)"
	@echo "$(GREEN)4)$(NC) 코드 품질 검사 (quality)"
	@echo "$(GREEN)5)$(NC) 시스템 정보 (system-info)"
	@echo "$(GREEN)6)$(NC) 로그 확인 (logs)"
	@echo "$(GREEN)7)$(NC) 정리 (clean)"
	@echo "$(GREEN)8)$(NC) 도움말 (help)"
	@echo "$(GREEN)q)$(NC) 종료"
	@echo ""
	@read -p "선택하세요 (1-8, q): " choice; \
	case $$choice in \
		1) make run ;; \
		2) make check ;; \
		3) make test ;; \
		4) make quality ;; \
		5) make system-info ;; \
		6) make logs ;; \
		7) make clean ;; \
		8) make help ;; \
		q|Q) echo "$(GREEN)👋 안녕히 가세요!$(NC)" ;; \
		*) echo "$(RED)❌ 잘못된 선택입니다$(NC)" ;; \
	esac

# 🏃 빠른 명령어 별칭
quick-start: dev-setup run ## ⚡ 빠른 시작 (개발환경 + 실행)
	@echo "$(BOLD)$(GREEN)⚡ 빠른 시작 완료$(NC)"

full-check: quality health-check env-check ## 🔍 전체 시스템 검사
	@echo "$(BOLD)$(GREEN)🔍 전체 시스템 검사 완료$(NC)"

daily-routine: clean format test quality backup ## 📅 일일 루틴
	@echo "$(BOLD)$(GREEN)📅 일일 루틴 완료$(NC)"

# 기본 타겟 설정
.DEFAULT_GOAL := help 