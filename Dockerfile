# AI Trading System Dockerfile
# Python 3.11 기반 엔터프라이즈급 트레이딩 시스템

FROM python:3.11-slim

# 메타데이터
LABEL maintainer="AI Trading System Team"
LABEL version="1.0.0"
LABEL description="AI 기반 자동매매 시스템"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    # 시스템 도구
    curl \
    wget \
    git \
    vim \
    htop \
    # 개발 도구
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # 수학 라이브러리
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # 이미지 처리
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # 데이터베이스
    libpq-dev \
    # 기타
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 가상환경 설정
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 시스템 파일 복사
COPY . .

# 애플리케이션 디렉토리 생성
RUN mkdir -p /app/logs /app/data /app/models /app/backups /app/cache

# 권한 설정
RUN chmod +x /app/scripts/*.py
RUN chown -R nobody:nogroup /app
USER nobody

# 헬스 체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 포트 노출
EXPOSE 8000

# 기본 명령어
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 