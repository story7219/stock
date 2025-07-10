# AI Trading System ML/DL Dockerfile
# GPU 지원 ML/DL 모델 학습 및 추론용

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 메타데이터
LABEL maintainer="AI Trading System ML Team"
LABEL version="1.0.0"
LABEL description="AI 트레이딩 시스템 ML/DL 모델 학습 및 추론"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    # Python 및 기본 도구
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3-venv \
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
    # CUDA 관련
    cuda-toolkit-11-8 \
    cuda-runtime-11-8 \
    cuda-drivers \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 가상환경 설정
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 지원 라이브러리 설치
RUN pip install --no-cache-dir \
    nvidia-ml-py3 \
    cupy-cuda11x==11.6.0

# ML/DL 라이브러리 설치
RUN pip install --no-cache-dir \
    # TensorFlow GPU
    tensorflow==2.15.0 \
    # PyTorch GPU
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    # 기타 ML 라이브러리
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    lightgbm==4.1.0 \
    catboost==1.2.2 \
    optuna==3.5.0 \
    mlflow==2.8.1 \
    # 데이터 처리
    pandas==2.1.4 \
    numpy==1.24.3 \
    polars==0.20.2 \
    # 시각화
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.17.0 \
    # 기타
    jupyter==1.0.0 \
    ipykernel==6.27.1

# 일반 Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 시스템 파일 복사
COPY . .

# ML 모델 디렉토리 생성
RUN mkdir -p /app/models /app/data /app/logs /app/notebooks /app/experiments

# Jupyter 설정
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# 권한 설정
RUN chmod +x /app/scripts/*.py
RUN chown -R nobody:nogroup /app
USER nobody

# 헬스 체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# 포트 노출
EXPOSE 8888 8000

# 기본 명령어 (Jupyter Notebook)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 