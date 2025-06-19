import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# 프로젝트 루트 경로 설정 (이 파일의 상위 디렉토리)
# utils/logger.py -> ../
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import LOG_FILE, LOG_LEVEL

def get_logger(name):
    """설정된 로거를 반환합니다."""
    
    logger = logging.getLogger(name)
    
    # 로거에 핸들러가 이미 설정되어 있는지 확인 (중복 추가 방지)
    if logger.handlers:
        return logger
        
    logger.setLevel(LOG_LEVEL)

    # 포매터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 콘솔 핸들러 설정
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 파일 핸들러 설정 (로그 파일 로테이션)
    # 로그 파일 경로에서 디렉토리 부분만 추출
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1024*1024*5, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 전역 로거 (필요 시 간단히 사용)
default_logger = get_logger("default") 