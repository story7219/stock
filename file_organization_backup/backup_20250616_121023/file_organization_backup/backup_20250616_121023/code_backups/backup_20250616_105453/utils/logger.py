# logger.py
# 로깅 및 예외처리 등 공통 유틸리티 함수 정의 

"""
로깅 유틸리티
"""
import logging
from datetime import datetime
import os
import sys
import re

# 로그 디렉토리 생성
if not os.path.exists('logs'):
    os.makedirs('logs')

# 로거 설정
logger = logging.getLogger('trading_system')
logger.setLevel(logging.INFO)

# 기존 핸들러 제거
if logger.hasHandlers():
    logger.handlers.clear()

# 파일 핸들러
file_handler = logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log')
file_handler.setLevel(logging.INFO)

# UTF-8 인코딩으로 스트림 핸들러 생성
stream_handler = logging.StreamHandler(sys.stdout)
try:
    stream_handler.stream.reconfigure(encoding='utf-8')
except Exception:
    # Python < 3.7 또는 reconfigure 미지원 환경에서는 무시
    pass

# 포맷터
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def remove_emoji(text):
    # 이모지 및 특수문자 제거 (간단 버전)
    return re.sub(r'[^\w\s.,:()\[\]가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]', '', text)

def log_event(level: str, message: str):
    # 콘솔에는 이모지 제거 후 출력, 파일에는 원본 저장
    clean_message = remove_emoji(message)
    if level == "INFO":
        logger.info(clean_message)
    elif level == "WARNING":
        logger.warning(clean_message)
    elif level == "ERROR":
        logger.error(clean_message)
    elif level == "CRITICAL":
        logger.critical(clean_message)
    elif level == "SUCCESS":
        logger.info("성공: " + clean_message)
    else:
        logger.info(clean_message) 