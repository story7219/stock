"""
🔧 안전한 로깅 시스템
버퍼 분리 오류 방지
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "AdvancedTrader", level: int = logging.INFO) -> logging.Logger:
    """안전한 로거 설정"""
    
    # 로거 생성
    logger = logging.getLogger(name)
    
    # 이미 설정된 경우 반환
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 콘솔 핸들러 (안전한 출력)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 파일 핸들러 (로그 파일 저장)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 중복 로그 방지
    logger.propagate = False
    
    return logger

class SafeLogger:
    """안전한 로거 래퍼"""
    
    def __init__(self, name: str = "SafeLogger"):
        self.logger = setup_logger(name)
    
    def info(self, message: str):
        """안전한 정보 로그"""
        try:
            self.logger.info(message)
        except:
            print(f"INFO: {message}")
    
    def error(self, message: str):
        """안전한 오류 로그"""
        try:
            self.logger.error(message)
        except:
            print(f"ERROR: {message}")
    
    def warning(self, message: str):
        """안전한 경고 로그"""
        try:
            self.logger.warning(message)
        except:
            print(f"WARNING: {message}") 