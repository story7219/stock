import logging
import logging.handlers
import os

# 이 설정값들은 config.py에서 직접 주입받거나, 여기서 기본값을 정의할 수 있습니다.
# 여기서는 의존성을 낮추기 위해 직접 기본값을 정의합니다.
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'logs/trading_system.log')
LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

def setup_logging():
    """구조화된 로깅 시스템 설정"""
    # 로그 포맷 설정
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (회전 로그)
    try:
        # 로그 디렉토리 생성
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH, 
            maxBytes=LOG_MAX_SIZE, 
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"✅ 로그 파일 설정 완료: {LOG_FILE_PATH}")
    except Exception as e:
        logging.warning(f"⚠️ 로그 파일 설정 실패: {e}")
    
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    
    return root_logger 