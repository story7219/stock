# logger.py
# 로깅 및 예외처리 등 공통 유틸리티 함수 정의 

def log_event(event_type, message):
    """
    이벤트 및 예외 상황 로깅 함수
    """
    print(f"[{event_type}] {message}") 