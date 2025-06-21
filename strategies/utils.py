"""
유틸리티 함수들
- 로깅 설정
- 공통 헬퍼 함수들
- 데이터 변환 함수들
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """로깅 시스템 설정"""
    try:
        # 로그 디렉토리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 로그 파일명 (날짜별)
        log_filename = log_dir / f"personal_blackrock_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 로깅 레벨 설정
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # 로깅 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # PersonalBlackRock 전용 로거 반환
        logger = logging.getLogger('personal_blackrock')
        logger.info(f"✅ 로깅 시스템 초기화 완료 - 레벨: {log_level}")
        
        return logger
        
    except Exception as e:
        print(f"❌ 로깅 설정 실패: {e}")
        # 기본 로거 반환
        return logging.getLogger('personal_blackrock')

def format_number(value: float, decimal_places: int = 2) -> str:
    """숫자를 한국어 형식으로 포맷팅"""
    try:
        if value >= 100000000:  # 1억 이상
            return f"{value/100000000:.{decimal_places}f}억"
        elif value >= 10000:  # 1만 이상
            return f"{value/10000:.{decimal_places}f}만"
        else:
            return f"{value:,.{decimal_places}f}"
    except:
        return str(value)

def format_currency(value: float) -> str:
    """통화 형식으로 포맷팅"""
    try:
        return f"{value:,.0f}원"
    except:
        return f"{value}원"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """퍼센트 형식으로 포맷팅"""
    try:
        return f"{value:.{decimal_places}f}%"
    except:
        return f"{value}%"

def safe_float(value: Any, default: float = 0.0) -> float:
    """안전한 float 변환"""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """안전한 int 변환"""
    try:
        if value is None or value == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "") -> str:
    """안전한 str 변환"""
    try:
        if value is None:
            return default
        return str(value).strip()
    except:
        return default

def calculate_change_rate(current: float, previous: float) -> float:
    """변화율 계산"""
    try:
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    except:
        return 0.0

def get_market_status() -> str:
    """시장 상태 확인"""
    try:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()  # 0=월요일, 6=일요일
        
        # 주말
        if weekday >= 5:  # 토요일, 일요일
            return "휴장"
        
        # 평일 장중 시간 (9:00 ~ 15:30)
        if (hour == 9 and minute >= 0) or (9 < hour < 15) or (hour == 15 and minute <= 30):
            return "장중"
        elif hour < 9:
            return "장전"
        else:
            return "장후"
            
    except:
        return "알 수 없음"

def validate_stock_code(stock_code: str) -> bool:
    """종목코드 유효성 검사"""
    try:
        if not stock_code:
            return False
        
        # 6자리 숫자 형식 확인
        if len(stock_code) == 6 and stock_code.isdigit():
            return True
            
        return False
    except:
        return False

def create_directory_if_not_exists(directory_path: str) -> bool:
    """디렉토리가 없으면 생성"""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 디렉토리 생성 실패: {directory_path}, 오류: {e}")
        return False

def load_config_from_env(config_keys: List[str]) -> Dict[str, Optional[str]]:
    """환경변수에서 설정 로드"""
    config = {}
    for key in config_keys:
        config[key] = os.getenv(key)
    return config

def truncate_text(text: str, max_length: int = 50) -> str:
    """텍스트 길이 제한"""
    try:
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    except:
        return str(text)

def get_emoji_for_change(change_rate: float) -> str:
    """변화율에 따른 이모지 반환"""
    try:
        if change_rate > 5:
            return "🚀"
        elif change_rate > 0:
            return "📈"
        elif change_rate < -5:
            return "📉"
        elif change_rate < 0:
            return "🔻"
        else:
            return "➡️"
    except:
        return "➡️"

def get_risk_level_emoji(risk_level: str) -> str:
    """리스크 레벨에 따른 이모지 반환"""
    risk_emojis = {
        "매우낮음": "🟢",
        "낮음": "🟡", 
        "보통": "🟠",
        "높음": "🔴",
        "매우높음": "⚫"
    }
    return risk_emojis.get(risk_level, "⚪")

def format_time_elapsed(start_time: datetime) -> str:
    """경과 시간 포맷팅"""
    try:
        elapsed = datetime.now() - start_time
        total_seconds = int(elapsed.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}초"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}분 {seconds}초"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}시간 {minutes}분"
    except:
        return "알 수 없음"

class SimpleCache:
    """간단한 메모리 캐시"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Any:
        """캐시에서 값 조회"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """캐시에 값 저장"""
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """캐시 초기화"""
        self.cache.clear()
    
    def size(self) -> int:
        """캐시 크기 반환"""
        return len(self.cache) 