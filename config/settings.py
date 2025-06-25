"""
전역 설정 관리 모듈
모든 환경 변수와 설정을 중앙에서 관리
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

@dataclass
class APIConfig:
    """API 설정 클래스"""
    gemini_api_key: str
    alpha_vantage_key: Optional[str] = None
    finnhub_key: Optional[str] = None

@dataclass
class DataConfig:
    """데이터 수집 설정 클래스"""
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    timeout: int = 30
    
@dataclass
class AnalysisConfig:
    """분석 설정 클래스"""
    technical_indicators: List[str] = None
    lookback_period: int = 252  # 1년
    min_volume: int = 100000
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = [
                'RSI', 'MACD', 'BB', 'SMA', 'EMA', 'STOCH', 'ADX'
            ]

@dataclass
class ReportConfig:
    """리포트 설정 클래스"""
    output_formats: List[str] = None
    output_dir: str = "reports"
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['html', 'csv']

class Settings:
    """전역 설정 클래스"""
    
    def __init__(self):
        self.api = APIConfig(
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            finnhub_key=os.getenv('FINNHUB_API_KEY')
        )
        
        self.data = DataConfig(
            cache_enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            cache_ttl=int(os.getenv('CACHE_TTL', '3600'))
        )
        
        self.analysis = AnalysisConfig()
        
        self.report = ReportConfig(
            output_formats=os.getenv('REPORT_FORMAT', 'html,csv').split(','),
            output_dir=os.getenv('OUTPUT_DIR', 'reports')
        )
        
        # 로깅 설정
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
        
    def validate(self) -> bool:
        """설정 유효성 검사"""
        if not self.api.gemini_api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        return True

# 전역 설정 인스턴스
settings = Settings()