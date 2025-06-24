#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra Stock Analysis System - 설정 관리
.env 파일의 환경변수를 불러와서 시스템 전체에서 사용할 수 있도록 관리
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드
load_dotenv()

class Config:
    """시스템 설정 클래스"""
    
    # ===========================================
    # GitHub API 설정 (자동화용)
    # ===========================================
    GITHUB_API_TOKEN = os.getenv('GITHUB_API_TOKEN', '')
    
    # ===========================================
    # Gemini AI 설정
    # ===========================================
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-8b')
    GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.03'))
    GEMINI_MAX_TOKENS = int(os.getenv('GEMINI_MAX_TOKENS', '8192'))
    
    # ===========================================
    # 한국투자증권 API 설정
    # ===========================================
    # Mock Trading 모드 체크
    IS_MOCK = os.getenv('IS_MOCK', 'true').lower() == 'true'
    
    # Mock/Live 모드에 따른 API 키 선택
    MOCK_KIS_APP_KEY = os.getenv('MOCK_KIS_APP_KEY', '')
    MOCK_KIS_APP_SECRET = os.getenv('MOCK_KIS_APP_SECRET', '')
    MOCK_KIS_ACCOUNT_NUMBER = os.getenv('MOCK_KIS_ACCOUNT_NUMBER', '')
    
    LIVE_KIS_APP_KEY = os.getenv('LIVE_KIS_APP_KEY', '')
    LIVE_KIS_APP_SECRET = os.getenv('LIVE_KIS_APP_SECRET', '')
    LIVE_KIS_ACCOUNT_NUMBER = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '')
    
    # 현재 모드에 따른 API 키 설정
    @property
    def KIS_APP_KEY(self):
        return self.MOCK_KIS_APP_KEY if self.IS_MOCK else self.LIVE_KIS_APP_KEY
    
    @property
    def KIS_APP_SECRET(self):
        return self.MOCK_KIS_APP_SECRET if self.IS_MOCK else self.LIVE_KIS_APP_SECRET
    
    @property
    def KIS_ACCOUNT_NUMBER(self):
        return self.MOCK_KIS_ACCOUNT_NUMBER if self.IS_MOCK else self.LIVE_KIS_ACCOUNT_NUMBER
    
    KIS_ACCOUNT_PRODUCT_CODE = os.getenv('KIS_ACCOUNT_PRODUCT_CODE', '01')
    
    # ===========================================
    # 추가 API 설정 (화면에서 본 것들)
    # ===========================================
    DART_API_KEY = os.getenv('DART_API_KEY', '')
    ZAPIER_NLA_API_KEY = os.getenv('ZAPIER_NLA_API_KEY', '')
    
    # Google 설정
    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', '')
    GOOGLE_SPREADSHEET_ID = os.getenv('GOOGLE_SPREADSHEET_ID', '')
    GOOGLE_WORKSHEET_NAME = os.getenv('GOOGLE_WORKSHEET_NAME', '매매기록')
    
    # ===========================================
    # 데이터 수집 설정
    # ===========================================
    YAHOO_DELAY = float(os.getenv('YAHOO_DELAY', '0.1'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    ANALYSIS_LIMIT = int(os.getenv('ANALYSIS_LIMIT', '200'))
    TECHNICAL_ANALYSIS_DAYS = int(os.getenv('TECHNICAL_ANALYSIS_DAYS', '60'))
    
    # ===========================================
    # 텔레그램 설정
    # ===========================================
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
    
    # ===========================================
    # Google Sheets 설정 (기존 호환성 유지)
    # ===========================================
    GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH', 'credentials.json')
    GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID', '')
    GOOGLE_SHEETS_ENABLED = os.getenv('GOOGLE_SHEETS_ENABLED', 'false').lower() == 'true'
    GOOGLE_SHEETS_BACKUP_INTERVAL = int(os.getenv('GOOGLE_SHEETS_BACKUP_INTERVAL', '24'))
    
    # ===========================================
    # 보안 및 제한 설정
    # ===========================================
    DAILY_API_LIMIT = int(os.getenv('DAILY_API_LIMIT', '1000'))
    API_DELAY = float(os.getenv('API_DELAY', '0.1'))
    DATA_CACHE_TIMEOUT = int(os.getenv('DATA_CACHE_TIMEOUT', '3600'))
    
    # ===========================================
    # 로깅 설정
    # ===========================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/ultra_hts.log')
    
    # ===========================================
    # GUI 설정
    # ===========================================
    WINDOW_WIDTH = int(os.getenv('WINDOW_WIDTH', '1600'))
    WINDOW_HEIGHT = int(os.getenv('WINDOW_HEIGHT', '1000'))
    
    # ===========================================
    # 테스트 모드 설정
    # ===========================================
    TESTING = os.getenv('TESTING', 'false').lower() == 'true'
    
    def validate_config(self):
        """필수 설정값 검증"""
        required_configs = {
            'GITHUB_API_TOKEN': self.GITHUB_API_TOKEN,
            'GEMINI_API_KEY': self.GEMINI_API_KEY,
        }
        
        missing_configs = [key for key, value in required_configs.items() if not value]
        
        if missing_configs:
            print(f"⚠️ 누락된 필수 설정: {', '.join(missing_configs)}")
            return False
        
        print("✅ 모든 필수 설정이 완료되었습니다.")
        return True
    
    def get_config_summary(self):
        """설정 요약 정보 반환"""
        return {
            'github_token_set': bool(self.GITHUB_API_TOKEN),
            'gemini_api_set': bool(self.GEMINI_API_KEY),
            'kis_api_set': bool(self.KIS_APP_KEY and self.KIS_APP_SECRET),
            'dart_api_set': bool(self.DART_API_KEY),
            'telegram_enabled': self.TELEGRAM_ENABLED,
            'google_sheets_enabled': self.GOOGLE_SHEETS_ENABLED,
            'mock_mode': self.IS_MOCK,
            'testing_mode': self.TESTING,
            'current_kis_key': self.KIS_APP_KEY[:10] + '...' if self.KIS_APP_KEY else 'None'
        }


# 전역 설정 인스턴스
config = Config()

# 설정 검증 (모듈 로드 시 자동 실행)
if __name__ == "__main__":
    print("🔧 설정 파일 로드 완료")
    print(f"📊 설정 요약: {config.get_config_summary()}")
    config.validate_config()
else:
    # 다른 모듈에서 import 시에는 조용히 로드
    pass 