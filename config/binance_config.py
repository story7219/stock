#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
바이낸스 API 설정 관리
.env 파일에서 API 키를 읽어오는 설정 모듈
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

class BinanceSettings:
    """바이낸스 API 설정 클래스"""
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """바이낸스 API 키 반환"""
        return os.getenv('BINANCE_API_KEY')
    
    @staticmethod
    def get_api_secret() -> Optional[str]:
        """바이낸스 API 시크릿 반환"""
        return os.getenv('BINANCE_SECRET')
    
    @staticmethod
    def get_testnet() -> bool:
        """테스트넷 사용 여부 반환"""
        return os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    @staticmethod
    def get_rate_limit() -> int:
        """API 요청 제한 반환"""
        return int(os.getenv('BINANCE_RATE_LIMIT', '1200'))
    
    @staticmethod
    def is_configured() -> bool:
        """API 키가 설정되어 있는지 확인"""
        return bool(BinanceSettings.get_api_key() and BinanceSettings.get_api_secret())

def get_binance_config():
    """바이낸스 설정 딕셔너리 반환"""
    return {
        'api_key': BinanceSettings.get_api_key(),
        'api_secret': BinanceSettings.get_api_secret(),
        'testnet': BinanceSettings.get_testnet(),
        'rate_limit': BinanceSettings.get_rate_limit()
    } 