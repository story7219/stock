"""
Config 모듈 - 설정 관리
"""

from .settings import Settings

# 전역 설정 인스턴스
settings = Settings()

__all__ = ['settings', 'Settings'] 