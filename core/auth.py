"""
API 인증 및 토큰 관리를 담당하는 모듈
"""
import asyncio
import os
from datetime import datetime, timedelta
import aiohttp
import logging

from core_legacy.config import KIS_BASE_URL

logger = logging.getLogger(__name__)

class KISAuth:
    """한국투자증권 API 인증 및 토큰 관리를 담당"""
    
    _token_data: dict = {}
    _token_lock = asyncio.Lock()

    def __init__(self, app_key: str, app_secret: str):
        if not app_key or not app_secret:
            raise ValueError("APP_KEY와 APP_SECRET은 필수입니다.")
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = KIS_BASE_URL

    async def _issue_token(self) -> dict:
        """새로운 접근 토큰을 발급받습니다."""
        path = "/oauth2/tokenP"
        url = f"{self.base_url}{path}"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=body) as response:
                    response.raise_for_status()
                    token_data = await response.json()
                    
                    # 만료 시간 계산 (API 응답의 expires_in은 초 단위)
                    expires_in = int(token_data.get("expires_in", 86400))
                    token_data["expires_at"] = datetime.now() + timedelta(seconds=expires_in - 300) # 5분 여유
                    
                    logger.info(f"새로운 접근 토큰 발급 성공. 만료 예정: {token_data['expires_at']}")
                    return token_data
            except aiohttp.ClientError as e:
                logger.error(f"API 토큰 발급 실패: {e}")
                raise
            except Exception as e:
                logger.error(f"토큰 발급 중 예상치 못한 오류: {e}")
                raise

    async def get_valid_token(self) -> str:
        """유효한 접근 토큰을 반환합니다. 필요시 자동으로 갱신합니다."""
        async with self._token_lock:
            # 토큰이 없거나, 만료 5분 전이면 새로 발급
            if not self._token_data or self._token_data.get("expires_at", datetime.now()) <= datetime.now():
                logger.info("토큰이 만료되었거나 존재하지 않습니다. 새로 발급합니다.")
                self._token_data = await self._issue_token()

            return self._token_data.get("access_token", "")

    async def get_token_status(self) -> dict:
        """현재 토큰의 상태를 반환합니다."""
        async with self._token_lock:
            if not self._token_data or not self._token_data.get("access_token"):
                return {"status": "토큰 없음", "valid": False}
            
            expires_at = self._token_data.get("expires_at")
            if expires_at <= datetime.now():
                return {"status": "만료됨", "valid": False, "expires_at": expires_at}
                
            time_left = expires_at - datetime.now()
            return {
                "status": "유효함",
                "valid": True,
                "expires_at": expires_at.strftime('%Y-%m-%d %H:%M:%S'),
                "time_left": str(time_left).split('.')[0]
            }

    async def invalidate_token(self):
        """현재 토큰을 강제로 무효화합니다 (테스트용)."""
        async with self._token_lock:
            self._token_data = {}
            logger.info("토큰이 강제로 무효화되었습니다.") 