#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📱 텔레그램 알림 래퍼 모듈
=====================

텔레그램 봇 알림 시스템을 관리합니다.
"""

import asyncio
import logging
import config
from utils.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

class TelegramNotifierWrapper:
    """📱 텔레그램 알림 시스템 래퍼 클래스"""
    
    def __init__(self):
        try:
            self.bot_token = getattr(config, 'TELEGRAM_BOT_TOKEN', None)
            self.chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)
            
            if self.bot_token and self.chat_id:
                self.notifier = TelegramNotifier(self.bot_token, self.chat_id)
                self.enabled = True
                logger.info("📱 텔레그램 알림 활성화")
            else:
                self.notifier = None
                self.enabled = False
                logger.warning("⚠️ 텔레그램 설정 없음 - 알림 비활성화")
        except Exception as e:
            logger.error(f"❌ 텔레그램 초기화 실패: {e}")
            self.notifier = None
            self.enabled = False
    
    async def send_message(self, message: str, urgent: bool = False):
        """📱 텔레그램 메시지 전송"""
        if not self.enabled:
            logger.info(f"📱 [텔레그램 비활성화] {message}")
            return
        
        try:
            if urgent:
                message = f"🚨 긴급 알림 🚨\n{message}"
            
            success = await self.notifier.send_message(message)
            if success:
                logger.debug("📱 텔레그램 메시지 전송 성공")
            else:
                logger.error("❌ 텔레그램 전송 실패")
                
        except Exception as e:
            logger.error(f"❌ 텔레그램 알림 오류: {e}") 