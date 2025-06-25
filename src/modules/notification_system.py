#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📱 실시간 알림 시스템 v2.0
텔레그램 알림 및 로깅 지원
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests
import telegram
from telegram import Bot
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class NotificationConfig:
    """알림 설정 데이터 클래스"""
    telegram_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

@dataclass
class NotificationMessage:
    """알림 메시지 데이터 클래스"""
    title: str
    content: str
    priority: str  # HIGH, MEDIUM, LOW
    category: str  # TRADE, ANALYSIS, SYSTEM, ERROR
    timestamp: datetime
    symbols: List[str] = None
    image_path: Optional[str] = None
    action_required: bool = False

class NotificationSystem:
    """실시간 알림 시스템"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """알림 시스템 초기화"""
        self.config = config or {
            'telegram_enabled': False,
            'default_level': 'INFO'
        }
        self.notification_queue = asyncio.Queue()
        self.is_running = False
        logger.info("알림 시스템 초기화 완료")
        self.telegram_bot = None
        self.message_queue = asyncio.Queue()
        self.sent_messages = {}  # 중복 방지
        self.initialize_services()
    
    def initialize_services(self):
        """알림 서비스 초기화"""
        try:
            if self.config.get('telegram_enabled') and self.config.get('telegram_bot_token'):
                self.telegram_bot = Bot(token=self.config['telegram_bot_token'])
                logger.info("텔레그램 봇 초기화 완료")
                
        except Exception as e:
            logger.error(f"알림 서비스 초기화 오류: {e}")
    
    async def send_notification(self, message: str, level: str = "INFO"):
        """알림 전송 (테스트용 간단 버전)"""
        try:
            logger.info(f"알림 전송: [{level}] {message}")
            
            # Mock 알림 전송
            notification_data = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'status': 'sent'
            }
            
            # 큐에 추가
            await self.notification_queue.put(notification_data)
            
            logger.info("알림 전송 완료")
            return True
            
        except Exception as e:
            logger.error(f"알림 전송 중 오류: {e}")
            return False
    
    async def _send_immediate(self, message: NotificationMessage):
        """즉시 알림 전송"""
        tasks = []
        
        if self.config.get('telegram_enabled'):
            tasks.append(self._send_telegram(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_telegram(self, message: NotificationMessage):
        """텔레그램 알림 전송"""
        try:
            if not self.telegram_bot:
                return
            
            # 메시지 포맷팅
            formatted_message = self._format_telegram_message(message)
            
            # 이미지가 있는 경우
            if message.image_path and os.path.exists(message.image_path):
                with open(message.image_path, 'rb') as photo:
                    await self.telegram_bot.send_photo(
                        chat_id=self.config['telegram_chat_id'],
                        photo=photo,
                        caption=formatted_message,
                        parse_mode='HTML'
                    )
            else:
                await self.telegram_bot.send_message(
                    chat_id=self.config['telegram_chat_id'],
                    text=formatted_message,
                    parse_mode='HTML'
                )
            
            logger.info(f"텔레그램 알림 전송 완료: {message.title}")
            
        except Exception as e:
            logger.error(f"텔레그램 알림 전송 실패: {e}")
    
    def _format_telegram_message(self, message: NotificationMessage) -> str:
        """텔레그램 메시지 포맷팅"""
        priority_emoji = {
            "HIGH": "🚨",
            "MEDIUM": "⚠️",
            "LOW": "ℹ️"
        }
        
        category_emoji = {
            "TRADE": "💰",
            "ANALYSIS": "📊",
            "SYSTEM": "⚙️",
            "ERROR": "❌"
        }
        
        emoji = priority_emoji.get(message.priority, "ℹ️")
        cat_emoji = category_emoji.get(message.category, "📝")
        
        formatted = f"{emoji} <b>{message.title}</b>\n\n"
        formatted += f"{cat_emoji} {message.content}\n\n"
        
        if message.symbols:
            formatted += f"🏷️ 관련 종목: {', '.join(message.symbols)}\n"
        
        formatted += f"⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return formatted
    
    def _generate_message_hash(self, message: NotificationMessage) -> str:
        """메시지 해시 생성"""
        content = f"{message.title}_{message.content}_{message.category}"
        return str(hash(content))
    
    def _is_duplicate_message(self, message_hash: str) -> bool:
        """중복 메시지 체크"""
        if message_hash in self.sent_messages:
            # 1시간 이내 중복 메시지는 무시
            sent_time = self.sent_messages[message_hash]
            if datetime.now() - sent_time < timedelta(hours=1):
                return True
        return False
    
    async def start_queue_processor(self):
        """큐 처리기 시작"""
        while True:
            try:
                # 큐에서 메시지 가져오기 (최대 10초 대기)
                message = await asyncio.wait_for(self.message_queue.get(), timeout=10.0)
                await self._send_immediate(message)
                self.message_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"큐 처리 오류: {e}")
    
    def send_trade_signal(self, symbol: str, action: str, price: float, reason: str):
        """매매 신호 알림"""
        message = NotificationMessage(
            title=f"매매 신호: {symbol}",
            content=f"행동: {action}\n가격: {price:,.0f}원\n사유: {reason}",
            priority="HIGH",
            category="TRADE",
            timestamp=datetime.now(),
            symbols=[symbol],
            action_required=True
        )
        asyncio.create_task(self.send_notification(f"매매 신호: {symbol} {action} @ {price:,.0f}원", "HIGH"))
    
    def send_analysis_result(self, title: str, content: str, symbols: List[str] = None):
        """분석 결과 알림"""
        message = NotificationMessage(
            title=title,
            content=content,
            priority="MEDIUM",
            category="ANALYSIS",
            timestamp=datetime.now(),
            symbols=symbols or []
        )
        asyncio.create_task(self.send_notification(f"{title}: {content}", "MEDIUM"))
    
    def send_system_alert(self, title: str, content: str, priority: str = "LOW"):
        """시스템 알림"""
        message = NotificationMessage(
            title=title,
            content=content,
            priority=priority,
            category="SYSTEM",
            timestamp=datetime.now()
        )
        asyncio.create_task(self.send_notification(f"{title}: {content}", priority))
    
    def send_error_alert(self, error_msg: str, traceback_info: str = ""):
        """오류 알림"""
        message = NotificationMessage(
            title="시스템 오류 발생",
            content=f"오류: {error_msg}\n\n{traceback_info}",
            priority="HIGH",
            category="ERROR",
            timestamp=datetime.now()
        )
        asyncio.create_task(self.send_notification(f"시스템 오류: {error_msg}", "HIGH"))

# 전역 알림 시스템 인스턴스
notification_system = None

def initialize_notification_system(config: NotificationConfig):
    """알림 시스템 초기화"""
    global notification_system
    notification_system = NotificationSystem(config)
    return notification_system

def get_notification_system() -> Optional[NotificationSystem]:
    """알림 시스템 인스턴스 반환"""
    return notification_system 