#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
텔레그램 알림 래퍼 클래스
core_trader.py에서 사용하기 위한 호환성 래퍼
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from personal_blackrock.notifier import TelegramNotifier

logger = logging.getLogger(__name__)

class TelegramNotifierWrapper:
    """
    TelegramNotifier를 래핑하여 core_trader.py와의 호환성 제공
    """
    
    def __init__(self):
        self.notifier = TelegramNotifier()
        self._initialized = False
        logger.info("TelegramNotifierWrapper 초기화")
    
    async def initialize(self):
        """비동기 초기화"""
        try:
            await self.notifier.initialize()
            self._initialized = True
            logger.info("✅ TelegramNotifierWrapper 초기화 완료")
        except Exception as e:
            logger.error(f"❌ TelegramNotifierWrapper 초기화 실패: {e}")
            self._initialized = False
    
    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """동기 메시지 발송 (비동기 호출을 동기적으로 래핑)"""
        try:
            if not self._initialized:
                # 초기화되지 않았다면 동기적으로 초기화 시도
                asyncio.create_task(self.initialize())
                
            # 비동기 메서드를 동기적으로 실행
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 이벤트 루프가 있다면 태스크로 실행
                task = asyncio.create_task(self.notifier.send_message(message, parse_mode))
                return True  # 즉시 True 반환 (실제 결과는 비동기적으로 처리)
            else:
                # 새 이벤트 루프에서 실행
                return asyncio.run(self.notifier.send_message(message, parse_mode))
                
        except Exception as e:
            logger.error(f"❌ 메시지 발송 실패: {e}")
            # 콘솔에라도 출력
            print(f"📱 [텔레그램 알림] {message}")
            return False
    
    async def send_message_async(self, message: str, parse_mode: str = "Markdown") -> bool:
        """비동기 메시지 발송"""
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self.notifier.send_message(message, parse_mode)
            
        except Exception as e:
            logger.error(f"❌ 비동기 메시지 발송 실패: {e}")
            return False
    
    def send_order_notification(self, order_data: Dict[str, Any]) -> bool:
        """주문 알림 발송"""
        try:
            symbol = order_data.get('symbol', 'N/A')
            order_type = order_data.get('order_type', 'N/A')
            quantity = order_data.get('quantity', 0)
            price = order_data.get('price', 0)
            
            message = f"📊 **주문 실행 알림**\n\n"
            message += f"📈 종목: {symbol}\n"
            message += f"🔄 주문 유형: {order_type}\n"
            message += f"📦 수량: {quantity:,}주\n"
            message += f"💰 가격: {price:,}원\n"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ 주문 알림 발송 실패: {e}")
            return False
    
    def send_analysis_notification(self, analysis_data: Dict[str, Any]) -> bool:
        """분석 결과 알림 발송"""
        try:
            strategy = analysis_data.get('strategy', 'N/A')
            results = analysis_data.get('results', [])
            
            if not results:
                return False
                
            message = f"🎯 **{strategy} 분석 완료**\n\n"
            
            for i, result in enumerate(results[:3], 1):
                stock_name = result.get('name', 'N/A')
                stock_code = result.get('code', 'N/A')
                score = result.get('score', 0)
                
                message += f"{i}. {stock_name} ({stock_code}) - {score:.1f}점\n"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ 분석 알림 발송 실패: {e}")
            return False
    
    def is_available(self) -> bool:
        """알림 서비스 사용 가능 여부"""
        return self._initialized and self.notifier.is_available()
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.notifier:
                await self.notifier.cleanup()
            logger.info("✅ TelegramNotifierWrapper 정리 완료")
        except Exception as e:
            logger.error(f"❌ TelegramNotifierWrapper 정리 중 오류: {e}") 