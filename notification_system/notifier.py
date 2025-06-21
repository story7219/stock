"""
텔레그램 알림 시스템
- 분석 결과 알림
- 실시간 모니터링 알림
- 급변 상황 알림
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json
import aiohttp

# 로깅 설정
logger = logging.getLogger(__name__)

class TelegramNotifier:
    """텔레그램 알림 서비스"""
    
    def __init__(self):
        self.bot_token = None
        self.chat_id = None
        self.is_enabled = False
        self.session = None
        
    async def initialize(self):
        """텔레그램 봇 초기화"""
        try:
            # 환경변수에서 텔레그램 설정 읽기
            self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if self.bot_token and self.chat_id:
                self.is_enabled = True
                logger.info("✅ 텔레그램 알림 서비스 활성화")
            else:
                logger.warning("⚠️ 텔레그램 설정이 없습니다. 알림 기능을 비활성화합니다.")
                self.is_enabled = False
                
        except Exception as e:
            logger.error(f"❌ 텔레그램 초기화 실패: {e}")
            self.is_enabled = False
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """텔레그램 메시지 발송"""
        try:
            if not self.is_enabled:
                logger.debug("텔레그램 알림이 비활성화되어 있습니다.")
                return False
            
            # 실제 텔레그램 API 호출
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"✅ 텔레그램 메시지 전송 성공")
                        print(f"📱 [텔레그램 전송 완료] {message[:50]}...")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 텔레그램 API 오류 {response.status}: {error_text}")
                        print(f"📱 [텔레그램 전송 실패] {message[:50]}...")
                        return False
            
        except Exception as e:
            logger.error(f"❌ 텔레그램 메시지 발송 실패: {e}")
            print(f"📱 [텔레그램 오류] {str(e)}")
            return False
    
    async def send_analysis_result(self, strategy_name: str, results: list):
        """분석 결과 알림 발송"""
        try:
            if not results:
                return False
                
            message = f"🎯 **{strategy_name} 전략 분석 완료**\n\n"
            message += f"📊 코스피 200 분석 결과 TOP 5:\n\n"
            
            for i, result in enumerate(results[:5], 1):
                stock_name = result.get('name', 'N/A')
                stock_code = result.get('code', 'N/A')
                current_price = result.get('current_price', 0)
                change_rate = result.get('change_rate', 0)
                score = result.get('score', 0)
                
                message += f"{i}. **{stock_name}** ({stock_code})\n"
                message += f"   💰 현재가: {current_price:,}원 ({change_rate:+.2f}%)\n"
                message += f"   ⭐ 점수: {score:.1f}점\n\n"
            
            message += f"🤖 PersonalBlackRock AI 분석 시스템\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ 분석 결과 알림 발송 실패: {e}")
            return False
    
    async def send_monitoring_alert(self, alert_data: Dict[str, Any]):
        """모니터링 알림 발송"""
        try:
            stock_name = alert_data.get('stock_name', 'N/A')
            stock_code = alert_data.get('stock_code', 'N/A')
            alert_type = alert_data.get('alert_type', '알림')
            alert_message = alert_data.get('message', '')
            
            message = f"🚨 **실시간 모니터링 알림**\n\n"
            message += f"📊 **{stock_name}** ({stock_code})\n"
            message += f"🔔 **{alert_type}**\n\n"
            message += f"{alert_message}\n\n"
            message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ 모니터링 알림 발송 실패: {e}")
            return False
    
    async def send_system_notification(self, title: str, message: str):
        """시스템 알림 발송"""
        try:
            notification = f"🤖 **{title}**\n\n"
            notification += f"{message}\n\n"
            notification += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return await self.send_message(notification)
            
        except Exception as e:
            logger.error(f"❌ 시스템 알림 발송 실패: {e}")
            return False
    
    def is_available(self) -> bool:
        """텔레그램 알림 사용 가능 여부"""
        return self.is_enabled
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.session:
                await self.session.close()
            logger.info("✅ 텔레그램 알림 서비스 정리 완료")
        except Exception as e:
            logger.error(f"❌ 텔레그램 서비스 정리 중 오류: {e}")

    def __str__(self):
        return f"TelegramNotifier(enabled={self.is_enabled})"

# main.py에서 사용할 수 있도록 Notifier 클래스 추가
class Notifier:
    """
    메인 시스템에서 사용하는 통합 알림 클래스
    """
    
    def __init__(self):
        self.telegram_notifier = TelegramNotifier()
        self._initialized = False
        logger.info("Notifier 초기화")
    
    async def initialize(self):
        """알림 시스템 초기화"""
        try:
            await self.telegram_notifier.initialize()
            self._initialized = True
            logger.info("✅ 통합 알림 시스템 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 통합 알림 시스템 초기화 실패: {e}")
            self._initialized = False
    
    async def send_analysis_results(self, results: Dict[str, Any]):
        """분석 결과 알림 통합 발송"""
        try:
            if not self._initialized:
                await self.initialize()
            
            message = "🎯 **코스피 200 전체 종목 분석 완료**\n\n"
            
            for strategy_name, stocks in results.items():
                if not stocks:
                    continue
                    
                message += f"📈 **{strategy_name} 전략 TOP 3:**\n"
                
                for i, stock in enumerate(stocks[:3], 1):
                    stock_name = stock.get('name', 'N/A')
                    stock_code = stock.get('code', 'N/A')
                    score = stock.get('score', 0)
                    
                    message += f"  {i}. {stock_name} ({stock_code}) - {score:.1f}점\n"
                
                message += "\n"
            
            message += f"🤖 Enhanced Token Manager 시스템\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return await self.telegram_notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ 분석 결과 통합 알림 발송 실패: {e}")
            return False
    
    async def send_system_alert(self, title: str, message: str):
        """시스템 알림 발송"""
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self.telegram_notifier.send_system_notification(title, message)
            
        except Exception as e:
            logger.error(f"❌ 시스템 알림 발송 실패: {e}")
            return False
    
    async def send_monitoring_alert(self, alert_data: Dict[str, Any]):
        """모니터링 알림 발송"""
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self.telegram_notifier.send_monitoring_alert(alert_data)
            
        except Exception as e:
            logger.error(f"❌ 모니터링 알림 발송 실패: {e}")
            return False
    
    def is_available(self) -> bool:
        """알림 서비스 사용 가능 여부"""
        return self._initialized and self.telegram_notifier.is_available()
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.telegram_notifier:
                await self.telegram_notifier.cleanup()
            logger.info("✅ 통합 알림 시스템 정리 완료")
        except Exception as e:
            logger.error(f"❌ 통합 알림 시스템 정리 중 오류: {e}")
    
    def __str__(self):
        return f"Notifier(initialized={self._initialized})" 