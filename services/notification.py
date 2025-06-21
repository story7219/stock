# -*- coding: utf-8 -*-
"""
📱 텔레그램 알림 서비스

주식 분석 결과를 텔레그램으로 전송하는 모듈입니다.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class TelegramNotifier:
    """
    텔레그램 알림 서비스
    
    Features:
    - 비동기 메시지 전송
    - HTML 포맷 지원
    - 이모지 포함 메시지
    - 오류 복구 메커니즘
    - 메시지 길이 자동 분할
    """
    
    def __init__(self):
        """텔레그램 알림 서비스 초기화"""
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ 텔레그램 설정이 불완전합니다. 알림이 비활성화됩니다.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ 텔레그램 알림 서비스 초기화 완료")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.max_message_length = 4096  # 텔레그램 최대 메시지 길이
        
        # 통계
        self.sent_messages = 0
        self.failed_messages = 0
    
    async def send_analysis_result(self, analysis_result: Dict[str, Any]) -> bool:
        """
        분석 결과 전송
        
        Args:
            analysis_result: AI 분석 결과
            
        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            logger.debug("텔레그램 알림이 비활성화되어 있습니다.")
            return False
        
        try:
            message = self._format_analysis_message(analysis_result)
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"분석 결과 전송 실패: {e}")
            self.failed_messages += 1
            return False
    
    async def send_top_stocks(self, top_stocks: List[Dict[str, Any]], strategy_name: str) -> bool:
        """
        TOP 종목 리스트 전송
        
        Args:
            top_stocks: TOP 종목 리스트
            strategy_name: 전략명
            
        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            return False
        
        try:
            message = self._format_top_stocks_message(top_stocks, strategy_name)
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"TOP 종목 전송 실패: {e}")
            self.failed_messages += 1
            return False
    
    async def send_system_notification(self, title: str, content: str, level: str = "INFO") -> bool:
        """
        시스템 알림 전송
        
        Args:
            title: 알림 제목
            content: 알림 내용
            level: 알림 레벨 (INFO, WARNING, ERROR)
            
        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            return False
        
        try:
            message = self._format_system_message(title, content, level)
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"시스템 알림 전송 실패: {e}")
            self.failed_messages += 1
            return False
    
    def _format_analysis_message(self, result: Dict[str, Any]) -> str:
        """분석 결과 메시지 포맷팅"""
        # 기본 정보 추출
        name = result.get('name', 'N/A')
        stock_code = result.get('stock_code', 'N/A')
        strategy = result.get('strategy', 'N/A')
        score = result.get('점수', 0)
        grade = result.get('추천 등급_한글', result.get('추천 등급', 'N/A'))
        confidence = result.get('신뢰도', 0.0)
        
        # 이모지 선택
        grade_emoji = self._get_grade_emoji(grade)
        confidence_emoji = self._get_confidence_emoji(confidence)
        
        # 메시지 구성
        message = f"""
🎯 <b>{strategy} 전략 분석 결과</b>

📊 <b>{name} ({stock_code})</b>
━━━━━━━━━━━━━━━━━━━━

{grade_emoji} <b>추천 등급: {grade}</b>
⭐ <b>점수: {score}점</b>
{confidence_emoji} <b>신뢰도: {confidence:.1%}</b>

💡 <b>추천 이유:</b>
{result.get('추천 이유', 'N/A')}

🎯 <b>진입 가격:</b> {result.get('진입 가격', 'N/A')}
🚀 <b>목표 가격:</b> {result.get('목표 가격', 'N/A')}

📈 <b>상세 분석:</b>
{result.get('분석', 'N/A')[:200]}...

⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """.strip()
        
        return message
    
    def _format_top_stocks_message(self, top_stocks: List[Dict[str, Any]], strategy_name: str) -> str:
        """TOP 종목 리스트 메시지 포맷팅"""
        message = f"""
🏆 <b>{strategy_name} 전략 TOP 5</b>
━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, stock in enumerate(top_stocks, 1):
            name = stock.get('name', 'N/A')
            score = stock.get('점수', 0)
            grade = stock.get('추천 등급_한글', stock.get('추천 등급', 'N/A'))
            confidence = stock.get('신뢰도', 0.0)
            
            rank_emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1] if i <= 5 else f"{i}️⃣"
            grade_emoji = self._get_grade_emoji(grade)
            
            message += f"""
{rank_emoji} <b>{name}</b>
   {grade_emoji} {grade} | ⭐ {score}점 | 📊 {confidence:.1%}
   💡 {stock.get('추천 이유', 'N/A')[:50]}...

"""
        
        message += f"⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return message.strip()
    
    def _format_system_message(self, title: str, content: str, level: str) -> str:
        """시스템 메시지 포맷팅"""
        level_emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        
        emoji = level_emojis.get(level, "📢")
        
        message = f"""
{emoji} <b>{title}</b>
━━━━━━━━━━━━━━━━━━━━

{content}

⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """.strip()
        
        return message
    
    def _get_grade_emoji(self, grade: str) -> str:
        """추천 등급별 이모지"""
        grade_emojis = {
            "매수": "🚀",
            "강력매수": "🔥",
            "적정매수": "📈",
            "보유": "⏸️",
            "비중축소": "📉",
            "매도": "⛔",
            "BUY": "🚀",
            "STRONG BUY": "🔥",
            "MODERATE BUY": "📈",
            "HOLD": "⏸️",
            "REDUCE": "📉",
            "SELL": "⛔"
        }
        return grade_emojis.get(grade, "📊")
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """신뢰도별 이모지"""
        if confidence >= 0.8:
            return "🎯"
        elif confidence >= 0.6:
            return "👍"
        elif confidence >= 0.4:
            return "👌"
        else:
            return "🤔"
    
    async def _send_message(self, message: str, max_retries: int = 3) -> bool:
        """
        메시지 전송 (재시도 포함)
        
        Args:
            message: 전송할 메시지
            max_retries: 최대 재시도 횟수
            
        Returns:
            전송 성공 여부
        """
        # 메시지 길이 확인 및 분할
        messages = self._split_message(message)
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    for msg in messages:
                        url = f"{self.base_url}/sendMessage"
                        payload = {
                            "chat_id": self.chat_id,
                            "text": msg,
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True
                        }
                        
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                self.sent_messages += 1
                                logger.debug("✅ 텔레그램 메시지 전송 성공")
                            else:
                                error_text = await response.text()
                                raise Exception(f"HTTP {response.status}: {error_text}")
                
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"⚠️ 텔레그램 전송 실패 (시도 {attempt + 1}/{max_retries}), {wait_time}초 후 재시도: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ 텔레그램 전송 최종 실패: {e}")
                    self.failed_messages += 1
                    return False
    
    def _split_message(self, message: str) -> List[str]:
        """긴 메시지를 분할"""
        if len(message) <= self.max_message_length:
            return [message]
        
        messages = []
        current_message = ""
        
        lines = message.split('\n')
        for line in lines:
            if len(current_message + line + '\n') <= self.max_message_length:
                current_message += line + '\n'
            else:
                if current_message:
                    messages.append(current_message.strip())
                    current_message = line + '\n'
                else:
                    # 한 줄이 너무 긴 경우
                    while len(line) > self.max_message_length:
                        messages.append(line[:self.max_message_length])
                        line = line[self.max_message_length:]
                    current_message = line + '\n'
        
        if current_message:
            messages.append(current_message.strip())
        
        return messages
    
    async def test_connection(self) -> bool:
        """텔레그램 연결 테스트"""
        if not self.enabled:
            logger.info("텔레그램이 비활성화되어 있습니다.")
            return False
        
        test_message = f"""
🧪 <b>텔레그램 연결 테스트</b>

✅ 봇 토큰: 정상
✅ 채팅 ID: 정상
✅ 메시지 전송: 성공

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        success = await self._send_message(test_message)
        if success:
            logger.info("✅ 텔레그램 연결 테스트 성공")
        else:
            logger.error("❌ 텔레그램 연결 테스트 실패")
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """전송 통계 조회"""
        total_attempts = self.sent_messages + self.failed_messages
        success_rate = (self.sent_messages / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "enabled": self.enabled,
            "sent_messages": self.sent_messages,
            "failed_messages": self.failed_messages,
            "success_rate": f"{success_rate:.1f}%",
            "status": "정상" if success_rate > 90 else "주의" if success_rate > 70 else "경고"
        } 