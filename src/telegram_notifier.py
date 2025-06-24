#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 텔레그램 알림 시스템
투자 분석 결과 및 시스템 상태 알림을 위한 안정화된 텔레그램 봇

Features:
- 강화된 메시지 타입 안전성
- 자동 재시도 및 오류 복구
- 메시지 길이 제한 및 분할 처리
- 비동기 처리 최적화
- 텔레그램 API 속도 제한 준수

Author: AI Assistant
Version: 7.0 (Code Quality Enhanced)
License: MIT
"""

import asyncio
import logging
import os
import traceback
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import aiohttp
import json
from urllib.parse import quote

# 로거 설정
logger = logging.getLogger(__name__)

# 텔레그램 API 상수
TELEGRAM_API_BASE = "https://api.telegram.org/bot"
MAX_MESSAGE_LENGTH = 4096
MAX_CAPTION_LENGTH = 1024
MAX_RETRY_ATTEMPTS = 3
RATE_LIMIT_DELAY = 0.5  # 초
REQUEST_TIMEOUT = 30  # 초


@dataclass
class TelegramConfig:
    """텔레그램 설정"""

    bot_token: str
    chat_id: str
    max_message_length: int = MAX_MESSAGE_LENGTH
    max_retry_attempts: int = MAX_RETRY_ATTEMPTS
    rate_limit_delay: float = RATE_LIMIT_DELAY
    request_timeout: int = REQUEST_TIMEOUT
    enable_markdown: bool = True
    enable_html: bool = False


@dataclass
class MessageMetrics:
    """메시지 전송 메트릭"""

    total_sent: int = 0
    total_failed: int = 0
    retry_count: int = 0
    rate_limit_hits: int = 0
    last_sent_time: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)


class TelegramApiError(Exception):
    """텔레그램 API 오류"""

    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code


class RateLimitError(TelegramApiError):
    """속도 제한 오류"""

    pass


class MessageTooLongError(TelegramApiError):
    """메시지 길이 초과 오류"""

    pass


class TelegramNotifier:
    """
    향상된 텔레그램 알림 시스템

    안전성과 성능이 강화된 텔레그램 봇으로 투자 분석 결과 및 시스템 상태를 알림
    """

    def __init__(self, config: Optional[TelegramConfig] = None):
        """
        텔레그램 알림 시스템 초기화

        Args:
            config: 텔레그램 설정. None이면 환경변수에서 로드
        """
        self.config = config or self._load_config_from_env()
        self.metrics = MessageMetrics()
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(1)
        self._last_request_time = 0.0

        # 설정 검증
        self._validate_config()

        logger.info("📱 텔레그램 알림 시스템 초기화 완료")
        logger.debug(
            f"설정: 봇토큰={self.config.bot_token[:10]}..., 채팅ID={self.config.chat_id}"
        )

    def _load_config_from_env(self) -> TelegramConfig:
        """환경변수에서 설정 로드"""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN 환경변수가 설정되지 않았습니다")
        if not chat_id:
            raise ValueError("TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다")

        return TelegramConfig(bot_token=bot_token, chat_id=chat_id)

    def _validate_config(self) -> None:
        """설정 유효성 검증"""
        if not self.config.bot_token or len(self.config.bot_token) < 10:
            raise ValueError("올바르지 않은 봇 토큰입니다")

        if not self.config.chat_id:
            raise ValueError("채팅 ID가 설정되지 않았습니다")

        try:
            # 채팅 ID가 숫자인지 확인 (개인 채팅의 경우)
            if not self.config.chat_id.startswith("@"):
                int(self.config.chat_id)
        except ValueError:
            raise ValueError("올바르지 않은 채팅 ID입니다")

    async def health_check(self) -> bool:
        """
        텔레그램 봇 상태 확인

        Returns:
            bool: 텔레그램 봇이 정상 작동하는지 여부
        """
        try:
            async with self._get_session() as session:
                url = f"{TELEGRAM_API_BASE}{self.config.bot_token}/getMe"

                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            bot_info = data.get("result", {})
                            logger.debug(
                                f"✅ 텔레그램 봇 정상: {bot_info.get('username', 'Unknown')}"
                            )
                            return True

                    logger.warning(
                        f"⚠️ 텔레그램 봇 상태 확인 실패: HTTP {response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(f"❌ 텔레그램 봇 헬스 체크 실패: {e}")
            return False
        
    @asynccontextmanager
    async def _get_session(self):
        """HTTP 세션 컨텍스트 매니저"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "StockAnalysisBot/7.0",
                },
            )

        try:
            yield self._session
        except Exception as e:
            logger.error(f"❌ HTTP 세션 오류: {e}")
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None
            raise

    async def _rate_limit_wait(self) -> None:
        """속도 제한 대기"""
        async with self._rate_limiter:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self.config.rate_limit_delay:
                wait_time = self.config.rate_limit_delay - time_since_last
                await asyncio.sleep(wait_time)

            self._last_request_time = time.time()

    async def _send_api_request(
        self, method: str, data: Dict[str, Any], retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        텔레그램 API 요청 전송 (안전성 강화)

        Args:
            method: API 메서드명
            data: 요청 데이터
            retry_count: 재시도 횟수

        Returns:
            API 응답 데이터 또는 None
        """
        try:
            # 속도 제한 준수
            await self._rate_limit_wait()

            url = f"{TELEGRAM_API_BASE}{self.config.bot_token}/{method}"

            async with self._get_session() as session:
                async with session.post(url, json=data) as response:
                    response_data = await response.json()

                    if response.status == 200 and response_data.get("ok"):
                        self.metrics.total_sent += 1
                        self.metrics.last_sent_time = datetime.now()
                        return response_data

                    # 오류 처리
                    error_description = response_data.get(
                        "description", "Unknown error"
                    )
                    error_code = response_data.get("error_code")

                    # 속도 제한 감지
                    if error_code == 429:
                        self.metrics.rate_limit_hits += 1
                        retry_after = response_data.get("parameters", {}).get(
                            "retry_after", 60
                        )

                        if retry_count < self.config.max_retry_attempts:
                            logger.warning(
                                f"⚠️ 텔레그램 속도 제한. {retry_after}초 후 재시도..."
                            )
                            await asyncio.sleep(retry_after)
                            return await self._send_api_request(
                                method, data, retry_count + 1
                            )
                        else:
                            raise RateLimitError(f"속도 제한 초과: {error_description}")

                    # 메시지 길이 초과
                    if "message is too long" in error_description.lower():
                        raise MessageTooLongError(
                            f"메시지 길이 초과: {error_description}"
                        )

                    # 기타 오류
                    raise TelegramApiError(f"API 오류: {error_description}", error_code)

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retry_count < self.config.max_retry_attempts:
                self.metrics.retry_count += 1
                wait_time = 2**retry_count  # 지수 백오프
                logger.warning(f"⚠️ 네트워크 오류로 {wait_time}초 후 재시도: {e}")
                await asyncio.sleep(wait_time)
                return await self._send_api_request(method, data, retry_count + 1)
            else:
                logger.error(f"❌ 네트워크 오류 재시도 횟수 초과: {e}")
                raise

        except Exception as e:
            self.metrics.total_failed += 1
            self.metrics.error_messages.append(str(e))
            logger.error(f"❌ API 요청 실패: {e}")
            raise

    def _split_long_message(self, message: str) -> List[str]:
        """
        긴 메시지를 여러 부분으로 분할

        Args:
            message: 분할할 메시지

        Returns:
            분할된 메시지 리스트
        """
        if len(message) <= self.config.max_message_length:
            return [message]

        # 줄 단위로 분할 시도
        lines = message.split("\n")
        chunks = []
        current_chunk = ""

        for line in lines:
            # 단일 줄이 너무 긴 경우
            if len(line) > self.config.max_message_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 강제로 문자 단위 분할
                for i in range(0, len(line), self.config.max_message_length - 10):
                    chunk = line[i : i + self.config.max_message_length - 10]
                    if i > 0:
                        chunk = "..." + chunk
                    if i + self.config.max_message_length - 10 < len(line):
                        chunk = chunk + "..."
                    chunks.append(chunk)
                continue

            # 현재 청크에 추가할 수 있는지 확인
            test_chunk = current_chunk + ("\n" if current_chunk else "") + line
            if len(test_chunk) <= self.config.max_message_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [message[: self.config.max_message_length]]

    def _format_message_safe(self, content: Any) -> str:
        """
        안전한 메시지 포맷팅

        Args:
            content: 포맷할 내용

        Returns:
            포맷된 문자열
        """
        try:
            if content is None:
                return "내용이 없습니다."

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, dict):
                return json.dumps(content, ensure_ascii=False, indent=2)

            if isinstance(content, (list, tuple)):
                return "\n".join(str(item) for item in content)

            return str(content)
            
        except Exception as e:
            logger.warning(f"⚠️ 메시지 포맷팅 실패: {e}")
            return f"메시지 포맷 오류: {str(content)[:100]}..."

    async def send_message(
        self,
        message: Union[str, Dict, List, Any],
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> bool:
        """
        안전한 메시지 전송

        Args:
            message: 전송할 메시지
            parse_mode: 파싱 모드 ('Markdown', 'HTML' 등)
            disable_notification: 무음 알림 여부

        Returns:
            전송 성공 여부
        """
        try:
            # 메시지 포맷팅
            formatted_message = self._format_message_safe(message)

            if not formatted_message.strip():
                logger.warning("⚠️ 빈 메시지는 전송하지 않습니다")
        return False
    
            # 파싱 모드 설정
            if parse_mode is None:
                parse_mode = "Markdown" if self.config.enable_markdown else None

            # 메시지 분할
            message_chunks = self._split_long_message(formatted_message)

            # 각 청크 전송
            success_count = 0
            for i, chunk in enumerate(message_chunks):
                try:
        data = {
                        "chat_id": self.config.chat_id,
                        "text": chunk,
                        "disable_notification": disable_notification,
                    }

                    if parse_mode:
                        data["parse_mode"] = parse_mode

                    result = await self._send_api_request("sendMessage", data)

                    if result:
                        success_count += 1
                        logger.debug(
                            f"✅ 메시지 청크 {i+1}/{len(message_chunks)} 전송 완료"
                        )

                    # 여러 청크인 경우 잠시 대기
                    if len(message_chunks) > 1 and i < len(message_chunks) - 1:
                        await asyncio.sleep(0.5)

                except MessageTooLongError:
                    # 메시지가 여전히 너무 긴 경우 강제 단축
                    short_chunk = (
                        chunk[: self.config.max_message_length - 100]
                        + "...\n\n[메시지가 잘림]"
                    )
                    data["text"] = short_chunk
                    result = await self._send_api_request("sendMessage", data)
                    if result:
                        success_count += 1

                except Exception as e:
                    logger.error(f"❌ 메시지 청크 {i+1} 전송 실패: {e}")
                    continue

            total_success = success_count == len(message_chunks)
            if total_success:
                logger.debug(f"✅ 메시지 전송 완료 ({len(message_chunks)}개 청크)")
                    else:
                logger.warning(
                    f"⚠️ 부분 전송: {success_count}/{len(message_chunks)} 성공"
                )

            return total_success

        except Exception as e:
            logger.error(f"❌ 메시지 전송 실패: {e}")
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            return False
        
    async def send_error(
        self, error_message: str, include_traceback: bool = False
    ) -> bool:
        """
        오류 메시지 전송

        Args:
            error_message: 오류 메시지
            include_traceback: 스택 트레이스 포함 여부

        Returns:
            전송 성공 여부
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            formatted_error = f"""
🚨 **시스템 오류 알림**

⏰ **시간**: {timestamp}
❌ **오류**: {error_message}

🔧 시스템 관리자에게 문의하세요.
"""

            if include_traceback:
                tb = traceback.format_exc()
                if tb and tb != "NoneType: None\n":
                    formatted_error += (
                        f"\n📋 **스택 트레이스**:\n```\n{tb[:1000]}...\n```"
                    )

            return await self.send_message(formatted_error, disable_notification=False)

        except Exception as e:
            logger.error(f"❌ 오류 메시지 전송 실패: {e}")
            return False
        
    async def send_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        상태 업데이트 전송

        Args:
            status: 상태 메시지
            details: 추가 상세 정보

        Returns:
            전송 성공 여부
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            message = f"""
📊 **시스템 상태 업데이트**

⏰ **시간**: {timestamp}
🔄 **상태**: {status}
"""

            if details:
                message += "\n📋 **상세 정보**:\n"
                for key, value in details.items():
                    message += f"   • {key}: {value}\n"

            return await self.send_message(message, disable_notification=True)

        except Exception as e:
            logger.error(f"❌ 상태 업데이트 전송 실패: {e}")
            return False
        
    async def notify_analysis_results_upgraded(
        self, analysis_result: Dict[str, Any]
    ) -> bool:
        """
        향상된 분석 결과 알림

        Args:
            analysis_result: 분석 결과

        Returns:
            전송 성공 여부
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 기본 정보 추출
            korean_stocks = analysis_result.get("korean_market_top5", [])
            us_stocks = analysis_result.get("us_market_top5", [])
            analysis_summary = analysis_result.get("analysis_summary", {})

            # 헤더 메시지
            header_message = f"""
🎯 **투자 대가 전략 분석 완료**

⏰ **분석 시간**: {timestamp}
🔍 **분석 전략**: 10개 투자 대가 통합 전략
📊 **분석 결과**: 한국 {len(korean_stocks)}개, 미국 {len(us_stocks)}개 종목 선정

🚀 **AI 품질 보장**: Gemini 1.5 Flash 기반 정교한 분석
"""

            # 한국 시장 Top5
            if korean_stocks:
                korean_message = "\n🇰🇷 **한국시장 Top5**\n\n"
                for i, stock in enumerate(korean_stocks[:5], 1):
                    korean_message += self._format_stock_info(i, stock)

                await self.send_message(header_message + korean_message)
                await asyncio.sleep(1)  # 메시지 간 간격

            # 미국 시장 Top5
            if us_stocks:
                us_message = "\n🇺🇸 **미국시장 Top5**\n\n"
                for i, stock in enumerate(us_stocks[:5], 1):
                    us_message += self._format_stock_info(i, stock)

                await self.send_message(us_message)
                await asyncio.sleep(1)

            # 분석 요약
            if analysis_summary:
                summary_message = self._format_analysis_summary(analysis_summary)
                await self.send_message(summary_message)

            # 성공 메시지
            success_message = f"""
✅ **분석 완료 알림**

📈 총 {len(korean_stocks) + len(us_stocks)}개 우수 종목 선정
🎯 각 종목별 진입가, 목표가, 예상수익률 제시
🛡️ 리스크 평가 및 투자 전략 포함

📊 상세 분석 결과는 구글 시트에서 확인하세요.
"""

            await self.send_message(success_message)

            logger.info("✅ 향상된 분석 결과 알림 전송 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 분석 결과 알림 전송 실패: {e}")
            await self.send_error(f"분석 결과 알림 전송 실패: {str(e)[:100]}")
            return False
        
    def _format_stock_info(self, rank: int, stock: Dict[str, Any]) -> str:
        """주식 정보 포맷팅"""
        try:
            symbol = stock.get("symbol", "Unknown")
            name = stock.get("name", "Unknown")
            current_price = stock.get("current_price", 0)
            entry_price = stock.get("entry_price", 0)
            target_price = stock.get("target_price", 0)
            expected_return = stock.get("expected_return_pct", 0)
            reasoning = stock.get("selection_reasoning", "분석 중")

            # 가격 포맷팅
            if isinstance(current_price, (int, float)) and current_price > 0:
                if current_price >= 1000:
                    price_str = f"{current_price:,.0f}"
                else:
                    price_str = f"{current_price:.2f}"
        else:
                price_str = "확인 중"

            return f"""
**{rank}. {name} ({symbol})**
💰 현재가: {price_str}
🎯 목표가: {target_price:,.0f} ({expected_return:+.1f}%)
📊 추천사유: {reasoning[:80]}...

"""
        except Exception as e:
            logger.warning(f"⚠️ 주식 정보 포맷팅 실패: {e}")
            return f"{rank}. {stock.get('symbol', 'Unknown')} - 정보 처리 중\n\n"

    def _format_analysis_summary(self, summary: Dict[str, Any]) -> str:
        """분석 요약 포맷팅"""
        try:
            message = "\n📋 **분석 요약**\n\n"

            # 시장 상황
            market_condition = summary.get("market_condition", "분석 중")
            message += f"📈 **시장 상황**: {market_condition}\n\n"

            # 주요 전략
            top_strategies = summary.get("top_performing_strategies", [])
            if top_strategies:
                message += "🏆 **우수 전략**:\n"
                for strategy in top_strategies[:3]:
                    message += f"   • {strategy}\n"
                message += "\n"

            # 리스크 요인
            risk_factors = summary.get("key_risk_factors", [])
            if risk_factors:
                message += "⚠️ **주요 리스크**:\n"
                for risk in risk_factors[:3]:
                    message += f"   • {risk}\n"
                message += "\n"

            # 투자 조언
            advice = summary.get("investment_advice", "")
            if advice:
                message += f"💡 **투자 조언**: {advice[:200]}...\n\n"

            return message
            
        except Exception as e:
            logger.warning(f"⚠️ 분석 요약 포맷팅 실패: {e}")
            return "\n📋 **분석 요약**: 처리 중...\n\n"
    
    async def send_system_start(self) -> bool:
        """시스템 시작 알림"""
        message = f"""
🚀 **투자 분석 시스템 시작**

⏰ **시작 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🔄 **상태**: 데이터 수집 및 분석 시작
🤖 **AI 엔진**: Gemini 1.5 Flash
📊 **분석 대상**: 코스피200, 나스닥100, S&P500

분석이 완료되면 결과를 알려드리겠습니다.
"""
        return await self.send_message(message)
    
    async def send_shutdown_notification(self) -> bool:
        """시스템 종료 알림"""
        message = f"""
🛑 **투자 분석 시스템 종료**

⏰ **종료 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
📊 **전송 통계**: 
   • 성공: {self.metrics.total_sent}건
   • 실패: {self.metrics.total_failed}건
   • 재시도: {self.metrics.retry_count}회

다음 분석을 기다려 주세요.
"""
        return await self.send_message(message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 정보 반환"""
        return {
            "total_sent": self.metrics.total_sent,
            "total_failed": self.metrics.total_failed,
            "retry_count": self.metrics.retry_count,
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "last_sent_time": (
                self.metrics.last_sent_time.isoformat()
                if self.metrics.last_sent_time
                else None
            ),
            "error_count": len(self.metrics.error_messages),
            "recent_errors": (
                self.metrics.error_messages[-5:] if self.metrics.error_messages else []
            ),
        }

    async def close(self) -> None:
        """리소스 정리"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                logger.debug("✅ HTTP 세션 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 텔레그램 리소스 정리 실패: {e}")


# 사용 예시 및 테스트
async def main():
    """테스트 실행"""
        notifier = TelegramNotifier()
        
    # 헬스 체크
    if await notifier.health_check():
        print("✅ 텔레그램 봇 연결 정상")

        # 테스트 메시지
        await notifier.send_message("🧪 텔레그램 알림 시스템 테스트")

        # 메트릭 출력
        metrics = notifier.get_metrics()
        print(f"📊 메트릭: {metrics}")
    else:
        print("❌ 텔레그램 봇 연결 실패")

    await notifier.close()


if __name__ == "__main__":
    asyncio.run(main())
