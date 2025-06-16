"""
🚀 고급 자동매매 시스템 - 메인 실행 파일 (import 수정)
전략 로직 유지하면서 최적화된 구조
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 절대 import로 변경
from core.trader import AdvancedTrader
from utils.logger import setup_logger, SafeLogger

# 환경 설정 로드
load_dotenv()

class TradingSystem:
    """매매 시스템 관리자"""
    
    def __init__(self):
        self.logger = SafeLogger("TradingSystem")
        self.trader = None
        self.running = False
    
    async def start(self):
        """시스템 시작"""
        self.logger.info("🚀 고급 자동매매 시스템 시작")
        
        try:
            # 트레이더 초기화
            self.trader = AdvancedTrader()
            await self.trader.initialize()
            
            # 시그널 핸들러 설정
            self._setup_signal_handlers()
            
            # 트레이더 실행
            self.running = True
            await self.trader.run()
            
        except KeyboardInterrupt:
            self.logger.info("👋 사용자에 의해 종료됨")
        except Exception as e:
            self.logger.error(f"❌ 시스템 오류: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """시스템 중지"""
        if not self.running:
            return
            
        self.running = False
        self.logger.info("🛑 시스템 종료 중...")
        
        try:
            if self.trader:
                await self.trader.stop()
        except Exception as e:
            self.logger.error(f"❌ 종료 중 오류: {e}")
        finally:
            self.logger.info("🔚 시스템 종료 완료")
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            self.logger.info(f"📡 시그널 수신: {signum}")
            asyncio.create_task(self.stop())
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except:
            pass  # Windows에서 일부 시그널 지원 안함

async def main():
    """메인 실행 함수"""
    system = TradingSystem()
    await system.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 프로그램 종료")
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
    finally:
        print("🔚 프로그램 완전 종료") 