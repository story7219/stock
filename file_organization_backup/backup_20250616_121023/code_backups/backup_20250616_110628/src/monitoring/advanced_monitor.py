"""
고급 모니터링 시스템
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class AlertConfig:
    """알림 설정"""
    profit_threshold: float = 0.05  # 5% 수익률
    loss_threshold: float = -0.03   # -3% 손실률
    volume_spike_threshold: float = 3.0  # 거래량 3배 급증
    price_change_threshold: float = 0.1  # 10% 가격 변동

class AdvancedMonitor:
    """고급 모니터링 시스템"""
    
    def __init__(self, api_client, telegram_notifier, config: AlertConfig = None):
        self.api_client = api_client
        self.telegram_notifier = telegram_notifier
        self.config = config or AlertConfig()
        
        self.monitoring_active = False
        self.alert_history = []
        
    async def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring_active = True
        logging.info("🔍 고급 모니터링 시작")
        
        while self.monitoring_active:
            try:
                await self._run_monitoring_cycle()
                await asyncio.sleep(30)  # 30초마다 체크
            except Exception as e:
                logging.error(f"❌ 모니터링 오류: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    async def _run_monitoring_cycle(self):
        """모니터링 사이클 실행"""
        # 포트폴리오 모니터링
        await self._monitor_portfolio()
        
        # 시장 이상 징후 감지
        await self._detect_market_anomalies()
        
        # 시스템 상태 체크
        await self._check_system_health()
        
        # 성과 추적
        await self._track_performance()
    
    async def _monitor_portfolio(self):
        """포트폴리오 실시간 모니터링"""
        try:
            portfolio = await self._get_current_portfolio()
            
            for stock_code, position in portfolio.items():
                current_price = await self._get_current_price(stock_code)
                if not current_price:
                    continue
                
                # 수익률 계산
                profit_rate = (current_price - position['avg_price']) / position['avg_price']
                
                # 알림 조건 확인
                if profit_rate >= self.config.profit_threshold:
                    await self._send_profit_alert(stock_code, profit_rate, position)
                elif profit_rate <= self.config.loss_threshold:
                    await self._send_loss_alert(stock_code, profit_rate, position)
                
        except Exception as e:
            logging.error(f"❌ 포트폴리오 모니터링 오류: {e}")
    
    async def _send_profit_alert(self, stock_code: str, profit_rate: float, position: Dict):
        """수익 알림"""
        message = f"""
🎉 <b>수익 달성 알림</b>

📊 종목: {stock_code}
📈 수익률: {profit_rate:.2%}
💰 평균단가: {position['avg_price']:,}원
📊 보유수량: {position['quantity']}주
💵 평가손익: {int(position['quantity'] * position['avg_price'] * profit_rate):,}원

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.telegram_notifier.send_message(message) 