            import uvicorn
from __future__ import annotations
from dashboard.trading_dashboard import (
from dataclasses import dataclass
from datetime import datetime, timedelta
from kis_config import KISConfig
from kis_trader import KISTrader
from typing import Dict, List, Optional, Any
import asyncio
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: kis_integration.py
모듈: KIS API 대시보드 통합
목적: KIS API와 대시보드 실시간 연동

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - aiohttp, asyncio
    - kis_config, kis_trader
"""



    TradingDashboard, PositionInfo, TradingSignal,
    MarketSummary, DashboardState, SignalType, RiskLevel
)

logger = logging.getLogger(__name__)

@dataclass
class KISPosition:
    """KIS 포지션 정보"""
    symbol: str
    name: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    market_value: float
    entry_time: datetime
    last_update: datetime = datetime.now()

class KISDashboardIntegration:
    """KIS API 대시보드 통합"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kis_config = KISConfig()
        self.kis_trader = KISTrader(self.kis_config)
        self.dashboard = TradingDashboard()

        # 상태 관리
        self.positions: Dict[str, KISPosition] = {}
        self.market_data: Dict[str, Dict] = {}
        self.update_interval = 1.0  # 1초

        # 실시간 데이터 수집 설정
        self.symbols_to_track = config.get('symbols_to_track', [])
        self.market_indices = ['KOSPI', 'KOSDAQ']

    async def start_integration(self):
        """통합 시작"""
        logger.info("Starting KIS-Dashboard integration...")

        # 대시보드 실행
        dashboard_task = asyncio.create_task(self._run_dashboard())

        # 실시간 데이터 수집
        data_task = asyncio.create_task(self._collect_real_time_data())

        # 포지션 모니터링
        position_task = asyncio.create_task(self._monitor_positions())

        # 시장 데이터 모니터링
        market_task = asyncio.create_task(self._monitor_market_data())

        try:
            await asyncio.gather(dashboard_task, data_task, position_task, market_task)
        except Exception as e:
            logger.error(f"Integration error: {e}")

    async def _run_dashboard(self):
        """대시보드 실행"""
        try:
            # 대시보드 서버 실행
            config = uvicorn.Config(
                self.dashboard.app,
                host="0.0.0.0",
                port=8000,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Dashboard error: {e}")

    async def _collect_real_time_data(self):
        """실시간 데이터 수집"""
        while True:
            try:
                # KIS API에서 실시간 데이터 수집
                await self._fetch_real_time_data()

                # 대시보드 업데이트
                await self._update_dashboard()

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Real-time data collection error: {e}")
                await asyncio.sleep(5.0)

    async def _fetch_real_time_data(self):
        """KIS API에서 실시간 데이터 가져오기"""
        try:
            # 현재가 조회
            for symbol in self.symbols_to_track:
                try:
                    # KIS API 호출 (실제 구현 필요)
                    current_data = await self.kis_trader.get_current_price(symbol)

                    if current_data:
                        self.market_data[symbol] = {
                            'current_price': current_data.get('stck_prpr', 0),
                            'change_rate': current_data.get('prdy_vrss', 0),
                            'volume': current_data.get('acml_vol', 0),
                            'high': current_data.get('stck_hgpr', 0),
                            'low': current_data.get('stck_lwpr', 0),
                            'timestamp': datetime.now()
                        }

                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")

            # 시장 지수 조회
            await self._fetch_market_indices()

        except Exception as e:
            logger.error(f"Failed to fetch real-time data: {e}")

    async def _fetch_market_indices(self):
        """시장 지수 조회"""
        try:
            # KOSPI, KOSDAQ 지수 조회 (실제 구현 필요)
            # 예시 데이터
            self.market_data['KOSPI'] = {
                'current_price': 2500.0,
                'change_rate': 0.5,
                'timestamp': datetime.now()
            }

            self.market_data['KOSDAQ'] = {
                'current_price': 850.0,
                'change_rate': 0.3,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to fetch market indices: {e}")

    async def _monitor_positions(self):
        """포지션 모니터링"""
        while True:
            try:
                # KIS API에서 포지션 정보 조회
                await self._fetch_positions()

                # 포지션 정보를 대시보드 형식으로 변환
                dashboard_positions = await self._convert_to_dashboard_positions()

                # 대시보드 업데이트
                await self.dashboard.update_positions(dashboard_positions)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _fetch_positions(self):
        """KIS API에서 포지션 정보 조회"""
        try:
            # KIS API 호출 (실제 구현 필요)
            positions_data = await self.kis_trader.get_positions()

            for pos_data in positions_data:
                symbol = pos_data.get('pdno')  # 종목코드

                if symbol:
                    # 현재가 조회
                    current_price = self.market_data.get(symbol, {}).get('current_price', 0)

                    # 포지션 정보 계산
                    quantity = int(pos_data.get('hldg_qty', 0))
                    avg_price = float(pos_data.get('pchs_avg_pric', 0))

                    if quantity > 0 and avg_price > 0:
                        pnl = (current_price - avg_price) * quantity
                        pnl_pct = (current_price - avg_price) / avg_price
                        market_value = current_price * quantity

                        self.positions[symbol] = KISPosition(
                            symbol=symbol,
                            name=pos_data.get('prdt_name', symbol),
                            quantity=quantity,
                            avg_price=avg_price,
                            current_price=current_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            market_value=market_value,
                            entry_time=datetime.now() - timedelta(hours=2)  # 예시
                        )

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")

    async def _convert_to_dashboard_positions(self) -> List[PositionInfo]:
        """포지션 정보를 대시보드 형식으로 변환"""
        dashboard_positions = []

        for symbol, position in self.positions.items():
            # AI 신호 강도 계산 (실제로는 AI 모델에서 가져와야 함)
            signal_strength = 0.7  # 예시

            # 리스크 레벨 결정
            risk_level = self._determine_risk_level(position.pnl_pct)

            dashboard_positions.append(PositionInfo(
                symbol=position.symbol,
                name=position.name,
                quantity=position.quantity,
                avg_price=position.avg_price,
                current_price=position.current_price,
                pnl=position.pnl,
                pnl_pct=position.pnl_pct,
                market_value=position.market_value,
                entry_time=position.entry_time,
                signal_strength=signal_strength,
                risk_level=risk_level
            ))

        return dashboard_positions

    def _determine_risk_level(self, pnl_pct: float) -> RiskLevel:
        """손익률 기반 리스크 레벨 결정"""
        if pnl_pct < -0.05:
            return RiskLevel.CRITICAL
        elif pnl_pct < -0.02:
            return RiskLevel.HIGH
        elif pnl_pct < 0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _monitor_market_data(self):
        """시장 데이터 모니터링"""
        while True:
            try:
                # 시장 요약 생성
                market_summary = await self._create_market_summary()

                # 대시보드 상태 생성
                dashboard_state = await self._create_dashboard_state()

                # 대시보드 업데이트
                await self.dashboard.update_market_summary(market_summary)
                await self.dashboard.update_dashboard_state(dashboard_state)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Market data monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _create_market_summary(self) -> MarketSummary:
        """시장 요약 생성"""
        kospi_data = self.market_data.get('KOSPI', {})
        kosdaq_data = self.market_data.get('KOSDAQ', {})

        return MarketSummary(
            kospi=kospi_data.get('current_price', 2500.0),
            kospi_change=kospi_data.get('change_rate', 0.0),
            kosdaq=kosdaq_data.get('current_price', 850.0),
            kosdaq_change=kosdaq_data.get('change_rate', 0.0),
            vix=25.0,  # VIX API 연동 필요
            market_regime="NORMAL",
            trading_volume=1000000000
        )

    async def _create_dashboard_state(self) -> DashboardState:
        """대시보드 상태 생성"""
        total_value = sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.pnl for pos in self.positions.values())
        total_pnl_pct = total_pnl / total_value if total_value > 0 else 0

        # 일일 손익 계산 (실제로는 거래 히스토리 기반)
        daily_pnl = total_pnl * 0.6  # 예시

        return DashboardState(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl / total_value if total_value > 0 else 0,
            win_rate=0.65,  # 실제 계산 필요
            active_positions=len(self.positions),
            risk_level=RiskLevel.LOW,
            trading_allowed=True
        )

    async def _update_dashboard(self):
        """대시보드 업데이트"""
        try:
            # 매매 신호 생성 (실제로는 AI 모델에서 가져와야 함)
            signals = await self._generate_trading_signals()

            # 대시보드 업데이트
            await self.dashboard.update_signals(signals)

        except Exception as e:
            logger.error(f"Dashboard update error: {e}")

    async def _generate_trading_signals(self) -> List[TradingSignal]:
        """매매 신호 생성"""
        signals = []

        # 예시 신호 생성
        for symbol, data in self.market_data.items():
            if symbol in ['KOSPI', 'KOSDAQ']:
                continue

            # 간단한 신호 생성 로직 (실제로는 AI 모델 사용)
            change_rate = data.get('change_rate', 0)

            if change_rate > 0.03:  # 3% 이상 상승
                signals.append(TradingSignal(
                    symbol=symbol,
                    name=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.8,
                    target_price=data.get('current_price', 0) * 1.02,
                    stop_loss=data.get('current_price', 0) * 0.98,
                    reason="급등 신호 감지"
                ))
            elif change_rate < -0.03:  # 3% 이상 하락
                signals.append(TradingSignal(
                    symbol=symbol,
                    name=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.7,
                    target_price=data.get('current_price', 0) * 0.98,
                    stop_loss=data.get('current_price', 0) * 1.02,
                    reason="급락 신호 감지"
                ))

        return signals[:10]  # 최대 10개 신호만 반환

# 사용 예시
async def main():
    """메인 실행 함수"""

    config = {
        'symbols_to_track': ['005930', '000660', '035420'],  # 삼성전자, SK하이닉스, NAVER
        'update_interval': 1.0
    }

    integration = KISDashboardIntegration(config)
    await integration.start_integration()

if __name__ == "__main__":
    asyncio.run(main())

