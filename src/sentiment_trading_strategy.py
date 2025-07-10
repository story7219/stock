from __future__ import annotations
from aiohttp import ClientSession
from core.models import Signal, StrategyType, TradeType
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from kis_trader import KISTrader
from push_notifications import PushNotificationService
from typing import Dict, List, Optional, Any
import asyncio
import logging
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: sentiment_trading_strategy.py
모듈: 개인 투자자 심리(감성) 역이용 전략
목적: 공포 매수, 탐욕 매도, 뉴스 반응 선제 대응

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, aiohttp, numpy, pandas
    - kis_trader, push_notifications
"""





logger = logging.getLogger(__name__)

class SentimentSignalType(Enum):
    FEAR_BUY = "fear_buy"
    GREED_SELL = "greed_sell"
    NEWS_ANTICIPATE = "news_anticipate"

@dataclass
class SentimentSignal:
    symbol: str
    name: str
    signal_type: SentimentSignalType
    confidence: float
    reason: str
    target_price: float
    stop_loss: float
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment_score: float = 0.0
    vix_level: float = 0.0
    news_score: float = 0.0
    retail_flow: float = 0.0

class SentimentTradingStrategy:
    """개인 투자자 심리(감성) 역이용 전략"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kis_trader = KISTrader(config.get('kis_config', {}))
        self.push_service = PushNotificationService(config.get('push_config', {}))
        self.session: Optional[ClientSession] = None
        self.active_signals: List[SentimentSignal] = []
        self.vix_threshold = config.get('vix_threshold', 25.0)
        self.greed_threshold = config.get('greed_threshold', 80.0)
        self.news_sentiment_threshold = config.get('news_sentiment_threshold', 0.7)
        self.max_position_size = config.get('max_position_size', 1000000)

    async def start_strategy(self):
        logger.info("Starting Sentiment Trading Strategy...")
        self.session = ClientSession()
        try:
            tasks = [
                asyncio.create_task(self._monitor_fear_buy()),
                asyncio.create_task(self._monitor_greed_sell()),
                asyncio.create_task(self._monitor_news_anticipate()),
                asyncio.create_task(self._execute_signals())
            ]
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Sentiment strategy error: {e}")
        finally:
            if self.session:
                await self.session.close()

    async def _monitor_fear_buy(self):
        """공포 매수: VIX 급등 시 역발상 매수"""
        while True:
            try:
                vix = await self._get_vix_level()
                if vix >= self.vix_threshold:
                    candidates = await self._get_fear_buy_candidates()
                    for stock in candidates:
                        signal = await self._create_fear_buy_signal(stock, vix)
                        if signal:
                            self.active_signals.append(signal)
                            await self.push_service.send_trading_signal_notification(
                                symbol=signal.symbol,
                                signal_type="FEAR_BUY",
                                confidence=signal.confidence,
                                reason=signal.reason
                            )
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Fear buy monitor error: {e}")
                await asyncio.sleep(60)

    async def _monitor_greed_sell(self):
        """탐욕 매도: 과열 구간에서 조기 매도"""
        while True:
            try:
                greed_index = await self._get_greed_index()
                if greed_index >= self.greed_threshold:
                    positions = await self.kis_trader.get_positions()
                    for position in positions:
                        signal = await self._create_greed_sell_signal(position, greed_index)
                        if signal:
                            self.active_signals.append(signal)
                            await self.push_service.send_trading_signal_notification(
                                symbol=signal.symbol,
                                signal_type="GREED_SELL",
                                confidence=signal.confidence,
                                reason=signal.reason
                            )
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Greed sell monitor error: {e}")
                await asyncio.sleep(60)

    async def _monitor_news_anticipate(self):
        """뉴스 반응: 개인들 반응 패턴 선제 대응"""
        while True:
            try:
                news_events = await self._get_news_events()
                for news in news_events:
                    if news['sentiment'] > self.news_sentiment_threshold:
                        signal = await self._create_news_anticipate_signal(news)
                        if signal:
                            self.active_signals.append(signal)
                            await self.push_service.send_trading_signal_notification(
                                symbol=signal.symbol,
                                signal_type="NEWS_ANTICIPATE",
                                confidence=signal.confidence,
                                reason=signal.reason
                            )
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"News anticipate monitor error: {e}")
                await asyncio.sleep(30)

    async def _execute_signals(self):
        """신호 실행"""
        while True:
            try:
                for signal in self.active_signals:
                    if signal.signal_type == SentimentSignalType.FEAR_BUY:
                        await self._execute_fear_buy(signal)
                    elif signal.signal_type == SentimentSignalType.GREED_SELL:
                        await self._execute_greed_sell(signal)
                    elif signal.signal_type == SentimentSignalType.NEWS_ANTICIPATE:
                        await self._execute_news_anticipate(signal)
                # 신호 소멸(예: 10분 경과) 처리
                self.active_signals = [s for s in self.active_signals if (datetime.now() - s.timestamp).seconds < 600]
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Signal execution error: {e}")
                await asyncio.sleep(5)

    async def _get_vix_level(self) -> float:
        """VIX(공포지수) 조회 (실제 구현 필요)"""
        # TODO: 실제 VIX API 연동
        return 28.5

    async def _get_fear_buy_candidates(self) -> List[Dict[str, Any]]:
        """공포 매수 후보 종목 조회 (실제 구현 필요)"""
        # TODO: 실제 데이터 연동
        return [
            {'symbol': '005930', 'name': '삼성전자', 'price': 70000.0},
            {'symbol': '000660', 'name': 'SK하이닉스', 'price': 120000.0}
        ]

    async def _create_fear_buy_signal(self, stock: Dict[str, Any], vix: float) -> Optional[SentimentSignal]:
        try:
            target_price = stock['price'] * 1.04
            stop_loss = stock['price'] * 0.96
            return SentimentSignal(
                symbol=stock['symbol'],
                name=stock['name'],
                signal_type=SentimentSignalType.FEAR_BUY,
                confidence=min((vix - self.vix_threshold) / 10, 1.0),
                reason=f"VIX {vix:.1f} 급등, 역발상 매수",
                target_price=target_price,
                stop_loss=stop_loss,
                sentiment_score=-1.0 * vix,
                vix_level=vix
            )
        except Exception as e:
            logger.error(f"Error creating fear buy signal: {e}")
            return None

    async def _get_greed_index(self) -> float:
        """탐욕(과열) 지수 조회 (실제 구현 필요)"""
        # TODO: 실제 데이터 연동 (개인 순매수, 급등, 뉴스 과열 등)
        return 85.0

    async def _create_greed_sell_signal(self, position: Dict[str, Any], greed_index: float) -> Optional[SentimentSignal]:
        try:
            current_price = position.get('current_price', position.get('avg_price', 100000.0))
            target_price = current_price * 0.98
            stop_loss = current_price * 1.02
            return SentimentSignal(
                symbol=position['symbol'],
                name=position.get('name', position['symbol']),
                signal_type=SentimentSignalType.GREED_SELL,
                confidence=min((greed_index - self.greed_threshold) / 20, 1.0),
                reason=f"탐욕지수 {greed_index:.1f} 과열, 조기 매도",
                target_price=target_price,
                stop_loss=stop_loss,
                sentiment_score=greed_index,
                retail_flow=1.0
            )
        except Exception as e:
            logger.error(f"Error creating greed sell signal: {e}")
            return None

    async def _get_news_events(self) -> List[Dict[str, Any]]:
        """뉴스 이벤트 및 감성 점수 조회 (실제 구현 필요)"""
        # TODO: 뉴스/댓글/커뮤니티 감성 분석 연동
        return [
            {'symbol': '035420', 'name': 'NAVER', 'sentiment': 0.85, 'news_score': 0.9}
        ]

    async def _create_news_anticipate_signal(self, news: Dict[str, Any]) -> Optional[SentimentSignal]:
        try:
            target_price = 400000.0 * 1.03
            stop_loss = 400000.0 * 0.97
            return SentimentSignal(
                symbol=news['symbol'],
                name=news['name'],
                signal_type=SentimentSignalType.NEWS_ANTICIPATE,
                confidence=news['sentiment'],
                reason=f"개인 반응 패턴 선제 대응 (감성 {news['sentiment']:.2f})",
                target_price=target_price,
                stop_loss=stop_loss,
                news_score=news['news_score'],
                sentiment_score=news['sentiment']
            )
        except Exception as e:
            logger.error(f"Error creating news anticipate signal: {e}")
            return None

    async def _execute_fear_buy(self, signal: SentimentSignal):
        try:
            logger.info(f"Executing fear buy for {signal.symbol}: {signal.reason}")
            order_result = await self.kis_trader.place_order(
                symbol=signal.symbol,
                order_type='buy',
                quantity=self._calculate_position_size(signal),
                price=signal.target_price
            )
            if order_result:
                await self.push_service.send_trading_signal_notification(
                    symbol=signal.symbol,
                    signal_type="FEAR_BUY_EXECUTED",
                    confidence=signal.confidence,
                    reason=f"공포 매수 실행 완료: {signal.reason}"
                )
        except Exception as e:
            logger.error(f"Error executing fear buy: {e}")

    async def _execute_greed_sell(self, signal: SentimentSignal):
        try:
            logger.info(f"Executing greed sell for {signal.symbol}: {signal.reason}")
            order_result = await self.kis_trader.place_order(
                symbol=signal.symbol,
                order_type='sell',
                quantity=1,  # 실제 포지션 수량 연동 필요
                price=signal.target_price
            )
            if order_result:
                await self.push_service.send_trading_signal_notification(
                    symbol=signal.symbol,
                    signal_type="GREED_SELL_EXECUTED",
                    confidence=signal.confidence,
                    reason=f"탐욕 매도 실행 완료: {signal.reason}"
                )
        except Exception as e:
            logger.error(f"Error executing greed sell: {e}")

    async def _execute_news_anticipate(self, signal: SentimentSignal):
        try:
            logger.info(f"Executing news anticipate for {signal.symbol}: {signal.reason}")
            order_result = await self.kis_trader.place_order(
                symbol=signal.symbol,
                order_type='buy',
                quantity=self._calculate_position_size(signal),
                price=signal.target_price
            )
            if order_result:
                await self.push_service.send_trading_signal_notification(
                    symbol=signal.symbol,
                    signal_type="NEWS_ANTICIPATE_EXECUTED",
                    confidence=signal.confidence,
                    reason=f"뉴스 반응 선제 진입 완료: {signal.reason}"
                )
        except Exception as e:
            logger.error(f"Error executing news anticipate: {e}")

    def _calculate_position_size(self, signal: SentimentSignal) -> int:
        base_amount = self.max_position_size
        confidence_multiplier = signal.confidence
        return max(100000, min(1000000, int(base_amount * confidence_multiplier)))

    def get_strategy_status(self) -> Dict[str, Any]:
        return {
            'active_signals': len(self.active_signals),
            'vix_threshold': self.vix_threshold,
            'greed_threshold': self.greed_threshold,
            'news_sentiment_threshold': self.news_sentiment_threshold,
            'last_update': datetime.now().isoformat()
        }

    async def generate_signals(self, market_data: dict, start_date, end_date) -> list[Signal]:
        """백테스트용 신호 생성 (Signal 모델 반환)"""
        # 예시: VIX, 뉴스, 개인 순매수 등 데이터에서 신호 생성
        signals = []
        # 예시: 날짜별로 공포 매수 신호 생성
        chart_data = market_data.get('chart_data', {})
        target_stocks = market_data.get('target_stocks', list(chart_data.keys()))
        for stock_code in target_stocks:
            df = chart_data.get(stock_code)
            if df is None or df.empty:
                continue
            for date, row in df.iterrows():
                # 예시: 임의로 VIX 28 이상이면 공포 매수 신호
                vix = 28.5  # 실제로는 market_data에서 가져와야 함
                if vix >= self.vix_threshold:
                    signals.append(Signal(
                        stock_code=stock_code,
                        strategy_type=StrategyType.SHORT_TERM_OPTIMIZED,
                        signal_type=TradeType.BUY,
                        confidence_score=min((vix - self.vix_threshold) / 10, 1.0),
                        target_price=row['close_price'] * 1.04,
                        stop_loss=row['close_price'] * 0.96,
                        take_profit=None,
                        reasoning=f"VIX {vix:.1f} 급등, 역발상 매수",
                        created_at=date
                    ))
        return signals

# 사용 예시
async def main():
    config = {
        'kis_config': {},
        'push_config': {
            'enabled': True,
            'device_tokens': ['your_device_token'],
            'signal_threshold': 0.7,
            'risk_threshold': 0.6
        },
        'vix_threshold': 25.0,
        'greed_threshold': 80.0,
        'news_sentiment_threshold': 0.7,
        'max_position_size': 1000000
    }
    strategy = SentimentTradingStrategy(config)
    await strategy.start_strategy()

if __name__ == "__main__":
    asyncio.run(main())

