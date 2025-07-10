from __future__ import annotations
from aiohttp import ClientSession
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from kis_config import KISConfig
from kis_trader import KISTrader
from push_notifications import PushNotificationService
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
import logging
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: real_time_data_pipeline.py
모듈: 실시간 데이터 파이프라인
목적: KIS API, 뉴스 API, 공시 API, 기술적 지표 통합

Author: AI Assistant
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - asyncio, aiohttp, pandas, numpy
    - kis_config, kis_trader
"""





logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """시장 데이터"""
    symbol: str
    current_price: float
    change_rate: float
    volume: int
    high: float
    low: float
    open_price: float
    prev_close: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class NewsData:
    """뉴스 데이터"""
    title: str
    content: str
    sentiment: float
    impact_score: float
    source: str
    published_at: datetime
    related_symbols: List[str] = field(default_factory=list)

@dataclass
class DisclosureData:
    """공시 데이터"""
    title: str
    content: str
    disclosure_type: str
    impact_score: float
    published_at: datetime
    related_symbols: List[str] = field(default_factory=list)

@dataclass
class TechnicalIndicators:
    """기술적 지표"""
    symbol: str
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    volume_ma: float
    price_ma_5: float
    price_ma_20: float
    timestamp: datetime = field(default_factory=datetime.now)

class RealTimeDataPipeline:
    """실시간 데이터 파이프라인"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kis_config = KISConfig()
        self.kis_trader = KISTrader(self.kis_config)

        # 데이터 저장소
        self.market_data: Dict[str, MarketData] = {}
        self.news_data: List[NewsData] = []
        self.disclosure_data: List[DisclosureData] = []
        self.technical_indicators: Dict[str, TechnicalIndicators] = {}

        # 설정
        self.symbols_to_track = config.get('symbols_to_track', [])
        self.update_interval = config.get('update_interval', 1.0)
        self.news_api_key = config.get('news_api_key', '')
        self.dart_api_key = config.get('dart_api_key', '')

        # 푸시 알림 서비스
        self.push_service = PushNotificationService(config.get('push_config', {}))

        # 데이터 히스토리 (기술적 지표 계산용)
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}

        # 세션 관리
        self.session: Optional[ClientSession] = None

    async def start_pipeline(self):
        """파이프라인 시작"""
        logger.info("Starting real-time data pipeline...")

        # HTTP 세션 생성
        self.session = ClientSession()

        try:
            # 여러 데이터 수집 태스크 동시 실행
            tasks = [
                asyncio.create_task(self._collect_market_data()),
                asyncio.create_task(self._collect_news_data()),
                asyncio.create_task(self._collect_disclosure_data()),
                asyncio.create_task(self._calculate_technical_indicators()),
                asyncio.create_task(self._process_alerts())
            ]

            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            if self.session:
                await self.session.close()

    async def _collect_market_data(self):
        """시장 데이터 수집"""
        while True:
            try:
                for symbol in self.symbols_to_track:
                    await self._fetch_market_data(symbol)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Market data collection error: {e}")
                await asyncio.sleep(5.0)

    async def _fetch_market_data(self, symbol: str):
        """개별 종목 시장 데이터 수집"""
        try:
            # KIS API 호출 (실제 구현 필요)
            # current_data = await self.kis_trader.get_current_price(symbol)

            # 예시 데이터
            current_data = {
                'stck_prpr': 75000,  # 현재가
                'prdy_vrss': 0.02,   # 등락률
                'acml_vol': 1000000, # 거래량
                'stck_hgpr': 75500,  # 고가
                'stck_lwpr': 74500,  # 저가
                'stck_oprc': 74800,  # 시가
                'prdy_clpr': 73500   # 전일종가
            }

            if current_data:
                market_data = MarketData(
                    symbol=symbol,
                    current_price=current_data.get('stck_prpr', 0),
                    change_rate=current_data.get('prdy_vrss', 0),
                    volume=current_data.get('acml_vol', 0),
                    high=current_data.get('stck_hgpr', 0),
                    low=current_data.get('stck_lwpr', 0),
                    open_price=current_data.get('stck_oprc', 0),
                    prev_close=current_data.get('prdy_clpr', 0)
                )

                self.market_data[symbol] = market_data

                # 가격 히스토리 업데이트
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(market_data.current_price)

                # 거래량 히스토리 업데이트
                if symbol not in self.volume_history:
                    self.volume_history[symbol] = []
                self.volume_history[symbol].append(market_data.volume)

                # 히스토리 길이 제한 (최근 100개)
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
                if len(self.volume_history[symbol]) > 100:
                    self.volume_history[symbol] = self.volume_history[symbol][-100:]

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")

    async def _collect_news_data(self):
        """뉴스 데이터 수집"""
        while True:
            try:
                # 뉴스 API 호출 (실제 구현 필요)
                # news_data = await self._fetch_news_from_api()

                # 예시 뉴스 데이터
                example_news = [
                    {
                        'title': '삼성전자, 신제품 출시 예정',
                        'content': '삼성전자가 새로운 스마트폰을 출시할 예정입니다.',
                        'sentiment': 0.8,
                        'impact_score': 0.7,
                        'source': '경제일보',
                        'published_at': datetime.now(),
                        'related_symbols': ['005930']
                    }
                ]

                for news in example_news:
                    news_data = NewsData(**news)
                    self.news_data.append(news_data)

                    # 중요 뉴스는 푸시 알림
                    if news_data.impact_score > 0.7:
                        await self.push_service.send_market_alert_notification(
                            alert_type="중요 뉴스",
                            message=news_data.title,
                            market_data={
                                'sentiment': news_data.sentiment,
                                'impact_score': news_data.impact_score,
                                'symbols': news_data.related_symbols
                            }
                        )

                # 뉴스 데이터 정리 (최근 100개만 유지)
                if len(self.news_data) > 100:
                    self.news_data = self.news_data[-100:]

                await asyncio.sleep(60.0)  # 1분마다 뉴스 수집

            except Exception as e:
                logger.error(f"News data collection error: {e}")
                await asyncio.sleep(60.0)

    async def _collect_disclosure_data(self):
        """공시 데이터 수집"""
        while True:
            try:
                # DART API 호출 (실제 구현 필요)
                # disclosure_data = await self._fetch_disclosure_from_api()

                # 예시 공시 데이터
                example_disclosure = [
                    {
                        'title': '주요사항보고서',
                        'content': '회사의 주요 경영사항이 변경되었습니다.',
                        'disclosure_type': '주요사항',
                        'impact_score': 0.6,
                        'published_at': datetime.now(),
                        'related_symbols': ['005930']
                    }
                ]

                for disclosure in example_disclosure:
                    disclosure_data = DisclosureData(**disclosure)
                    self.disclosure_data.append(disclosure_data)

                    # 중요 공시는 푸시 알림
                    if disclosure_data.impact_score > 0.6:
                        await self.push_service.send_market_alert_notification(
                            alert_type="중요 공시",
                            message=disclosure_data.title,
                            market_data={
                                'disclosure_type': disclosure_data.disclosure_type,
                                'impact_score': disclosure_data.impact_score,
                                'symbols': disclosure_data.related_symbols
                            }
                        )

                # 공시 데이터 정리 (최근 50개만 유지)
                if len(self.disclosure_data) > 50:
                    self.disclosure_data = self.disclosure_data[-50:]

                await asyncio.sleep(300.0)  # 5분마다 공시 수집

            except Exception as e:
                logger.error(f"Disclosure data collection error: {e}")
                await asyncio.sleep(300.0)

    async def _calculate_technical_indicators(self):
        """기술적 지표 계산"""
        while True:
            try:
                for symbol in self.symbols_to_track:
                    if symbol in self.price_history and len(self.price_history[symbol]) >= 20:
                        indicators = await self._calculate_indicators_for_symbol(symbol)
                        self.technical_indicators[symbol] = indicators

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Technical indicators calculation error: {e}")
                await asyncio.sleep(5.0)

    async def _calculate_indicators_for_symbol(self, symbol: str) -> TechnicalIndicators:
        """개별 종목 기술적 지표 계산"""
        prices = self.price_history[symbol]
        volumes = self.volume_history[symbol]

        if len(prices) < 20:
            return TechnicalIndicators(symbol=symbol)

        # RSI 계산
        rsi = self._calculate_rsi(prices)

        # MACD 계산
        macd, macd_signal = self._calculate_macd(prices)

        # 볼린저 밴드 계산
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)

        # 이동평균 계산
        ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        volume_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]

        return TechnicalIndicators(
            symbol=symbol,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            bollinger_middle=bb_middle,
            volume_ma=volume_ma,
            price_ma_5=ma_5,
            price_ma_20=ma_20
        )

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """MACD 계산"""
        if len(prices) < slow:
            return 0.0, 0.0

        prices_array = np.array(prices)

        # EMA 계산
        ema_fast = self._calculate_ema(prices_array, fast)
        ema_slow = self._calculate_ema(prices_array, slow)

        # MACD 라인
        macd_line = ema_fast - ema_slow

        # 시그널 라인 (MACD의 EMA)
        macd_values = []
        for i in range(len(prices)):
            if i >= slow - 1:
                macd_values.append(macd_line[i])

        if len(macd_values) >= signal:
            signal_line = self._calculate_ema(np.array(macd_values), signal)
            return float(macd_line[-1]), float(signal_line[-1])

        return float(macd_line[-1]), 0.0

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """지수이동평균 계산"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """볼린저 밴드 계산"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]

        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return float(upper), float(middle), float(lower)

    async def _process_alerts(self):
        """알림 처리"""
        while True:
            try:
                for symbol in self.symbols_to_track:
                    if symbol in self.market_data and symbol in self.technical_indicators:
                        await self._check_alerts_for_symbol(symbol)

                await asyncio.sleep(10.0)  # 10초마다 알림 체크

            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(10.0)

    async def _check_alerts_for_symbol(self, symbol: str):
        """개별 종목 알림 체크"""
        market_data = self.market_data[symbol]
        indicators = self.technical_indicators[symbol]

        # 급등/급락 알림
        if abs(market_data.change_rate) > 0.05:  # 5% 이상 변화
            await self.push_service.send_market_alert_notification(
                alert_type="급등/급락",
                message=f"{symbol}: {market_data.change_rate:+.2%} 변화",
                market_data={
                    'symbol': symbol,
                    'change_rate': market_data.change_rate,
                    'current_price': market_data.current_price
                }
            )

        # 기술적 지표 알림
        if indicators.rsi > 80:  # RSI 과매수
            await self.push_service.send_market_alert_notification(
                alert_type="RSI 과매수",
                message=f"{symbol}: RSI {indicators.rsi:.1f}",
                market_data={
                    'symbol': symbol,
                    'rsi': indicators.rsi,
                    'current_price': market_data.current_price
                }
            )
        elif indicators.rsi < 20:  # RSI 과매도
            await self.push_service.send_market_alert_notification(
                alert_type="RSI 과매도",
                message=f"{symbol}: RSI {indicators.rsi:.1f}",
                market_data={
                    'symbol': symbol,
                    'rsi': indicators.rsi,
                    'current_price': market_data.current_price
                }
            )

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """시장 데이터 조회"""
        return self.market_data.get(symbol)

    def get_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """기술적 지표 조회"""
        return self.technical_indicators.get(symbol)

    def get_recent_news(self, hours: int = 24) -> List[NewsData]:
        """최근 뉴스 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [news for news in self.news_data if news.published_at >= cutoff_time]

    def get_recent_disclosures(self, hours: int = 24) -> List[DisclosureData]:
        """최근 공시 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [disclosure for disclosure in self.disclosure_data if disclosure.published_at >= cutoff_time]

    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        return {
            'active_symbols': len(self.market_data),
            'technical_indicators': len(self.technical_indicators),
            'recent_news': len(self.get_recent_news()),
            'recent_disclosures': len(self.get_recent_disclosures()),
            'price_history_length': {symbol: len(prices) for symbol, prices in self.price_history.items()},
            'last_update': datetime.now().isoformat()
        }

# 사용 예시
async def main():
    """메인 실행 함수"""

    config = {
        'symbols_to_track': ['005930', '000660', '035420'],  # 삼성전자, SK하이닉스, NAVER
        'update_interval': 1.0,
        'news_api_key': 'your_news_api_key',
        'dart_api_key': 'your_dart_api_key',
        'push_config': {
            'enabled': True,
            'device_tokens': ['device_token_1'],
            'signal_threshold': 0.8,
            'risk_threshold': 0.7
        }
    }

    pipeline = RealTimeDataPipeline(config)
    await pipeline.start_pipeline()

if __name__ == "__main__":
    asyncio.run(main())

