#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇺🇸 미국 주식 파생상품 실시간 데이터 수집 시스템
===========================================
실시간 옵션/선물 데이터 스트리밍 및 분석
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import aiohttp
import websockets
import yfinance as yf
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class USDerivativeData:
    """미국 파생상품 데이터 구조"""
    symbol: str
    underlying: str
    derivative_type: str  # 'option', 'future'
    contract_type: str = ""  # 'call', 'put' for options
    strike_price: float = 0.0
    expiry_date: str = ""
    current_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'derivative_type': self.derivative_type,
            'contract_type': self.contract_type,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date,
            'current_price': self.current_price,
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'change': self.change,
            'change_percent': self.change_percent,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class RealTimeConfig:
    """실시간 데이터 설정"""
    # API 키들
    polygon_api_key: str = ""
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""
    tradier_token: str = ""
    
    # WebSocket 설정
    max_connections: int = 5
    reconnect_delay: int = 5
    heartbeat_interval: int = 30
    
    # 데이터 필터링
    min_volume: int = 100
    min_open_interest: int = 50
    max_spread_percent: float = 5.0
    
    # 모니터링 대상
    watch_symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'DIA',  # ETF
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # 대형주
        'NVDA', 'META', 'NFLX', 'AMD', 'CRM'  # 기술주
    ])
    
    @classmethod
    def from_env(cls) -> 'RealTimeConfig':
        """환경 변수에서 설정 로드"""
        return cls(
            polygon_api_key=os.getenv('POLYGON_API_KEY', ''),
            alpha_vantage_api_key=os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            finnhub_api_key=os.getenv('FINNHUB_API_KEY', ''),
            tradier_token=os.getenv('TRADIER_TOKEN', ''),
            max_connections=int(os.getenv('MAX_WS_CONNECTIONS', '5')),
            reconnect_delay=int(os.getenv('WS_RECONNECT_DELAY', '5')),
            heartbeat_interval=int(os.getenv('WS_HEARTBEAT_INTERVAL', '30'))
        )

class USRealTimeDerivatives:
    """미국 파생상품 실시간 데이터 수집기"""
    
    def __init__(self, config: Optional[RealTimeConfig] = None):
        self.config = config or RealTimeConfig.from_env()
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: Dict[str, Any] = {}
        self.data_callbacks: List[Callable] = []
        self.is_running = False
        self.last_heartbeat = {}
        
        # 데이터 캐시
        self.options_cache: Dict[str, List[USDerivativeData]] = {}
        self.futures_cache: Dict[str, List[USDerivativeData]] = {}
        
        logger.info("🇺🇸 미국 파생상품 실시간 수집기 초기화")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop_all_streams()
        if self.session:
            await self.session.close()
    
    def add_data_callback(self, callback: Callable[[USDerivativeData], None]):
        """데이터 콜백 추가"""
        self.data_callbacks.append(callback)
    
    async def _notify_callbacks(self, data: USDerivativeData):
        """콜백 함수들에 데이터 전달"""
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"콜백 실행 오류: {e}")
    
    async def get_options_chain_realtime(self, symbol: str) -> List[USDerivativeData]:
        """실시간 옵션 체인 데이터"""
        try:
            # 1순위: Tradier API (실시간)
            if self.config.tradier_token:
                options = await self._get_tradier_options(symbol)
                if options:
                    self.options_cache[symbol] = options
                    return options
            
            # 2순위: Polygon API
            if self.config.polygon_api_key:
                options = await self._get_polygon_options(symbol)
                if options:
                    self.options_cache[symbol] = options
                    return options
            
            # 3순위: Yahoo Finance (백업)
            options = await self._get_yahoo_options(symbol)
            if options:
                self.options_cache[symbol] = options
                return options
            
            return self.options_cache.get(symbol, [])
            
        except Exception as e:
            logger.error(f"옵션 체인 조회 오류 {symbol}: {e}")
            return self.options_cache.get(symbol, [])
    
    async def _get_tradier_options(self, symbol: str) -> List[USDerivativeData]:
        """Tradier API를 통한 옵션 데이터"""
        if not self.config.tradier_token:
            return []
        
        try:
            # 옵션 만료일 조회
            expiry_url = f"https://api.tradier.com/v1/markets/options/expirations"
            headers = {
                'Authorization': f'Bearer {self.config.tradier_token}',
                'Accept': 'application/json'
            }
            
            async with self.session.get(
                expiry_url, 
                headers=headers, 
                params={'symbol': symbol}
            ) as response:
                if response.status != 200:
                    return []
                
                expiry_data = await response.json()
                expirations = expiry_data.get('expirations', {}).get('date', [])
                
                if not expirations:
                    return []
                
                # 가장 가까운 만료일 선택
                nearest_expiry = expirations[0] if isinstance(expirations, list) else expirations
                
                # 옵션 체인 조회
                chain_url = f"https://api.tradier.com/v1/markets/options/chains"
                params = {
                    'symbol': symbol,
                    'expiration': nearest_expiry,
                    'greeks': 'true'
                }
                
                async with self.session.get(
                    chain_url, 
                    headers=headers, 
                    params=params
                ) as chain_response:
                    if chain_response.status != 200:
                        return []
                    
                    chain_data = await chain_response.json()
                    options_data = chain_data.get('options', {}).get('option', [])
                    
                    if not isinstance(options_data, list):
                        options_data = [options_data]
                    
                    options = []
                    for opt in options_data:
                        if opt.get('volume', 0) < self.config.min_volume:
                            continue
                        
                        derivative = USDerivativeData(
                            symbol=opt.get('symbol', ''),
                            underlying=symbol,
                            derivative_type='option',
                            contract_type=opt.get('option_type', '').lower(),
                            strike_price=float(opt.get('strike', 0)),
                            expiry_date=opt.get('expiration_date', ''),
                            current_price=float(opt.get('last', 0)),
                            bid=float(opt.get('bid', 0)),
                            ask=float(opt.get('ask', 0)),
                            volume=int(opt.get('volume', 0)),
                            open_interest=int(opt.get('open_interest', 0)),
                            implied_volatility=float(opt.get('greeks', {}).get('smv_vol', 0)),
                            delta=float(opt.get('greeks', {}).get('delta', 0)),
                            gamma=float(opt.get('greeks', {}).get('gamma', 0)),
                            theta=float(opt.get('greeks', {}).get('theta', 0)),
                            vega=float(opt.get('greeks', {}).get('vega', 0)),
                            change=float(opt.get('change', 0)),
                            change_percent=float(opt.get('change_percentage', 0))
                        )
                        options.append(derivative)
                        
                        # 콜백 알림
                        await self._notify_callbacks(derivative)
                    
                    return options
                    
        except Exception as e:
            logger.error(f"Tradier 옵션 조회 오류 {symbol}: {e}")
            return []
    
    async def _get_polygon_options(self, symbol: str) -> List[USDerivativeData]:
        """Polygon API를 통한 옵션 데이터"""
        if not self.config.polygon_api_key:
            return []
        
        try:
            # 현재 날짜 기준 옵션 조회
            today = datetime.now().strftime('%Y-%m-%d')
            url = f"https://api.polygon.io/v3/reference/options/contracts"
            
            params = {
                'underlying_ticker': symbol,
                'contract_type': 'call',  # call과 put 따로 조회
                'expiration_date.gte': today,
                'limit': 100,
                'apikey': self.config.polygon_api_key
            }
            
            options = []
            
            # Call 옵션 조회
            for contract_type in ['call', 'put']:
                params['contract_type'] = contract_type
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        continue
                    
                    data = await response.json()
                    contracts = data.get('results', [])
                    
                    for contract in contracts[:20]:  # 상위 20개만
                        # 실시간 시세 조회
                        quote = await self._get_polygon_option_quote(
                            contract.get('ticker', '')
                        )
                        
                        if not quote:
                            continue
                        
                        derivative = USDerivativeData(
                            symbol=contract.get('ticker', ''),
                            underlying=symbol,
                            derivative_type='option',
                            contract_type=contract_type,
                            strike_price=float(contract.get('strike_price', 0)),
                            expiry_date=contract.get('expiration_date', ''),
                            current_price=quote.get('last_price', 0),
                            bid=quote.get('bid', 0),
                            ask=quote.get('ask', 0),
                            volume=quote.get('volume', 0),
                            change=quote.get('change', 0),
                            change_percent=quote.get('change_percent', 0)
                        )
                        options.append(derivative)
                        
                        # 콜백 알림
                        await self._notify_callbacks(derivative)
            
            return options
            
        except Exception as e:
            logger.error(f"Polygon 옵션 조회 오류 {symbol}: {e}")
            return []
    
    async def _get_polygon_option_quote(self, option_symbol: str) -> Optional[Dict]:
        """Polygon 옵션 실시간 시세"""
        try:
            url = f"https://api.polygon.io/v2/last/trade/{option_symbol}"
            params = {'apikey': self.config.polygon_api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', {})
                return None
                
        except Exception as e:
            logger.error(f"Polygon 옵션 시세 조회 오류 {option_symbol}: {e}")
            return None
    
    async def _get_yahoo_options(self, symbol: str) -> List[USDerivativeData]:
        """Yahoo Finance 옵션 데이터 (백업)"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return []
            
            # 가장 가까운 만료일
            nearest_expiry = expirations[0]
            option_chain = ticker.option_chain(nearest_expiry)
            
            options = []
            
            # NaN 값을 안전하게 처리하는 헬퍼 함수들
            def safe_float(value, default=0.0):
                """NaN 값을 안전하게 float로 변환"""
                try:
                    if pd.isna(value) or value == '' or value is None:
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                """NaN 값을 안전하게 int로 변환"""
                try:
                    if pd.isna(value) or value == '' or value is None:
                        return default
                    return int(float(value))
                except (ValueError, TypeError):
                    return default
            
            # 콜 옵션 처리
            for _, row in option_chain.calls.iterrows():
                try:
                    option = USDerivativeData(
                        symbol=f"{symbol}_{safe_float(row.get('strike', 0)):.0f}C_{nearest_expiry}",
                        underlying=symbol,
                        derivative_type='option',
                        contract_type='call',
                        strike_price=safe_float(row.get('strike', 0)),
                        expiry_date=nearest_expiry,
                        current_price=safe_float(row.get('lastPrice', 0)),
                        bid=safe_float(row.get('bid', 0)),
                        ask=safe_float(row.get('ask', 0)),
                        volume=safe_int(row.get('volume', 0)),
                        open_interest=safe_int(row.get('openInterest', 0)),
                        implied_volatility=safe_float(row.get('impliedVolatility', 0)),
                        delta=safe_float(row.get('delta', 0)),
                        gamma=safe_float(row.get('gamma', 0)),
                        theta=safe_float(row.get('theta', 0)),
                        vega=safe_float(row.get('vega', 0)),
                        timestamp=datetime.now()
                    )
                    options.append(option)
                except Exception as e:
                    logger.warning(f"콜 옵션 처리 오류: {e}")
                    continue
            
            # 풋 옵션 처리
            for _, row in option_chain.puts.iterrows():
                try:
                    option = USDerivativeData(
                        symbol=f"{symbol}_{safe_float(row.get('strike', 0)):.0f}P_{nearest_expiry}",
                        underlying=symbol,
                        derivative_type='option',
                        contract_type='put',
                        strike_price=safe_float(row.get('strike', 0)),
                        expiry_date=nearest_expiry,
                        current_price=safe_float(row.get('lastPrice', 0)),
                        bid=safe_float(row.get('bid', 0)),
                        ask=safe_float(row.get('ask', 0)),
                        volume=safe_int(row.get('volume', 0)),
                        open_interest=safe_int(row.get('openInterest', 0)),
                        implied_volatility=safe_float(row.get('impliedVolatility', 0)),
                        delta=safe_float(row.get('delta', 0)),
                        gamma=safe_float(row.get('gamma', 0)),
                        theta=safe_float(row.get('theta', 0)),
                        vega=safe_float(row.get('vega', 0)),
                        timestamp=datetime.now()
                    )
                    options.append(option)
                except Exception as e:
                    logger.warning(f"풋 옵션 처리 오류: {e}")
                    continue
                    
            return options
            
        except Exception as e:
            logger.error(f"Yahoo Finance 옵션 데이터 오류: {e}")
            return []
    
    async def start_websocket_stream(self, symbols: List[str] = None):
        """WebSocket 실시간 스트림 시작"""
        symbols = symbols or self.config.watch_symbols
        self.is_running = True
        
        logger.info(f"🔴 실시간 스트림 시작: {symbols}")
        
        # 여러 WebSocket 연결을 병렬로 시작
        tasks = []
        
        # Polygon WebSocket
        if self.config.polygon_api_key:
            tasks.append(self._start_polygon_websocket(symbols))
        
        # Finnhub WebSocket
        if self.config.finnhub_api_key:
            tasks.append(self._start_finnhub_websocket(symbols))
        
        # 병렬 실행
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _start_polygon_websocket(self, symbols: List[str]):
        """Polygon WebSocket 연결"""
        uri = f"wss://socket.polygon.io/options"
        
        while self.is_running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websocket_connections['polygon'] = websocket
                    
                    # 인증
                    auth_msg = {
                        "action": "auth",
                        "params": self.config.polygon_api_key
                    }
                    await websocket.send(json.dumps(auth_msg))
                    
                    # 구독
                    for symbol in symbols:
                        subscribe_msg = {
                            "action": "subscribe",
                            "params": f"T.{symbol}*"  # 모든 옵션
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                    
                    logger.info("✅ Polygon WebSocket 연결 완료")
                    
                    # 메시지 수신
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_polygon_message(data)
                        except Exception as e:
                            logger.error(f"Polygon 메시지 처리 오류: {e}")
                            
            except Exception as e:
                logger.error(f"Polygon WebSocket 오류: {e}")
                if self.is_running:
                    await asyncio.sleep(self.config.reconnect_delay)
    
    async def _start_finnhub_websocket(self, symbols: List[str]):
        """Finnhub WebSocket 연결"""
        uri = f"wss://ws.finnhub.io?token={self.config.finnhub_api_key}"
        
        while self.is_running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websocket_connections['finnhub'] = websocket
                    
                    # 구독
                    for symbol in symbols:
                        subscribe_msg = {
                            "type": "subscribe",
                            "symbol": symbol
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                    
                    logger.info("✅ Finnhub WebSocket 연결 완료")
                    
                    # 메시지 수신
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_finnhub_message(data)
                        except Exception as e:
                            logger.error(f"Finnhub 메시지 처리 오류: {e}")
                            
            except Exception as e:
                logger.error(f"Finnhub WebSocket 오류: {e}")
                if self.is_running:
                    await asyncio.sleep(self.config.reconnect_delay)
    
    async def _process_polygon_message(self, data: List[Dict]):
        """Polygon 메시지 처리"""
        for item in data:
            if item.get('ev') == 'T':  # Trade 이벤트
                # 옵션 심볼 파싱
                symbol = item.get('sym', '')
                if not symbol:
                    continue
                
                # 기존 캐시에서 해당 옵션 업데이트
                for underlying, options in self.options_cache.items():
                    for option in options:
                        if option.symbol == symbol:
                            option.current_price = item.get('p', option.current_price)
                            option.volume = item.get('s', option.volume)
                            option.timestamp = datetime.now()
                            
                            await self._notify_callbacks(option)
                            break
    
    async def _process_finnhub_message(self, data: Dict):
        """Finnhub 메시지 처리"""
        if data.get('type') == 'trade':
            trades = data.get('data', [])
            for trade in trades:
                symbol = trade.get('s', '')
                price = trade.get('p', 0)
                volume = trade.get('v', 0)
                
                # 기본 주식 데이터로 옵션 가격 영향 분석
                logger.debug(f"📊 {symbol}: ${price:.2f} (거래량: {volume:,})")
    
    async def stop_all_streams(self):
        """모든 스트림 중지"""
        self.is_running = False
        
        for name, ws in self.websocket_connections.items():
            try:
                if ws and not ws.closed:
                    await ws.close()
                logger.info(f"🔴 {name} WebSocket 연결 종료")
            except Exception as e:
                logger.error(f"{name} WebSocket 종료 오류: {e}")
        
        self.websocket_connections.clear()
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """시장 파생상품 종합 현황"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_options': 0,
                'total_volume': 0,
                'total_open_interest': 0,
                'put_call_ratio': 0.0,
                'high_iv_options': [],
                'most_active_options': [],
                'unusual_activity': []
            }
            
            all_options = []
            for options in self.options_cache.values():
                all_options.extend(options)
            
            if not all_options:
                return summary
            
            # 통계 계산
            summary['total_options'] = len(all_options)
            summary['total_volume'] = sum(opt.volume for opt in all_options)
            summary['total_open_interest'] = sum(opt.open_interest for opt in all_options)
            
            # Put/Call 비율
            puts = [opt for opt in all_options if opt.contract_type == 'put']
            calls = [opt for opt in all_options if opt.contract_type == 'call']
            
            put_volume = sum(opt.volume for opt in puts)
            call_volume = sum(opt.volume for opt in calls)
            
            if call_volume > 0:
                summary['put_call_ratio'] = put_volume / call_volume
            
            # 고변동성 옵션 (상위 10개)
            high_iv = sorted(all_options, key=lambda x: x.implied_volatility, reverse=True)[:10]
            summary['high_iv_options'] = [opt.to_dict() for opt in high_iv]
            
            # 거래량 많은 옵션 (상위 10개)
            most_active = sorted(all_options, key=lambda x: x.volume, reverse=True)[:10]
            summary['most_active_options'] = [opt.to_dict() for opt in most_active]
            
            return summary
            
        except Exception as e:
            logger.error(f"시장 종합 현황 오류: {e}")
            return {}

# 헬퍼 함수
async def get_us_realtime_derivatives(config: Optional[RealTimeConfig] = None) -> USRealTimeDerivatives:
    """미국 실시간 파생상품 수집기 팩토리"""
    return USRealTimeDerivatives(config)

# 테스트용 메인 함수
async def main():
    """테스트 실행"""
    print("🇺🇸 미국 파생상품 실시간 데이터 테스트")
    print("=" * 50)
    
    config = RealTimeConfig.from_env()
    
    async with USRealTimeDerivatives(config) as collector:
        # 데이터 콜백 등록
        def on_data(data: USDerivativeData):
            print(f"📊 {data.symbol}: ${data.current_price:.2f} "
                  f"({data.change_percent:+.2f}%) Vol: {data.volume:,}")
        
        collector.add_data_callback(on_data)
        
        # 옵션 체인 조회 테스트
        print("\n1️⃣ 옵션 체인 조회 테스트...")
        for symbol in ['SPY', 'QQQ', 'AAPL']:
            options = await collector.get_options_chain_realtime(symbol)
            print(f"✅ {symbol} 옵션: {len(options)}개")
            
            if options:
                # 상위 3개 출력
                for opt in options[:3]:
                    print(f"   📈 {opt.symbol}: ${opt.current_price:.2f} "
                          f"(IV: {opt.implied_volatility:.1%})")
        
        print("\n2️⃣ 시장 종합 현황...")
        summary = await collector.get_market_summary()
        print(f"✅ 총 옵션: {summary.get('total_options', 0):,}개")
        print(f"✅ 총 거래량: {summary.get('total_volume', 0):,}")
        print(f"✅ Put/Call 비율: {summary.get('put_call_ratio', 0):.3f}")
        
        # 실시간 스트림 테스트 (10초간)
        print("\n3️⃣ 실시간 스트림 테스트 (10초)...")
        stream_task = asyncio.create_task(
            collector.start_websocket_stream(['SPY', 'QQQ'])
        )
        
        await asyncio.sleep(10)
        await collector.stop_all_streams()
        
        try:
            stream_task.cancel()
            await stream_task
        except asyncio.CancelledError:
            pass
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main()) 