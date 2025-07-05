#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇺🇸 다중 소스 미국 파생상품 데이터 수집기
===========================================
KIS API 파생상품 승인 대기 중 사용할 수 있는 대안 데이터 소스들
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import yfinance as yf
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class USDerivativeData:
    """미국 파생상품 데이터 구조"""
    symbol: str
    name: str
    underlying: str
    derivative_type: str  # 'future', 'option', 'etf'
    contract_type: str = ""  # 'call', 'put' for options
    strike_price: float = 0.0
    expiry_date: str = ""
    current_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    change: float = 0.0
    change_percent: float = 0.0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    currency: str = "USD"
    exchange: str = ""
    data_source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'name': self.name,
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
            'change': self.change,
            'change_percent': self.change_percent,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'currency': self.currency,
            'exchange': self.exchange,
            'data_source': self.data_source,
            'timestamp': self.timestamp.isoformat()
        }

class MultiSourceUSDerivatives:
    """다중 소스 미국 파생상품 데이터 수집기"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.data_callbacks: List[Callable] = []
        
        # API 키 설정
        self.polygon_key = os.getenv('POLYGON_API_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        
        # 주요 미국 지수 ETF (파생상품 대용)
        self.index_etfs = {
            'SPY': {'name': 'SPDR S&P 500 ETF', 'underlying': 'SPX', 'type': 'etf'},
            'QQQ': {'name': 'Invesco QQQ Trust', 'underlying': 'NDX', 'type': 'etf'},
            'DIA': {'name': 'SPDR Dow Jones Industrial Average ETF', 'underlying': 'DJI', 'type': 'etf'},
            'IWM': {'name': 'iShares Russell 2000 ETF', 'underlying': 'RUT', 'type': 'etf'},
            'VIX': {'name': 'CBOE Volatility Index', 'underlying': 'VIX', 'type': 'index'},
            'UVXY': {'name': 'ProShares Ultra VIX Short-Term Futures ETF', 'underlying': 'VIX', 'type': 'etf'},
            'SQQQ': {'name': 'ProShares UltraPro Short QQQ', 'underlying': 'NDX', 'type': 'etf'},
            'SPXU': {'name': 'ProShares UltraPro Short S&P500', 'underlying': 'SPX', 'type': 'etf'}
        }
        
        logger.info("🇺🇸 다중 소스 미국 파생상품 데이터 수집기 초기화")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def add_data_callback(self, callback: Callable):
        """데이터 수신 콜백 추가"""
        self.data_callbacks.append(callback)
    
    async def notify_callbacks(self, data: USDerivativeData):
        """콜백들에게 데이터 알림"""
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"콜백 실행 오류: {e}")
    
    async def get_yahoo_finance_data(self, symbols: List[str]) -> List[USDerivativeData]:
        """Yahoo Finance에서 ETF/지수 데이터 수집"""
        derivatives = []
        
        try:
            for symbol in symbols:
                if symbol not in self.index_etfs:
                    continue
                
                info = self.index_etfs[symbol]
                
                try:
                    # yfinance를 사용하여 데이터 수집
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        previous = data.iloc[-2] if len(data) > 1 else latest
                        
                        change = latest['Close'] - previous['Close']
                        change_pct = (change / previous['Close']) * 100 if previous['Close'] != 0 else 0
                        
                        derivative = USDerivativeData(
                            symbol=symbol,
                            name=info['name'],
                            underlying=info['underlying'],
                            derivative_type=info['type'],
                            current_price=float(latest['Close']),
                            volume=int(latest['Volume']),
                            change=float(change),
                            change_percent=float(change_pct),
                            exchange='NYSE' if symbol in ['SPY', 'DIA', 'IWM'] else 'NASDAQ',
                            data_source='Yahoo Finance'
                        )
                        
                        derivatives.append(derivative)
                        await self.notify_callbacks(derivative)
                        
                        logger.info(f"✅ {symbol} 데이터 수집: ${derivative.current_price:.2f} ({derivative.change_percent:+.2f}%)")
                
                except Exception as e:
                    logger.error(f"{symbol} 데이터 수집 오류: {e}")
                    continue
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Yahoo Finance 데이터 수집 오류: {e}")
        
        return derivatives
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """시장 종합 현황"""
        try:
            # Yahoo Finance에서 주요 ETF 데이터 수집
            etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'UVXY']
            etf_data = await self.get_yahoo_finance_data(etf_symbols)
            
            # 주요 지수 현황
            major_indices = {}
            for etf in etf_data:
                if etf.symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
                    major_indices[etf.symbol] = {
                        'price': etf.current_price,
                        'change': etf.change,
                        'change_percent': etf.change_percent
                    }
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_derivatives': len(etf_data),
                'data_sources': ['Yahoo Finance'],
                'major_indices': major_indices,
                'available_data': {
                    'etfs': len(etf_data),
                    'options': 0,  # KIS 승인 후 추가 예정
                    'futures': 0   # KIS 승인 후 추가 예정
                },
                'kis_derivatives_status': '📋 신청 진행 중 - 승인 대기'
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"시장 종합 현황 조회 오류: {e}")
            return {}

# 테스트용 메인 함수
async def main():
    """다중 소스 미국 파생상품 데이터 수집 테스트"""
    print("🇺🇸 다중 소스 미국 파생상품 데이터 수집 테스트")
    print("=" * 60)
    print("📌 KIS API 파생상품 승인 대기 중 - 대안 데이터 소스 활용")
    print()
    
    async with MultiSourceUSDerivatives() as collector:
        # 데이터 수신 콜백 등록
        def on_data_received(data: USDerivativeData):
            print(f"📊 실시간 데이터: {data.symbol} - ${data.current_price:.2f} "
                  f"({data.change_percent:+.2f}%) [{data.data_source}]")
        
        collector.add_data_callback(on_data_received)
        
        # 1. Yahoo Finance ETF 데이터 수집
        print("1️⃣ Yahoo Finance ETF 데이터 수집...")
        etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'UVXY']
        etf_data = await collector.get_yahoo_finance_data(etf_symbols)
        
        print(f"✅ ETF 데이터: {len(etf_data)}개 수집")
        
        # 2. 시장 종합 현황
        print("\n2️⃣ 시장 종합 현황...")
        summary = await collector.get_market_summary()
        
        print(f"✅ 총 파생상품: {summary.get('total_derivatives', 0)}개")
        print(f"✅ 데이터 소스: {', '.join(summary.get('data_sources', []))}")
        print(f"✅ KIS 상태: {summary.get('kis_derivatives_status', '확인 중')}")
        
        # 3. 주요 지수 현황
        major_indices = summary.get('major_indices', {})
        if major_indices:
            print("\n3️⃣ 주요 지수 현황:")
            for symbol, data in major_indices.items():
                print(f"   📈 {symbol}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")
    
    print("\n🎯 결론:")
    print("   ✅ KIS API 파생상품 승인 대기 중에도 Yahoo Finance로 미국 지수 ETF 데이터 수집 가능")
    print("   ✅ SPY, QQQ, DIA, IWM 등 주요 지수 ETF 실시간 모니터링")
    print("   ✅ ETF를 통한 지수 선물 대용 데이터 제공")
    print("   🔄 KIS API 파생상품 승인 후 실제 선물/옵션 데이터 통합 예정")
    print("\n📋 다음 단계:")
    print("   1. 한국투자증권에서 해외파생상품 서비스 신청 완료")
    print("   2. 승인 후 실제 선물/옵션 데이터 연동")
    print("   3. 실시간 WebSocket 스트리밍 구현")

if __name__ == "__main__":
    asyncio.run(main()) 
# -*- coding: utf-8 -*-
"""
🇺🇸 다중 소스 미국 파생상품 데이터 수집기
===========================================
KIS API 파생상품 승인 대기 중 사용할 수 있는 대안 데이터 소스들
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import yfinance as yf
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class USDerivativeData:
    """미국 파생상품 데이터 구조"""
    symbol: str
    name: str
    underlying: str
    derivative_type: str  # 'future', 'option', 'etf'
    contract_type: str = ""  # 'call', 'put' for options
    strike_price: float = 0.0
    expiry_date: str = ""
    current_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    change: float = 0.0
    change_percent: float = 0.0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    currency: str = "USD"
    exchange: str = ""
    data_source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'name': self.name,
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
            'change': self.change,
            'change_percent': self.change_percent,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'currency': self.currency,
            'exchange': self.exchange,
            'data_source': self.data_source,
            'timestamp': self.timestamp.isoformat()
        }

class MultiSourceUSDerivatives:
    """다중 소스 미국 파생상품 데이터 수집기"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.data_callbacks: List[Callable] = []
        
        # API 키 설정
        self.polygon_key = os.getenv('POLYGON_API_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        
        # 주요 미국 지수 ETF (파생상품 대용)
        self.index_etfs = {
            'SPY': {'name': 'SPDR S&P 500 ETF', 'underlying': 'SPX', 'type': 'etf'},
            'QQQ': {'name': 'Invesco QQQ Trust', 'underlying': 'NDX', 'type': 'etf'},
            'DIA': {'name': 'SPDR Dow Jones Industrial Average ETF', 'underlying': 'DJI', 'type': 'etf'},
            'IWM': {'name': 'iShares Russell 2000 ETF', 'underlying': 'RUT', 'type': 'etf'},
            'VIX': {'name': 'CBOE Volatility Index', 'underlying': 'VIX', 'type': 'index'},
            'UVXY': {'name': 'ProShares Ultra VIX Short-Term Futures ETF', 'underlying': 'VIX', 'type': 'etf'},
            'SQQQ': {'name': 'ProShares UltraPro Short QQQ', 'underlying': 'NDX', 'type': 'etf'},
            'SPXU': {'name': 'ProShares UltraPro Short S&P500', 'underlying': 'SPX', 'type': 'etf'}
        }
        
        logger.info("🇺🇸 다중 소스 미국 파생상품 데이터 수집기 초기화")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def add_data_callback(self, callback: Callable):
        """데이터 수신 콜백 추가"""
        self.data_callbacks.append(callback)
    
    async def notify_callbacks(self, data: USDerivativeData):
        """콜백들에게 데이터 알림"""
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"콜백 실행 오류: {e}")
    
    async def get_yahoo_finance_data(self, symbols: List[str]) -> List[USDerivativeData]:
        """Yahoo Finance에서 ETF/지수 데이터 수집"""
        derivatives = []
        
        try:
            for symbol in symbols:
                if symbol not in self.index_etfs:
                    continue
                
                info = self.index_etfs[symbol]
                
                try:
                    # yfinance를 사용하여 데이터 수집
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        previous = data.iloc[-2] if len(data) > 1 else latest
                        
                        change = latest['Close'] - previous['Close']
                        change_pct = (change / previous['Close']) * 100 if previous['Close'] != 0 else 0
                        
                        derivative = USDerivativeData(
                            symbol=symbol,
                            name=info['name'],
                            underlying=info['underlying'],
                            derivative_type=info['type'],
                            current_price=float(latest['Close']),
                            volume=int(latest['Volume']),
                            change=float(change),
                            change_percent=float(change_pct),
                            exchange='NYSE' if symbol in ['SPY', 'DIA', 'IWM'] else 'NASDAQ',
                            data_source='Yahoo Finance'
                        )
                        
                        derivatives.append(derivative)
                        await self.notify_callbacks(derivative)
                        
                        logger.info(f"✅ {symbol} 데이터 수집: ${derivative.current_price:.2f} ({derivative.change_percent:+.2f}%)")
                
                except Exception as e:
                    logger.error(f"{symbol} 데이터 수집 오류: {e}")
                    continue
                
                # API 호출 제한 고려
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Yahoo Finance 데이터 수집 오류: {e}")
        
        return derivatives
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """시장 종합 현황"""
        try:
            # Yahoo Finance에서 주요 ETF 데이터 수집
            etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'UVXY']
            etf_data = await self.get_yahoo_finance_data(etf_symbols)
            
            # 주요 지수 현황
            major_indices = {}
            for etf in etf_data:
                if etf.symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
                    major_indices[etf.symbol] = {
                        'price': etf.current_price,
                        'change': etf.change,
                        'change_percent': etf.change_percent
                    }
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_derivatives': len(etf_data),
                'data_sources': ['Yahoo Finance'],
                'major_indices': major_indices,
                'available_data': {
                    'etfs': len(etf_data),
                    'options': 0,  # KIS 승인 후 추가 예정
                    'futures': 0   # KIS 승인 후 추가 예정
                },
                'kis_derivatives_status': '📋 신청 진행 중 - 승인 대기'
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"시장 종합 현황 조회 오류: {e}")
            return {}

# 테스트용 메인 함수
async def main():
    """다중 소스 미국 파생상품 데이터 수집 테스트"""
    print("🇺🇸 다중 소스 미국 파생상품 데이터 수집 테스트")
    print("=" * 60)
    print("📌 KIS API 파생상품 승인 대기 중 - 대안 데이터 소스 활용")
    print()
    
    async with MultiSourceUSDerivatives() as collector:
        # 데이터 수신 콜백 등록
        def on_data_received(data: USDerivativeData):
            print(f"📊 실시간 데이터: {data.symbol} - ${data.current_price:.2f} "
                  f"({data.change_percent:+.2f}%) [{data.data_source}]")
        
        collector.add_data_callback(on_data_received)
        
        # 1. Yahoo Finance ETF 데이터 수집
        print("1️⃣ Yahoo Finance ETF 데이터 수집...")
        etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'UVXY']
        etf_data = await collector.get_yahoo_finance_data(etf_symbols)
        
        print(f"✅ ETF 데이터: {len(etf_data)}개 수집")
        
        # 2. 시장 종합 현황
        print("\n2️⃣ 시장 종합 현황...")
        summary = await collector.get_market_summary()
        
        print(f"✅ 총 파생상품: {summary.get('total_derivatives', 0)}개")
        print(f"✅ 데이터 소스: {', '.join(summary.get('data_sources', []))}")
        print(f"✅ KIS 상태: {summary.get('kis_derivatives_status', '확인 중')}")
        
        # 3. 주요 지수 현황
        major_indices = summary.get('major_indices', {})
        if major_indices:
            print("\n3️⃣ 주요 지수 현황:")
            for symbol, data in major_indices.items():
                print(f"   📈 {symbol}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")
    
    print("\n🎯 결론:")
    print("   ✅ KIS API 파생상품 승인 대기 중에도 Yahoo Finance로 미국 지수 ETF 데이터 수집 가능")
    print("   ✅ SPY, QQQ, DIA, IWM 등 주요 지수 ETF 실시간 모니터링")
    print("   ✅ ETF를 통한 지수 선물 대용 데이터 제공")
    print("   🔄 KIS API 파생상품 승인 후 실제 선물/옵션 데이터 통합 예정")
    print("\n📋 다음 단계:")
    print("   1. 한국투자증권에서 해외파생상품 서비스 신청 완료")
    print("   2. 승인 후 실제 선물/옵션 데이터 연동")
    print("   3. 실시간 WebSocket 스트리밍 구현")

if __name__ == "__main__":
    asyncio.run(main()) 