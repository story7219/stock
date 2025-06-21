#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 시장 데이터 제공자
- 실시간 시장 데이터 수집 및 관리
- 다양한 데이터 소스 통합
- 캐싱 및 최적화된 데이터 전달
- v2.5.0 (2024-12-24): NumPy 호환성 개선
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import time
from concurrent.futures import ThreadPoolExecutor
import requests

# NumPy 호환성 처리
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """시장 데이터 구조"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    value: int = 0  # 거래대금
    change: float = 0.0  # 전일대비 변화액
    change_rate: float = 0.0  # 전일대비 변화율
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketIndex:
    """시장 지수 데이터"""
    index_name: str
    current_value: float
    change: float
    change_rate: float
    timestamp: datetime
    volume: int = 0
    high: float = 0.0
    low: float = 0.0

class MarketDataProvider:
    """시장 데이터 통합 제공자"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5분 캐시
        self.last_update = {}
        
        # 데이터 소스 설정
        self.data_sources = {
            'primary': 'kis_api',  # 주 데이터 소스
            'fallback': 'public_api'  # 백업 데이터 소스
        }
        
        logger.info("📊 시장 데이터 제공자 초기화 완료")
    
    async def get_current_market_data(self, symbol: str, use_cache: bool = True) -> Optional[MarketData]:
        """현재 시장 데이터 조회"""
        try:
            # 캐시 확인
            if use_cache and self._is_cache_valid(symbol):
                logger.debug(f"📋 캐시에서 {symbol} 데이터 반환")
                return self.cache[symbol]['data']
            
            # 실시간 데이터 수집
            market_data = await self._fetch_realtime_data(symbol)
            
            # 캐시 업데이트
            if market_data:
                self._update_cache(symbol, market_data)
                logger.debug(f"📊 {symbol} 실시간 데이터 수집 완료")
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 시장 데이터 조회 오류: {e}")
            return None
    
    async def get_multiple_stocks_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """다중 종목 데이터 조회"""
        results = {}
        
        # 병렬 처리로 효율성 증대
        tasks = [self.get_current_market_data(symbol) for symbol in symbols]
        market_data_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, data in zip(symbols, market_data_list):
            if isinstance(data, MarketData):
                results[symbol] = data
            elif isinstance(data, Exception):
                logger.warning(f"⚠️ {symbol} 데이터 수집 오류: {data}")
        
        logger.info(f"📊 다중 종목 데이터 수집 완료: {len(results)}/{len(symbols)}")
        return results
    
    async def get_market_indices(self) -> Dict[str, MarketIndex]:
        """주요 시장 지수 조회"""
        try:
            indices = {}
            
            # 주요 지수 리스트
            index_codes = {
                'KOSPI': '001',
                'KOSDAQ': '101',
                'KPI200': '028'
            }
            
            for index_name, code in index_codes.items():
                index_data = await self._fetch_index_data(index_name, code)
                if index_data:
                    indices[index_name] = index_data
            
            logger.info(f"📈 시장 지수 수집 완료: {len(indices)}개")
            return indices
                    
        except Exception as e:
            logger.error(f"❌ 시장 지수 조회 오류: {e}")
            return {}
    
    async def get_top_stocks_by_volume(self, market: str = 'ALL', limit: int = 20) -> List[MarketData]:
        """거래량 상위 종목 조회"""
        try:
            # 시뮬레이션 데이터 (실제로는 API에서 가져와야 함)
            top_stocks = []
            
            sample_stocks = [
                ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('035420', 'NAVER'),
                ('051910', 'LG화학'), ('006400', '삼성SDI'), ('035720', '카카오'),
                ('028260', '삼성물산'), ('066570', 'LG전자'), ('096770', 'SK이노베이션'),
                ('003550', 'LG'), ('055550', '신한지주'), ('105560', 'KB금융'),
                ('000270', '기아'), ('005380', '현대차'), ('012330', '현대모비스'),
                ('017670', 'SK텔레콤'), ('030200', 'KT'), ('009150', '삼성전기'),
                ('000810', '삼성화재'), ('018260', '삼성에스디에스')
            ]
            
            for symbol, name in sample_stocks[:limit]:
                # 시뮬레이션 데이터 생성
                import random
                
                base_price = random.randint(10000, 100000)
                volume = random.randint(1000000, 10000000)
                change_rate = random.uniform(-5.0, 5.0)
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open_price=base_price * 0.98,
                    high_price=base_price * 1.03,
                    low_price=base_price * 0.97,
                    close_price=base_price,
                    volume=volume,
                    value=volume * base_price,
                    change=base_price * change_rate / 100,
                    change_rate=change_rate,
                    additional_data={'name': name, 'market': 'KOSPI'}
                )
                
                top_stocks.append(market_data)
            
            # 거래량 기준 정렬
            top_stocks.sort(key=lambda x: x.volume, reverse=True)
            
            logger.info(f"📊 거래량 상위 {len(top_stocks)}개 종목 조회 완료")
            return top_stocks
            
        except Exception as e:
            logger.error(f"❌ 거래량 상위 종목 조회 오류: {e}")
            return []
    
    async def get_top_stocks_by_change(self, market: str = 'ALL', limit: int = 20, direction: str = 'up') -> List[MarketData]:
        """등락률 상위/하위 종목 조회"""
        try:
            # 거래량 상위 종목 데이터를 기반으로 등락률 정렬
            all_stocks = await self.get_top_stocks_by_volume(market, limit * 2)
            
            if direction == 'up':
                # 상승률 상위
                sorted_stocks = sorted(all_stocks, key=lambda x: x.change_rate, reverse=True)
            else:
                # 하락률 상위 (절댓값 기준)
                sorted_stocks = sorted(all_stocks, key=lambda x: x.change_rate)
            
            result = sorted_stocks[:limit]
            
            logger.info(f"📊 등락률 {direction} {len(result)}개 종목 조회 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ 등락률 상위 종목 조회 오류: {e}")
            return []

    async def _fetch_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """실시간 데이터 수집 (시뮬레이션)"""
        try:
            # 실제로는 KIS API나 다른 데이터 소스에서 가져와야 함
            # 여기서는 시뮬레이션 데이터 생성
            import random
            
            base_price = random.randint(10000, 100000)
            volume = random.randint(100000, 1000000)
            change_rate = random.uniform(-3.0, 3.0)
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=base_price * 0.99,
                high_price=base_price * 1.02,
                low_price=base_price * 0.98,
                close_price=base_price,
                volume=volume,
                value=volume * base_price,
                change=base_price * change_rate / 100,
                change_rate=change_rate
            )
            
        except Exception as e:
            logger.error(f"❌ {symbol} 실시간 데이터 수집 오류: {e}")
            return None

    async def _fetch_index_data(self, index_name: str, code: str) -> Optional[MarketIndex]:
        """지수 데이터 수집 (시뮬레이션)"""
        try:
            # 실제로는 API에서 가져와야 함
            import random
            
            if index_name == 'KOSPI':
                base_value = random.uniform(2400, 2600)
            elif index_name == 'KOSDAQ':
                base_value = random.uniform(700, 900)
            else:
                base_value = random.uniform(300, 400)
            
            change_rate = random.uniform(-2.0, 2.0)
            change = base_value * change_rate / 100
            
            return MarketIndex(
                index_name=index_name,
                current_value=round(base_value, 2),
                change=round(change, 2),
                change_rate=round(change_rate, 2),
                timestamp=datetime.now(),
                volume=random.randint(1000000, 5000000),
                high=base_value * 1.01,
                low=base_value * 0.99
            )

        except Exception as e:
            logger.error(f"❌ {index_name} 지수 데이터 수집 오류: {e}")
            return None
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """캐시 유효성 확인"""
        if symbol not in self.cache:
            return False
        
        cache_time = self.cache[symbol]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_ttl
    
    def _update_cache(self, symbol: str, data: MarketData) -> None:
        """캐시 업데이트"""
        self.cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """시장 전체 요약 정보"""
        try:
            # 주요 지수
            indices = await self.get_market_indices()
            
            # 거래량 상위 종목
            top_volume = await self.get_top_stocks_by_volume(limit=5)
            
            # 상승률 상위 종목
            top_gainers = await self.get_top_stocks_by_change(limit=5, direction='up')
            
            # 하락률 상위 종목
            top_losers = await self.get_top_stocks_by_change(limit=5, direction='down')
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'indices': {name: {
                    'value': idx.current_value,
                    'change': idx.change,
                    'change_rate': idx.change_rate
                } for name, idx in indices.items()},
                'top_volume': [{
                    'symbol': stock.symbol,
                    'name': stock.additional_data.get('name', stock.symbol),
                    'volume': stock.volume,
                    'price': stock.close_price,
                    'change_rate': stock.change_rate
                } for stock in top_volume],
                'top_gainers': [{
                    'symbol': stock.symbol,
                    'name': stock.additional_data.get('name', stock.symbol),
                    'change_rate': stock.change_rate,
                    'price': stock.close_price
                } for stock in top_gainers],
                'top_losers': [{
                    'symbol': stock.symbol,
                    'name': stock.additional_data.get('name', stock.symbol),
                    'change_rate': stock.change_rate,
                    'price': stock.close_price
                } for stock in top_losers]
            }
            
            logger.info("📊 시장 요약 정보 생성 완료")
            return summary
            
        except Exception as e:
            logger.error(f"❌ 시장 요약 정보 생성 오류: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """캐시 초기화"""
        self.cache.clear()
        logger.info("🗑️ 데이터 캐시 초기화 완료")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        return {
            'cached_symbols': len(self.cache),
            'cache_ttl': self.cache_ttl,
            'oldest_cache': min([v['timestamp'] for v in self.cache.values()]) if self.cache else None,
            'newest_cache': max([v['timestamp'] for v in self.cache.values()]) if self.cache else None
        }

# 테스트 함수
async def test_market_data_provider():
    """시장 데이터 제공자 테스트"""
    logger.info("🧪 시장 데이터 제공자 테스트 시작")
    
    provider = MarketDataProvider()
    
    # 1. 단일 종목 데이터 테스트
    print("\n1. 단일 종목 데이터 테스트")
    data = await provider.get_current_market_data('005930')
    if data:
        print(f"삼성전자: {data.close_price:,}원 ({data.change_rate:+.2f}%)")
    
    # 2. 다중 종목 데이터 테스트
    print("\n2. 다중 종목 데이터 테스트")
    symbols = ['005930', '000660', '035420']
    multi_data = await provider.get_multiple_stocks_data(symbols)
    for symbol, data in multi_data.items():
        print(f"{symbol}: {data.close_price:,}원 ({data.change_rate:+.2f}%)")
    
    # 3. 시장 지수 테스트
    print("\n3. 시장 지수 테스트")
    indices = await provider.get_market_indices()
    for name, index in indices.items():
        print(f"{name}: {index.current_value:.2f} ({index.change_rate:+.2f}%)")
    
    # 4. 거래량 상위 종목 테스트
    print("\n4. 거래량 상위 종목")
    top_volume = await provider.get_top_stocks_by_volume(limit=5)
    for i, stock in enumerate(top_volume, 1):
        name = stock.additional_data.get('name', stock.symbol)
        print(f"{i}. {name}: {stock.volume:,}주")
    
    # 5. 시장 요약 테스트
    print("\n5. 시장 요약")
    summary = await provider.get_market_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 6. 캐시 통계
    print("\n6. 캐시 통계")
    cache_stats = provider.get_cache_stats()
    print(f"캐시된 종목 수: {cache_stats['cached_symbols']}")
    print(f"캐시 TTL: {cache_stats['cache_ttl']}초")

if __name__ == "__main__":
    asyncio.run(test_market_data_provider()) 