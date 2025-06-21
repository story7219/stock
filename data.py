#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 최적화된 데이터 처리 모듈
- 비동기 데이터 수집
- 벡터화 계산
- 메모리 효율적 처리
- 안정성 우선 설계
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import time
from performance_core import get_performance_core, performance_monitor

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """주식 데이터 구조체 (메모리 최적화)"""
    symbol: str
    name: str = ""
    current_price: float = 0.0
    change_rate: float = 0.0
    volume: int = 0
    market_cap: int = 0
    
    # 기술적 지표 (핵심만)
    rsi: float = 0.0
    macd: float = 0.0
    bb_position: float = 0.0  # 볼린저 밴드 위치
    ma_trend: str = "NEUTRAL"  # 이동평균 추세
    
    # 재무 지표 (핵심만)
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    roe: float = 0.0
    debt_ratio: float = 0.0
    
    # 메타데이터
    last_updated: datetime = field(default_factory=datetime.now)
    data_quality: str = "GOOD"  # GOOD, PARTIAL, POOR

@dataclass
class AnalysisResult:
    """분석 결과 구조체"""
    stock_data: StockData
    score: float
    recommendation: str
    reason: str
    strategy: str
    confidence: float = 0.0

class OptimizedStockDataFetcher:
    """🚀 최적화된 주식 데이터 수집기"""
    
    def __init__(self):
        self.core = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # 나스닥100 & S&P500 심볼 캐시
        self._nasdaq100_symbols = None
        self._sp500_symbols = None
        
        logger.info("✅ 최적화된 주식 데이터 수집기 초기화")
    
    async def initialize(self):
        """초기화"""
        self.core = await get_performance_core()
        
        # 심볼 목록 미리 로드
        await self._load_symbols()
        
        logger.info("✅ 데이터 수집기 초기화 완료")
    
    async def _load_symbols(self):
        """심볼 목록 미리 로드 및 캐시"""
        try:
            # 캐시에서 확인
            nasdaq_cache_key = "nasdaq100_symbols"
            sp500_cache_key = "sp500_symbols"
            
            self._nasdaq100_symbols = self.core.cache.get(nasdaq_cache_key)
            self._sp500_symbols = self.core.cache.get(sp500_cache_key)
            
            if not self._nasdaq100_symbols or not self._sp500_symbols:
                logger.info("📋 주식 심볼 목록 로딩 중...")
                
                # 나스닥100 심볼 (실제 운영에서는 API에서 가져오지만, 여기서는 대표적인 것들)
                self._nasdaq100_symbols = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                    "ADBE", "CRM", "INTC", "CSCO", "CMCSA", "PEP", "COST", "TMUS",
                    "AVGO", "TXN", "QCOM", "CHTR", "GILD", "FISV", "BKNG", "REGN",
                    "ADP", "ISRG", "MDLZ", "VRTX", "CSX", "SBUX", "ATVI", "ILMN",
                    "AMD", "MU", "AMAT", "LRCX", "ADI", "KLAC", "MRVL", "SNPS",
                    "CDNS", "ORLY", "WDAY", "CTAS", "FAST", "PAYX", "VRSK", "DXCM",
                    "BIIB", "DLTR", "ROST", "KDP", "SGEN", "MCHP", "CERN", "TEAM",
                    "IDXX", "ANSS", "ALGN", "CRWD", "MRNA", "ZM", "NTES", "DOCU",
                    "OKTA", "SPLK", "DDOG", "PTON", "ZS", "SNOW", "NET", "FTNT",
                    "PANW", "VRSN", "TTWO", "EA", "EBAY", "PYPL", "WBA", "SIRI",
                    "FOXA", "FOX", "DISH", "BMRN", "XLNX", "LULU", "MNST", "ASML",
                    "JD", "PDD", "BIDU", "NXPI", "MELI", "CPRT", "PCAR", "LCID",
                    "RIVN", "HOOD", "RBLX", "U"  # 최근 추가된 종목들
                ]
                
                # S&P500 심볼 (대표적인 것들, 실제로는 500개)
                self._sp500_symbols = [
                    # 기술주
                    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA",
                    "CRM", "ORCL", "ADBE", "NFLX", "INTC", "CSCO", "IBM", "QCOM",
                    
                    # 금융주
                    "BRK.B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW", "BLK",
                    "SPGI", "CME", "ICE", "MCO", "MSCI", "TRV", "AIG", "MET", "PRU", "AFL",
                    
                    # 헬스케어
                    "JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "DHR", "MRK", "BMY", "AMGN",
                    "GILD", "REGN", "VRTX", "BIIB", "ILMN", "MRNA", "ZTS", "ELV", "CVS", "CI",
                    
                    # 소비재
                    "AMZN", "TSLA", "HD", "MCD", "DIS", "NKE", "SBUX", "LOW", "TJX", "BKNG",
                    "COST", "WMT", "TGT", "KO", "PEP", "PG", "UL", "CL", "KMB", "GIS",
                    
                    # 산업재
                    "BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE", "EMR",
                    "ITW", "CSX", "UNP", "FDX", "NSC", "LUV", "DAL", "AAL", "UAL", "JBHT",
                    
                    # 에너지
                    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "KMI", "OKE",
                    "WMB", "EPD", "ET", "MPLX", "BKR", "HAL", "DVN", "FANG", "MRO", "APA",
                    
                    # 유틸리티
                    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
                    "FE", "ETR", "ES", "AWK", "DTE", "PPL", "CMS", "CNP", "NI", "LNT",
                    
                    # 부동산
                    "AMT", "CCI", "EQIX", "PLD", "WELL", "SPG", "EQR", "AVB", "DLR", "BXP",
                    "VTR", "ARE", "MAA", "UDR", "CPT", "HST", "REG", "FRT", "KIM", "AIV",
                    
                    # 소재
                    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "CTVA", "DD", "DOW", "PPG",
                    "NUE", "STLD", "VMC", "MLM", "PKG", "IP", "CF", "MOS", "FMC", "ALB"
                ]
                
                # 중복 제거
                self._nasdaq100_symbols = list(set(self._nasdaq100_symbols))
                self._sp500_symbols = list(set(self._sp500_symbols))
                
                # 캐시에 저장 (24시간)
                self.core.cache.set(nasdaq_cache_key, self._nasdaq100_symbols, ttl=86400)
                self.core.cache.set(sp500_cache_key, self._sp500_symbols, ttl=86400)
                
                logger.info(f"📋 심볼 로딩 완료: 나스닥100({len(self._nasdaq100_symbols)}), S&P500({len(self._sp500_symbols)})")
            
        except Exception as e:
            logger.error(f"❌ 심볼 로딩 실패: {e}")
            # 기본값 설정
            self._nasdaq100_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            self._sp500_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    @performance_monitor
    async def fetch_nasdaq100_data(self) -> List[StockData]:
        """나스닥100 데이터 수집 (병렬 처리)"""
        return await self._fetch_symbols_data(self._nasdaq100_symbols, "NASDAQ100")
    
    @performance_monitor
    async def fetch_sp500_data(self) -> List[StockData]:
        """S&P500 데이터 수집 (병렬 처리)"""
        return await self._fetch_symbols_data(self._sp500_symbols, "SP500")
    
    async def _fetch_symbols_data(self, symbols: List[str], index_name: str) -> List[StockData]:
        """심볼 목록 데이터 수집 (최적화된 병렬 처리)"""
        try:
            logger.info(f"📊 {index_name} 데이터 수집 시작 ({len(symbols)}개 종목)")
            
            # 배치 단위로 처리
            batch_size = 20
            all_results = []
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.info(f"📦 배치 {i//batch_size + 1} 처리 중... ({len(batch_symbols)}개 종목)")
                
                # 병렬 처리
                tasks = [self._fetch_single_stock_data(symbol) for symbol in batch_symbols]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 성공한 결과만 수집
                valid_results = [
                    result for result in batch_results 
                    if isinstance(result, StockData) and result.data_quality != "POOR"
                ]
                
                all_results.extend(valid_results)
                
                # 배치 간 짧은 대기 (API 제한 회피)
                if i + batch_size < len(symbols):
                    await asyncio.sleep(0.2)
            
            logger.info(f"✅ {index_name} 데이터 수집 완료: {len(all_results)}/{len(symbols)} 성공")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ {index_name} 데이터 수집 실패: {e}")
            return []
    
    async def _fetch_single_stock_data(self, symbol: str) -> Optional[StockData]:
        """단일 종목 데이터 수집 (캐시 활용)"""
        try:
            # 캐시 확인
            cache_key = f"stock_data_{symbol}"
            cached_data = self.core.cache.get(cache_key)
            
            if cached_data:
                logger.debug(f"📋 캐시 사용: {symbol}")
                return cached_data
            
            # 새로운 데이터 수집
            stock_data = await self._get_yfinance_data(symbol)
            
            if stock_data and stock_data.data_quality != "POOR":
                # 캐시에 저장 (5분)
                self.core.cache.set(cache_key, stock_data, ttl=300)
                logger.debug(f"💾 데이터 캐시 저장: {symbol}")
            
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 수집 실패: {e}")
            return None
    
    async def _get_yfinance_data(self, symbol: str) -> Optional[StockData]:
        """yfinance를 통한 데이터 수집 (스레드 풀 사용)"""
        try:
            # CPU 집약적 작업을 스레드 풀에서 실행
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.thread_pool, 
                self._sync_get_yfinance_data, 
                symbol
            )
            
            return data
            
        except Exception as e:
            logger.error(f"❌ {symbol} yfinance 데이터 수집 실패: {e}")
            return None
    
    def _sync_get_yfinance_data(self, symbol: str) -> Optional[StockData]:
        """동기 yfinance 데이터 수집"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")  # 6개월 데이터로 단축
            
            if hist.empty or len(hist) < 10:
                logger.warning(f"⚠️ {symbol} 히스토리 데이터 부족")
                return StockData(symbol=symbol, data_quality="POOR")
            
            # 기본 정보
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            previous_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
            change_rate = ((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0
            
            # 기술적 지표 계산 (최적화된 버전)
            rsi = self._calculate_rsi_fast(hist['Close'])
            macd = self._calculate_macd_fast(hist['Close'])
            bb_position = self._calculate_bb_position_fast(hist['Close'])
            ma_trend = self._calculate_ma_trend_fast(hist['Close'])
            
            # 재무 지표
            pe_ratio = info.get('trailingPE', 0) or 0
            pb_ratio = info.get('priceToBook', 0) or 0
            roe = info.get('returnOnEquity', 0) or 0
            debt_ratio = info.get('debtToEquity', 0) or 0
            
            stock_data = StockData(
                symbol=symbol,
                name=info.get('longName', symbol),
                current_price=current_price,
                change_rate=change_rate,
                volume=info.get('volume', 0) or 0,
                market_cap=info.get('marketCap', 0) or 0,
                rsi=rsi,
                macd=macd,
                bb_position=bb_position,
                ma_trend=ma_trend,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                roe=roe,
                debt_ratio=debt_ratio,
                data_quality="GOOD" if all([current_price, change_rate is not None]) else "PARTIAL"
            )
            
            # 메모리 최적화
            self.core.memory_optimizer.track_allocation()
            
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 동기 데이터 수집 실패: {e}")
            return StockData(symbol=symbol, data_quality="POOR")
    
    def _calculate_rsi_fast(self, prices: pd.Series, period: int = 14) -> float:
        """빠른 RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_macd_fast(self, prices: pd.Series) -> float:
        """빠른 MACD 계산"""
        try:
            if len(prices) < 26:
                return 0.0
            
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            
            return macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_bb_position_fast(self, prices: pd.Series, period: int = 20) -> float:
        """빠른 볼린저 밴드 위치 계산"""
        try:
            if len(prices) < period:
                return 0.5
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            current_price = prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if pd.isna(current_upper) or pd.isna(current_lower):
                return 0.5
            
            # 0~1 사이의 위치 (0: 하단, 1: 상단)
            position = (current_price - current_lower) / (current_upper - current_lower)
            return max(0, min(1, position))
            
        except Exception:
            return 0.5
    
    def _calculate_ma_trend_fast(self, prices: pd.Series) -> str:
        """빠른 이동평균 추세 계산"""
        try:
            if len(prices) < 50:
                return "NEUTRAL"
            
            ma20 = prices.rolling(window=20).mean().iloc[-1]
            ma50 = prices.rolling(window=50).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            if pd.isna(ma20) or pd.isna(ma50):
                return "NEUTRAL"
            
            if current_price > ma20 > ma50:
                return "BULLISH"
            elif current_price < ma20 < ma50:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception:
            return "NEUTRAL"
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("✅ 데이터 수집기 정리 완료")
        except Exception as e:
            logger.error(f"❌ 데이터 수집기 정리 중 오류: {e}")

if __name__ == "__main__":
    async def test_data_fetcher():
        """데이터 수집기 테스트"""
        print("🧪 최적화된 데이터 수집기 테스트 시작...")
        
        fetcher = OptimizedStockDataFetcher()
        await fetcher.initialize()
        
        # 테스트용 소규모 데이터 수집
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        start_time = time.time()
        results = await fetcher._fetch_symbols_data(test_symbols, "TEST")
        end_time = time.time()
        
        print(f"⏱️ 수집 시간: {end_time - start_time:.2f}초")
        print(f"📊 수집 결과: {len(results)}개 종목")
        
        for stock in results[:3]:  # 상위 3개만 출력
            print(f"  - {stock.symbol}: ${stock.current_price:.2f} ({stock.change_rate:+.2f}%)")
        
        await fetcher.cleanup()
        print("✅ 테스트 완료!")
    
    asyncio.run(test_data_fetcher()) 