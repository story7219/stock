"""
Async data collectors for high-performance data gathering.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..core.cache_manager import CacheManager
from ..core.connection_pool import HTTPConnectionPool
from ..core.async_executor import AsyncExecutor
from ..core.memory_optimizer import MemoryOptimizer
from .models import StockData, MarketData, MarketType, DataStatus, TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Data collection configuration."""
    batch_size: int = 50
    max_concurrent: int = 20
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_ttl: int = 300  # 5 minutes
    enable_caching: bool = True
    enable_memory_optimization: bool = True


class AsyncDataCollector:
    """High-performance async data collector."""
    
    def __init__(self, 
                 config: Optional[CollectionConfig] = None,
                 cache_manager: Optional[CacheManager] = None,
                 connection_pool: Optional[HTTPConnectionPool] = None,
                 async_executor: Optional[AsyncExecutor] = None,
                 memory_optimizer: Optional[MemoryOptimizer] = None):
        """Initialize async data collector."""
        self.config = config or CollectionConfig()
        self.cache_manager = cache_manager
        self.connection_pool = connection_pool
        self.async_executor = async_executor
        self.memory_optimizer = memory_optimizer
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (asyncio.cpu_count() or 1) + 4)
        )
        
        # Market symbol mappings
        self.market_symbols = {
            MarketType.KOSPI200: self._get_kospi200_symbols(),
            MarketType.NASDAQ100: self._get_nasdaq100_symbols(),
            MarketType.SP500: self._get_sp500_symbols()
        }
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }
    
    def _get_kospi200_symbols(self) -> List[str]:
        """Get KOSPI 200 symbols."""
        # This would typically fetch from KRX API or a maintained list
        # For now, returning a sample list
        return [
            "005930.KS",  # Samsung Electronics
            "000660.KS",  # SK Hynix
            "035420.KS",  # NAVER
            "005490.KS",  # POSCO Holdings
            "051910.KS",  # LG Chem
            "006400.KS",  # Samsung SDI
            "035720.KS",  # Kakao
            "207940.KS",  # Samsung Biologics
            "068270.KS",  # Celltrion
            "028260.KS",  # Samsung C&T
            # Add more symbols as needed
        ]
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols."""
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA",
            "AVGO", "ORCL", "COST", "NFLX", "ADBE", "PEP", "ASML", "CSCO",
            "TMUS", "CMCSA", "TXN", "QCOM", "AMD", "INTC", "INTU", "AMAT",
            "HON", "BKNG", "AMGN", "ISRG", "VRTX", "ADP", "SBUX", "GILD",
            "MU", "ADI", "LRCX", "PYPL", "REGN", "ATVI", "MELI", "KLAC",
            "SNPS", "CDNS", "MAR", "ORLY", "CSX", "ABNB", "FTNT", "NXPI",
            "CHTR", "MRVL", "ADSK", "DXCM", "WDAY", "EXC", "LULU", "BIIB",
            "IDXX", "AEP", "FAST", "ROST", "ODFL", "KDP", "VRSK", "XEL",
            "CTSH", "CSGP", "EA", "GEHC", "KHC", "DLTR", "DDOG", "ANSS",
            "CPRT", "PAYX", "FANG", "ON", "MRNA", "SIRI", "TEAM", "CRWD",
            "ZS", "ALGN", "LCID", "SGEN", "MTCH", "WBD", "EBAY", "ENPH",
            "ZM", "RIVN", "DOCU", "ILMN", "JD", "OKTA", "SPLK", "PDD"
        ]
    
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols."""
        # This would typically fetch from a reliable source
        # For now, returning major symbols
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "BRK.B", "UNH", "JNJ",
            "XOM", "JPM", "V", "PG", "HD", "CVX", "MA", "BAC", "ABBV", "PFE",
            "AVGO", "KO", "TMO", "COST", "MRK", "WMT", "CSCO", "ACN", "LLY",
            "DHR", "VZ", "ABT", "ORCL", "CRM", "ADBE", "NKE", "TXN", "NEE",
            "WFC", "RTX", "QCOM", "AMGN", "PM", "T", "HON", "UNP", "IBM",
            "SPGI", "LOW", "INTC", "GS", "CAT", "INTU", "AMD", "AMAT", "C",
            "DE", "BKNG", "PLD", "TJX", "AXP", "BLK", "SYK", "MDLZ", "GILD",
            "ADI", "TMUS", "CB", "LRCX", "MO", "ISRG", "VRTX", "CI", "MMC",
            "SO", "DUK", "BSX", "CVS", "ZTS", "PYPL", "NOW", "EQIX", "SHW",
            "ITW", "AON", "MU", "CL", "CME", "FCX", "HUM", "ATVI", "APD",
            "GE", "PGR", "EL", "KLAC", "SNPS", "ICE", "NSC", "FIS", "USB",
            "MSI", "MCO", "D", "EMR", "DG", "COF", "WM", "GD", "TGT", "F"
        ]
    
    async def collect_market_data(self, 
                                 market_type: MarketType,
                                 symbols: Optional[List[str]] = None) -> MarketData:
        """Collect market data asynchronously."""
        start_time = time.time()
        
        try:
            # Use provided symbols or default market symbols
            target_symbols = symbols or self.market_symbols.get(market_type, [])
            
            if not target_symbols:
                logger.warning(f"No symbols found for market type: {market_type}")
                return MarketData(market_type=market_type, data_status=DataStatus.INVALID)
            
            # Check cache first
            cache_key = f"market_data:{market_type.value}:{hash(tuple(sorted(target_symbols)))}"
            if self.cache_manager and self.config.enable_caching:
                cached_data = await self.cache_manager.get(cache_key)
                if cached_data:
                    self.stats['cache_hits'] += 1
                    logger.info(f"Cache hit for {market_type.value}")
                    return MarketData.from_dict(cached_data)
                self.stats['cache_misses'] += 1
            
            # Collect stock data in batches
            stock_data_list = await self._collect_stocks_batch(target_symbols, market_type)
            
            # Create market data
            market_data = MarketData(
                market_type=market_type,
                stocks=stock_data_list
            )
            
            # Calculate market statistics
            market_data.calculate_market_stats()
            
            # Cache the result
            if self.cache_manager and self.config.enable_caching:
                await self.cache_manager.set(
                    cache_key, 
                    market_data.to_dict(), 
                    ttl=self.config.cache_ttl
                )
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + 
                 execution_time) / self.stats['total_requests']
            )
            
            logger.info(f"Collected {len(stock_data_list)} stocks for {market_type.value} "
                       f"in {execution_time:.2f}s")
            
            return market_data
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Failed to collect market data for {market_type.value}: {e}")
            return MarketData(
                market_type=market_type,
                data_status=DataStatus.INVALID,
                last_updated=datetime.now()
            )
        finally:
            self.stats['total_requests'] += 1
    
    async def _collect_stocks_batch(self, 
                                   symbols: List[str], 
                                   market_type: MarketType) -> List[StockData]:
        """Collect stock data in batches."""
        stock_data_list = []
        
        # Split symbols into batches
        batches = [symbols[i:i + self.config.batch_size] 
                  for i in range(0, len(symbols), self.config.batch_size)]
        
        # Process batches concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_batch(batch_symbols):
            async with semaphore:
                return await self._collect_batch_data(batch_symbols, market_type)
        
        # Execute all batches
        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Collect results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
                continue
            stock_data_list.extend(result)
        
        return stock_data_list
    
    async def _collect_batch_data(self, 
                                 symbols: List[str], 
                                 market_type: MarketType) -> List[StockData]:
        """Collect data for a batch of symbols."""
        if not symbols:
            return []
        
        try:
            # Use thread pool for yfinance operations
            loop = asyncio.get_event_loop()
            stock_data_list = await loop.run_in_executor(
                self.thread_pool,
                self._fetch_yfinance_data,
                symbols,
                market_type
            )
            
            return stock_data_list
            
        except Exception as e:
            logger.error(f"Failed to collect batch data: {e}")
            return []
    
    def _fetch_yfinance_data(self, 
                            symbols: List[str], 
                            market_type: MarketType) -> List[StockData]:
        """Fetch data using yfinance (CPU-bound operation)."""
        stock_data_list = []
        
        try:
            # Download data for all symbols at once
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    
                    # Get basic info
                    info = ticker.info
                    
                    # Get current price data
                    hist = ticker.history(period="2d")
                    if hist.empty:
                        logger.warning(f"No historical data for {symbol}")
                        continue
                    
                    current_data = hist.iloc[-1]
                    previous_data = hist.iloc[-2] if len(hist) > 1 else current_data
                    
                    # Create StockData object
                    stock_data = StockData(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        market_type=market_type,
                        current_price=float(current_data['Close']),
                        open_price=float(current_data['Open']),
                        high_price=float(current_data['High']),
                        low_price=float(current_data['Low']),
                        close_price=float(current_data['Close']),
                        previous_close=float(previous_data['Close']),
                        volume=int(current_data['Volume']),
                        market_cap=info.get('marketCap'),
                        data=hist,
                        info=info
                    )
                    
                    stock_data_list.append(stock_data)
                    
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    continue
            
            return stock_data_list
            
        except Exception as e:
            logger.error(f"Failed to fetch yfinance data: {e}")
            return []
    
    async def collect_kospi200_data(self) -> List[StockData]:
        """Collect KOSPI 200 data."""
        market_data = await self.collect_market_data(MarketType.KOSPI200)
        return market_data.stocks
    
    async def collect_nasdaq100_data(self) -> List[StockData]:
        """Collect NASDAQ 100 data."""
        market_data = await self.collect_market_data(MarketType.NASDAQ100)
        return market_data.stocks
    
    async def collect_sp500_data(self) -> List[StockData]:
        """Collect S&P 500 data."""
        market_data = await self.collect_market_data(MarketType.SP500)
        return market_data.stocks
    
    async def collect_single_stock(self, symbol: str, market_type: MarketType) -> Optional[StockData]:
        """Collect data for a single stock."""
        # Check cache first
        cache_key = f"stock_data:{symbol}"
        if self.cache_manager and self.config.enable_caching:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return StockData.from_dict(cached_data)
        
        # Fetch data
        stock_data_list = await self._collect_batch_data([symbol], market_type)
        
        if stock_data_list:
            stock_data = stock_data_list[0]
            
            # Cache the result
            if self.cache_manager and self.config.enable_caching:
                await self.cache_manager.set(
                    cache_key,
                    stock_data.to_dict(),
                    ttl=self.config.cache_ttl
                )
            
            return stock_data
        
        return None
    
    async def update_technical_indicators(self, stock_data: StockData) -> StockData:
        """Update technical indicators for stock data."""
        if stock_data.data is None or stock_data.data.empty:
            return stock_data
        
        try:
            # Use thread pool for CPU-intensive calculations
            loop = asyncio.get_event_loop()
            indicators = await loop.run_in_executor(
                self.thread_pool,
                self._calculate_technical_indicators,
                stock_data.data
            )
            
            stock_data.technical_indicators = indicators
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators for {stock_data.symbol}: {e}")
            return stock_data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate technical indicators (CPU-bound operation)."""
        try:
            import ta
            
            # Calculate indicators
            indicators = TechnicalIndicators()
            
            # Moving averages
            indicators.sma_20 = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
            indicators.sma_50 = ta.trend.sma_indicator(df['Close'], window=50).iloc[-1] if len(df) >= 50 else None
            indicators.sma_200 = ta.trend.sma_indicator(df['Close'], window=200).iloc[-1] if len(df) >= 200 else None
            indicators.ema_12 = ta.trend.ema_indicator(df['Close'], window=12).iloc[-1]
            indicators.ema_26 = ta.trend.ema_indicator(df['Close'], window=26).iloc[-1]
            
            # Momentum indicators
            indicators.rsi = ta.momentum.rsi(df['Close']).iloc[-1]
            macd_line = ta.trend.macd(df['Close'])
            macd_signal = ta.trend.macd_signal(df['Close'])
            indicators.macd = macd_line.iloc[-1]
            indicators.macd_signal = macd_signal.iloc[-1]
            indicators.macd_histogram = (macd_line - macd_signal).iloc[-1]
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(df['Close'])
            bb_mid = ta.volatility.bollinger_mavg(df['Close'])
            bb_low = ta.volatility.bollinger_lband(df['Close'])
            indicators.bollinger_upper = bb_high.iloc[-1]
            indicators.bollinger_middle = bb_mid.iloc[-1]
            indicators.bollinger_lower = bb_low.iloc[-1]
            
            # ATR
            indicators.atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).iloc[-1]
            
            # Volume indicators
            indicators.volume_sma = ta.volume.volume_sma(df['Close'], df['Volume']).iloc[-1]
            indicators.obv = ta.volume.on_balance_volume(df['Close'], df['Volume']).iloc[-1]
            
            # Trend indicators
            indicators.adx = ta.trend.adx(df['High'], df['Low'], df['Close']).iloc[-1]
            indicators.cci = ta.trend.cci(df['High'], df['Low'], df['Close']).iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return TechnicalIndicators()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset collection statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.connection_pool:
            await self.connection_pool.close()
        
        if self.memory_optimizer:
            self.memory_optimizer.force_cleanup() 