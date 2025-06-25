"""
Data processors for cleaning, transforming, and enriching stock data.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

from ..core.cache_manager import CacheManager
from ..core.memory_optimizer import MemoryOptimizer
from .models import StockData, MarketData, TechnicalIndicators, DataStatus

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    enable_outlier_detection: bool = True
    outlier_threshold: float = 3.0
    enable_missing_data_imputation: bool = True
    enable_data_normalization: bool = True
    enable_feature_engineering: bool = True
    min_data_points: int = 10
    max_missing_ratio: float = 0.3
    enable_caching: bool = True
    cache_ttl: int = 1800  # 30 minutes


class DataProcessor:
    """High-performance data processor with async capabilities."""
    
    def __init__(self,
                 config: Optional[ProcessingConfig] = None,
                 cache_manager: Optional[CacheManager] = None,
                 memory_optimizer: Optional[MemoryOptimizer] = None):
        """Initialize data processor."""
        self.config = config or ProcessingConfig()
        self.cache_manager = cache_manager
        self.memory_optimizer = memory_optimizer
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(16, (asyncio.cpu_count() or 1) + 2)
        )
        
        # Processing statistics
        self.stats = {
            'processed_stocks': 0,
            'cleaned_records': 0,
            'imputed_values': 0,
            'detected_outliers': 0,
            'processing_time': 0.0
        }
    
    async def process_market_data(self, market_data: MarketData) -> MarketData:
        """Process entire market data asynchronously."""
        if not market_data.stocks:
            return market_data
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process stocks in parallel batches
            processed_stocks = await self._process_stocks_batch(market_data.stocks)
            
            # Update market data
            market_data.stocks = processed_stocks
            market_data.data_status = DataStatus.VALID
            market_data.last_updated = datetime.now()
            
            # Recalculate market statistics
            market_data.calculate_market_stats()
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats['processing_time'] += processing_time
            
            logger.info(f"Processed {len(processed_stocks)} stocks in {processing_time:.2f}s")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to process market data: {e}")
            market_data.data_status = DataStatus.INVALID
            return market_data
    
    async def _process_stocks_batch(self, stocks: List[StockData]) -> List[StockData]:
        """Process stocks in batches."""
        batch_size = 20
        batches = [stocks[i:i + batch_size] for i in range(0, len(stocks), batch_size)]
        
        # Process batches concurrently
        semaphore = asyncio.Semaphore(5)  # Limit concurrent batches
        
        async def process_batch(batch):
            async with semaphore:
                return await asyncio.gather(
                    *[self.process_stock_data(stock) for stock in batch],
                    return_exceptions=True
                )
        
        # Execute all batches
        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Collect results
        processed_stocks = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            
            for result in batch_result:
                if isinstance(result, Exception):
                    logger.error(f"Stock processing failed: {result}")
                    continue
                processed_stocks.append(result)
        
        return processed_stocks
    
    async def process_stock_data(self, stock_data: StockData) -> StockData:
        """Process individual stock data."""
        # Check cache first
        cache_key = f"processed_stock:{stock_data.symbol}:{hash(str(stock_data.last_updated))}"
        if self.cache_manager and self.config.enable_caching:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return StockData.from_dict(cached_data)
        
        try:
            # Process in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            processed_stock = await loop.run_in_executor(
                self.thread_pool,
                self._process_stock_sync,
                stock_data
            )
            
            # Cache the result
            if self.cache_manager and self.config.enable_caching:
                await self.cache_manager.set(
                    cache_key,
                    processed_stock.to_dict(),
                    ttl=self.config.cache_ttl
                )
            
            self.stats['processed_stocks'] += 1
            return processed_stock
            
        except Exception as e:
            logger.error(f"Failed to process stock {stock_data.symbol}: {e}")
            return stock_data
    
    def _process_stock_sync(self, stock_data: StockData) -> StockData:
        """Synchronous stock data processing (CPU-bound)."""
        try:
            # Clean price data
            stock_data = self._clean_price_data(stock_data)
            
            # Handle missing data
            if self.config.enable_missing_data_imputation:
                stock_data = self._impute_missing_data(stock_data)
            
            # Detect and handle outliers
            if self.config.enable_outlier_detection:
                stock_data = self._detect_outliers(stock_data)
            
            # Normalize data
            if self.config.enable_data_normalization:
                stock_data = self._normalize_data(stock_data)
            
            # Feature engineering
            if self.config.enable_feature_engineering:
                stock_data = self._engineer_features(stock_data)
            
            # Validate processed data
            stock_data = self._validate_processed_data(stock_data)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Sync processing failed for {stock_data.symbol}: {e}")
            return stock_data
    
    def _clean_price_data(self, stock_data: StockData) -> StockData:
        """Clean price data for inconsistencies."""
        if stock_data.data is None or stock_data.data.empty:
            return stock_data
        
        try:
            df = stock_data.data.copy()
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Ensure positive prices
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = df[col].clip(lower=0.01)  # Minimum price
            
            # Ensure High >= Low
            if 'High' in df.columns and 'Low' in df.columns:
                # Fix inverted high/low
                mask = df['High'] < df['Low']
                df.loc[mask, ['High', 'Low']] = df.loc[mask, ['Low', 'High']].values
            
            # Ensure volume is non-negative
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].clip(lower=0)
            
            # Remove duplicate timestamps
            df = df[~df.index.duplicated(keep='last')]
            
            # Sort by date
            df = df.sort_index()
            
            stock_data.data = df
            self.stats['cleaned_records'] += len(df)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to clean price data for {stock_data.symbol}: {e}")
            return stock_data
    
    def _impute_missing_data(self, stock_data: StockData) -> StockData:
        """Impute missing data using various strategies."""
        if stock_data.data is None or stock_data.data.empty:
            return stock_data
        
        try:
            df = stock_data.data.copy()
            original_missing = df.isnull().sum().sum()
            
            # Price columns - forward fill then backward fill
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Volume - use median imputation
            if 'Volume' in df.columns:
                median_volume = df['Volume'].median()
                df['Volume'] = df['Volume'].fillna(median_volume)
            
            # For remaining missing values, use interpolation
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # Final cleanup - drop rows with too many missing values
            missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
            df = df[missing_ratio <= self.config.max_missing_ratio]
            
            final_missing = df.isnull().sum().sum()
            imputed_count = original_missing - final_missing
            
            stock_data.data = df
            self.stats['imputed_values'] += imputed_count
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to impute missing data for {stock_data.symbol}: {e}")
            return stock_data
    
    def _detect_outliers(self, stock_data: StockData) -> StockData:
        """Detect and handle outliers using statistical methods."""
        if stock_data.data is None or stock_data.data.empty:
            return stock_data
        
        try:
            df = stock_data.data.copy()
            outlier_count = 0
            
            # Use IQR method for outlier detection
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in numeric_columns:
                if col not in df.columns or df[col].isnull().all():
                    continue
                
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                # Identify outliers
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count += outlier_mask.sum()
                
                # Handle outliers by capping
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
            
            stock_data.data = df
            self.stats['detected_outliers'] += outlier_count
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to detect outliers for {stock_data.symbol}: {e}")
            return stock_data
    
    def _normalize_data(self, stock_data: StockData) -> StockData:
        """Normalize data for better analysis."""
        if stock_data.data is None or stock_data.data.empty:
            return stock_data
        
        try:
            df = stock_data.data.copy()
            
            # Calculate returns
            if 'Close' in df.columns:
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Calculate price ranges
            if all(col in df.columns for col in ['High', 'Low']):
                df['Price_Range'] = df['High'] - df['Low']
                df['Price_Range_Pct'] = df['Price_Range'] / df['Close'] * 100
            
            # Normalize volume
            if 'Volume' in df.columns:
                df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            stock_data.data = df
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to normalize data for {stock_data.symbol}: {e}")
            return stock_data
    
    def _engineer_features(self, stock_data: StockData) -> StockData:
        """Engineer additional features for analysis."""
        if stock_data.data is None or stock_data.data.empty:
            return stock_data
        
        try:
            df = stock_data.data.copy()
            
            # Price momentum features
            if 'Close' in df.columns:
                # Moving averages
                for window in [5, 10, 20, 50]:
                    if len(df) >= window:
                        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                        df[f'Price_vs_MA_{window}'] = df['Close'] / df[f'MA_{window}'] - 1
                
                # Price acceleration
                df['Price_Velocity'] = df['Close'].diff()
                df['Price_Acceleration'] = df['Price_Velocity'].diff()
            
            # Volatility features
            if 'Returns' in df.columns:
                # Rolling volatility
                for window in [10, 20, 50]:
                    if len(df) >= window:
                        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
                
                # Volatility regime
                if len(df) >= 20:
                    vol_20 = df['Returns'].rolling(window=20).std()
                    vol_50 = df['Returns'].rolling(window=50).std()
                    df['Vol_Regime'] = vol_20 / vol_50
            
            # Volume features
            if 'Volume' in df.columns:
                # Volume trend
                df['Volume_Trend'] = df['Volume'].rolling(window=10).mean() / df['Volume'].rolling(window=30).mean()
                
                # Volume spike detection
                vol_mean = df['Volume'].rolling(window=20).mean()
                vol_std = df['Volume'].rolling(window=20).std()
                df['Volume_Spike'] = (df['Volume'] - vol_mean) / vol_std
            
            # Market structure features
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                # Support and resistance levels
                df['High_20'] = df['High'].rolling(window=20).max()
                df['Low_20'] = df['Low'].rolling(window=20).min()
                df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
            
            stock_data.data = df
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to engineer features for {stock_data.symbol}: {e}")
            return stock_data
    
    def _validate_processed_data(self, stock_data: StockData) -> StockData:
        """Validate processed data quality."""
        if stock_data.data is None or stock_data.data.empty:
            stock_data.data_status = DataStatus.INVALID
            return stock_data
        
        try:
            df = stock_data.data
            
            # Check minimum data points
            if len(df) < self.config.min_data_points:
                stock_data.data_status = DataStatus.INVALID
                return stock_data
            
            # Check for excessive missing data
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_ratio > self.config.max_missing_ratio:
                stock_data.data_status = DataStatus.PARTIAL
            else:
                stock_data.data_status = DataStatus.VALID
            
            # Update metadata
            stock_data.last_updated = datetime.now()
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to validate data for {stock_data.symbol}: {e}")
            stock_data.data_status = DataStatus.INVALID
            return stock_data
    
    async def calculate_correlations(self, stocks: List[StockData]) -> pd.DataFrame:
        """Calculate correlation matrix between stocks."""
        try:
            # Extract returns data
            returns_data = {}
            for stock in stocks:
                if (stock.data is not None and 
                    not stock.data.empty and 
                    'Returns' in stock.data.columns):
                    returns_data[stock.symbol] = stock.data['Returns'].dropna()
            
            if not returns_data:
                return pd.DataFrame()
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            loop = asyncio.get_event_loop()
            correlation_matrix = await loop.run_in_executor(
                self.thread_pool,
                lambda: returns_df.corr()
            )
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")
            return pd.DataFrame()
    
    async def detect_anomalies(self, stock_data: StockData) -> Dict[str, Any]:
        """Detect anomalies in stock data."""
        if stock_data.data is None or stock_data.data.empty:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            anomalies = await loop.run_in_executor(
                self.thread_pool,
                self._detect_anomalies_sync,
                stock_data
            )
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies for {stock_data.symbol}: {e}")
            return {}
    
    def _detect_anomalies_sync(self, stock_data: StockData) -> Dict[str, Any]:
        """Synchronous anomaly detection."""
        anomalies = {
            'price_gaps': [],
            'volume_spikes': [],
            'unusual_patterns': []
        }
        
        try:
            df = stock_data.data
            
            # Detect price gaps
            if 'Close' in df.columns and len(df) > 1:
                price_changes = df['Close'].pct_change().abs()
                gap_threshold = price_changes.quantile(0.95)
                gap_mask = price_changes > gap_threshold
                anomalies['price_gaps'] = df.index[gap_mask].tolist()
            
            # Detect volume spikes
            if 'Volume' in df.columns and len(df) > 20:
                vol_mean = df['Volume'].rolling(window=20).mean()
                vol_std = df['Volume'].rolling(window=20).std()
                z_scores = (df['Volume'] - vol_mean) / vol_std
                spike_mask = z_scores.abs() > 3
                anomalies['volume_spikes'] = df.index[spike_mask].tolist()
            
            # Detect unusual price patterns
            if 'Returns' in df.columns and len(df) > 50:
                returns = df['Returns'].dropna()
                unusual_threshold = returns.std() * 2.5
                unusual_mask = returns.abs() > unusual_threshold
                anomalies['unusual_patterns'] = returns.index[unusual_mask].tolist()
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Sync anomaly detection failed: {e}")
            return anomalies
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'processed_stocks': 0,
            'cleaned_records': 0,
            'imputed_values': 0,
            'detected_outliers': 0,
            'processing_time': 0.0
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True) 