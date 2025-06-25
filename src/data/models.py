"""
Data models for the investment system.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class MarketType(Enum):
    """Market type enumeration."""
    KOSPI200 = "KOSPI200"
    NASDAQ100 = "NASDAQ100"
    SP500 = "SP500"


class DataStatus(Enum):
    """Data status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    STALE = "stale"


@dataclass
class TechnicalIndicators:
    """Technical indicators for a stock."""
    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None
    
    # Volume indicators
    volume_sma: Optional[float] = None
    obv: Optional[float] = None
    
    # Trend indicators
    adx: Optional[float] = None
    cci: Optional[float] = None
    
    # Support/Resistance
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    
    # Custom indicators
    momentum_score: Optional[float] = None
    volatility_score: Optional[float] = None
    trend_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def is_valid(self) -> bool:
        """Check if indicators are valid."""
        return any(v is not None for v in self.__dict__.values())


@dataclass
class StockData:
    """Stock data container."""
    symbol: str
    name: str
    market_type: MarketType
    
    # Price data
    current_price: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    previous_close: Optional[float] = None
    
    # Volume and market cap
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    market_cap: Optional[float] = None
    
    # Price changes
    change_amount: Optional[float] = None
    change_percent: Optional[float] = None
    
    # Historical data
    data: Optional[pd.DataFrame] = None
    
    # Technical indicators
    technical_indicators: Optional[TechnicalIndicators] = None
    
    # Fundamental data (if needed)
    info: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    last_updated: Optional[datetime] = None
    data_status: DataStatus = DataStatus.VALID
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.last_updated is None:
            self.last_updated = datetime.now()
        
        # Calculate change if not provided
        if (self.change_amount is None and 
            self.current_price is not None and 
            self.previous_close is not None):
            self.change_amount = self.current_price - self.previous_close
        
        if (self.change_percent is None and 
            self.change_amount is not None and 
            self.previous_close is not None and 
            self.previous_close != 0):
            self.change_percent = (self.change_amount / self.previous_close) * 100
    
    def is_valid(self) -> bool:
        """Check if stock data is valid."""
        return (self.data_status == DataStatus.VALID and
                self.current_price is not None and
                self.current_price > 0)
    
    def is_stale(self, max_age_minutes: int = 60) -> bool:
        """Check if data is stale."""
        if self.last_updated is None:
            return True
        
        age = (datetime.now() - self.last_updated).total_seconds() / 60
        return age > max_age_minutes
    
    def get_price_data(self) -> Dict[str, float]:
        """Get price data as dictionary."""
        return {
            'current_price': self.current_price,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'previous_close': self.previous_close,
            'change_amount': self.change_amount,
            'change_percent': self.change_percent
        }
    
    def get_volume_data(self) -> Dict[str, Union[int, float]]:
        """Get volume data as dictionary."""
        return {
            'volume': self.volume,
            'avg_volume': self.avg_volume,
            'market_cap': self.market_cap
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'symbol': self.symbol,
            'name': self.name,
            'market_type': self.market_type.value,
            'current_price': self.current_price,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'previous_close': self.previous_close,
            'volume': self.volume,
            'avg_volume': self.avg_volume,
            'market_cap': self.market_cap,
            'change_amount': self.change_amount,
            'change_percent': self.change_percent,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'data_status': self.data_status.value,
            'error_message': self.error_message
        }
        
        # Add technical indicators if available
        if self.technical_indicators:
            result['technical_indicators'] = self.technical_indicators.to_dict()
        
        # Add info
        if self.info:
            result['info'] = self.info
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockData':
        """Create from dictionary."""
        # Extract technical indicators
        technical_indicators = None
        if 'technical_indicators' in data:
            technical_indicators = TechnicalIndicators(**data['technical_indicators'])
        
        # Parse datetime
        last_updated = None
        if data.get('last_updated'):
            last_updated = datetime.fromisoformat(data['last_updated'])
        
        return cls(
            symbol=data['symbol'],
            name=data['name'],
            market_type=MarketType(data['market_type']),
            current_price=data.get('current_price'),
            open_price=data.get('open_price'),
            high_price=data.get('high_price'),
            low_price=data.get('low_price'),
            close_price=data.get('close_price'),
            previous_close=data.get('previous_close'),
            volume=data.get('volume'),
            avg_volume=data.get('avg_volume'),
            market_cap=data.get('market_cap'),
            change_amount=data.get('change_amount'),
            change_percent=data.get('change_percent'),
            technical_indicators=technical_indicators,
            info=data.get('info', {}),
            last_updated=last_updated,
            data_status=DataStatus(data.get('data_status', 'valid')),
            error_message=data.get('error_message')
        )


@dataclass
class MarketData:
    """Market-wide data container."""
    market_type: MarketType
    stocks: List[StockData] = field(default_factory=list)
    
    # Market indices
    index_value: Optional[float] = None
    index_change: Optional[float] = None
    index_change_percent: Optional[float] = None
    
    # Market statistics
    total_volume: Optional[int] = None
    advancing_stocks: Optional[int] = None
    declining_stocks: Optional[int] = None
    unchanged_stocks: Optional[int] = None
    
    # Market sentiment
    market_sentiment: Optional[str] = None
    volatility_index: Optional[float] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    data_status: DataStatus = DataStatus.VALID
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def add_stock(self, stock: StockData) -> None:
        """Add stock to market data."""
        self.stocks.append(stock)
    
    def get_valid_stocks(self) -> List[StockData]:
        """Get only valid stocks."""
        return [stock for stock in self.stocks if stock.is_valid()]
    
    def get_stocks_by_symbol(self, symbols: List[str]) -> List[StockData]:
        """Get stocks by symbols."""
        symbol_set = set(symbols)
        return [stock for stock in self.stocks if stock.symbol in symbol_set]
    
    def calculate_market_stats(self) -> None:
        """Calculate market statistics."""
        valid_stocks = self.get_valid_stocks()
        
        if not valid_stocks:
            return
        
        # Calculate advancing/declining stocks
        self.advancing_stocks = sum(1 for stock in valid_stocks 
                                  if stock.change_percent and stock.change_percent > 0)
        self.declining_stocks = sum(1 for stock in valid_stocks 
                                  if stock.change_percent and stock.change_percent < 0)
        self.unchanged_stocks = sum(1 for stock in valid_stocks 
                                  if stock.change_percent and stock.change_percent == 0)
        
        # Calculate total volume
        volumes = [stock.volume for stock in valid_stocks if stock.volume]
        self.total_volume = sum(volumes) if volumes else None
        
        # Determine market sentiment
        if self.advancing_stocks and self.declining_stocks:
            ratio = self.advancing_stocks / (self.advancing_stocks + self.declining_stocks)
            if ratio > 0.6:
                self.market_sentiment = "bullish"
            elif ratio < 0.4:
                self.market_sentiment = "bearish"
            else:
                self.market_sentiment = "neutral"
    
    def get_top_performers(self, n: int = 10) -> List[StockData]:
        """Get top performing stocks."""
        valid_stocks = self.get_valid_stocks()
        return sorted(valid_stocks, 
                     key=lambda x: x.change_percent or 0, 
                     reverse=True)[:n]
    
    def get_worst_performers(self, n: int = 10) -> List[StockData]:
        """Get worst performing stocks."""
        valid_stocks = self.get_valid_stocks()
        return sorted(valid_stocks, 
                     key=lambda x: x.change_percent or 0)[:n]
    
    def get_most_active(self, n: int = 10) -> List[StockData]:
        """Get most active stocks by volume."""
        valid_stocks = [stock for stock in self.get_valid_stocks() if stock.volume]
        return sorted(valid_stocks, 
                     key=lambda x: x.volume, 
                     reverse=True)[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'market_type': self.market_type.value,
            'stocks': [stock.to_dict() for stock in self.stocks],
            'index_value': self.index_value,
            'index_change': self.index_change,
            'index_change_percent': self.index_change_percent,
            'total_volume': self.total_volume,
            'advancing_stocks': self.advancing_stocks,
            'declining_stocks': self.declining_stocks,
            'unchanged_stocks': self.unchanged_stocks,
            'market_sentiment': self.market_sentiment,
            'volatility_index': self.volatility_index,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'data_status': self.data_status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary."""
        # Parse stocks
        stocks = []
        for stock_data in data.get('stocks', []):
            stocks.append(StockData.from_dict(stock_data))
        
        # Parse datetime
        last_updated = None
        if data.get('last_updated'):
            last_updated = datetime.fromisoformat(data['last_updated'])
        
        return cls(
            market_type=MarketType(data['market_type']),
            stocks=stocks,
            index_value=data.get('index_value'),
            index_change=data.get('index_change'),
            index_change_percent=data.get('index_change_percent'),
            total_volume=data.get('total_volume'),
            advancing_stocks=data.get('advancing_stocks'),
            declining_stocks=data.get('declining_stocks'),
            unchanged_stocks=data.get('unchanged_stocks'),
            market_sentiment=data.get('market_sentiment'),
            volatility_index=data.get('volatility_index'),
            last_updated=last_updated,
            data_status=DataStatus(data.get('data_status', 'valid'))
        )


@dataclass
class StrategyResult:
    """Strategy analysis result."""
    stock: StockData
    strategy_name: str
    score: float
    rank: int
    reasoning: str
    confidence: float = 0.0
    
    # Detailed scores
    technical_score: Optional[float] = None
    momentum_score: Optional[float] = None
    volatility_score: Optional[float] = None
    volume_score: Optional[float] = None
    
    # Additional metadata
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.stock.symbol,
            'name': self.stock.name,
            'strategy_name': self.strategy_name,
            'score': self.score,
            'rank': self.rank,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'technical_score': self.technical_score,
            'momentum_score': self.momentum_score,
            'volatility_score': self.volatility_score,
            'volume_score': self.volume_score,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        } 