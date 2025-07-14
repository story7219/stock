#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: src/backtesting/core/data_handler.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import datetime
from typing import Any
import Dict, List, Optional, Tuple, Union, Deque
from ..events import MarketEvent
import pandas_market_calendars as mcal

class DataHandler:
    def __init__(self, data_path: str, config: Dict[str, Any]):
        self.data_path = data_path
        self.config = config
        self.data = self._load_and_filter_data()
        self.event_stream = self._create_event_stream()

    def _load_and_filter_data(self) -> pd.DataFrame:
        """
        Loads data and filters it to include only valid KRX trading days.
        """
        df = pd.read_parquet(self.data_path)
        if 'datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('datetime')

        if df.empty:
            self.lookup_data = df
            return df

        # Get KRX market calendar
        krx = mcal.get_calendar('KRX')
        start_date = pd.to_datetime(df.index.min()).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(df.index.max()).strftime('%Y-%m-%d')
        schedule = krx.schedule(start_date=start_date, end_date=end_date)

        # Filter the dataframe to only include valid trading days
        valid_trading_days = schedule.index.tz_localize(None) # Remove timezone info for comparison
        df = df[df.index.isin(valid_trading_days)]

        # Create a multi-index version for quick lookups
        self.lookup_data = df.set_index(['symbol', df.index])
        return df

    def _create_event_stream(self):
        # Create a generator for market events from the dataframe
        for index, row in self.data.iterrows():
            event_data = {
                'price': row.get('close', 0),
                'open': row.get('open', 0),
                'high': row.get('high', 0),
                'low': row.get('low', 0),
                'volume': row.get('volume', 0),
                'volatility': row.get('volatility', 0.02) # Example
            }
            yield MarketEvent(
                symbol=row.get('symbol', 'UNKNOWN'),
                datetime=index,
                data=event_data
            )

    def load_events_into_queue(self, event_queue: Deque[MarketEvent]):
        """
        Loads all historical data as MarketEvents into the event queue.
        """
        for event in self.event_stream:
            event_queue.append(event)

    def get_latest_data(self, symbol: str, dt: datetime) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest market data for a given symbol and datetime.
        """
        try:
            # Use the multi-index for efficient lookup
            data_row = self.lookup_data.loc[(symbol, dt)]
            return data_row.to_dict()
        except (KeyError, IndexError):
            # No data found for this exact symbol/datetime
            return None

    def reset(self):
        # Re-create the generator
        self.event_stream = self._create_event_stream()

    def get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame:
        if self.pointer < lookback:
            return pd.DataFrame()
        return self.data.iloc[self.pointer - lookback : self.pointer]
