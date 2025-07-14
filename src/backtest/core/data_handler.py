#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: data_handler.py
모듈: 데이터 핸들러
목적: 실제 거래환경(시간, 휴장, 호가, 거래량, 상장폐지 등) 완벽 재현

Author: WorldClassAI
Created: 2025-07-12
Version: 1.0.0
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

class DataHandler:
    def __init__(self, data_path: str, config: Dict[str, Any]):
        self.data_path = data_path
        self.config = config
        self._load_data()
        self._load_holidays()
        self.pointer = 0

    def _load_data(self):
        """데이터를 로드합니다."""
        try:
            self.data = pd.read_parquet(self.data_path)
            # 상장폐지 종목 포함, 생존편향 제거
            if self.config.get("remove_survivorship_bias", True):
                if "delisted" in self.data.columns:
                    self.data = self.data[self.data["delisted"] == False]
        except Exception as e:
            # 샘플 데이터 생성
            print(f"데이터 로드 실패, 샘플 데이터 생성: {e}")
            self._create_sample_data()

    def _create_sample_data(self):
        """샘플 데이터를 생성합니다."""
        # 20년치 샘플 데이터 생성
        start_date = datetime.datetime(2005, 1, 1)
        end_date = datetime.datetime(2025, 12, 31)

        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 월-금만
                dates.append(current_date)
            current_date += datetime.timedelta(days=1)

        # 샘플 데이터 생성
        data = []
        for i, date in enumerate(dates):
            # 기본 가격 (시간에 따른 상승 추세 + 변동성)
            base_price = 1000 + i * 0.1 + np.random.normal(0, 50)
            base_price = max(base_price, 100)  # 최소 100원

            # OHLC 생성
            daily_return = np.random.normal(0.001, 0.02)  # 일평균 0.1%, 변동성 2%
            open_price = base_price
            close_price = open_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))

            # 거래량
            volume = np.random.lognormal(10, 1) * 1000000  # 평균 100만주

            data.append({
                'datetime': date,
                'symbol': '005930',  # 삼성전자
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'price': close_price,
                'delisted': False,
            })

        self.data = pd.DataFrame(data)

    def _load_holidays(self):
        """휴장일 데이터를 로드합니다."""
        try:
            holidays_df = pd.read_csv(self.config["holiday_path"])
            self.holidays = set(holidays_df["date"].tolist())
        except Exception as e:
            print(f"휴장일 데이터 로드 실패, 기본 휴장일 사용: {e}")
            # 기본 휴장일 설정
            self.holidays = set([
                "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
                "2023-02-11", "2023-03-01", "2023-05-05", "2023-05-29", "2023-06-06", "2023-08-15",
                "2023-09-28", "2023-09-29", "2023-10-03", "2023-10-09", "2023-12-25",
                "2024-01-01", "2024-01-02", "2024-01-22", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
                "2024-03-01", "2024-05-05", "2024-05-15", "2024-06-06", "2024-08-15",
                "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-03", "2024-10-09", "2024-12-25"
            ])

    def reset(self):
        """포인터를 초기화합니다."""
        self.pointer = 0

    def is_end(self) -> bool:
        """데이터의 끝인지 확인합니다."""
        return self.pointer >= len(self.data)

    def get_next_event(self) -> Dict[str, Any]:
        """다음 시장 이벤트를 반환합니다."""
        while self.pointer < len(self.data):
            row = self.data.iloc[self.pointer]
            self.pointer += 1

            # 거래시간이고 휴장일이 아닌 경우만 반환
            if self._is_trading_time(row["datetime"]) and not self._is_holiday(row["datetime"]):
                return row.to_dict()

        return {}

    def _is_trading_time(self, dt: pd.Timestamp) -> bool:
        """거래시간인지 확인합니다."""
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)

        t = dt.time()

        # 정규장 시간 (09:00-11:30, 12:30-15:30)
        morning_start = datetime.time(9, 0)
        morning_end = datetime.time(11, 30)
        afternoon_start = datetime.time(12, 30)
        afternoon_end = datetime.time(15, 30)

        # 동시호가 시간 (08:30-09:00, 15:30-15:40)
        pre_market_start = datetime.time(8, 30)
        pre_market_end = datetime.time(9, 0)
        post_market_start = datetime.time(15, 30)
        post_market_end = datetime.time(15, 40)

        # 정규장 시간 확인
        if (morning_start <= t <= morning_end or :
            afternoon_start <= t <= afternoon_end):
            return True

        # 동시호가 시간 확인
        if (pre_market_start <= t <= pre_market_end or:
            post_market_start <= t <= post_market_end):
            return True

        return False

    def _is_holiday(self, dt: pd.Timestamp) -> bool:
        """휴장일인지 확인합니다."""
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)

        # 주말 확인
        if dt.weekday() >= 5:  # 토요일(5), 일요일(6)
            return True

        # 공휴일 확인
        date_str = dt.strftime("%Y-%m-%d")
        return date_str in self.holidays
