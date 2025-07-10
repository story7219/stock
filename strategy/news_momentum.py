from __future__ import annotations
from core.config import config
from core.logger import get_logger, log_function_call
from core.models import News, NewsCategory, SentimentType, Signal, StrategyType, TradeType
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: news_momentum.py
모듈: 뉴스 모멘텀 전략 엔진
목적: 뉴스 감성분석 + DART 공시 기반 매매 신호 생성

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas==2.1.4
    - numpy==1.24.0

Performance:
    - 신호 생성: < 1초 (100종목 기준)
    - 메모리사용량: < 50MB

Security:
    - 데이터 검증
    - 에러 처리
    - 로깅

License: MIT
"""





logger = get_logger(__name__)


class NewsMomentumStrategy:
    """뉴스 모멘텀 전략"""

    def __init__(self):
        self.strategy_type = StrategyType.NEWS_MOMENTUM
        self.weight = config.trading.news_momentum_weight
        self.sentiment_threshold = 0.3  # 감성 점수 임계값
        self.news_count_threshold = 3   # 최소 뉴스 개수
        self.time_window_hours = 24     # 분석 시간 윈도우
        self.momentum_decay_factor = 0.9  # 모멘텀 감쇠 계수
        self.keyword_weights = {
            "실적": 1.5, "호재": 1.3, "성장": 1.2, "확대": 1.1, "진출": 1.1,
            "개발": 1.0, "혁신": 1.0, "악재": -1.3, "손실": -1.2, "축소": -1.1,
            "철수": -1.1, "규제": -1.0
        }

    async def generate_signals(self, news_list: List[News],
                             stock_data: Optional[Dict[str, pd.DataFrame]] = None,
                             target_stocks: Optional[List[str]] = None) -> List[Signal]:
        processed_news = self._preprocess_news(news_list)
        stock_momentum = self._calculate_stock_momentum(processed_news)
        signals = []
        for stock_code, momentum_data in stock_momentum.items():
            if target_stocks and stock_code not in target_stocks:
                continue
            try:
                signal = self._create_signal(stock_code, momentum_data, stock_data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"종목 '{stock_code}' 신호 생성 실패", error=str(e))
                continue
        signals.sort(key=lambda x: x.confidence_score, reverse=True)
        return signals

    def _preprocess_news(self, news_list: List[News]) -> List[News]:
        processed_news = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.time_window_hours)
        for news in news_list:
            if news.published_at < cutoff_time:
                continue
            if not news.related_stocks:
                continue
            if abs(news.sentiment_score) < 0.1:
                continue
            processed_news.append(news)
        return processed_news

    def _calculate_stock_momentum(self, news_list: List[News]) -> Dict[str, Dict[str, Any]]:
        stock_momentum = {}
        for news in news_list:
            for stock_code in news.related_stocks:
                if stock_code not in stock_momentum:
                    stock_momentum[stock_code] = {
                        'positive_news': [], 'negative_news': [],
                        'total_score': 0.0, 'news_count': 0,
                        'latest_news_time': None
                    }
                momentum_data = stock_momentum[stock_code]
                if news.sentiment == SentimentType.POSITIVE:
                    momentum_data['positive_news'].append(news)
                elif news.sentiment == SentimentType.NEGATIVE:
                    momentum_data['negative_news'].append(news)
                base_score = news.sentiment_score
                keyword_multiplier = self._calculate_keyword_multiplier(news.title + " " + news.content)
                time_decay = self._calculate_time_decay(news.published_at)
                weighted_score = base_score * keyword_multiplier * time_decay
                momentum_data['total_score'] += weighted_score
                momentum_data['news_count'] += 1
                if momentum_data['latest_news_time'] is None or news.published_at > momentum_data['latest_news_time']:
                    momentum_data['latest_news_time'] = news.published_at
        return stock_momentum


    # ... (other methods) ...
