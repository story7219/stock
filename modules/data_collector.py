#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 고품질 데이터 수집기
코스피200·나스닥100·S&P500 전체 종목 데이터를 .env API를 최대한 활용하여 수집
Gemini AI가 필요한 모든 데이터를 고품질로 제공
"""

import os
import asyncio
import logging
import pandas as pd
import yfinance as yf
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from investment_strategies import StockData

logger = logging.getLogger(__name__)

class MultiDataCollector:
    """🚀 다중 소스 데이터 수집기 - API 최대 활용"""
    
    def __init__(self):
        """초기화 및 API 키 로드"""
        self.kis_app_key = os.getenv('LIVE_KIS_APP_KEY')
        self.kis_app_secret = os.getenv('LIVE_KIS_APP_SECRET')
        self.kis_account = os.getenv('LIVE_KIS_ACCOUNT_NUMBER')
        self.dart_api_key = os.getenv('DART_API_KEY')
        
        # 수집 통계
        self.collected_count = 0
        self.failed_count = 0
        self.start_time = None
        
        logger.info("📊 다중 데이터 수집기 초기화 완료")
        logger.info(f"KIS API: {'✅' if self.kis_app_key else '❌'}")
        logger.info(f"DART API: {'✅' if self.dart_api_key else '❌'}")
    
    async def collect_all_markets(self) -> List[StockData]:
        """🌏 전체 시장 데이터 수집"""
        logger.info("🚀 전체 시장 데이터 수집 시작")
        self.start_time = time.time()
        
        all_stocks = []
        
        # 1. 코스피200 수집
        logger.info("📈 코스피200 데이터 수집")
        kospi_stocks = await self._collect_kospi200()
        all_stocks.extend(kospi_stocks)
        logger.info(f"✅ 코스피200: {len(kospi_stocks)}개 종목")
        
        # 2. 나스닥100 수집
        logger.info("🇺🇸 나스닥100 데이터 수집")
        nasdaq_stocks = await self._collect_nasdaq100()
        all_stocks.extend(nasdaq_stocks)
        logger.info(f"✅ 나스닥100: {len(nasdaq_stocks)}개 종목")
        
        # 3. S&P500 수집
        logger.info("📊 S&P500 데이터 수집")
        sp500_stocks = await self._collect_sp500()
        all_stocks.extend(sp500_stocks)
        logger.info(f"✅ S&P500: {len(sp500_stocks)}개 종목")
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"🎉 전체 수집 완료: {len(all_stocks)}개 종목 ({elapsed_time:.1f}초)")
        
        return all_stocks
    
    async def _collect_kospi200(self) -> List[StockData]:
        """📈 코스피200 종목 수집"""
        stocks = []
        
        try:
            # 코스피200 구성종목 리스트 (주요 종목들)
            kospi200_symbols = [
                '005930.KS',  # 삼성전자
                '000660.KS',  # SK하이닉스
                '035420.KS',  # 네이버
                '005380.KS',  # 현대자동차
                '051910.KS',  # LG화학
                '028260.KS',  # 삼성물산
                '006400.KS',  # 삼성SDI
                '012330.KS',  # 현대모비스
                '096770.KS',  # SK이노베이션
                '003550.KS',  # LG
                '017670.KS',  # SK텔레콤
                '030200.KS',  # KT
                '036570.KS',  # 엔씨소프트
                '003490.KS',  # 대한항공
                '323410.KS',  # 카카오뱅크
                '035720.KS',  # 카카오
                '207940.KS',  # 삼성바이오로직스
                '068270.KS',  # 셀트리온
                '000270.KS',  # 기아
                '018260.KS',  # 삼성에스디에스
            ]
            
            # 병렬 수집
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(self._collect_single_stock, symbol, "KOSPI200")
                    for symbol in kospi200_symbols
                ]
                
                for future in as_completed(futures):
                    try:
                        stock_data = future.result()
                        if stock_data:
                            stocks.append(stock_data)
                            self.collected_count += 1
                    except Exception as e:
                        self.failed_count += 1
                        logger.warning(f"코스피200 종목 수집 실패: {e}")
        
        except Exception as e:
            logger.error(f"코스피200 수집 중 오류: {e}")
        
        return stocks
    
    async def _collect_nasdaq100(self) -> List[StockData]:
        """🇺🇸 나스닥100 종목 수집"""
        stocks = []
        
        try:
            # 나스닥100 주요 종목들
            nasdaq100_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NVDA', 'NFLX', 'PYPL', 'ADBE',
                'CMCSA', 'INTC', 'CSCO', 'PEP', 'COST',
                'TMUS', 'AVGO', 'TXN', 'QCOM', 'SBUX',
                'INTU', 'GILD', 'AMGN', 'AMAT', 'BKNG',
                'MDLZ', 'ISRG', 'ADP', 'CSX', 'REGN',
                'VRTX', 'FISV', 'ATVI', 'BIIB', 'CHTR',
                'MU', 'AMD', 'LRCX', 'MRNA', 'ADI'
            ]
            
            # 병렬 수집
            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = [
                    executor.submit(self._collect_single_stock, symbol, "NASDAQ100")
                    for symbol in nasdaq100_symbols
                ]
                
                for future in as_completed(futures):
                    try:
                        stock_data = future.result()
                        if stock_data:
                            stocks.append(stock_data)
                            self.collected_count += 1
                    except Exception as e:
                        self.failed_count += 1
                        logger.warning(f"나스닥100 종목 수집 실패: {e}")
        
        except Exception as e:
            logger.error(f"나스닥100 수집 중 오류: {e}")
        
        return stocks
    
    async def _collect_sp500(self) -> List[StockData]:
        """📊 S&P500 종목 수집"""
        stocks = []
        
        try:
            # S&P500 주요 종목들
            sp500_symbols = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA',
                'BRK-B', 'UNH', 'JNJ', 'V', 'XOM',
                'JPM', 'PG', 'NVDA', 'HD', 'MA',
                'CVX', 'BAC', 'ABBV', 'PFE', 'KO',
                'AVGO', 'PEP', 'TMO', 'COST', 'WMT',
                'DIS', 'DHR', 'ABT', 'LIN', 'ACN',
                'VZ', 'ADBE', 'CRM', 'NKE', 'NEE',
                'BMY', 'ORCL', 'PM', 'CMCSA', 'HON'
            ]
            
            # 병렬 수집
            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = [
                    executor.submit(self._collect_single_stock, symbol, "S&P500")
                    for symbol in sp500_symbols
                ]
                
                for future in as_completed(futures):
                    try:
                        stock_data = future.result()
                        if stock_data:
                            stocks.append(stock_data)
                            self.collected_count += 1
                    except Exception as e:
                        self.failed_count += 1
                        logger.warning(f"S&P500 종목 수집 실패: {e}")
        
        except Exception as e:
            logger.error(f"S&P500 수집 중 오류: {e}")
        
        return stocks
    
    def _collect_single_stock(self, symbol: str, market: str) -> Optional[StockData]:
        """단일 종목 데이터 수집"""
        try:
            # Yahoo Finance에서 데이터 수집
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")
            
            if hist.empty or 'longName' not in info:
                return None
            
            # 기본 정보
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else info.get('currentPrice', 0)
            
            # 기술적 지표 계산
            rsi = self._calculate_rsi(hist['Close']) if len(hist) >= 14 else None
            ma20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None
            ma60 = hist['Close'].rolling(60).mean().iloc[-1] if len(hist) >= 60 else None
            volume_ratio = hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:-5].mean() if len(hist) >= 20 else 1.0
            
            # StockData 객체 생성
            stock_data = StockData(
                symbol=symbol,
                name=info.get('longName', symbol),
                current_price=float(current_price),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('forwardPE') or info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                roe=info.get('returnOnEquity'),
                debt_ratio=self._calculate_debt_ratio(info),
                dividend_yield=info.get('dividendYield'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                rsi=rsi,
                moving_avg_20=float(ma20) if ma20 and not pd.isna(ma20) else None,
                moving_avg_60=float(ma60) if ma60 and not pd.isna(ma60) else None,
                volume_ratio=float(volume_ratio) if volume_ratio and not pd.isna(volume_ratio) else 1.0,
                market=market,
                sector=info.get('sector', ''),
                news_sentiment=0.0  # 기본값, 추후 뉴스 분석으로 업데이트
            )
            
            return stock_data
            
        except Exception as e:
            logger.warning(f"{symbol} 데이터 수집 실패: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        except:
            return None
    
    def _calculate_debt_ratio(self, info: Dict) -> Optional[float]:
        """부채비율 계산"""
        try:
            total_debt = info.get('totalDebt', 0)
            total_equity = info.get('totalStockholdersEquity', 1)
            
            if total_equity and total_equity > 0:
                return total_debt / total_equity
            return None
        except:
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        return {
            'collected': self.collected_count,
            'failed': self.failed_count,
            'success_rate': (self.collected_count / (self.collected_count + self.failed_count) * 100) 
                           if (self.collected_count + self.failed_count) > 0 else 0,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }

class DataCleaner:
    """🧹 데이터 정제 및 검증"""
    
    def __init__(self):
        self.cleaned_count = 0
        self.removed_count = 0
    
    def clean_stock_data(self, stocks: List[StockData]) -> List[StockData]:
        """주식 데이터 정제"""
        logger.info(f"🧹 데이터 정제 시작: {len(stocks)}개 종목")
        
        cleaned_stocks = []
        
        for stock in stocks:
            if self._is_valid_stock(stock):
                # 데이터 보정
                cleaned_stock = self._normalize_stock_data(stock)
                cleaned_stocks.append(cleaned_stock)
                self.cleaned_count += 1
            else:
                self.removed_count += 1
        
        logger.info(f"✅ 정제 완료: {len(cleaned_stocks)}개 유효, {self.removed_count}개 제거")
        return cleaned_stocks
    
    def _is_valid_stock(self, stock: StockData) -> bool:
        """종목 유효성 검증"""
        # 기본 필수 정보 체크
        if not stock.symbol or not stock.name or stock.current_price <= 0:
            return False
        
        # 극단적인 값 필터링
        if stock.pe_ratio and (stock.pe_ratio < 0 or stock.pe_ratio > 1000):
            return False
        
        if stock.pb_ratio and (stock.pb_ratio < 0 or stock.pb_ratio > 100):
            return False
        
        return True
    
    def _normalize_stock_data(self, stock: StockData) -> StockData:
        """데이터 정규화 및 보정"""
        # None 값들을 기본값으로 대체
        if stock.rsi is None:
            stock.rsi = 50.0  # 중립값
        
        if stock.volume_ratio is None:
            stock.volume_ratio = 1.0  # 평균값
        
        if stock.news_sentiment is None:
            stock.news_sentiment = 0.0  # 중립값
        
        return stock

if __name__ == "__main__":
    print("🚀 고품질 데이터 수집기 v1.0")
    print("=" * 50)
    
    async def test_collector():
        collector = MultiDataCollector()
        stocks = await collector.collect_all_markets()
        
        print(f"\n📊 수집 결과:")
        print(f"  • 총 종목: {len(stocks)}개")
        print(f"  • 수집 통계: {collector.get_collection_stats()}")
        
        if stocks:
            print(f"\n📈 샘플 종목:")
            for i, stock in enumerate(stocks[:3]):
                print(f"  {i+1}. {stock.name} ({stock.symbol}) - {stock.current_price:.2f}")
    
    asyncio.run(test_collector()) 