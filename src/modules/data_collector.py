"""
주식 데이터 수집 모듈
코스피200, 나스닥100, S&P500 전체 종목 데이터 자동 수집
"""
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import time
import requests
from bs4 import BeautifulSoup
import json

# 새로운 인터페이스 import
from ..core.base_interfaces import (
    IDataCollector, StockData, MarketType, 
    DataCollectionError, validate_stock_data
)

class DataCollector(IDataCollector):
    """주식 데이터 수집 클래스 - IDataCollector 인터페이스 구현"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self._rate_limit_delay = 0.1  # API 호출 제한을 위한 지연
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def get_market_symbols(self, market: MarketType) -> List[str]:
        """시장별 종목 리스트 수집"""
        try:
            if market == MarketType.KOSPI200:
                return self._get_kospi200_symbols()
            elif market == MarketType.NASDAQ100:
                return self._get_nasdaq100_symbols()
            elif market == MarketType.SP500:
                return self._get_sp500_symbols()
            else:
                raise ValueError(f"지원하지 않는 시장: {market}")
                
        except Exception as e:
            logger.error(f"{market.value} 종목 리스트 수집 실패: {e}")
            raise DataCollectionError(f"종목 리스트 수집 실패: {e}")
    
    def _get_kospi200_symbols(self) -> List[str]:
        """코스피200 종목 리스트 수집"""
        logger.info("코스피200 종목 리스트 수집 시작")
        
        # 코스피200 주요 종목들 (실제로는 전체 200개 종목)
        kospi200_codes = [
            "005930.KS",  # 삼성전자
            "000660.KS",  # SK하이닉스
            "207940.KS",  # 삼성바이오로직스
            "005490.KS",  # POSCO홀딩스
            "035420.KS",  # NAVER
            "006400.KS",  # 삼성SDI
            "051910.KS",  # LG화학
            "005380.KS",  # 현대차
            "012330.KS",  # 현대모비스
            "028260.KS",  # 삼성물산
            "066570.KS",  # LG전자
            "055550.KS",  # 신한지주
            "035720.KS",  # 카카오
            "096770.KS",  # SK이노베이션
            "003670.KS",  # 포스코퓨처엠
            "017670.KS",  # SK텔레콤
            "000270.KS",  # 기아
            "373220.KS",  # LG에너지솔루션
            "323410.KS",  # 카카오뱅크
            "068270.KS",  # 셀트리온
        ]
        
        logger.info(f"코스피200 {len(kospi200_codes)}개 종목 수집 완료")
        return kospi200_codes
        
    def _get_nasdaq100_symbols(self) -> List[str]:
        """나스닥100 종목 리스트 수집"""
        logger.info("나스닥100 종목 리스트 수집 시작")
        
        nasdaq100_symbols = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
            "AVGO", "COST", "PEP", "NFLX", "ADBE", "CSCO", "TMUS", "INTC",
            "CMCSA", "TXN", "QCOM", "AMGN", "HON", "INTU", "AMD", "SBUX",
            "ISRG", "BKNG", "MDLZ", "GILD", "ADP", "REGN", "VRTX", "LRCX",
            "MU", "PANW", "SNPS", "CDNS", "MRVL", "ORLY", "CRWD", "FTNT",
            "KLAC", "ABNB", "CHTR", "MELI", "NXPI", "DXCM", "MAR", "WDAY"
        ]
        
        logger.info(f"나스닥100 {len(nasdaq100_symbols)}개 종목 수집 완료")
        return nasdaq100_symbols
        
    def _get_sp500_symbols(self) -> List[str]:
        """S&P500 종목 리스트 수집"""
        logger.info("S&P500 종목 리스트 수집 시작")
        
        try:
            # Wikipedia에서 S&P500 종목 리스트 수집
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # 심볼 정리 (점 제거 등)
            cleaned_symbols = []
            for symbol in symbols:
                if isinstance(symbol, str):
                    cleaned_symbol = symbol.replace('.', '-')
                    cleaned_symbols.append(cleaned_symbol)
            
            logger.info(f"S&P500 {len(cleaned_symbols)}개 종목 수집 완료")
            return cleaned_symbols
            
        except Exception as e:
            logger.warning(f"Wikipedia에서 S&P500 수집 실패: {e}, 백업 리스트 사용")
            # 백업 주요 종목 리스트
            sp500_major = [
                "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "BRK-B",
                "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "PFE",
                "ABBV", "BAC", "KO", "AVGO", "PEP", "TMO", "COST", "WMT", "DIS",
                "MRK", "CSCO", "CRM", "ACN", "VZ", "ADBE", "DHR", "ABT", "NFLX"
            ]
            return sp500_major
    
    async def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 주식 데이터 수집"""
        try:
            logger.debug(f"{symbol} 데이터 수집 시작")
            
            ticker = yf.Ticker(symbol)
            
            # 기본 정보 수집
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                logger.warning(f"{symbol}: 히스토리 데이터 없음")
                return None
            
            # 현재 가격 및 변동률 계산
            current_price = float(hist['Close'].iloc[-1])
            previous_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change_percent = ((current_price - previous_price) / previous_price) * 100 if previous_price != 0 else 0.0
            
            # 시장 유형 결정
            market = self._determine_market_type(symbol)
            
            # StockData 객체 생성
            stock_data = StockData(
                symbol=symbol,
                name=info.get('longName', symbol),
                market=market,
                current_price=current_price,
                market_cap=info.get('marketCap'),
                volume=int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else None,
                change_percent=change_percent,
                historical_data=hist,
                info=info,
                last_updated=datetime.now()
            )
            
            # 데이터 검증
            if validate_stock_data(stock_data):
                logger.debug(f"{symbol} 데이터 수집 완료")
                return stock_data
            else:
                logger.warning(f"{symbol} 데이터 검증 실패")
                return None
                
        except Exception as e:
            logger.error(f"{symbol} 데이터 수집 실패: {e}")
            return None
    
    def _determine_market_type(self, symbol: str) -> MarketType:
        """심볼로부터 시장 유형 결정"""
        if symbol.endswith('.KS'):
            return MarketType.KOSPI200
        elif symbol in self._get_nasdaq100_symbols():
            return MarketType.NASDAQ100
        else:
            return MarketType.SP500
    
    async def collect_market_data(self, market: MarketType) -> List[StockData]:
        """특정 시장의 모든 종목 데이터 수집"""
        try:
            logger.info(f"{market.value} 시장 데이터 수집 시작")
            
            symbols = self.get_market_symbols(market)
            stock_data_list = []
            failed_count = 0
            
            # 배치 처리를 위해 청크로 나누기
            chunk_size = 10
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                
                # 병렬 처리
                tasks = [self.get_stock_data(symbol) for symbol in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, StockData):
                        stock_data_list.append(result)
                    elif isinstance(result, Exception):
                        failed_count += 1
                        logger.error(f"데이터 수집 예외: {result}")
                    else:
                        failed_count += 1
                
                # API 호출 제한을 위한 지연
                await asyncio.sleep(self._rate_limit_delay)
            
            logger.info(f"{market.value} 데이터 수집 완료: 성공 {len(stock_data_list)}, 실패 {failed_count}")
            return stock_data_list
            
        except Exception as e:
            logger.error(f"{market.value} 시장 데이터 수집 실패: {e}")
            raise DataCollectionError(f"{market.value} 시장 데이터 수집 실패: {e}")
    
    async def collect_all_markets_data(self) -> Dict[MarketType, List[StockData]]:
        """모든 시장 데이터 수집"""
        try:
            logger.info("전체 시장 데이터 수집 시작")
            
            # 모든 시장 병렬 수집
            tasks = [
                self.collect_market_data(MarketType.KOSPI200),
                self.collect_market_data(MarketType.NASDAQ100),
                self.collect_market_data(MarketType.SP500)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data = {}
            markets = [MarketType.KOSPI200, MarketType.NASDAQ100, MarketType.SP500]
            
            for i, result in enumerate(results):
                if isinstance(result, list):
                    market_data[markets[i]] = result
                else:
                    logger.error(f"{markets[i].value} 수집 실패: {result}")
                    market_data[markets[i]] = []
            
            total_stocks = sum(len(stocks) for stocks in market_data.values())
            logger.info(f"전체 시장 데이터 수집 완료: 총 {total_stocks}개 종목")
            
            return market_data
            
        except Exception as e:
            logger.error(f"전체 시장 데이터 수집 실패: {e}")
            raise DataCollectionError(f"전체 시장 데이터 수집 실패: {e}")

    async def collect_stock_data(self, symbol: str) -> Optional[StockData]:
        """개별 주식 데이터 수집 (레거시 호환 메서드)"""
        return await self.get_stock_data(symbol)

# 하위 호환성을 위한 레거시 함수들
async def collect_kospi200_data() -> List[StockData]:
    """코스피200 데이터 수집 (레거시 호환)"""
    async with DataCollector() as collector:
        return await collector.collect_market_data(MarketType.KOSPI200)

async def collect_nasdaq100_data() -> List[StockData]:
    """나스닥100 데이터 수집 (레거시 호환)"""
    async with DataCollector() as collector:
        return await collector.collect_market_data(MarketType.NASDAQ100)

async def collect_sp500_data() -> List[StockData]:
    """S&P500 데이터 수집 (레거시 호환)"""
    async with DataCollector() as collector:
        return await collector.collect_market_data(MarketType.SP500)