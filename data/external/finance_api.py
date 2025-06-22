#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 금융 데이터 API 클라이언트
Yahoo Finance, DART, 네이버금융 등에서 주식 데이터 수집
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import json
import re

logger = logging.getLogger(__name__)

class FinanceAPIClient:
    """통합 금융 데이터 API 클라이언트"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.dart_api_key = None  # DART API 키 (선택사항)
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_stock_data(self, symbol: str, market: str = "KR") -> Dict[str, Any]:
        """종목 기본 데이터 수집"""
        try:
            if market == "KR":
                return await self._get_korean_stock_data(symbol)
            else:
                return await self._get_us_stock_data(symbol)
        except Exception as e:
            logger.error(f"주식 데이터 수집 오류 ({symbol}): {e}")
            return {}
    
    async def _get_korean_stock_data(self, symbol: str) -> Dict[str, Any]:
        """한국 주식 데이터 수집"""
        data = {}
        
        # 네이버 금융에서 기본 정보 수집
        naver_data = await self._get_naver_stock_data(symbol)
        data.update(naver_data)
        
        # Yahoo Finance에서 추가 데이터 수집 (한국 종목)
        yahoo_symbol = f"{symbol}.KS" if len(symbol) == 6 else f"{symbol}.KQ"
        yahoo_data = await self._get_yahoo_stock_data(yahoo_symbol)
        data.update(yahoo_data)
        
        return data
    
    async def _get_us_stock_data(self, symbol: str) -> Dict[str, Any]:
        """미국 주식 데이터 수집"""
        return await self._get_yahoo_stock_data(symbol)
    
    async def get_us_stock_data(self, symbol: str) -> Dict[str, Any]:
        """미국 주식 데이터 수집 (외부 호출용)"""
        return await self._get_us_stock_data(symbol)
    
    async def _get_naver_stock_data(self, symbol: str) -> Dict[str, Any]:
        """네이버 금융에서 데이터 수집"""
        try:
            url = f"https://finance.naver.com/item/main.naver?code={symbol}"
            
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        html = await response.text()
            else:
                async with self.session.get(url) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            data = {
                'symbol': symbol,
                'source': 'naver'
            }
            
            # 기본 정보 추출
            try:
                # 종목명
                name_elem = soup.select_one('.wrap_company h2 a')
                if name_elem:
                    data['name'] = name_elem.text.strip()
                
                # 현재가
                price_elem = soup.select_one('.no_today .blind')
                if price_elem:
                    price_text = price_elem.text.replace(',', '')
                    data['price'] = float(price_text)
                
                # 시가총액
                market_cap_elem = soup.select_one('table.tb_type1 tr:nth-child(3) td:nth-child(2)')
                if market_cap_elem:
                    market_cap_text = market_cap_elem.text.replace(',', '').replace('억원', '')
                    data['market_cap'] = float(market_cap_text) * 100  # 억원을 백만원으로
                
                # PER, PBR 등 추출
                ratio_table = soup.select('table.tb_type1 tr')
                for row in ratio_table:
                    cells = row.select('td')
                    if len(cells) >= 2:
                        label = cells[0].text.strip()
                        value_text = cells[1].text.strip()
                        
                        if 'PER' in label:
                            try:
                                data['pe_ratio'] = float(value_text.replace(',', ''))
                            except:
                                pass
                        elif 'PBR' in label:
                            try:
                                data['pb_ratio'] = float(value_text.replace(',', ''))
                            except:
                                pass
                
            except Exception as e:
                logger.warning(f"네이버 데이터 파싱 오류 ({symbol}): {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"네이버 데이터 수집 오류 ({symbol}): {e}")
            return {'symbol': symbol, 'source': 'naver', 'error': str(e)}
    
    async def _get_yahoo_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Yahoo Finance에서 데이터 수집"""
        try:
            # yfinance는 동기 라이브러리이므로 비동기로 실행
            def get_yahoo_data():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1y")
                
                return info, hist
            
            info, hist = await asyncio.to_thread(get_yahoo_data)
            
            data = {
                'symbol': symbol,
                'source': 'yahoo'
            }
            
            # 기본 정보
            if 'longName' in info:
                data['name'] = info['longName']
            elif 'shortName' in info:
                data['name'] = info['shortName']
            
            if 'currentPrice' in info:
                data['price'] = info['currentPrice']
            elif 'regularMarketPrice' in info:
                data['price'] = info['regularMarketPrice']
            
            if 'marketCap' in info:
                data['market_cap'] = info['marketCap'] / 1000000  # 백만원 단위
            
            # 재무 지표
            financial_metrics = {
                'pe_ratio': ['trailingPE', 'forwardPE'],
                'pb_ratio': ['priceToBook'],
                'roe': ['returnOnEquity'],
                'roa': ['returnOnAssets'],
                'debt_ratio': ['debtToEquity'],
                'current_ratio': ['currentRatio'],
                'dividend_yield': ['dividendYield'],
                'revenue_growth': ['revenueGrowth'],
                'profit_growth': ['earningsGrowth']
            }
            
            for metric, keys in financial_metrics.items():
                for key in keys:
                    if key in info and info[key] is not None:
                        value = info[key]
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            if metric in ['roe', 'roa', 'dividend_yield', 'revenue_growth', 'profit_growth']:
                                data[metric] = value * 100  # 퍼센트로 변환
                            else:
                                data[metric] = value
                        break
            
            # 섹터/업종 정보
            if 'sector' in info:
                data['sector'] = info['sector']
            if 'industry' in info:
                data['industry'] = info['industry']
            
            # 가격 모멘텀 계산
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                price_3m_ago = hist['Close'].iloc[-60] if len(hist) > 60 else hist['Close'].iloc[0]
                data['price_momentum_3m'] = ((current_price - price_3m_ago) / price_3m_ago) * 100
            
            return data
            
        except Exception as e:
            logger.error(f"Yahoo Finance 데이터 수집 오류 ({symbol}): {e}")
            return {'symbol': symbol, 'source': 'yahoo', 'error': str(e)}
    
    async def get_market_data(self, market: str = "KR") -> Dict[str, Any]:
        """시장 지수 데이터 수집"""
        try:
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'market': market
            }
            
            if market == "KR":
                # 한국 시장 지수
                kospi_data = await self._get_yahoo_stock_data("^KS11")
                kosdaq_data = await self._get_yahoo_stock_data("^KQ11")
                
                market_data.update({
                    'kospi': kospi_data.get('price'),
                    'kosdaq': kosdaq_data.get('price')
                })
            else:
                # 미국 시장 지수
                sp500_data = await self._get_yahoo_stock_data("^GSPC")
                nasdaq_data = await self._get_yahoo_stock_data("^IXIC")
                
                market_data.update({
                    'sp500': sp500_data.get('price'),
                    'nasdaq': nasdaq_data.get('price')
                })
            
            # 환율 정보
            usd_krw_data = await self._get_yahoo_stock_data("USDKRW=X")
            market_data['usd_krw'] = usd_krw_data.get('price')
            
            # VIX 지수
            vix_data = await self._get_yahoo_stock_data("^VIX")
            market_data['vix'] = vix_data.get('price')
            
            return market_data
            
        except Exception as e:
            logger.error(f"시장 데이터 수집 오류: {e}")
            return {'error': str(e)}
    
    async def get_batch_stock_data(self, symbols: List[str], market: str = "KR") -> List[Dict[str, Any]]:
        """여러 종목 데이터 일괄 수집"""
        semaphore = asyncio.Semaphore(10)  # 동시 요청 제한
        
        async def get_single_stock(symbol):
            async with semaphore:
                await asyncio.sleep(0.1)  # 요청 간격 조절
                return await self.get_stock_data(symbol, market)
        
        tasks = [get_single_stock(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 오류 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"배치 수집 오류 ({symbols[i]}): {result}")
                processed_results.append({
                    'symbol': symbols[i],
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_top_stocks_list(self, market: str = "KR", count: int = 100) -> List[str]:
        """상위 시가총액 종목 리스트 반환"""
        if market == "KR":
            # 한국 주요 종목 (KOSPI 200 기준)
            return [
                "005930",  # 삼성전자
                "000660",  # SK하이닉스
                "035420",  # NAVER
                "051910",  # LG화학
                "006400",  # 삼성SDI
                "035720",  # 카카오
                "207940",  # 삼성바이오로직스
                "068270",  # 셀트리온
                "028260",  # 삼성물산
                "066570",  # LG전자
                "003550",  # LG
                "096770",  # SK이노베이션
                "033780",  # KT&G
                "015760",  # 한국전력
                "017670",  # SK텔레콤
                "030200",  # KT
                "036570",  # 엔씨소프트
                "018260",  # 삼성에스디에스
                "032830",  # 삼성생명
                "011200",  # HMM
                "009150",  # 삼성전기
                "010950",  # S-Oil
                "086790",  # 하나금융지주
                "316140",  # 우리금융지주
                "055550",  # 신한지주
                "105560",  # KB금융
                "000810",  # 삼성화재
                "012330",  # 현대모비스
                "005380",  # 현대차
                "000270",  # 기아
            ][:count]
        else:
            # 미국 주요 종목 (S&P 500 기준)
            return [
                "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
                "META", "NVDA", "BRK-B", "UNH", "JNJ",
                "V", "WMT", "JPM", "PG", "MA",
                "HD", "BAC", "ABBV", "PFE", "KO",
                "AVGO", "PEP", "TMO", "COST", "DIS",
                "ABT", "ACN", "VZ", "ADBE", "CRM"
            ][:count]
    
    async def get_sector_stocks_kr(self, sector: str, limit: int = 50) -> List[str]:
        """한국 시장 섹터별 종목 목록 조회"""
        # 간단한 섹터별 종목 매핑 (실제로는 더 정교한 API 필요)
        sector_mapping = {
            '반도체': ['005930', '000660', '006400', '042700'],
            '바이오': ['207940', '068270', '326030', '214150'],
            '게임': ['035720', '036570', '251270', '112040'],
            '금융': ['055550', '086790', '316140', '105560'],
            '자동차': ['005380', '000270', '012330', '161390']
        }
        return sector_mapping.get(sector, self.get_top_stocks_list('KR', limit))[:limit]
    
    async def get_sector_stocks_us(self, sector: str, limit: int = 50) -> List[str]:
        """미국 시장 섹터별 종목 목록 조회"""
        # 간단한 섹터별 종목 매핑 (실제로는 더 정교한 API 필요)
        sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABT', 'TMO'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'PG', 'KO'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        }
        return sector_mapping.get(sector, self.get_top_stocks_list('US', limit))[:limit]

# 편의 함수들
async def collect_stock_data(symbols: List[str], market: str = "KR") -> List[Dict[str, Any]]:
    """주식 데이터 수집 편의 함수"""
    async with FinanceAPIClient() as client:
        return await client.get_batch_stock_data(symbols, market)

async def collect_market_data(market: str = "KR") -> Dict[str, Any]:
    """시장 데이터 수집 편의 함수"""
    async with FinanceAPIClient() as client:
        return await client.get_market_data(market)

def get_recommended_stocks(market: str = "KR", count: int = 50) -> List[str]:
    """추천 종목 리스트 반환"""
    client = FinanceAPIClient()
    return client.get_top_stocks_list(market, count) 