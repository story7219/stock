#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 통합 데이터 로더
외부 API → 데이터 정제 → AI 전처리까지 전체 파이프라인 관리
"""

import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# 로컬 모듈 임포트
from data.external.finance_api import FinanceAPIClient
from data.processed.data_cleaner import DataCleaner, CleanedStockData
from ai_integration.ai_preprocessor import AIDataPreprocessor

logger = logging.getLogger(__name__)

class DataLoader:
    """통합 데이터 로더 클래스"""
    
    def __init__(self):
        self.api_client = FinanceAPIClient()
        self.data_cleaner = DataCleaner()
        self.ai_preprocessor = AIDataPreprocessor()
        
        # 캐시 설정
        self.cache_dir = Path("data/processed/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=1)  # 1시간 캐시
        
        # 데이터 품질 설정
        self.min_data_quality = 60.0
        
        # 시장별 기본 종목 리스트
        self.default_symbols = {
            'KR': [
                '005930',  # 삼성전자
                '000660',  # SK하이닉스
                '035420',  # NAVER
                '207940',  # 삼성바이오로직스
                '005380',  # 현대차
                '051910',  # LG화학
                '006400',  # 삼성SDI
                '035720',  # 카카오
                '068270',  # 셀트리온
                '028260',  # 삼성물산
            ],
            'US': [
                'AAPL',    # Apple
                'MSFT',    # Microsoft
                'GOOGL',   # Alphabet
                'AMZN',    # Amazon
                'TSLA',    # Tesla
                'META',    # Meta
                'NVDA',    # NVIDIA
                'NFLX',    # Netflix
                'DIS',     # Disney
                'PYPL',    # PayPal
            ]
        }
    
    async def load_market_data(self, market: str = 'KR', symbols: Optional[List[str]] = None, 
                              use_cache: bool = True) -> List[CleanedStockData]:
        """시장 데이터 로드 (전체 파이프라인)"""
        try:
            logger.info(f"{market} 시장 데이터 로드 시작")
            
            # 1단계: 심볼 목록 결정
            if symbols is None:
                symbols = self.default_symbols.get(market, [])
            
            # 2단계: 캐시 확인
            if use_cache:
                cached_data = self._load_from_cache(market, symbols)
                if cached_data:
                    logger.info(f"캐시에서 데이터 로드: {len(cached_data)} 종목")
                    return cached_data
            
            # 3단계: 외부 API에서 데이터 수집
            raw_data = await self._fetch_raw_data(market, symbols)
            
            # 4단계: 데이터 정제
            cleaned_data = self._clean_data(raw_data)
            
            # 5단계: 품질 필터링
            filtered_data = self._filter_by_quality(cleaned_data)
            
            # 6단계: 캐시 저장
            if use_cache:
                self._save_to_cache(market, symbols, filtered_data)
            
            logger.info(f"{market} 시장 데이터 로드 완료: {len(filtered_data)} 종목")
            return filtered_data
            
        except Exception as e:
            logger.error(f"시장 데이터 로드 오류 ({market}): {e}")
            raise
    
    async def load_stock_data(self, symbol: str, market: str = 'KR') -> Optional[CleanedStockData]:
        """개별 종목 데이터 로드"""
        try:
            # 외부 API에서 데이터 수집
            async with self.api_client as client:
                raw_data = await client.get_stock_data(symbol, market)
            
            if not raw_data:
                return None
            
            # 데이터 정제
            cleaned_data = self.data_cleaner.clean_stock_data(raw_data)
            
            # 품질 검증
            if cleaned_data and cleaned_data.data_quality >= self.min_data_quality:
                return cleaned_data
            else:
                logger.warning(f"품질 기준 미달: {symbol} (품질: {cleaned_data.data_quality if cleaned_data else 0})")
                return None
                
        except Exception as e:
            logger.error(f"종목 데이터 로드 오류 ({symbol}): {e}")
            return None
    
    async def load_sector_data(self, sector: str, market: str = 'KR', limit: int = 50) -> List[CleanedStockData]:
        """섹터별 데이터 로드"""
        try:
            logger.info(f"{market} 시장 {sector} 섹터 데이터 로드 시작")
            
            # 외부 API에서 섹터 종목 목록 가져오기
            async with self.api_client as client:
                if market == 'KR':
                    sector_symbols = await client.get_sector_stocks_kr(sector, limit)
                else:
                    sector_symbols = await client.get_sector_stocks_us(sector, limit)
            
            if not sector_symbols:
                logger.warning(f"섹터 종목 목록이 비어있음: {sector}")
                return []
            
            # 각 종목의 데이터 로드
            sector_data = []
            for symbol in sector_symbols:
                stock_data = await self.load_stock_data(symbol, market)
                if stock_data:
                    sector_data.append(stock_data)
            
            logger.info(f"{sector} 섹터 데이터 로드 완료: {len(sector_data)} 종목")
            return sector_data
            
        except Exception as e:
            logger.error(f"섹터 데이터 로드 오류 ({sector}): {e}")
            return []
    
    def prepare_for_ai_analysis(self, stocks: List[CleanedStockData], 
                               analysis_type: str = 'investment') -> Dict[str, Any]:
        """AI 분석을 위한 데이터 전처리"""
        try:
            # AI 전처리기를 통해 데이터 변환
            ai_ready_data = self.ai_preprocessor.prepare_for_analysis(
                stocks, analysis_type
            )
            
            logger.info(f"AI 분석용 데이터 준비 완료: {len(stocks)} 종목")
            return ai_ready_data
            
        except Exception as e:
            logger.error(f"AI 데이터 전처리 오류: {e}")
            raise
    
    async def _fetch_raw_data(self, market: str, symbols: List[str]) -> List[Dict[str, Any]]:
        """외부 API에서 원시 데이터 수집"""
        raw_data = []
        
        async with self.api_client as client:
            # 병렬로 데이터 수집
            tasks = []
            for symbol in symbols:
                task = client.get_stock_data(symbol, market)
                tasks.append(task)
            
            # 모든 태스크 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"데이터 수집 오류 ({symbols[i]}): {result}")
                elif result:
                    raw_data.append(result)
        
        logger.info(f"원시 데이터 수집 완료: {len(raw_data)}/{len(symbols)} 종목")
        return raw_data
    
    def _clean_data(self, raw_data: List[Dict[str, Any]]) -> List[CleanedStockData]:
        """데이터 정제"""
        return self.data_cleaner.clean_batch_data(raw_data)
    
    def _filter_by_quality(self, stocks: List[CleanedStockData]) -> List[CleanedStockData]:
        """품질 기준으로 필터링"""
        return self.data_cleaner.filter_by_quality(stocks, self.min_data_quality)
    
    def _get_cache_key(self, market: str, symbols: List[str]) -> str:
        """캐시 키 생성"""
        symbols_str = "_".join(sorted(symbols))
        return f"{market}_{hash(symbols_str)}"
    
    def _get_cache_filename(self, cache_key: str) -> str:
        """캐시 파일명 생성"""
        return f"cache_{cache_key}.json"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """캐시 유효성 검사"""
        if not cache_file.exists():
            return False
        
        # 파일 수정 시간 확인
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < self.cache_duration
    
    def _load_from_cache(self, market: str, symbols: List[str]) -> Optional[List[CleanedStockData]]:
        """캐시에서 데이터 로드"""
        try:
            cache_key = self._get_cache_key(market, symbols)
            cache_file = self.cache_dir / self._get_cache_filename(cache_key)
            
            if not self._is_cache_valid(cache_file):
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # CleanedStockData 객체로 변환
            stocks = []
            for stock_data in cache_data.get('stocks', []):
                stock = CleanedStockData(**stock_data)
                stocks.append(stock)
            
            return stocks
            
        except Exception as e:
            logger.debug(f"캐시 로드 실패: {e}")
            return None
    
    def _save_to_cache(self, market: str, symbols: List[str], stocks: List[CleanedStockData]):
        """캐시에 데이터 저장"""
        try:
            cache_key = self._get_cache_key(market, symbols)
            cache_file = self.cache_dir / self._get_cache_filename(cache_key)
            
            cache_data = {
                "market": market,
                "symbols": symbols,
                "cached_at": datetime.now().isoformat(),
                "stocks": [stock.to_dict() for stock in stocks]
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"캐시 저장 완료: {cache_file}")
            
        except Exception as e:
            logger.error(f"캐시 저장 오류: {e}")
    
    def clear_cache(self, market: str = None):
        """캐시 삭제"""
        try:
            cache_files = list(self.cache_dir.glob("cache_*.json"))
            
            deleted_count = 0
            for cache_file in cache_files:
                if market is None:
                    # 모든 캐시 삭제
                    cache_file.unlink()
                    deleted_count += 1
                else:
                    # 특정 시장 캐시만 삭제
                    if cache_file.name.startswith(f"cache_{market}_"):
                        cache_file.unlink()
                        deleted_count += 1
            
            logger.info(f"캐시 삭제 완료: {deleted_count} 파일")
            
        except Exception as e:
            logger.error(f"캐시 삭제 오류: {e}")
    
    def get_data_statistics(self, stocks: List[CleanedStockData]) -> Dict[str, Any]:
        """데이터 통계 정보"""
        if not stocks:
            return {}
        
        # 기본 통계
        stats = self.data_cleaner.get_quality_statistics(stocks)
        
        # 추가 통계
        markets = {}
        sectors = {}
        
        for stock in stocks:
            # 시장별 분포
            market = stock.market
            markets[market] = markets.get(market, 0) + 1
            
            # 섹터별 분포
            if stock.sector:
                sector = stock.sector
                sectors[sector] = sectors.get(sector, 0) + 1
        
        stats.update({
            "시장별_분포": markets,
            "섹터별_분포": sectors,
            "평균_시가총액": round(np.mean([s.market_cap for s in stocks if s.market_cap]), 2),
            "평균_주가": round(np.mean([s.price for s in stocks if s.price]), 2)
        })
        
        return stats
    
    async def get_market_overview(self, market: str = 'KR') -> Dict[str, Any]:
        """시장 개요 정보"""
        try:
            # 주요 종목 데이터 로드
            stocks = await self.load_market_data(market)
            
            if not stocks:
                return {"error": f"{market} 시장 데이터를 불러올 수 없습니다"}
            
            # 시장 통계 계산
            overview = {
                "시장": market,
                "분석_시간": datetime.now().isoformat(),
                "총_종목수": len(stocks),
                "데이터_통계": self.get_data_statistics(stocks)
            }
            
            # 상위 종목 (시가총액 기준)
            top_stocks = sorted(stocks, key=lambda x: x.market_cap or 0, reverse=True)[:10]
            overview["상위_종목"] = [
                {
                    "종목코드": stock.symbol,
                    "종목명": stock.name,
                    "시가총액": stock.market_cap,
                    "주가": stock.price,
                    "섹터": stock.sector
                }
                for stock in top_stocks
            ]
            
            return overview
            
        except Exception as e:
            logger.error(f"시장 개요 생성 오류 ({market}): {e}")
            return {"error": str(e)}

# 편의 함수들
async def load_korean_stocks(symbols: List[str] = None) -> List[CleanedStockData]:
    """한국 주식 데이터 로드"""
    loader = DataLoader()
    return await loader.load_market_data('KR', symbols)

async def load_us_stocks(symbols: List[str] = None) -> List[CleanedStockData]:
    """미국 주식 데이터 로드"""
    loader = DataLoader()
    return await loader.load_market_data('US', symbols)

async def load_stock_for_analysis(symbol: str, market: str = 'KR') -> Optional[Dict[str, Any]]:
    """AI 분석용 개별 종목 데이터 로드"""
    loader = DataLoader()
    stock_data = await loader.load_stock_data(symbol, market)
    
    if stock_data:
        return loader.prepare_for_ai_analysis([stock_data])
    return None 