"""
데이터 매니저 - 비동기 데이터 수집, 처리, 캐싱
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.cache_manager import cached, cache_manager
from core.database_manager import db_manager
from core.performance_monitor import monitor_performance
from config.settings import settings

logger = structlog.get_logger(__name__)


class DataManager:
    """데이터 매니저 - 주식 데이터 수집 및 관리"""
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.performance.max_workers)
        
        # 샘플 데이터 (실제 환경에서는 API에서 가져옴)
        self._sample_stocks = self._generate_sample_stocks()
    
    async def initialize(self):
        """데이터 매니저 초기화"""
        if self._initialized:
            return
        
        # HTTP 세션 생성
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Ultra-HTS-System/5.0'
            }
        )
        
        self._initialized = True
        logger.info("데이터 매니저 초기화 완료")
    
    def _generate_sample_stocks(self) -> Dict[str, List[Dict[str, Any]]]:
        """샘플 주식 데이터 생성"""
        
        kospi_stocks = [
            {"name": "삼성전자", "code": "005930", "sector": "반도체", "base_price": 75000},
            {"name": "SK하이닉스", "code": "000660", "sector": "반도체", "base_price": 120000},
            {"name": "NAVER", "code": "035420", "sector": "인터넷", "base_price": 180000},
            {"name": "카카오", "code": "035720", "sector": "인터넷", "base_price": 95000},
            {"name": "LG에너지솔루션", "code": "373220", "sector": "배터리", "base_price": 450000},
            {"name": "삼성바이오로직스", "code": "207940", "sector": "바이오", "base_price": 850000},
            {"name": "현대차", "code": "005380", "sector": "자동차", "base_price": 190000},
            {"name": "기아", "code": "000270", "sector": "자동차", "base_price": 85000},
            {"name": "POSCO홀딩스", "code": "005490", "sector": "철강", "base_price": 380000},
            {"name": "LG화학", "code": "051910", "sector": "화학", "base_price": 420000},
        ]
        
        nasdaq_stocks = [
            {"name": "Apple Inc.", "code": "AAPL", "sector": "Technology", "base_price": 175},
            {"name": "Microsoft Corp.", "code": "MSFT", "sector": "Technology", "base_price": 380},
            {"name": "NVIDIA Corp.", "code": "NVDA", "sector": "Semiconductors", "base_price": 450},
            {"name": "Tesla Inc.", "code": "TSLA", "sector": "Automotive", "base_price": 250},
            {"name": "Amazon.com Inc.", "code": "AMZN", "sector": "E-commerce", "base_price": 145},
            {"name": "Alphabet Inc.", "code": "GOOGL", "sector": "Internet", "base_price": 140},
            {"name": "Meta Platforms", "code": "META", "sector": "Social Media", "base_price": 350},
            {"name": "Netflix Inc.", "code": "NFLX", "sector": "Streaming", "base_price": 450},
            {"name": "Adobe Inc.", "code": "ADBE", "sector": "Software", "base_price": 520},
            {"name": "PayPal Holdings", "code": "PYPL", "sector": "Fintech", "base_price": 65},
        ]
        
        sp500_stocks = [
            {"name": "Berkshire Hathaway", "code": "BRK.B", "sector": "Conglomerate", "base_price": 350},
            {"name": "JPMorgan Chase", "code": "JPM", "sector": "Banking", "base_price": 145},
            {"name": "Johnson & Johnson", "code": "JNJ", "sector": "Healthcare", "base_price": 160},
            {"name": "Visa Inc.", "code": "V", "sector": "Payment", "base_price": 250},
            {"name": "Procter & Gamble", "code": "PG", "sector": "Consumer Goods", "base_price": 150},
            {"name": "Mastercard Inc.", "code": "MA", "sector": "Payment", "base_price": 380},
            {"name": "UnitedHealth Group", "code": "UNH", "sector": "Healthcare", "base_price": 520},
            {"name": "Home Depot", "code": "HD", "sector": "Retail", "base_price": 320},
            {"name": "Coca-Cola Co.", "code": "KO", "sector": "Beverages", "base_price": 58},
            {"name": "Walt Disney Co.", "code": "DIS", "sector": "Entertainment", "base_price": 95},
        ]
        
        return {
            "KOSPI 200": kospi_stocks,
            "NASDAQ-100": nasdaq_stocks,
            "S&P 500": sp500_stocks
        }
    
    @monitor_performance("get_stocks_by_index")
    @cached(ttl=60, key_prefix="stocks_by_index")
    async def get_stocks_by_index(self, index_name: str) -> List[Dict[str, Any]]:
        """지수별 주식 데이터 조회"""
        try:
            if index_name not in self._sample_stocks:
                return []
            
            # 병렬로 각 종목의 실시간 데이터 생성
            base_stocks = self._sample_stocks[index_name]
            
            # 비동기 배치 처리
            tasks = [
                self._generate_realtime_data(stock) 
                for stock in base_stocks
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 에러가 아닌 결과만 필터링
            valid_results = [
                result for result in results 
                if not isinstance(result, Exception)
            ]
            
            # 등락률 기준 정렬
            valid_results.sort(key=lambda x: x['change_rate'], reverse=True)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"지수별 주식 조회 실패 {index_name}: {e}")
            return []
    
    async def _generate_realtime_data(self, base_stock: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 주식 데이터 생성 (시뮬레이션)"""
        # 시장 시간 확인
        now = datetime.now()
        is_market_open = self._is_market_open(now)
        
        # 변동률 생성 (시장 개장 시간에 따라 다르게)
        if is_market_open:
            change_rate = np.random.normal(0, 2.5)  # 평균 0%, 표준편차 2.5%
        else:
            change_rate = np.random.normal(0, 0.5)  # 시장 외 시간에는 변동 적음
        
        base_price = base_stock["base_price"]
        current_price = base_price * (1 + change_rate / 100)
        
        # 거래량 생성
        volume = int(np.random.uniform(100000, 5000000))
        
        # 시가총액 계산 (대략적)
        shares_outstanding = np.random.uniform(100000000, 1000000000)  # 발행주식수
        market_cap = current_price * shares_outstanding / 100000000  # 억원 단위
        
        return {
            "name": base_stock["name"],
            "code": base_stock["code"],
            "sector": base_stock["sector"],
            "price": round(current_price, 0 if "KRW" in str(current_price) else 2),
            "change_rate": round(change_rate, 2),
            "volume": volume,
            "market_cap": round(market_cap, 0),
            "updated_at": now.isoformat()
        }
    
    def _is_market_open(self, dt: datetime) -> bool:
        """시장 개장 시간 확인"""
        # 한국 시장: 09:00-15:30 (월-금)
        # 미국 시장: 23:30-06:00 (한국시간, 월-금)
        
        weekday = dt.weekday()  # 0=월요일, 6=일요일
        if weekday >= 5:  # 토요일, 일요일
            return False
        
        hour = dt.hour
        minute = dt.minute
        
        # 한국 시장 시간 (09:00-15:30)
        korean_open = (9 <= hour < 15) or (hour == 15 and minute <= 30)
        
        # 미국 시장 시간 (23:30-06:00, 다음날)
        us_open = (hour >= 23 and minute >= 30) or (hour < 6)
        
        return korean_open or us_open
    
    @monitor_performance("get_stock_by_name")
    @cached(ttl=30, key_prefix="stock_by_name")
    async def get_stock_by_name(self, stock_name: str) -> Optional[Dict[str, Any]]:
        """종목명으로 주식 데이터 조회"""
        try:
            # 모든 지수에서 검색
            for index_name, stocks in self._sample_stocks.items():
                for stock in stocks:
                    if stock["name"] == stock_name:
                        return await self._generate_realtime_data(stock)
            
            return None
            
        except Exception as e:
            logger.error(f"종목 조회 실패 {stock_name}: {e}")
            return None
    
    @monitor_performance("get_stock_by_code")
    @cached(ttl=30, key_prefix="stock_by_code")
    async def get_stock_by_code(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """종목 코드로 주식 데이터 조회"""
        try:
            # 모든 지수에서 검색
            for index_name, stocks in self._sample_stocks.items():
                for stock in stocks:
                    if stock["code"] == stock_code:
                        return await self._generate_realtime_data(stock)
            
            return None
            
        except Exception as e:
            logger.error(f"종목 조회 실패 {stock_code}: {e}")
            return None
    
    @monitor_performance("get_market_summary")
    @cached(ttl=300, key_prefix="market_summary")
    async def get_market_summary(self) -> Dict[str, Any]:
        """시장 요약 정보 조회"""
        try:
            # 각 지수별 요약 정보 병렬 수집
            tasks = [
                self._get_index_summary(index_name)
                for index_name in self._sample_stocks.keys()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            summary = {}
            for i, index_name in enumerate(self._sample_stocks.keys()):
                if not isinstance(results[i], Exception):
                    summary[index_name] = results[i]
            
            return summary
            
        except Exception as e:
            logger.error(f"시장 요약 조회 실패: {e}")
            return {}
    
    async def _get_index_summary(self, index_name: str) -> Dict[str, Any]:
        """지수 요약 정보 생성"""
        stocks = await self.get_stocks_by_index(index_name)
        
        if not stocks:
            return {"error": "데이터 없음"}
        
        # 통계 계산
        prices = [stock["price"] for stock in stocks]
        change_rates = [stock["change_rate"] for stock in stocks]
        volumes = [stock["volume"] for stock in stocks]
        
        return {
            "total_stocks": len(stocks),
            "avg_price": round(np.mean(prices), 2),
            "avg_change_rate": round(np.mean(change_rates), 2),
            "total_volume": sum(volumes),
            "gainers": len([r for r in change_rates if r > 0]),
            "losers": len([r for r in change_rates if r < 0]),
            "unchanged": len([r for r in change_rates if r == 0]),
            "top_gainer": max(change_rates) if change_rates else 0,
            "top_loser": min(change_rates) if change_rates else 0,
            "updated_at": datetime.now().isoformat()
        }
    
    @monitor_performance("search_stocks")
    async def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """주식 검색"""
        try:
            query_lower = query.lower()
            results = []
            
            # 모든 지수에서 검색
            for index_name, stocks in self._sample_stocks.items():
                for stock in stocks:
                    # 이름 또는 코드로 검색
                    if (query_lower in stock["name"].lower() or 
                        query_lower in stock["code"].lower() or
                        query_lower in stock["sector"].lower()):
                        
                        realtime_data = await self._generate_realtime_data(stock)
                        realtime_data["index"] = index_name
                        results.append(realtime_data)
                        
                        if len(results) >= limit:
                            break
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"주식 검색 실패 {query}: {e}")
            return []
    
    @monitor_performance("get_sector_analysis")
    @cached(ttl=600, key_prefix="sector_analysis")
    async def get_sector_analysis(self) -> Dict[str, Any]:
        """섹터별 분석 데이터"""
        try:
            sector_data = {}
            
            # 모든 종목 데이터 수집
            all_stocks = []
            for index_name in self._sample_stocks.keys():
                stocks = await self.get_stocks_by_index(index_name)
                all_stocks.extend(stocks)
            
            # 섹터별 그룹화
            for stock in all_stocks:
                sector = stock.get("sector", "기타")
                if sector not in sector_data:
                    sector_data[sector] = {
                        "stocks": [],
                        "total_market_cap": 0,
                        "avg_change_rate": 0,
                        "total_volume": 0
                    }
                
                sector_data[sector]["stocks"].append(stock)
                sector_data[sector]["total_market_cap"] += stock.get("market_cap", 0)
                sector_data[sector]["total_volume"] += stock.get("volume", 0)
            
            # 섹터별 통계 계산
            for sector, data in sector_data.items():
                stocks = data["stocks"]
                change_rates = [s["change_rate"] for s in stocks]
                
                data["stock_count"] = len(stocks)
                data["avg_change_rate"] = round(np.mean(change_rates), 2)
                data["best_performer"] = max(stocks, key=lambda x: x["change_rate"])
                data["worst_performer"] = min(stocks, key=lambda x: x["change_rate"])
                
                # 주식 리스트는 제거 (메모리 절약)
                del data["stocks"]
            
            return sector_data
            
        except Exception as e:
            logger.error(f"섹터 분석 실패: {e}")
            return {}
    
    async def update_database(self):
        """샘플 주식 데이터를 데이터베이스에 업데이트"""
        try:
            all_stocks = []
            for market, stocks in self._sample_stocks.items():
                for stock_info in stocks:
                    # 실시간 데이터 생성
                    realtime_data = await self._generate_realtime_data(stock_info)
                    # market 정보 추가
                    realtime_data['market'] = market
                    all_stocks.append(realtime_data)
            
            if all_stocks:
                await db_manager.bulk_insert_stock_data(all_stocks)
                logger.info(f"{len(all_stocks)}개의 주식 정보 DB 업데이트 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 업데이트 실패: {e}")
    
    async def get_technical_indicators(self, stock_code: str) -> Dict[str, Any]:
        """기술적 지표 계산"""
        try:
            # 주가 이력 데이터 조회
            price_history = await db_manager.get_price_history(stock_code)
            
            if not price_history or len(price_history) < 20:
                # 데이터가 없거나 부족한 경우 현재 주가 기반으로 기본 지표 생성
                current_stock = await self.get_stock_by_code(stock_code)
                if current_stock:
                    current_price = current_stock.get('price', 50000)
                    return {
                        'ma5': current_price * 0.99,
                        'ma20': current_price * 0.98,
                        'ma60': current_price * 0.97,
                        'rsi': 50.0,
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_histogram': 0.0,
                        'bollinger_upper': current_price * 1.05,
                        'bollinger_lower': current_price * 0.95,
                        'bollinger_middle': current_price,
                        'volume_avg': 1000000
                    }
                else:
                    # 기본 지표 반환
                    return {
                        'ma5': 50000,
                        'ma20': 49000,
                        'ma60': 48000,
                        'rsi': 50.0,
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_histogram': 0.0,
                        'bollinger_upper': 52500,
                        'bollinger_lower': 47500,
                        'bollinger_middle': 50000,
                        'volume_avg': 1000000
                    }
            
            # DataFrame으로 변환
            df = pd.DataFrame(price_history)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 기술적 지표 계산
            indicators = {}
            
            # 이동평균
            if len(df) >= 5:
                indicators['ma5'] = df['close_price'].rolling(5).mean().iloc[-1]
            else:
                indicators['ma5'] = df['close_price'].iloc[-1] if len(df) > 0 else 50000
                
            if len(df) >= 20:
                indicators['ma20'] = df['close_price'].rolling(20).mean().iloc[-1]
            else:
                indicators['ma20'] = df['close_price'].iloc[-1] if len(df) > 0 else 49000
                
            if len(df) >= 60:
                indicators['ma60'] = df['close_price'].rolling(60).mean().iloc[-1]
            else:
                indicators['ma60'] = df['close_price'].iloc[-1] if len(df) > 0 else 48000
            
            # RSI
            if len(df) >= 14:
                delta = df['close_price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            else:
                indicators['rsi'] = 50.0
            
            # MACD
            if len(df) >= 26:
                exp1 = df['close_price'].ewm(span=12).mean()
                exp2 = df['close_price'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                indicators['macd'] = macd.iloc[-1]
                indicators['macd_signal'] = signal.iloc[-1]
                indicators['macd_histogram'] = (macd - signal).iloc[-1]
            else:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0
            
            # 볼린저 밴드
            if len(df) >= 20:
                ma20 = df['close_price'].rolling(20).mean()
                std20 = df['close_price'].rolling(20).std()
                indicators['bollinger_upper'] = (ma20 + 2 * std20).iloc[-1]
                indicators['bollinger_lower'] = (ma20 - 2 * std20).iloc[-1]
                indicators['bollinger_middle'] = ma20.iloc[-1]
            else:
                current_price = df['close_price'].iloc[-1] if len(df) > 0 else 50000
                indicators['bollinger_upper'] = current_price * 1.05
                indicators['bollinger_lower'] = current_price * 0.95
                indicators['bollinger_middle'] = current_price
            
            # NaN 값 처리
            for key, value in indicators.items():
                if pd.isna(value):
                    indicators[key] = 0.0
                else:
                    indicators[key] = round(float(value), 2)
            
            return indicators
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패 {stock_code}: {e}")
            # 오류 발생 시 기본 지표 반환
            return {
                'ma5': 50000,
                'ma20': 49000,
                'ma60': 48000,
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'bollinger_upper': 52500,
                'bollinger_lower': 47500,
                'bollinger_middle': 50000,
                'volume_avg': 1000000
            }
    
    async def cleanup(self):
        """데이터 매니저 정리"""
        if self._session:
            await self._session.close()
        
        self.thread_pool.shutdown(wait=False)
        logger.info("데이터 매니저 정리 완료") 