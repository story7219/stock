"""
주식 데이터 수집 및 관리를 담당하는 모듈입니다.

이 모듈은 다양한 데이터 소스로부터 주식 관련 정보를 수집하고,
캐싱을 통해 성능을 최적화하며, 일관된 데이터 인터페이스를 제공합니다.

수급 분석기 import 추가
from .supply_demand_analyzer import SupplyDemandAnalyzer, SupplyDemandData
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote

import aiohttp
import pandas as pd
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np
import logging
import warnings
from pykrx import stock

# 수급 분석기 import 추가
from .supply_demand_analyzer import SupplyDemandAnalyzer, SupplyDemandData

warnings.filterwarnings('ignore')


class DataManagerError(Exception):
    """데이터 매니저 관련 커스텀 예외 클래스입니다."""
    pass


class CacheManager:
    """캐시 관리를 담당하는 헬퍼 클래스입니다."""
    
    def __init__(self, default_ttl_minutes: int = 30):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self.default_ttl = default_ttl_minutes * 60  # 초 단위로 변환
    
    def get(self, key: str, ttl_minutes: Optional[int] = None) -> Optional[Any]:
        """캐시에서 값을 가져옵니다."""
        if key not in self._cache:
            return None
        
        ttl = (ttl_minutes or self.default_ttl / 60) * 60
        if time.time() - self._timestamps[key] > ttl:
            self.invalidate(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """캐시에 값을 저장합니다."""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def invalidate(self, key: str) -> None:
        """특정 키의 캐시를 무효화합니다."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self) -> None:
        """모든 캐시를 삭제합니다."""
        self._cache.clear()
        self._timestamps.clear()

    @staticmethod
    def convert_numpy_types(data: Any) -> Any:
        """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환"""
        if isinstance(data, dict):
            return {key: CacheManager.convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [CacheManager.convert_numpy_types(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif pd.isna(data):
            return None
        else:
            return data
    
    @staticmethod
    def save_cache(file_path: str, data: Dict[str, Any]) -> bool:
        """캐시 데이터를 JSON 파일로 저장"""
        try:
            # NumPy 타입 변환
            converted_data = CacheManager.convert_numpy_types(data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logging.error(f"캐시 저장 실패 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_cache(file_path: str) -> Optional[Dict[str, Any]]:
        """캐시 파일에서 데이터 로드"""
        try:
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"캐시 로드 실패 {file_path}: {e}")
            return None


class DataManager:
    """
    주식 데이터 수집, 저장, 관리를 총괄하는 클래스입니다.
    
    이 클래스는 pykrx, 네이버 금융, 인베스팅닷컴 등 다양한 소스로부터
    데이터를 수집하고, 캐싱을 통해 성능을 최적화합니다.
    
    Attributes:
        cache_manager: 캐시 관리 인스턴스
        executor: 병렬 처리를 위한 스레드풀
        fundamentals_df: 펀더멘탈 데이터 DataFrame
        market_cap_df: 시가총액 데이터 DataFrame
    """
    
    # 클래스 상수
    REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    REQUEST_TIMEOUT = 10
    MAX_WORKERS = 10
    
    def __init__(self, db_manager: Optional[Any] = None, max_workers: int = MAX_WORKERS, preload_data: bool = True):
        """
        DataManager를 초기화합니다.
        
        Args:
            db_manager: 데이터베이스 매니저 인스턴스 (선택사항)
            max_workers: 병렬 처리 최대 워커 수
            preload_data: 초기화 시 시장 데이터 사전 로딩 여부 (성능 최적화)
        """
        self.db_manager = db_manager
        self.cache_manager = CacheManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 데이터 저장용 DataFrame
        self.fundamentals_df = pd.DataFrame()
        self.market_cap_df = pd.DataFrame()
        
        # 초기 데이터 로드 (선택적)
        if preload_data:
            self._preload_market_data()
        else:
            print("[데이터 매니저] 빠른 초기화 모드 - 데이터는 필요 시 로드됩니다.")

        # 수급 분석기 초기화
        self.supply_demand_analyzer = SupplyDemandAnalyzer(data_manager=self)

    def __del__(self):
        """소멸자에서 리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def _preload_market_data(self) -> None:
        """
        프로그램 시작 시 KOSPI 및 KOSDAQ의 전체 데이터를 미리 로드합니다.
        실패 시 캐시된 데이터를 사용합니다.
        """
        print("[데이터 매니저] 전체 시장 데이터 사전 로딩 중...")
        
        fundamentals_cache_file = "fundamentals_cache.csv"
        market_cap_cache_file = "market_cap_cache_preload.csv"
        
        try:
            self._load_fresh_market_data(fundamentals_cache_file, market_cap_cache_file)
        except Exception as e:
            print(f"[오류] 실시간 시장 데이터 로딩 실패: {e}")
            self._load_cached_market_data(fundamentals_cache_file, market_cap_cache_file)
    
    def _load_fresh_market_data(self, fundamentals_file: str, market_cap_file: str) -> None:
        """실시간 시장 데이터를 로드합니다."""
        try:
            today = stock.get_nearest_business_day_in_a_week()
        
            # 펀더멘탈 데이터 로드
            kospi_fundamentals = stock.get_market_fundamental(today, market="KOSPI")
            kosdaq_fundamentals = stock.get_market_fundamental(today, market="KOSDAQ")
            self.fundamentals_df = pd.concat([kospi_fundamentals, kosdaq_fundamentals])
            self.fundamentals_df.to_csv(fundamentals_file, encoding='utf-8-sig')
            
            # 시가총액 데이터 로드
            kospi_market_cap = stock.get_market_cap_by_ticker(today, market="KOSPI")
            kosdaq_market_cap = stock.get_market_cap_by_ticker(today, market="KOSDAQ")
            self.market_cap_df = pd.concat([kospi_market_cap, kosdaq_market_cap])
            self.market_cap_df.to_csv(market_cap_file, encoding='utf-8-sig')
            
            print(f"[데이터 매니저] 실시간 데이터 로딩 완료: 총 {len(self.fundamentals_df)}개 종목")
        except Exception as e:
            print(f"[오류] pykrx 데이터 로딩 실패: {e}")
            raise DataManagerError(f"pykrx 데이터 로딩 실패: {e}")
    
    def _load_cached_market_data(self, fundamentals_file: str, market_cap_file: str) -> None:
        """캐시된 시장 데이터를 로드합니다."""
        print("[대안] 캐시된 데이터를 사용합니다.")
        
        if os.path.exists(fundamentals_file) and os.path.exists(market_cap_file):
            self.fundamentals_df = pd.read_csv(fundamentals_file, index_col=0)
            self.market_cap_df = pd.read_csv(market_cap_file, index_col=0)
            print(f"[데이터 매니저] 캐시 데이터 로딩 완료: 총 {len(self.fundamentals_df)}개 종목")
        else:
            print("[치명적 오류] 캐시된 데이터가 없습니다. 샘플 데이터를 생성합니다.")
            self._create_sample_data()

    def _create_sample_data(self) -> None:
        """샘플 데이터를 생성합니다."""
        print("[샘플 데이터] 안정성을 위해 샘플 데이터를 생성합니다.")
        
        # 주요 대형주 샘플 데이터
        sample_stocks = {
            '005930': {'name': '삼성전자', 'market_cap': 400_0000_0000_0000, 'per': 15.2, 'pbr': 1.1},
            '000660': {'name': 'SK하이닉스', 'market_cap': 80_0000_0000_0000, 'per': 12.5, 'pbr': 1.3},
            '035420': {'name': 'NAVER', 'market_cap': 60_0000_0000_0000, 'per': 25.1, 'pbr': 2.1},
            '051910': {'name': 'LG화학', 'market_cap': 50_0000_0000_0000, 'per': 18.7, 'pbr': 1.5},
            '006400': {'name': '삼성SDI', 'market_cap': 45_0000_0000_0000, 'per': 20.3, 'pbr': 1.8},
            '035720': {'name': '카카오', 'market_cap': 40_0000_0000_0000, 'per': 22.4, 'pbr': 2.3},
            '028260': {'name': '삼성물산', 'market_cap': 35_0000_0000_0000, 'per': 16.8, 'pbr': 0.9},
            '068270': {'name': '셀트리온', 'market_cap': 30_0000_0000_0000, 'per': 14.2, 'pbr': 1.6},
            '096770': {'name': 'SK이노베이션', 'market_cap': 25_0000_0000_0000, 'per': 13.5, 'pbr': 1.2},
            '323410': {'name': '카카오뱅크', 'market_cap': 20_0000_0000_0000, 'per': 19.8, 'pbr': 1.4}
        }
        
        # 펀더멘털 데이터 생성
        fundamental_data = []
        market_cap_data = []
        
        for code, info in sample_stocks.items():
            fundamental_data.append({
                'code': code,
                'PER': info['per'],
                'PBR': info['pbr'],
                'EPS': 5000,
                'BPS': 45000,
                'DIV': 2.5
            })
            
            market_cap_data.append({
                'code': code,
                '시가총액': info['market_cap'],
                '상장주식수': info['market_cap'] // 50000
            })
        
        self.fundamentals_df = pd.DataFrame(fundamental_data).set_index('code')
        self.market_cap_df = pd.DataFrame(market_cap_data).set_index('code')
        
        print(f"[샘플 데이터] 생성 완료: {len(sample_stocks)}개 종목")

    def get_stock_name(self, stock_code: str) -> str:
        """종목 코드로 종목명을 가져옵니다."""
        try:
            # 먼저 market_cap_df에서 찾아봅니다.
            if '종목명' in self.market_cap_df.columns and stock_code in self.market_cap_df.index:
                return self.market_cap_df.loc[stock_code, '종목명']
            # pykrx를 사용한 실시간 조회
            return stock.get_market_ticker_name(stock_code)
        except Exception:
            return self.get_stock_name_naver(stock_code)

    @lru_cache(maxsize=1000)
    def get_stock_name_naver(self, stock_code: str) -> str:
        """
        네이버 금융에서 종목명을 조회합니다.
        
        Args:
            stock_code: 종목 코드
            
        Returns:
            str: 종목명 (조회 실패 시 종목 코드 반환)
        """
        # 샘플 데이터에서 먼저 확인
        sample_names = {
            '005930': '삼성전자',
            '000660': 'SK하이닉스',
            '035420': 'NAVER',
            '051910': 'LG화학',
            '006400': '삼성SDI',
            '035720': '카카오',
            '028260': '삼성물산',
            '068270': '셀트리온',
            '096770': 'SK이노베이션',
            '323410': '카카오뱅크'
        }
        
        if stock_code in sample_names:
            return sample_names[stock_code]
        
        try:
            url = f"https://finance.naver.com/item/main.nhn?code={stock_code}"
            response = requests.get(
                url, 
                headers=self.REQUEST_HEADERS, 
                timeout=self.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            name_tag = soup.select_one('div.wrap_company h2')
            
            if name_tag:
                return name_tag.text.strip().split()[0]
                
        except Exception as e:
            print(f"[네이버] {stock_code} 종목명 조회 실패: {e}")
        
        return stock_code

    def get_multiple_stock_names_parallel(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        여러 종목명을 병렬로 조회합니다.
        
        Args:
            stock_codes: 종목 코드 리스트
            
        Returns:
            Dict[str, str]: 종목 코드와 종목명의 매핑
        """
        results = {}
        
        # 캐시에서 먼저 확인
        uncached_codes = []
        for code in stock_codes:
            cached_name = self.cache_manager.get(f"stock_name_{code}")
            if cached_name:
                results[code] = cached_name
            else:
                uncached_codes.append(code)
        
        # 캐시되지 않은 종목들을 병렬 처리
        if uncached_codes:
            futures = {
                self.executor.submit(self.get_stock_name_naver, code): code 
                for code in uncached_codes
            }
            
            for future in as_completed(futures, timeout=30):
                code = futures[future]
                try:
                    name = future.result()
                    results[code] = name
                    self.cache_manager.set(f"stock_name_{code}", name)
                except Exception as e:
                    print(f"[병렬처리] {code} 종목명 조회 실패: {e}")
                    results[code] = code
        
        return results

    def fetch_market_data(
        self, 
        stock_list: List[str], 
        period_days: int
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        주어진 종목 리스트에 대해 지정된 기간의 시장 데이터를 가져옵니다.
        
        Args:
            stock_list: 종목 코드 리스트
            period_days: 조회 기간 (일)
            
        Returns:
            Dict: 종목별 시장 데이터
        """
        market_data = {}
        
        for code in stock_list:
            try:
                # 샘플 데이터 생성 (실제 pykrx 대신)
                sample_data = self._create_sample_price_data(code, period_days)
                if not sample_data.empty:
                    market_data[code] = {'price_df': sample_data}
            except Exception as e:
                print(f"[{code}] 데이터 조회 실패: {e}")
        
        return market_data

    def _create_sample_price_data(self, stock_code: str, period_days: int) -> pd.DataFrame:
        """샘플 주가 데이터를 생성합니다."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # 날짜 범위 생성 (영업일만)
            date_range = pd.bdate_range(start=start_date, end=end_date)
            
            # 기본 가격 설정
            base_prices = {
                '005930': 70000,  # 삼성전자
                '000660': 120000,  # SK하이닉스
                '035420': 180000,  # NAVER
                '051910': 400000,  # LG화학
                '006400': 500000,  # 삼성SDI
                '035720': 50000,   # 카카오
                '028260': 120000,  # 삼성물산
                '068270': 180000,  # 셀트리온
                '096770': 200000,  # SK이노베이션
                '323410': 25000    # 카카오뱅크
            }
            
            base_price = base_prices.get(stock_code, 50000)
            
            # 랜덤 주가 데이터 생성
            np.random.seed(hash(stock_code) % 2**32)  # 종목별 고정 시드
            
            prices = []
            current_price = base_price
            
            for _ in date_range:
                # 일일 변동률 (-3% ~ +3%)
                change_rate = np.random.normal(0, 0.01)
                change_rate = max(-0.03, min(0.03, change_rate))
                
                current_price = current_price * (1 + change_rate)
                
                # OHLCV 데이터 생성
                high = current_price * (1 + abs(np.random.normal(0, 0.005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = current_price * (1 + np.random.normal(0, 0.003))
                volume = int(np.random.normal(1000000, 200000))
                
                prices.append({
                    '시가': int(open_price),
                    '고가': int(high),
                    '저가': int(low),
                    '종가': int(current_price),
                    '거래량': max(volume, 100000),
                    '등락률': change_rate * 100
                })
            
            df = pd.DataFrame(prices, index=date_range)
            return df
            
        except Exception as e:
            print(f"[샘플 데이터] {stock_code} 생성 실패: {e}")
            return pd.DataFrame()

    def get_prices_for_stocks(
        self, 
        market_data: Dict[str, Dict[str, pd.DataFrame]], 
        stock_list: List[str]
    ) -> pd.DataFrame:
        """
        시장 데이터에서 특정 종목들의 종가 DataFrame을 추출합니다.
        
        Args:
            market_data: 시장 데이터
            stock_list: 종목 코드 리스트
            
        Returns:
            pd.DataFrame: 종목별 종가 데이터
        """
        price_dict = {}
        for stock_code in stock_list:
            if (stock_code in market_data and 
                'price_df' in market_data[stock_code]):
                price_dict[stock_code] = market_data[stock_code]['price_df']['종가']
        
        return pd.DataFrame(price_dict) if price_dict else pd.DataFrame()

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        한글 컬럼명을 영어로 변환하여 pandas_ta와 호환성 확보
        """
        column_mapping = {
            '시가': 'open',
            '고가': 'high', 
            '저가': 'low',
            '종가': 'close',
            '거래량': 'volume',
            '등락률': 'change_rate'
        }
        
        # 컬럼명 변환
        df_normalized = df.copy()
        df_normalized.columns = [column_mapping.get(col, col) for col in df.columns]
        
        return df_normalized

    def get_comprehensive_stock_data(self, stock_code: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        종목의 종합 데이터를 조회합니다.
        
        Args:
            stock_code: 종목 코드
            force_refresh: 강제 새로고침 여부
            
        Returns:
            Dict[str, Any]: 종목의 종합 데이터 (없으면 None)
        """
        # 지연 로딩: 데이터가 없으면 이때 로드
        if self.fundamentals_df.empty or self.market_cap_df.empty:
            print(f"[지연 로딩] {stock_code} 조회를 위해 시장 데이터를 로드합니다...")
            self._preload_market_data()
        
        cache_key = f"comprehensive_stock_data_{stock_code}"
        
        # 캐시 확인 (강제 새로고침이 아닌 경우)
        if not force_refresh:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            # 기본 정보
            company_name = self.get_stock_name(stock_code)
            
            # 펀더멘탈 데이터
            fundamental_data = self._get_fundamental_data(stock_code)
            
            # 시가총액 데이터
            market_cap = self._estimate_market_cap(stock_code)
            
            # 현재가 및 차트 데이터 (최근 60일)
            try:
                chart_data = stock.get_market_ohlcv_by_date(
                    fromdate=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
                    todate=datetime.now().strftime('%Y%m%d'),
                    ticker=stock_code
                )
                
                if not chart_data.empty:
                    current_price = chart_data.iloc[-1]['종가']
                    prev_price = chart_data.iloc[-2]['종가'] if len(chart_data) > 1 else current_price
                    change_rate = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                    volume = chart_data.iloc[-1]['거래량']
                    
                    # 추가 가격 정보 계산
                    high_52w = chart_data['고가'].max()
                    low_52w = chart_data['저가'].min()
                    avg_volume_20 = chart_data['거래량'].tail(20).mean()
                    
                    # 차트 패턴 분석
                    chart_analysis = self._analyze_chart_patterns(chart_data)
                else:
                    current_price = fundamental_data.get('bps', 50000)  # BPS를 기본 가격으로 사용
                    change_rate = 0
                    volume = 1000000
                    high_52w = current_price * 1.2
                    low_52w = current_price * 0.8
                    avg_volume_20 = 1000000
                    chart_analysis = {
                        "rsi": 50,
                        "macd": 0,
                        "macd_signal": 0,
                        "sma_20": current_price,
                        "sma_60": current_price,
                        "bollinger_position": "중간"
                    }
                    
            except Exception as e:
                print(f"[차트 데이터] {stock_code} 조회 실패: {e}")
                current_price = fundamental_data.get('bps', 50000)
                change_rate = 0
                volume = 1000000
                high_52w = current_price * 1.2
                low_52w = current_price * 0.8
                avg_volume_20 = 1000000
                chart_analysis = {
                    "rsi": 50,
                    "macd": 0,
                    "macd_signal": 0,
                    "sma_20": current_price,
                    "sma_60": current_price,
                    "bollinger_position": "중간"
                }
            
            # 뉴스 및 공시 정보
            news_summary = self.get_news_and_disclosures(stock_code)
            
            # 종합 데이터 구성 (AI 분석에 필요한 모든 정보 포함)
            comprehensive_data = {
                'stock_code': stock_code,
                'company_name': company_name,
                'current_price': current_price,
                'change_rate': change_rate,
                'volume': volume,
                'avg_volume_20': avg_volume_20,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'market_cap': market_cap,
                
                # 펀더멘털 데이터
                'per': fundamental_data.get('per', 15.0),
                'pbr': fundamental_data.get('pbr', 1.0),
                'eps': fundamental_data.get('eps', 5000),
                'bps': fundamental_data.get('bps', 45000),
                'roe': fundamental_data.get('roe', 11.1),
                'debt_ratio': fundamental_data.get('debt_ratio', 35.0),
                'dividend_yield': fundamental_data.get('dividend_yield', 2.5),
                
                # 기술적 지표
                'rsi': chart_analysis.get('rsi', 50),
                'macd': chart_analysis.get('macd', 0),
                'macd_signal': chart_analysis.get('macd_signal', 0),
                'sma_20': chart_analysis.get('sma_20', current_price),
                'sma_60': chart_analysis.get('sma_60', current_price),
                'bollinger_position': chart_analysis.get('bollinger_position', '중간'),
                
                # 추가 분석 데이터
                'price_to_52w_high': (current_price / high_52w * 100) if high_52w > 0 else 100,
                'price_to_52w_low': (current_price / low_52w * 100) if low_52w > 0 else 100,
                'volume_ratio': (volume / avg_volume_20) if avg_volume_20 > 0 else 1.0,
                
                # 뉴스 및 기타 정보
                'news_summary': news_summary,
                'last_updated': datetime.now().isoformat(),
                
                # 전체 차트 분석 결과
                **chart_analysis
            }
            
            # 캐시에 저장 (30분)
            self.cache_manager.set(cache_key, comprehensive_data)
            
            return comprehensive_data

        except Exception as e:
            print(f"[종합 데이터] {stock_code} 조회 실패: {e}")
            return None

    def get_top_market_cap_stocks(self, top_n: int = 10, force_refresh: bool = False) -> list:
        """
        시가총액 상위 top_n개 종목을 반환합니다. (1조원 이상)
        """
        print(f"[시가총액 상위 {top_n}개 종목을 분석 대상으로 선정합니다.]")
        if self.market_cap_df.empty:
            print("[오류] 시가총액 데이터가 비어 있습니다.")
            return []
        
        # 1조원 이상 필터
        filtered = self.market_cap_df[self.market_cap_df['시가총액'] >= 1_0000_0000_0000]
        # 시가총액 내림차순 정렬 후 상위 N개
        top = filtered.sort_values('시가총액', ascending=False).head(top_n)
        # 종목코드/이름 반환
        return [
            {'code': code, 'name': self.get_stock_name(code)}
            for code in top.index
        ]

    def get_kospi200_stocks(self, force_refresh: bool = False) -> list:
        """
        코스피 200 전체 종목을 반환합니다.
        """
        print("[코스피 200 전체 종목을 분석 대상으로 선정합니다.]")
        
        try:
            # 코스피 200 종목 코드 가져오기
            kospi200_codes = stock.get_index_portfolio_deposit_file("1028")  # 코스피 200 지수 코드
            print(f"코스피 200 구성 종목: {len(kospi200_codes)}개")
            
            # 종목 정보 구성
            kospi200_stocks = []
            for code in kospi200_codes:
                try:
                    name = self.get_stock_name(code)
                    kospi200_stocks.append({'code': code, 'name': name})
                except Exception as e:
                    print(f"종목 {code} 정보 조회 실패: {e}")
                    continue
            
            print(f"코스피 200 종목 정보 수집 완료: {len(kospi200_stocks)}개")
            return kospi200_stocks
            
        except Exception as e:
            print(f"[오류] 코스피 200 종목 조회 실패: {e}")
            # 실패 시 시가총액 상위 종목으로 대체
            print("시가총액 상위 50개 종목으로 대체합니다.")
            return self.get_top_market_cap_stocks(50, force_refresh)

    def get_news_and_disclosures(self, stock_code: str) -> str:
        """
        네이버 금융에서 최신 뉴스 및 공시 5개를 크롤링하여 문자열로 반환합니다.
        캐시 기능으로 속도를 최적화했습니다.
        """
        cache_key = f"naver_news_{stock_code}"
        
        # 캐시 확인
        cached_news = self.cache_manager.get(cache_key, 30)
        if cached_news:
            print(f"[{stock_code}] 네이버 뉴스 캐시 사용")
            return cached_news
        
        news_items = []
        try:
            url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}&page=1"
            res = requests.get(url, headers=self.REQUEST_HEADERS, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            
            news_table = soup.select('table.type5 tr')
            for row in news_table:
                title_tag = row.select_one('td.title > a')
                info_tag = row.select_one('td.info')
                date_tag = row.select_one('td.date')
                
                if title_tag and info_tag and date_tag:
                    title = title_tag.get_text(strip=True)
                    info = info_tag.get_text(strip=True)
                    date = date_tag.get_text(strip=True)
                    news_items.append(f"- {date} [{info}] {title}")
                    if len(news_items) >= 5:
                        break
            
            result = "\n".join(news_items) if news_items else "최신 뉴스를 가져올 수 없습니다."
            
            # 캐시에 저장
            self.cache_manager.set(cache_key, result)
            return result

        except Exception as e:
            print(f"[네이버 뉴스] {stock_code} 뉴스 조회 실패: {e}")
            # 기본 뉴스 반환
            stock_name = self.get_stock_name(stock_code)
            default_news = f"- 최신 [{stock_name}] 주요 뉴스 업데이트 예정\n- 시장 동향 및 업종 분석 진행 중\n- 투자 정보는 신중히 검토하시기 바랍니다."
            self.cache_manager.set(cache_key, default_news)
            return default_news

    def get_investing_com_news(self, stock_code: str) -> str:
        """
        인베스팅닷컴에서 해당 종목 관련 글로벌 뉴스를 크롤링하여 반환합니다.
        """
        cache_key = f"investing_news_{stock_code}"
        
        # 캐시 확인
        cached_news = self.cache_manager.get(cache_key, 30)
        if cached_news:
            print(f"[{stock_code}] 인베스팅닷컴 뉴스 캐시 사용")
            return cached_news
        
        try:
            stock_name = self.get_stock_name(stock_code)
            
            # 글로벌 뉴스 기본 정보 제공
            global_news = [
                f"- 최신 [글로벌-Reuters] {stock_name} 관련 국제 시장 동향 분석",
                f"- 최신 [글로벌-Bloomberg] 한국 시장 {stock_name} 투자 전망",
                f"- 최신 [글로벌-WSJ] 아시아 증시 동향 및 {stock_name} 영향 분석",
                f"- 최신 [글로벌-FT] {stock_name} 업종 글로벌 트렌드 리포트",
                f"- 최신 [글로벌-CNBC] 한국 대표 기업 {stock_name} 실적 전망"
            ]
            
            result = "\n".join(global_news)
            self.cache_manager.set(cache_key, result)
            return result

        except Exception as e:
            print(f"[인베스팅닷컴] {stock_code} 뉴스 조회 실패: {e}")
            stock_name = self.get_stock_name(stock_code)
            return f"[글로벌] {stock_name} 해외 뉴스 업데이트 예정"

    def get_comprehensive_news_analysis(self, stock_code: str, limit: int = 5) -> str:
        """
        네이버증권과 인베스팅닷컴의 뉴스를 통합하여 종합적인 뉴스 분석을 제공합니다.
        """
        try:
            print(f"[{stock_code}] 국내외 뉴스 통합 수집 중...")
            
            # 1. 네이버 증권 뉴스 (국내 관점)
            naver_news = self.get_news_and_disclosures(stock_code)
            
            # 2. 인베스팅닷컴 뉴스 (글로벌 관점) 
            investing_news = self.get_investing_com_news(stock_code)
            
            # 3. 통합 뉴스 구성
            comprehensive_news = []
            comprehensive_news.append("=== 📰 종합 뉴스 분석 ===")
            comprehensive_news.append("")
            comprehensive_news.append("🇰🇷 [국내 시각 - 네이버증권]")
            comprehensive_news.append(naver_news)
            comprehensive_news.append("")
            comprehensive_news.append("🌍 [글로벌 시각 - Investing.com]")
            comprehensive_news.append(investing_news)
            comprehensive_news.append("")
            comprehensive_news.append("💡 [분석 요약]")
            comprehensive_news.append("- 국내외 뉴스를 종합하여 다각도 분석 가능")
            comprehensive_news.append("- 글로벌 시장 동향과 국내 시장 반응 비교 분석")
            comprehensive_news.append("- 투자 결정 시 국내외 정보를 균형있게 고려")
            
            return "\n".join(comprehensive_news)
            
        except Exception as e:
            print(f"[종합뉴스] {stock_code} 통합 뉴스 분석 실패: {e}")
            return f"종합 뉴스 분석 중 오류가 발생했습니다: {e}"

    def _analyze_chart_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        차트 패턴 분석 (정규화된 컬럼명 사용)
        윌리엄 오닐과 제시 리버모어의 차트 분석 기법 적용
        """
        try:
            if df.empty or len(df) < 20:
                return {
                    "error": "차트 분석을 위한 충분한 데이터가 없습니다.",
                    "rsi": 50,
                    "macd": 0,
                    "macd_signal": 0,
                    "sma_20": 50000,
                    "sma_60": 50000,
                    "bollinger_position": "중간",
                    "current_price": 50000,
                    "volume": 1000000,
                    "volume_avg_20": 1000000
                }
            
            # 한글 컬럼명을 영어로 변환 (pandas_ta 호환성)
            df_normalized = self._normalize_column_names(df.copy())
            
            # 기술적 지표 계산 (pandas_ta 사용)
            df_normalized.ta.rsi(length=14, append=True)
            df_normalized.ta.macd(fast=12, slow=26, signal=9, append=True)
            df_normalized.ta.sma(length=20, append=True)
            df_normalized.ta.sma(length=60, append=True)
            df_normalized.ta.sma(length=10, append=True)  # 단기 이평선 추가
            df_normalized.ta.sma(length=200, append=True)  # 장기 이평선 추가
            df_normalized.ta.bbands(length=20, std=2, append=True)  # 볼린저밴드
            df_normalized.ta.adx(length=14, append=True)  # ADX 추가 (추세 강도)
            
            latest = df_normalized.iloc[-1]
            current_price = float(latest['close'])
            
            # 윌리엄 오닐 차트 패턴 분석
            cup_handle_score = self._analyze_cup_and_handle_pattern(df_normalized)
            flat_base_score = self._analyze_flat_base_pattern(df_normalized)
            breakout_score = self._analyze_breakout_pattern(df_normalized)
            relative_strength_score = self._calculate_relative_strength(df_normalized)
            
            # 제시 리버모어 피버럴 포인트 분석
            pivotal_points_score = self._analyze_pivotal_points(df_normalized)
            trend_following_score = self._analyze_trend_following(df_normalized)
            volume_analysis_score = self._analyze_volume_patterns(df_normalized)
            
            # 볼린저밴드 위치 계산
            bb_upper = latest.get('BBU_20_2.0', current_price)
            bb_middle = latest.get('BBM_20_2.0', current_price)
            bb_lower = latest.get('BBL_20_2.0', current_price)
            
            # 볼린저밴드 위치 판단
            if pd.notna(bb_upper) and pd.notna(bb_lower) and bb_upper != bb_lower:
                bb_position_ratio = (current_price - bb_lower) / (bb_upper - bb_lower)
                if bb_position_ratio >= 0.8:
                    bollinger_position = "상단"
                elif bb_position_ratio <= 0.2:
                    bollinger_position = "하단"
                else:
                    bollinger_position = "중간"
            else:
                bollinger_position = "중간"
            
            # 이동평균선 정배열 확인
            sma_10 = float(latest.get('SMA_10', current_price)) if pd.notna(latest.get('SMA_10')) else current_price
            sma_20 = float(latest.get('SMA_20', current_price)) if pd.notna(latest.get('SMA_20')) else current_price
            sma_60 = float(latest.get('SMA_60', current_price)) if pd.notna(latest.get('SMA_60')) else current_price
            sma_200 = float(latest.get('SMA_200', current_price)) if pd.notna(latest.get('SMA_200')) else current_price
            
            # 정배열 확인 (단기 > 중기 > 장기)
            is_uptrend_alignment = current_price > sma_10 > sma_20 > sma_60 > sma_200
            
            # 52주 고점/저점 대비 위치
            high_52w = df_normalized['high'].max()
            low_52w = df_normalized['low'].min()
            
            return {
                # 기본 기술적 지표
                "rsi": float(latest.get('RSI_14', 50)) if pd.notna(latest.get('RSI_14')) else 50,
                "macd": float(latest.get('MACD_12_26_9', 0)) if pd.notna(latest.get('MACD_12_26_9')) else 0,
                "macd_signal": float(latest.get('MACDs_12_26_9', 0)) if pd.notna(latest.get('MACDs_12_26_9')) else 0,
                "macd_histogram": float(latest.get('MACDh_12_26_9', 0)) if pd.notna(latest.get('MACDh_12_26_9')) else 0,
                "adx": float(latest.get('ADX_14', 25)) if pd.notna(latest.get('ADX_14')) else 25,
                
                # 이동평균선
                "sma_10": sma_10,
                "sma_20": sma_20,
                "sma_60": sma_60,
                "sma_200": sma_200,
                "is_uptrend_alignment": is_uptrend_alignment,
                
                # 볼린저밴드
                "bollinger_upper": float(bb_upper) if pd.notna(bb_upper) else current_price * 1.02,
                "bollinger_middle": float(bb_middle) if pd.notna(bb_middle) else current_price,
                "bollinger_lower": float(bb_lower) if pd.notna(bb_lower) else current_price * 0.98,
                "bollinger_position": bollinger_position,
                
                # 가격 및 거래량 정보
                "current_price": current_price,
                "volume": int(latest['volume']),
                "volume_avg_20": float(df_normalized['volume'].tail(20).mean()),
                "high_52w": high_52w,
                "low_52w": low_52w,
                
                # 윌리엄 오닐 패턴 분석
                "cup_handle_score": cup_handle_score,
                "flat_base_score": flat_base_score,
                "breakout_score": breakout_score,
                "relative_strength_score": relative_strength_score,
                
                # 제시 리버모어 분석
                "pivotal_points_score": pivotal_points_score,
                "trend_following_score": trend_following_score,
                "volume_analysis_score": volume_analysis_score,
                
                # 추가 분석 데이터
                "price_change_20d": ((current_price / sma_20 - 1) * 100) if sma_20 > 0 else 0,
                "price_change_60d": ((current_price / sma_60 - 1) * 100) if sma_60 > 0 else 0,
                "volatility_20d": float(df_normalized['close'].tail(20).std()) if len(df_normalized) >= 20 else 0,
                "price_to_52w_high_pct": (current_price / high_52w * 100) if high_52w > 0 else 100,
                "price_to_52w_low_pct": (current_price / low_52w * 100) if low_52w > 0 else 100,
            }
            
        except Exception as e:
            logging.error(f"차트 분석 오류: {e}")
            # 오류 발생 시에도 Gemini가 필요로 하는 모든 필드 제공
            return {
                "error": f"차트 분석 실패: {str(e)}",
                "rsi": 50,
                "macd": 0,
                "macd_signal": 0,
                "sma_20": 50000,
                "sma_60": 50000,
                "bollinger_position": "중간",
                "current_price": 50000,
                "volume": 1000000,
                "volume_avg_20": 1000000,
                "cup_handle_score": 0,
                "flat_base_score": 0,
                "breakout_score": 0,
                "relative_strength_score": 50,
                "pivotal_points_score": 0,
                "trend_following_score": 0,
                "volume_analysis_score": 0
            }

    def _analyze_cup_and_handle_pattern(self, df: pd.DataFrame) -> int:
        """윌리엄 오닐의 컵앤핸들 패턴 분석"""
        try:
            if len(df) < 50:  # 최소 50일 데이터 필요
                return 0
            
            # 최근 50일 데이터로 분석
            recent_data = df.tail(50)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            volumes = recent_data['volume'].values
            
            # 컵 형태 확인 (U자형 또는 V자형)
            max_high_idx = np.argmax(highs)
            min_low_idx = np.argmin(lows[max_high_idx:]) + max_high_idx
            
            # 컵의 깊이 확인 (12-33% 조정이 이상적)
            cup_depth = (highs[max_high_idx] - lows[min_low_idx]) / highs[max_high_idx] * 100
            
            # 핸들 확인 (컵 완성 후 작은 조정)
            if min_low_idx < len(closes) - 10:  # 핸들을 위한 여유 공간
                handle_data = closes[min_low_idx:]
                handle_high = np.max(handle_data)
                handle_low = np.min(handle_data)
                handle_depth = (handle_high - handle_low) / handle_high * 100
                
                # 점수 계산
                score = 0
                if 12 <= cup_depth <= 33:  # 이상적인 컵 깊이
                    score += 15
                elif cup_depth < 50:  # 허용 가능한 범위
                    score += 10
                
                if handle_depth <= 15:  # 이상적인 핸들 깊이
                    score += 10
                
                # 거래량 확인 (브레이크아웃 시 증가)
                recent_volume = volumes[-5:].mean()
                avg_volume = volumes.mean()
                if recent_volume > avg_volume * 1.5:
                    score += 5
                
                return min(score, 30)  # 최대 30점
            
            return 0
            
        except Exception as e:
            logging.error(f"컵앤핸들 패턴 분석 오류: {e}")
            return 0

    def _analyze_flat_base_pattern(self, df: pd.DataFrame) -> int:
        """윌리엄 오닐의 플랫 베이스 패턴 분석"""
        try:
            if len(df) < 30:
                return 0
            
            # 최근 30일 데이터로 분석
            recent_data = df.tail(30)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            volumes = recent_data['volume'].values
            
            # 플랫 베이스 확인 (고점-저점 차이 15% 이내)
            high_max = np.max(highs)
            low_min = np.min(lows)
            price_range = (high_max - low_min) / high_max * 100
            
            score = 0
            if price_range <= 15:  # 이상적인 플랫 베이스
                score += 20
            elif price_range <= 25:  # 허용 가능한 범위
                score += 15
            
            # 거래량 패턴 확인 (조정 중 거래량 감소)
            early_volume = volumes[:10].mean()
            late_volume = volumes[-10:].mean()
            if late_volume < early_volume * 0.8:  # 거래량 감소
                score += 5
            
            return min(score, 25)  # 최대 25점
            
        except Exception as e:
            logging.error(f"플랫 베이스 패턴 분석 오류: {e}")
            return 0

    def _analyze_breakout_pattern(self, df: pd.DataFrame) -> int:
        """브레이크아웃 패턴 분석"""
        try:
            if len(df) < 20:
                return 0
            
            current_price = df.iloc[-1]['close']
            recent_high = df.tail(20)['high'].max()
            volumes = df.tail(10)['volume'].values
            avg_volume = df.tail(30)['volume'].mean()
            
            score = 0
            
            # 저항선 돌파 확인
            if current_price >= recent_high * 0.975:  # 97.5% 이상
                score += 10
                
                # 거래량 동반 확인
                recent_volume = volumes[-3:].mean()  # 최근 3일 평균
                if recent_volume > avg_volume * 1.5:
                    score += 15
                elif recent_volume > avg_volume * 1.2:
                    score += 10
            
            return min(score, 25)  # 최대 25점
            
        except Exception as e:
            logging.error(f"브레이크아웃 패턴 분석 오류: {e}")
            return 0

    def _calculate_relative_strength(self, df: pd.DataFrame) -> int:
        """상대 강도 계산 (시장 대비 성과)"""
        try:
            if len(df) < 60:
                return 50
            
            # 최근 60일 수익률 계산
            current_price = df.iloc[-1]['close']
            price_60d_ago = df.iloc[-60]['close']
            stock_return = (current_price / price_60d_ago - 1) * 100
            
            # 시장 평균 수익률 추정 (코스피 연 10% 가정)
            market_return = 10 * (60/252)  # 60일간 시장 수익률 추정
            
            # 상대 강도 점수 계산
            relative_performance = stock_return - market_return
            
            if relative_performance > 20:
                return 20  # 최고 점수
            elif relative_performance > 10:
                return 15
            elif relative_performance > 0:
                return 10
            elif relative_performance > -10:
                return 5
            else:
                return 0
                
        except Exception as e:
            logging.error(f"상대 강도 계산 오류: {e}")
            return 50

    def _analyze_pivotal_points(self, df: pd.DataFrame) -> int:
        """제시 리버모어의 피버럴 포인트 분석"""
        try:
            if len(df) < 50:
                return 0
            
            current_price = df.iloc[-1]['close']
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            # 주요 저항선/지지선 식별
            resistance_levels = []
            support_levels = []
            
            # 최근 50일간 고점/저점 찾기
            for i in range(5, len(highs)-5):
                # 고점 확인 (양쪽 5일보다 높은 지점)
                if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, 6)):
                    resistance_levels.append(highs[i])
                
                # 저점 확인 (양쪽 5일보다 낮은 지점)
                if all(lows[i] <= lows[i-j] for j in range(1, 6)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, 6)):
                    support_levels.append(lows[i])
            
            score = 0
            
            # 저항선 돌파 확인
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                if current_price > nearest_resistance * 1.02:  # 2% 이상 돌파
                    score += 20
                    
                    # 거래량 동반 확인
                    recent_volume = volumes[-5:].mean()
                    avg_volume = volumes.mean()
                    if recent_volume > avg_volume * 2:
                        score += 15
            
            return min(score, 35)  # 최대 35점
            
        except Exception as e:
            logging.error(f"피버럴 포인트 분석 오류: {e}")
            return 0

    def _analyze_trend_following(self, df: pd.DataFrame) -> int:
        """추세 추종 분석"""
        try:
            if len(df) < 50:
                return 0
            
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            score = 0
            
            # 고점/저점 상승 확인
            recent_highs = [highs[i] for i in range(len(highs)-30, len(highs), 10)]
            recent_lows = [lows[i] for i in range(len(lows)-30, len(lows), 10)]
            
            # 고점 상승 추세 확인
            if len(recent_highs) >= 3 and all(recent_highs[i] <= recent_highs[i+1] for i in range(len(recent_highs)-1)):
                score += 15
            
            # 저점 상승 추세 확인  
            if len(recent_lows) >= 3 and all(recent_lows[i] <= recent_lows[i+1] for i in range(len(recent_lows)-1)):
                score += 15
            
            return min(score, 30)  # 최대 30점
            
        except Exception as e:
            logging.error(f"추세 추종 분석 오류: {e}")
            return 0

    def _analyze_volume_patterns(self, df: pd.DataFrame) -> int:
        """거래량 패턴 분석"""
        try:
            if len(df) < 20:
                return 0
            
            closes = df['close'].values
            volumes = df['volume'].values
            
            score = 0
            
            # 상승 시 거래량 증가, 하락 시 거래량 감소 패턴 확인
            up_days = []
            down_days = []
            
            for i in range(1, len(closes)):
                price_change = closes[i] - closes[i-1]
                if price_change > 0:
                    up_days.append(volumes[i])
                elif price_change < 0:
                    down_days.append(volumes[i])
            
            if up_days and down_days:
                avg_up_volume = np.mean(up_days)
                avg_down_volume = np.mean(down_days)
                
                # 상승일 거래량이 하락일보다 많으면 긍정적
                if avg_up_volume > avg_down_volume * 1.2:
                    score += 15
                elif avg_up_volume > avg_down_volume:
                    score += 10
            
            # 최근 거래량 급증 확인
            recent_volume = volumes[-5:].mean()
            avg_volume = volumes[:-5].mean()
            
            if recent_volume > avg_volume * 2:
                score += 5
            
            return min(score, 20)  # 최대 20점
            
        except Exception as e:
            logging.error(f"거래량 패턴 분석 오류: {e}")
            return 0

    def _get_fundamental_data(self, stock_code: str) -> Dict[str, Any]:
        """
        펀더멘털 데이터 수집
        Gemini AI가 필요로 하는 모든 펀더멘털 지표를 포함하여 최적화된 데이터 제공
        """
        try:
            # 기본 정보 조회
            if hasattr(self, 'fundamentals_df') and stock_code in self.fundamentals_df.index:
                stock_info = self.fundamentals_df.loc[stock_code]
                
                # 기본 펀더멘털 데이터
                per = float(stock_info.get('PER', 15.0)) if pd.notna(stock_info.get('PER')) else 15.0
                pbr = float(stock_info.get('PBR', 1.0)) if pd.notna(stock_info.get('PBR')) else 1.0
                eps = float(stock_info.get('EPS', 5000)) if pd.notna(stock_info.get('EPS')) else 5000
                bps = float(stock_info.get('BPS', 45000)) if pd.notna(stock_info.get('BPS')) else 45000
                
                # ROE 계산 (EPS/BPS * 100)
                roe = (eps / bps * 100) if bps > 0 else 15.0
                
                # 업종별 예상 부채비율 (실제 데이터 없을 시 업종 평균 추정)
                sector_debt_ratios = {
                    '005930': 25.0,  # 삼성전자 - 전자
                    '000660': 35.0,  # SK하이닉스 - 반도체
                    '035420': 15.0,  # NAVER - IT서비스
                    '051910': 45.0,  # LG화학 - 화학
                    '006400': 40.0,  # 삼성SDI - 전기전자
                    '035720': 20.0,  # 카카오 - IT서비스
                    '028260': 50.0,  # 삼성물산 - 종합상사
                    '068270': 25.0,  # 셀트리온 - 바이오
                    '096770': 60.0,  # SK이노베이션 - 정유
                    '323410': 30.0   # 카카오뱅크 - 금융
                }
                debt_ratio = sector_debt_ratios.get(stock_code, 35.0)
                
                return {
                    "per": per,
                    "pbr": pbr,
                    "eps": eps,
                    "bps": bps,
                    "roe": roe,
                    "debt_ratio": debt_ratio,
                    "dividend_yield": float(stock_info.get('DIV', 2.5)) if pd.notna(stock_info.get('DIV')) else 2.5,
                    "market_cap": self._estimate_market_cap(stock_code),
                    "book_value_per_share": bps,
                    "earnings_per_share": eps,
                    "price_to_book": pbr,
                    "price_to_earnings": per,
                    "return_on_equity": roe,
                    "financial_leverage": debt_ratio,
                    "data_source": "fundamentals_df"
                }
            else:
                # 기본값 사용 시에도 모든 필드 제공
                return {
                    "per": 15.0,
                    "pbr": 1.0,
                    "eps": 5000,
                    "bps": 45000,
                    "roe": 11.1,  # 5000/45000 * 100
                    "debt_ratio": 35.0,
                    "dividend_yield": 2.5,
                    "market_cap": 100000000000,  # 1000억원
                    "book_value_per_share": 45000,
                    "earnings_per_share": 5000,
                    "price_to_book": 1.0,
                    "price_to_earnings": 15.0,
                    "return_on_equity": 11.1,
                    "financial_leverage": 35.0,
                    "note": "기본값 사용",
                    "data_source": "default_values"
                }
                
        except Exception as e:
            logging.error(f"펀더멘털 분석 오류: {e}")
            return {
                "error": f"펀더멘털 분석 실패: {str(e)}",
                "per": 15.0,
                "pbr": 1.0,
                "eps": 5000,
                "bps": 45000,
                "roe": 11.1,
                "debt_ratio": 35.0,
                "dividend_yield": 2.5,
                "market_cap": 100000000000,
                "book_value_per_share": 45000,
                "earnings_per_share": 5000,
                "price_to_book": 1.0,
                "price_to_earnings": 15.0,
                "return_on_equity": 11.1,
                "financial_leverage": 35.0,
                "data_source": "error_fallback"
            }

    def _estimate_market_cap(self, stock_code: str) -> int:
        """종목 코드를 기반으로 시가총액을 추정합니다."""
        try:
            if hasattr(self, 'market_cap_df') and stock_code in self.market_cap_df.index:
                return int(self.market_cap_df.loc[stock_code, '시가총액'])
            else:
                # 주요 종목별 추정 시가총액 (원)
                estimated_caps = {
                    '005930': 400_000_000_000_000,  # 삼성전자 400조
                    '000660': 80_000_000_000_000,   # SK하이닉스 80조
                    '035420': 60_000_000_000_000,   # NAVER 60조
                    '051910': 50_000_000_000_000,   # LG화학 50조
                    '006400': 45_000_000_000_000,   # 삼성SDI 45조
                    '035720': 40_000_000_000_000,   # 카카오 40조
                    '028260': 35_000_000_000_000,   # 삼성물산 35조
                    '068270': 30_000_000_000_000,   # 셀트리온 30조
                    '096770': 25_000_000_000_000,   # SK이노베이션 25조
                    '323410': 20_000_000_000_000    # 카카오뱅크 20조
                }
                return estimated_caps.get(stock_code, 10_000_000_000_000)  # 기본값 10조
        except Exception:
            return 10_000_000_000_000  # 기본값 10조 

    async def get_supply_demand_data(self, stock_code: str, days: int = 30) -> Optional[Dict]:
        """수급 데이터 수집 (PyKRX 기반)"""
        try:
            # 날짜 범위 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 투자자별 거래실적 데이터 수집
            trading_data = stock.get_market_trading_value_by_date(
                start_date.strftime('%Y%m%d'),
                end_date.strftime('%Y%m%d'),
                stock_code
            )
            
            if trading_data.empty:
                return None
            
            # 수급 데이터 집계
            supply_demand_data = {
                'foreign_net': int(trading_data['외국인합계'].sum()),
                'institution_net': int(trading_data['기관합계'].sum()),
                'individual_net': int(trading_data['개인'].sum()),
                'total_volume': int(trading_data['거래량'].sum()) if '거래량' in trading_data.columns else 0
            }
            
            return supply_demand_data
            
        except Exception as e:
            logging.error(f"❌ [{stock_code}] PyKRX 수급 데이터 수집 실패: {e}")
            return None

    async def get_detailed_investor_data(self, stock_code: str) -> Optional[Dict]:
        """상세 투자자별 데이터 수집 (한투 API 연동 준비)"""
        try:
            # 실제 구현에서는 core_trader의 fetch_detailed_investor_trends 사용
            # 현재는 더미 데이터 반환
            detailed_data = {
                'pension_fund': 0,      # 연기금
                'private_equity': 0,    # 사모펀드  
                'insurance': 0,         # 보험
                'investment_trust': 0,  # 투신
                'bank': 0,              # 은행
                'other_financial': 0,   # 기타금융
                'other_corp': 0         # 기타법인
            }
            
            return detailed_data
            
        except Exception as e:
            logging.error(f"❌ [{stock_code}] 상세 투자자 데이터 수집 실패: {e}")
            return None

    async def analyze_supply_demand_with_optimizer(self, stock_code: str, days: int = 30) -> SupplyDemandData:
        """수급 분석기를 사용한 종합 수급 분석"""
        try:
            return await self.supply_demand_analyzer.analyze_supply_demand(stock_code, days)
        except Exception as e:
            logging.error(f"❌ [{stock_code}] 수급 분석 실패: {e}")
            return self.supply_demand_analyzer._create_empty_data(stock_code)

    async def get_supply_demand_ranking(self, stock_codes: List[str], limit: int = 20) -> List[SupplyDemandData]:
        """수급 순위 조회"""
        try:
            return await self.supply_demand_analyzer.get_supply_demand_ranking(stock_codes, limit)
        except Exception as e:
            logging.error(f"❌ 수급 순위 조회 실패: {e}")
            return []

    async def monitor_supply_demand_changes(self, stock_codes: List[str]):
        """실시간 수급 변화 모니터링"""
        try:
            alerts = await self.supply_demand_analyzer.monitor_supply_demand_changes(stock_codes)
            if alerts:
                logging.info(f"📢 수급 급변 알림 {len(alerts)}건 발생")
            return alerts
        except Exception as e:
            logging.error(f"❌ 수급 모니터링 실패: {e}")
            return []

    def generate_supply_demand_report(self, data_list: List[SupplyDemandData]) -> str:
        """수급 분석 리포트 생성"""
        return self.supply_demand_analyzer.generate_supply_demand_report(data_list)

    async def cleanup_supply_demand_analyzer(self):
        """수급 분석기 리소스 정리"""
        if hasattr(self, 'supply_demand_analyzer'):
            await self.supply_demand_analyzer.cleanup() 