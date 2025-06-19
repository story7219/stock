"""
🚀 AI 투자 판단을 위한 초고속 데이터 수집기 (DART API 최적화 버전)
- DART API를 최대한 활용한 신뢰성 높은 데이터 수집
- 재무제표, 공시, 기업정보 등 종합 분석
- 기존 스캘핑/단기매매 기능 + DART 통합 분석
"""
import logging
from datetime import datetime, timedelta
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache
import threading
from collections import deque, defaultdict
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from aiohttp import ClientSession
import FinanceDataReader as fdr
from functools import lru_cache
from cachetools import TTLCache
from bs4 import BeautifulSoup
import requests
import OpenDartReader # OpenDartReader 임포트

from chart_generator import create_stock_chart
from config import DART_API_KEY # DART API 키 임포트

logger = logging.getLogger(__name__)

@dataclass
class MarketSignal:
    """시장 신호 데이터 클래스 (메모리 최적화)"""
    symbol: str
    signal_type: str  # 'price_surge', 'volume_spike', 'orderbook_imbalance'
    strength: float   # 0-10 신호 강도
    timestamp: datetime
    data: Dict[str, Any] = None

class MarketType(Enum):
    """시장 구분"""
    KOSPI = "J"
    KOSDAQ = "Q"
    ALL = "ALL"

@dataclass
class StockInfo:
    """종목 정보 데이터 클래스 (DART 정보 포함)"""
    code: str                    # 종목코드
    name: str                    # 종목명
    current_price: int           # 현재가
    market_cap: int              # 시가총액 (억원)
    volume: int                  # 거래량
    volume_value: int            # 거래대금 (백만원)
    market_type: str             # 시장구분 (KOSPI/KOSDAQ)
    sector: str                  # 업종
    per: Optional[float] = None  # PER
    pbr: Optional[float] = None  # PBR
    roe: Optional[float] = None  # ROE
    debt_ratio: Optional[float] = None  # 부채비율
    score: float = 0.0           # AI 종합 점수
    # DART 추가 정보
    corp_code: Optional[str] = None      # DART 기업코드
    ceo_name: Optional[str] = None       # 대표이사명
    establishment_date: Optional[str] = None  # 설립일
    main_business: Optional[str] = None  # 주요사업
    employee_count: Optional[int] = None # 직원수
    recent_disclosure_count: int = 0     # 최근 공시 건수

@dataclass
class FilterCriteria:
    """필터링 기준"""
    min_market_cap: int = 500        # 최소 시가총액 (억원) - 5천억원 (테스트용 하향 조정)
    min_volume: int = 100000         # 최소 거래량 (주) (테스트용 하향 조정)
    min_volume_value: int = 1000     # 최소 거래대금 (백만원) - 10억원 (테스트용 하향 조정)
    market_types: List[str] = None   # 대상 시장
    exclude_sectors: List[str] = None # 제외 업종
    max_stocks: int = 50             # 최대 선별 종목 수

    def __post_init__(self):
        if self.market_types is None:
            self.market_types = ["KOSPI", "KOSDAQ"]
        if self.exclude_sectors is None:
            self.exclude_sectors = ["금융업", "보험업"]

@dataclass
class DartCompanyInfo:
    """DART 기업 정보 구조화"""
    corp_code: str
    corp_name: str
    corp_cls: str
    ceo_nm: str
    adres: str
    hm_url: str
    ir_url: str
    phn_no: str
    fax_no: str
    induty_code: str
    est_dt: str
    acc_mt: str

class AIDataCollector:
    """🚀 AI 투자 판단을 위한 초고속 핵심 데이터 수집 클래스 (DART API 최적화)"""
    
    def __init__(self, trader: 'CoreTrader'):
        """CoreTrader 인스턴스와 연동 + DART API 최적화 초기화"""
        self.trader = trader
        self.http_client = trader.http_client # trader로부터 http_client를 가져옴
        
        # DART API 리더 초기화 및 검증
        if DART_API_KEY:
            try:
                self.dart = OpenDartReader(DART_API_KEY)
                # DART API 연결 테스트 (더 안전한 방법)
                test_result = self.dart.list(start_dt='20240101', end_dt='20240110')
                logger.info("✅ DART API 리더가 성공적으로 초기화되고 연결 테스트가 완료되었습니다.")
                self.dart_available = True
            except Exception as e:
                logger.error(f"❌ DART API 초기화 실패: {e}")
                self.dart = None
                self.dart_available = False
        else:
            self.dart = None
            self.dart_available = False
            logger.warning("⚠️ DART_API_KEY가 설정되지 않아 DART 공시 조회를 비활성화합니다.")

        # 🚀 성능 최적화 설정
        self.cache_duration = 3  # 3초 캐시 (스캘핑용)
        self.price_cache = {}
        self.orderbook_cache = {}
        self.market_mood_cache = {'data': None, 'timestamp': None}
        
        # DART 전용 캐시 (더 긴 TTL)
        self.dart_company_cache = TTLCache(maxsize=1000, ttl=86400)  # 24시간
        self.dart_financial_cache = TTLCache(maxsize=500, ttl=21600)  # 6시간
        self.dart_disclosure_cache = TTLCache(maxsize=1000, ttl=3600) # 1시간
        
        # 병렬 처리 설정
        self.max_workers = 6  # 동시 API 호출 수
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 실시간 모니터링
        self.monitoring_symbols = set()
        self.signal_callbacks = []
        self.realtime_data = defaultdict(dict)
        self.is_monitoring = False
        
        # 임계값 (성능상 중요한 것만)
        self.thresholds = {
            'price_change': 1.5,      # 1.5% 이상만 (민감도 조정)
            'volume_spike': 150,      # 150% 이상만
            'orderbook_ratio': 2.0,   # 2:1 이상만
            'scalping_score': 6.0     # 6점 이상만 추천
        }

        self.exclude_sectors = ["금융업", "보험업"]
        self.news_cache = TTLCache(maxsize=512, ttl=1800) # 30분 캐시
        self.market_regime_cache = TTLCache(maxsize=1, ttl=3600) # 1시간 캐시
        self.theme_cache = TTLCache(maxsize=1, ttl=1800) # 30분 캐시

    # ===================================================================
    # 🏛️ DART API 최대 활용 메서드들 (신규 추가)
    # ===================================================================
    
    async def get_dart_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """
        🏛️ DART API를 최대한 활용한 종합 기업 분석 데이터
        - 기업 기본정보, 재무제표, 공시, 지분현황 등 모든 데이터 수집
        """
        logger.info(f"🏛️ [{symbol}] DART 종합 데이터 수집 시작...")
        
        if not self.dart_available:
            logger.warning("⚠️ DART API를 사용할 수 없습니다.")
            return {'error': 'DART API 사용 불가'}
        
        try:
            # 병렬로 모든 DART 데이터 수집
            tasks = {
                'company_info': self.get_dart_company_info(symbol),
                'financial_statements': self.get_dart_financial_statements(symbol),
                'recent_disclosures': self.get_dart_recent_disclosures(symbol),
                'major_shareholders': self.get_dart_major_shareholders(symbol),
                'business_report': self.get_dart_business_report(symbol),
                'audit_report': self.get_dart_audit_report(symbol),
                'dividend_info': self.get_dart_dividend_info(symbol),
                'capital_increase': self.get_dart_capital_increase_history(symbol)
            }
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # 결과 구조화
            comprehensive_data = {}
            for i, key in enumerate(tasks.keys()):
                if isinstance(results[i], Exception):
                    logger.warning(f"⚠️ [{symbol}] {key} 수집 실패: {results[i]}")
                    comprehensive_data[key] = None
                else:
                    comprehensive_data[key] = results[i]
            
            # AI 분석을 위한 종합 점수 계산
            comprehensive_data['dart_analysis_score'] = self._calculate_dart_analysis_score(comprehensive_data)
            comprehensive_data['collection_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"✅ [{symbol}] DART 종합 데이터 수집 완료")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"❌ [{symbol}] DART 종합 데이터 수집 실패: {e}")
            return {'error': str(e)}

    async def get_dart_company_info(self, symbol: str) -> Optional[Dict]:
        """🏢 DART 기업 기본정보 조회 (캐시 적용)"""
        cache_key = f"company_{symbol}"
        if cache_key in self.dart_company_cache:
            return self.dart_company_cache[cache_key]
        
        try:
            # 종목코드로 기업코드 찾기
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            # 기업 기본정보 조회
            company_info = await asyncio.to_thread(self.dart.company, corp_code)
            if company_info is None or company_info.empty:
                return None
            
            info_dict = company_info.iloc[0].to_dict()
            
            # 추가 상세 정보 조회
            corp_outline = await asyncio.to_thread(self.dart.company_outline, corp_code)
            
            result = {
                'corp_code': corp_code,
                'corp_name': info_dict.get('corp_name'),
                'ceo_nm': info_dict.get('ceo_nm'),
                'corp_cls': info_dict.get('corp_cls'),
                'adres': info_dict.get('adres'),
                'hm_url': info_dict.get('hm_url'),
                'ir_url': info_dict.get('ir_url'),
                'phn_no': info_dict.get('phn_no'),
                'est_dt': info_dict.get('est_dt'),
                'acc_mt': info_dict.get('acc_mt'),
                'induty_code': info_dict.get('induty_code')
            }
            
            # 기업개요 정보 추가
            if corp_outline is not None and not corp_outline.empty:
                outline_dict = corp_outline.iloc[0].to_dict()
                result.update({
                    'employee_count': outline_dict.get('emp_co'),
                    'main_business': outline_dict.get('bsn_sumry_ctnt'),
                    'capital_stock': outline_dict.get('cptl_stck_co')
                })
            
            self.dart_company_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 기업정보 조회 실패: {e}")
            return None

    async def get_dart_financial_statements(self, symbol: str) -> Optional[Dict]:
        """💰 DART 재무제표 종합 분석 (최근 5년)"""
        cache_key = f"financial_{symbol}"
        if cache_key in self.dart_financial_cache:
            return self.dart_financial_cache[cache_key]
        
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            current_year = datetime.now().year
            financial_data = {}
            
            # 최근 5년 재무제표 수집
            for year in range(current_year - 5, current_year):
                year_str = str(year)
                
                # 연간 재무제표 (사업보고서)
                fs_annual = await asyncio.to_thread(
                    self.dart.finstate, 
                    corp_code, 
                    bsns_year=year_str, 
                    reprt_code='11011'  # 사업보고서
                )
                
                if fs_annual is not None and not fs_annual.empty:
                    # 주요 계정 추출
                    fs_dict = {}
                    for _, row in fs_annual.iterrows():
                        account_name = row.get('account_nm', '')
                        current_amount = row.get('thstrm_amount', 0)
                        
                        # 주요 계정만 선별
                        if any(keyword in account_name for keyword in [
                            '매출액', '영업이익', '당기순이익', '자산총계', 
                            '부채총계', '자본총계', '현금및현금성자산'
                        ]):
                            try:
                                fs_dict[account_name] = int(str(current_amount).replace(',', '')) if current_amount else 0
                            except:
                                fs_dict[account_name] = 0
                    
                    financial_data[year_str] = fs_dict
            
            if not financial_data:
                return None
            
            # 재무 비율 계산
            financial_ratios = self._calculate_financial_ratios(financial_data)
            
            result = {
                'yearly_data': financial_data,
                'financial_ratios': financial_ratios,
                'trend_analysis': self._analyze_financial_trends(financial_data)
            }
            
            self.dart_financial_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 재무제표 조회 실패: {e}")
            return None

    async def get_dart_recent_disclosures(self, symbol: str, days: int = 30) -> List[Dict]:
        """📢 DART 최근 공시 조회"""
        cache_key = f"disclosures_{symbol}_{days}"
        if cache_key in self.dart_disclosure_cache:
            return self.dart_disclosure_cache[cache_key]
        
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return []
            
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            # 공시 목록 조회 (올바른 파라미터 사용)
            disclosure_list = await asyncio.to_thread(
                self.dart.list,
                start_dt=start_date,
                end_dt=end_date,
                corp_code=corp_code
            )
            
            if disclosure_list is None or disclosure_list.empty:
                return []
            
            # 중요 공시만 필터링 및 구조화
            important_disclosures = []
            for _, row in disclosure_list.iterrows():
                report_nm = row.get('report_nm', '')
                
                # 중요 공시 키워드 필터링
                if any(keyword in report_nm for keyword in [
                    '사업보고서', '반기보고서', '분기보고서', '공시정정',
                    '주요사항보고서', '증권발행', '합병', '분할', '영업양수도'
                ]):
                    important_disclosures.append({
                        'rcept_no': row.get('rcept_no'),
                        'corp_name': row.get('corp_name'),
                        'report_nm': report_nm,
                        'rcept_dt': row.get('rcept_dt'),
                        'flr_nm': row.get('flr_nm'),
                        'rm': row.get('rm', ''),
                        'importance_score': self._calculate_disclosure_importance(report_nm)
                    })
            
            # 중요도순 정렬
            important_disclosures.sort(key=lambda x: x['importance_score'], reverse=True)
            
            self.dart_disclosure_cache[cache_key] = important_disclosures
            return important_disclosures
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 공시 조회 실패: {e}")
            return []

    async def get_dart_major_shareholders(self, symbol: str) -> Optional[Dict]:
        """👥 DART 주요주주 현황"""
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            # 최근 사업보고서에서 주요주주 정보 추출
            current_year = datetime.now().year - 1  # 전년도 사업보고서
            
            major_shareholders = await asyncio.to_thread(
                self.dart.major_shareholders,
                corp_code,
                bsns_year=str(current_year)
            )
            
            if major_shareholders is None or major_shareholders.empty:
                return None
            
            # 주요주주 정보 구조화
            shareholders_data = []
            for _, row in major_shareholders.iterrows():
                shareholders_data.append({
                    'nm': row.get('nm', ''),  # 주주명
                    'relate': row.get('relate', ''),  # 관계
                    'stocks_co': row.get('stocks_co', 0),  # 보유주식수
                    'stocks_rt': row.get('stocks_rt', 0.0)  # 지분율
                })
            
            return {
                'reference_date': f"{current_year}년 사업보고서 기준",
                'major_shareholders': shareholders_data,
                'total_shareholders': len(shareholders_data)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 주요주주 조회 실패: {e}")
            return None

    async def get_dart_business_report(self, symbol: str) -> Optional[Dict]:
        """📊 DART 사업보고서 핵심 정보"""
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            current_year = datetime.now().year - 1
            
            # 사업보고서 조회
            business_report = await asyncio.to_thread(
                self.dart.report,
                corp_code,
                bsns_year=str(current_year),
                reprt_code='11011'  # 사업보고서
            )
            
            if business_report is None:
                return None
            
            return {
                'business_year': current_year,
                'report_summary': '사업보고서 데이터 수집 완료',
                'data_available': True
            }
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 사업보고서 조회 실패: {e}")
            return None

    async def get_dart_audit_report(self, symbol: str) -> Optional[Dict]:
        """🔍 DART 감사보고서 정보"""
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            current_year = datetime.now().year - 1
            
            # 감사보고서 의견 조회
            audit_opinion = await asyncio.to_thread(
                self.dart.audit_opinion,
                corp_code,
                bsns_year=str(current_year)
            )
            
            if audit_opinion is None or audit_opinion.empty:
                return None
            
            audit_data = []
            for _, row in audit_opinion.iterrows():
                audit_data.append({
                    'auditor': row.get('auditor', ''),
                    'opinion': row.get('opinion', ''),
                    'audit_dt': row.get('audit_dt', '')
                })
            
            return {
                'audit_year': current_year,
                'audit_opinions': audit_data
            }
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 감사보고서 조회 실패: {e}")
            return None

    async def get_dart_dividend_info(self, symbol: str) -> Optional[Dict]:
        """💎 DART 배당 정보"""
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            # 최근 3년 배당 정보
            dividend_data = {}
            current_year = datetime.now().year
            
            for year in range(current_year - 3, current_year):
                dividend = await asyncio.to_thread(
                    self.dart.dividend,
                    corp_code,
                    bsns_year=str(year)
                )
                
                if dividend is not None and not dividend.empty:
                    dividend_info = dividend.iloc[0].to_dict()
                    dividend_data[str(year)] = {
                        'cash_dividend': dividend_info.get('cash_dvdnd_per_shr', 0),
                        'stock_dividend': dividend_info.get('stk_dvdnd_rt', 0),
                        'dividend_yield': dividend_info.get('dvdnd_yld', 0)
                    }
            
            return dividend_data if dividend_data else None
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 배당정보 조회 실패: {e}")
            return None

    async def get_dart_capital_increase_history(self, symbol: str) -> Optional[List[Dict]]:
        """📈 DART 유상증자 이력"""
        try:
            corp_code = await self._get_corp_code_from_symbol(symbol)
            if not corp_code:
                return None
            
            # 최근 5년 유상증자 이력
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y%m%d')  # 5년
            
            capital_increase = await asyncio.to_thread(
                self.dart.list,
                corp_code=corp_code,
                start_dt=start_date,
                end_dt=end_date,
                kind='I'  # 발행공시
            )
            
            if capital_increase is None or capital_increase.empty:
                return None
            
            # 유상증자 관련 공시만 필터링
            increase_history = []
            for _, row in capital_increase.iterrows():
                report_nm = row.get('report_nm', '')
                if '유상증자' in report_nm or '신주발행' in report_nm:
                    increase_history.append({
                        'report_nm': report_nm,
                        'rcept_dt': row.get('rcept_dt'),
                        'flr_nm': row.get('flr_nm')
                    })
            
            return increase_history
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 유상증자 이력 조회 실패: {e}")
            return None

    # ===================================================================
    # 🧮 DART 데이터 분석 및 점수 계산 메서드들
    # ===================================================================
    
    def _calculate_dart_analysis_score(self, dart_data: Dict) -> float:
        """DART 데이터 기반 종합 분석 점수 계산"""
        score = 50.0  # 기본 점수
        
        try:
            # 1. 재무제표 점수 (30점)
            financial = dart_data.get('financial_statements')
            if financial:
                ratios = financial.get('financial_ratios', {})
                if ratios.get('roe', 0) > 15:
                    score += 10
                elif ratios.get('roe', 0) > 10:
                    score += 5
                
                if ratios.get('debt_ratio', 100) < 50:
                    score += 10
                elif ratios.get('debt_ratio', 100) < 70:
                    score += 5
                
                trend = financial.get('trend_analysis', {})
                if trend.get('revenue_growth', 0) > 10:
                    score += 10
            
            # 2. 공시 점수 (20점)
            disclosures = dart_data.get('recent_disclosures', [])
            if disclosures:
                # 최근 중요 공시가 많으면 감점 (불안정성)
                important_count = len([d for d in disclosures if d.get('importance_score', 0) > 7])
                if important_count > 3:
                    score -= 10
                elif important_count == 0:
                    score += 10  # 안정적
            
            # 3. 기업 정보 점수 (10점)
            company_info = dart_data.get('company_info')
            if company_info:
                if company_info.get('employee_count', 0) > 1000:
                    score += 5
                if company_info.get('est_dt'):
                    # 설립 연수 계산
                    try:
                        est_year = int(company_info['est_dt'][:4])
                        years = datetime.now().year - est_year
                        if years > 20:
                            score += 5
                    except:
                        pass
            
            # 4. 배당 점수 (10점)
            dividend_info = dart_data.get('dividend_info')
            if dividend_info:
                recent_dividends = list(dividend_info.values())
                if recent_dividends:
                    avg_yield = sum(d.get('dividend_yield', 0) for d in recent_dividends) / len(recent_dividends)
                    if avg_yield > 3:
                        score += 10
                    elif avg_yield > 1:
                        score += 5
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"⚠️ DART 분석 점수 계산 실패: {e}")
            return 50.0

    def _calculate_financial_ratios(self, financial_data: Dict) -> Dict:
        """재무 비율 계산"""
        ratios = {}
        
        try:
            latest_year = max(financial_data.keys())
            latest_data = financial_data[latest_year]
            
            # ROE 계산
            net_income = latest_data.get('당기순이익', 0)
            total_equity = latest_data.get('자본총계', 1)
            if total_equity > 0:
                ratios['roe'] = (net_income / total_equity) * 100
            
            # 부채비율 계산
            total_debt = latest_data.get('부채총계', 0)
            if total_equity > 0:
                ratios['debt_ratio'] = (total_debt / total_equity) * 100
            
            # 영업이익률 계산
            revenue = latest_data.get('매출액', 1)
            operating_income = latest_data.get('영업이익', 0)
            if revenue > 0:
                ratios['operating_margin'] = (operating_income / revenue) * 100
            
        except Exception as e:
            logger.warning(f"⚠️ 재무비율 계산 실패: {e}")
        
        return ratios

    def _analyze_financial_trends(self, financial_data: Dict) -> Dict:
        """재무 트렌드 분석"""
        trends = {}
        
        try:
            years = sorted(financial_data.keys())
            if len(years) < 2:
                return trends
            
            # 매출 성장률
            latest_revenue = financial_data[years[-1]].get('매출액', 0)
            prev_revenue = financial_data[years[-2]].get('매출액', 1)
            if prev_revenue > 0:
                trends['revenue_growth'] = ((latest_revenue - prev_revenue) / prev_revenue) * 100
            
            # 순이익 성장률
            latest_net_income = financial_data[years[-1]].get('당기순이익', 0)
            prev_net_income = financial_data[years[-2]].get('당기순이익', 1)
            if prev_net_income != 0:
                trends['net_income_growth'] = ((latest_net_income - prev_net_income) / abs(prev_net_income)) * 100
            
        except Exception as e:
            logger.warning(f"⚠️ 재무트렌드 분석 실패: {e}")
        
        return trends

    def _calculate_disclosure_importance(self, report_name: str) -> int:
        """공시 중요도 점수 계산"""
        importance_keywords = {
            '사업보고서': 10,
            '반기보고서': 8,
            '분기보고서': 6,
            '주요사항보고서': 9,
            '합병': 10,
            '분할': 9,
            '증권발행': 8,
            '영업양수도': 9,
            '공시정정': 7
        }
        
        for keyword, score in importance_keywords.items():
            if keyword in report_name:
                return score
        
        return 5  # 기본 점수

    async def _get_corp_code_from_symbol(self, symbol: str) -> Optional[str]:
        """종목코드로 DART 기업코드 조회"""
        try:
            # DART 기업코드 매핑 (주요 종목들)
            # 실제 운영시에는 더 완전한 매핑 테이블이나 API 활용 필요
            symbol_to_corp_code = {
                '005930': '00126380',  # 삼성전자
                '000660': '00119397',  # SK하이닉스
                '035420': '00164779',  # NAVER
                '207940': '00168099',  # 삼성바이오로직스
                '005380': '00126186',  # 현대차
                '051910': '00164593',  # LG화학
                '006400': '00127565',  # 삼성SDI
                '035720': '00164742',  # 카카오
                '028260': '00161289',  # 삼성물산
                '068270': '00165773',  # 셀트리온
            }
            
            corp_code = symbol_to_corp_code.get(symbol)
            if corp_code:
                return corp_code
            
            # 매핑에 없는 경우 DART에서 검색 시도
            corp_list = await asyncio.to_thread(self.dart.company, symbol)
            if corp_list is not None and not corp_list.empty:
                return corp_list.iloc[0]['corp_code']
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] 기업코드 조회 실패: {e}")
            return None

    # ===================================================================
    # 🎯 DART + KIS API 통합 분석 메서드들 (신규)
    # ===================================================================
    
    async def get_ultimate_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        🎯 DART + KIS API 완전 통합 분석
        - DART: 기업정보, 재무제표, 공시 등
        - KIS: 실시간 가격, 거래량, 투자자 동향 등
        - AI 점수: 모든 데이터를 종합한 최종 투자 판단
        """
        logger.info(f"🎯 [{symbol}] 완전 통합 분석 시작...")
        start_time = time.time()
        
        try:
            # 1. DART + KIS 데이터 병렬 수집
            dart_task = self.get_dart_comprehensive_data(symbol)
            kis_task = self.get_kis_comprehensive_data(symbol)
            
            dart_data, kis_data = await asyncio.gather(dart_task, kis_task, return_exceptions=True)
            
            # 2. 데이터 검증
            if isinstance(dart_data, Exception):
                logger.warning(f"⚠️ [{symbol}] DART 데이터 수집 실패: {dart_data}")
                dart_data = {}
            
            if isinstance(kis_data, Exception):
                logger.warning(f"⚠️ [{symbol}] KIS 데이터 수집 실패: {kis_data}")
                kis_data = {}
            
            # 3. 통합 분석 수행
            ultimate_analysis = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                
                # 원본 데이터
                'dart_data': dart_data,
                'kis_data': kis_data,
                
                # 통합 분석 결과
                'fundamental_score': self._calculate_fundamental_score(dart_data),
                'technical_score': self._calculate_technical_score(kis_data),
                'market_timing_score': self._calculate_market_timing_score(kis_data),
                
                # 최종 결과
                'ultimate_score': 0,
                'investment_recommendation': '',
                'risk_level': '',
                'key_strengths': [],
                'key_risks': [],
                'target_price_range': None
            }
            
            # 4. 최종 점수 및 추천 계산
            ultimate_analysis['ultimate_score'] = self._calculate_ultimate_score(ultimate_analysis)
            ultimate_analysis['investment_recommendation'] = self._get_ultimate_recommendation(ultimate_analysis)
            ultimate_analysis['risk_level'] = self._assess_risk_level(ultimate_analysis)
            ultimate_analysis['key_strengths'] = self._identify_key_strengths(dart_data, kis_data)
            ultimate_analysis['key_risks'] = self._identify_key_risks(dart_data, kis_data)
            ultimate_analysis['target_price_range'] = self._calculate_target_price(dart_data, kis_data)
            
            logger.info(f"✅ [{symbol}] 완전 통합 분석 완료 (점수: {ultimate_analysis['ultimate_score']:.1f}/100)")
            return ultimate_analysis
            
        except Exception as e:
            logger.error(f"❌ [{symbol}] 완전 통합 분석 실패: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }

    async def get_kis_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """🚀 KIS API 종합 데이터 수집"""
        try:
            # KIS API로 수집할 수 있는 모든 데이터를 병렬로 수집
            tasks = {
                'current_price': self.trader.get_current_price(symbol),
                'daily_history': self.trader.fetch_daily_price_history(symbol, 120),
                'minute_history': self.trader.fetch_minute_price_history(symbol),
                'news_headlines': self.trader.fetch_news_headlines(symbol),
                'investor_trends': self.trader.fetch_detailed_investor_trends(symbol)
            }
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            kis_data = {}
            for i, key in enumerate(tasks.keys()):
                if isinstance(results[i], Exception):
                    logger.warning(f"⚠️ [{symbol}] KIS {key} 수집 실패: {results[i]}")
                    kis_data[key] = None
                else:
                    kis_data[key] = results[i]
            
            # 기술적 지표 계산
            if kis_data['daily_history']:
                kis_data['technical_indicators'] = await self.get_technical_indicators(symbol)
            
            return kis_data
            
        except Exception as e:
            logger.error(f"❌ [{symbol}] KIS 종합 데이터 수집 실패: {e}")
            return {}

    async def get_dart_market_leaders(self, limit: int = 20) -> List[Dict]:
        """
        🏛️ DART 기반 시장 리더 종목 발굴
        - 재무 건전성이 우수한 종목들
        - 최근 공시가 긍정적인 종목들
        - 배당 수익률이 높은 종목들
        """
        logger.info(f"🏛️ DART 기반 시장 리더 {limit}개 종목 발굴...")
        
        if not self.dart_available:
            logger.warning("⚠️ DART API를 사용할 수 없어 시장 리더 발굴을 건너뜁니다.")
            return []
        
        try:
            # 주요 종목들의 DART 데이터 수집
            major_symbols = [
                '005930', '000660', '035420', '207940', '005380',  # 대형주
                '051910', '006400', '035720', '028260', '068270',
                '323410', '000270', '105560', '055550', '096770'   # 중형주
            ]
            
            # 병렬로 DART 데이터 수집
            tasks = [self.get_dart_comprehensive_data(symbol) for symbol in major_symbols]
            dart_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 유효한 결과만 필터링
            valid_results = []
            for i, result in enumerate(dart_results):
                if not isinstance(result, Exception) and result.get('dart_analysis_score', 0) > 0:
                    result['symbol'] = major_symbols[i]
                    valid_results.append(result)
            
            # DART 분석 점수순으로 정렬
            valid_results.sort(key=lambda x: x.get('dart_analysis_score', 0), reverse=True)
            
            # 상위 종목들에 대해 추가 정보 수집
            market_leaders = []
            for result in valid_results[:limit]:
                symbol = result['symbol']
                
                # KIS API로 현재 가격 정보 추가
                current_price_info = await self.trader.get_current_price(symbol)
                
                leader_info = {
                    'symbol': symbol,
                    'company_name': result.get('company_info', {}).get('corp_name', 'N/A'),
                    'dart_score': result.get('dart_analysis_score', 0),
                    'current_price': current_price_info.get('price', 0) if current_price_info else 0,
                    'financial_health': self._assess_financial_health(result.get('financial_statements')),
                    'disclosure_quality': self._assess_disclosure_quality(result.get('recent_disclosures', [])),
                    'dividend_attractiveness': self._assess_dividend_attractiveness(result.get('dividend_info')),
                    'key_highlights': self._extract_key_highlights(result)
                }
                
                market_leaders.append(leader_info)
            
            logger.info(f"✅ DART 기반 시장 리더 {len(market_leaders)}개 종목 발굴 완료")
            return market_leaders
            
        except Exception as e:
            logger.error(f"❌ DART 기반 시장 리더 발굴 실패: {e}")
            return []

    async def get_dart_risk_alerts(self, symbols: List[str]) -> List[Dict]:
        """
        ⚠️ DART 기반 리스크 알림
        - 최근 부정적 공시가 있는 종목
        - 재무 건전성이 악화된 종목
        - 감사 의견이 부정적인 종목
        """
        logger.info(f"⚠️ {len(symbols)}개 종목 DART 리스크 분석...")
        
        if not self.dart_available:
            return []
        
        risk_alerts = []
        
        try:
            # 병렬로 DART 데이터 수집
            tasks = [self.get_dart_comprehensive_data(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    continue
                
                symbol = symbols[i]
                risks = []
                
                # 1. 공시 리스크 체크
                disclosures = result.get('recent_disclosures', [])
                negative_disclosures = [d for d in disclosures if self._is_negative_disclosure(d['report_nm'])]
                if negative_disclosures:
                    risks.append({
                        'type': '부정적 공시',
                        'severity': 'HIGH',
                        'description': f"최근 {len(negative_disclosures)}건의 부정적 공시 발견"
                    })
                
                # 2. 재무 리스크 체크
                financial = result.get('financial_statements')
                if financial:
                    ratios = financial.get('financial_ratios', {})
                    if ratios.get('debt_ratio', 0) > 80:
                        risks.append({
                            'type': '높은 부채비율',
                            'severity': 'MEDIUM',
                            'description': f"부채비율 {ratios['debt_ratio']:.1f}% (위험 수준)"
                        })
                    
                    if ratios.get('roe', 0) < -10:
                        risks.append({
                            'type': '수익성 악화',
                            'severity': 'HIGH',
                            'description': f"ROE {ratios['roe']:.1f}% (적자 지속)"
                        })
                
                # 3. 감사 의견 리스크 체크
                audit_report = result.get('audit_report')
                if audit_report:
                    opinions = audit_report.get('audit_opinions', [])
                    for opinion in opinions:
                        if '한정' in opinion.get('opinion', '') or '부적정' in opinion.get('opinion', ''):
                            risks.append({
                                'type': '감사 의견 한정',
                                'severity': 'HIGH',
                                'description': f"감사 의견: {opinion.get('opinion', '')}"
                            })
                
                if risks:
                    risk_alerts.append({
                        'symbol': symbol,
                        'company_name': result.get('company_info', {}).get('corp_name', 'N/A'),
                        'risk_count': len(risks),
                        'risks': risks,
                        'overall_risk_level': self._calculate_overall_risk_level(risks)
                    })
            
            # 리스크 수준순으로 정렬
            risk_alerts.sort(key=lambda x: (
                {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['overall_risk_level'], 0),
                x['risk_count']
            ), reverse=True)
            
            logger.info(f"⚠️ {len(risk_alerts)}개 종목에서 리스크 발견")
            return risk_alerts
            
        except Exception as e:
            logger.error(f"❌ DART 리스크 분석 실패: {e}")
            return []

    # ===================================================================
    # 🧮 통합 분석 점수 계산 메서드들
    # ===================================================================
    
    def _calculate_fundamental_score(self, dart_data: Dict) -> float:
        """펀더멘털 점수 계산 (DART 데이터 기반)"""
        if not dart_data or 'error' in dart_data:
            return 50.0
        
        return dart_data.get('dart_analysis_score', 50.0)

    def _calculate_technical_score(self, kis_data: Dict) -> float:
        """기술적 점수 계산 (KIS 데이터 기반)"""
        score = 50.0
        
        try:
            technical = kis_data.get('technical_indicators', {})
            if not technical:
                return score
            
            # RSI 점수
            rsi = technical.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 10
            elif rsi < 20 or rsi > 80:
                score -= 10
            
            # 이동평균 점수
            if technical.get('is_golden_cross', False):
                score += 15
            
            # 거래량 점수
            current_vol = kis_data.get('current_price', {}).get('volume', 0)
            avg_vol = technical.get('volume_ma20', 1)
            if current_vol > avg_vol * 1.5:
                score += 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"⚠️ 기술적 점수 계산 실패: {e}")
            return 50.0

    def _calculate_market_timing_score(self, kis_data: Dict) -> float:
        """시장 타이밍 점수 계산"""
        score = 50.0
        
        try:
            # 투자자 동향 점수
            investor_trends = kis_data.get('investor_trends')
            if investor_trends:
                foreign_net = investor_trends.get('foreign_net_buy', 0)
                institution_net = investor_trends.get('institution_net_buy', 0)
                
                if foreign_net > 0 and institution_net > 0:
                    score += 20
                elif foreign_net > 0 or institution_net > 0:
                    score += 10
                elif foreign_net < 0 and institution_net < 0:
                    score -= 20
            
            # 뉴스 심리 점수 (간단한 키워드 분석)
            news = kis_data.get('news_headlines', [])
            if news:
                positive_count = sum(1 for n in news if any(keyword in n.get('title', '') 
                                   for keyword in ['상승', '호재', '성장', '확대', '증가']))
                negative_count = sum(1 for n in news if any(keyword in n.get('title', '')
                                   for keyword in ['하락', '악재', '감소', '위험', '우려']))
                
                if positive_count > negative_count:
                    score += 10
                elif negative_count > positive_count:
                    score -= 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"⚠️ 시장 타이밍 점수 계산 실패: {e}")
            return 50.0

    def _calculate_ultimate_score(self, analysis: Dict) -> float:
        """최종 통합 점수 계산"""
        fundamental = analysis.get('fundamental_score', 50.0)
        technical = analysis.get('technical_score', 50.0)
        timing = analysis.get('market_timing_score', 50.0)
        
        # 가중 평균 (펀더멘털 50%, 기술적 30%, 타이밍 20%)
        ultimate_score = (fundamental * 0.5) + (technical * 0.3) + (timing * 0.2)
        
        return round(ultimate_score, 1)

    def _get_ultimate_recommendation(self, analysis: Dict) -> str:
        """최종 투자 추천"""
        score = analysis.get('ultimate_score', 50.0)
        
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 60:
            return "WEAK_BUY"
        elif score >= 40:
            return "HOLD"
        elif score >= 30:
            return "WEAK_SELL"
        else:
            return "SELL"

    def _assess_risk_level(self, analysis: Dict) -> str:
        """리스크 수준 평가"""
        dart_data = analysis.get('dart_data', {})
        
        # 부채비율, 공시 내용 등을 종합하여 리스크 평가
        financial = dart_data.get('financial_statements', {})
        if financial:
            ratios = financial.get('financial_ratios', {})
            debt_ratio = ratios.get('debt_ratio', 50)
            
            if debt_ratio > 80:
                return "HIGH"
            elif debt_ratio > 60:
                return "MEDIUM"
            else:
                return "LOW"
        
        return "MEDIUM"

    def _identify_key_strengths(self, dart_data: Dict, kis_data: Dict) -> List[str]:
        """핵심 강점 식별"""
        strengths = []
        
        # DART 기반 강점
        if dart_data.get('financial_statements'):
            ratios = dart_data['financial_statements'].get('financial_ratios', {})
            if ratios.get('roe', 0) > 15:
                strengths.append("높은 자기자본수익률 (ROE > 15%)")
            if ratios.get('debt_ratio', 100) < 30:
                strengths.append("낮은 부채비율 (건전한 재무구조)")
        
        # KIS 기반 강점
        if kis_data.get('technical_indicators', {}).get('is_golden_cross'):
            strengths.append("골든크로스 형성 (상승 모멘텀)")
        
        return strengths

    def _identify_key_risks(self, dart_data: Dict, kis_data: Dict) -> List[str]:
        """핵심 리스크 식별"""
        risks = []
        
        # DART 기반 리스크
        if dart_data.get('financial_statements'):
            ratios = dart_data['financial_statements'].get('financial_ratios', {})
            if ratios.get('debt_ratio', 0) > 80:
                risks.append("높은 부채비율 (재무 리스크)")
            if ratios.get('roe', 0) < -5:
                risks.append("지속적인 적자 (수익성 악화)")
        
        # 부정적 공시 체크
        disclosures = dart_data.get('recent_disclosures', [])
        negative_disclosures = [d for d in disclosures if self._is_negative_disclosure(d.get('report_nm', ''))]
        if negative_disclosures:
            risks.append(f"최근 부정적 공시 {len(negative_disclosures)}건")
        
        return risks

    def _calculate_target_price(self, dart_data: Dict, kis_data: Dict) -> Optional[Dict]:
        """목표 주가 계산"""
        try:
            current_price = kis_data.get('current_price', {}).get('price', 0)
            if not current_price:
                return None
            
            # 간단한 목표가 계산 (실제로는 더 복잡한 모델 사용)
            financial = dart_data.get('financial_statements')
            if financial:
                ratios = financial.get('financial_ratios', {})
                roe = ratios.get('roe', 10)
                
                # ROE 기반 목표가 계산
                if roe > 15:
                    target_multiplier = 1.2
                elif roe > 10:
                    target_multiplier = 1.1
                else:
                    target_multiplier = 0.95
                
                target_price = int(current_price * target_multiplier)
                
                return {
                    'target_price': target_price,
                    'upside_potential': ((target_price - current_price) / current_price) * 100,
                    'method': 'ROE 기반 계산'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 목표 주가 계산 실패: {e}")
            return None

    # ===================================================================
    # 🔍 보조 분석 메서드들
    # ===================================================================
    
    def _assess_financial_health(self, financial_statements: Optional[Dict]) -> str:
        """재무 건전성 평가"""
        if not financial_statements:
            return "정보 없음"
        
        ratios = financial_statements.get('financial_ratios', {})
        debt_ratio = ratios.get('debt_ratio', 50)
        roe = ratios.get('roe', 0)
        
        if debt_ratio < 30 and roe > 15:
            return "매우 건전"
        elif debt_ratio < 50 and roe > 10:
            return "건전"
        elif debt_ratio < 70:
            return "보통"
        else:
            return "주의 필요"

    def _assess_disclosure_quality(self, disclosures: List[Dict]) -> str:
        """공시 품질 평가"""
        if not disclosures:
            return "공시 없음"
        
        important_count = len([d for d in disclosures if d.get('importance_score', 0) > 7])
        
        if important_count == 0:
            return "안정적"
        elif important_count <= 2:
            return "보통"
        else:
            return "불안정"

    def _assess_dividend_attractiveness(self, dividend_info: Optional[Dict]) -> str:
        """배당 매력도 평가"""
        if not dividend_info:
            return "배당 정보 없음"
        
        recent_yields = []
        for year_data in dividend_info.values():
            yield_val = year_data.get('dividend_yield', 0)
            if yield_val > 0:
                recent_yields.append(yield_val)
        
        if not recent_yields:
            return "배당 없음"
        
        avg_yield = sum(recent_yields) / len(recent_yields)
        
        if avg_yield > 4:
            return "매우 매력적"
        elif avg_yield > 2:
            return "매력적"
        elif avg_yield > 0:
            return "보통"
        else:
            return "배당 없음"

    def _extract_key_highlights(self, dart_result: Dict) -> List[str]:
        """핵심 하이라이트 추출"""
        highlights = []
        
        company_info = dart_result.get('company_info', {})
        if company_info.get('employee_count', 0) > 5000:
            highlights.append("대기업 (직원 5천명 이상)")
        
        financial = dart_result.get('financial_statements', {})
        if financial:
            trend = financial.get('trend_analysis', {})
            if trend.get('revenue_growth', 0) > 20:
                highlights.append("고성장 (매출 20% 이상 증가)")
        
        return highlights

    def _is_negative_disclosure(self, report_name: str) -> bool:
        """부정적 공시 판별"""
        negative_keywords = [
            '정정', '취소', '연기', '중단', '손실', '적자', '감축', 
            '구조조정', '법정관리', '회생', '파산', '영업정지'
        ]
        
        return any(keyword in report_name for keyword in negative_keywords)

    def _calculate_overall_risk_level(self, risks: List[Dict]) -> str:
        """전체 리스크 수준 계산"""
        if not risks:
            return "LOW"
        
        high_count = len([r for r in risks if r.get('severity') == 'HIGH'])
        medium_count = len([r for r in risks if r.get('severity') == 'MEDIUM'])
        
        if high_count > 0:
            return "HIGH"
        elif medium_count > 1:
            return "MEDIUM"
        else:
            return "LOW"

    async def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """📈 기술적 지표 계산"""
        try:
            # 일봉 데이터 가져오기
            daily_data = await self.trader.fetch_daily_price_history(symbol, 60)
            if not daily_data:
                return {}
            
            # 기본 기술적 지표 계산
            prices = [float(d.get('stck_clpr', 0)) for d in daily_data if d.get('stck_clpr')]
            volumes = [int(d.get('acml_vol', 0)) for d in daily_data if d.get('acml_vol')]
            
            if len(prices) < 20:
                return {}
            
            # 이동평균 계산
            ma5 = sum(prices[:5]) / 5 if len(prices) >= 5 else 0
            ma20 = sum(prices[:20]) / 20 if len(prices) >= 20 else 0
            ma60 = sum(prices[:60]) / 60 if len(prices) >= 60 else 0
            
            # RSI 계산 (단순화)
            rsi = self._calculate_rsi(prices)
            
            # 거래량 이동평균
            volume_ma20 = sum(volumes[:20]) / 20 if len(volumes) >= 20 else 0
            
            # 골든크로스 여부
            is_golden_cross = ma5 > ma20 and len(prices) >= 2 and prices[1] <= ma20
            
            return {
                'ma5': ma5,
                'ma20': ma20,
                'ma60': ma60,
                'rsi': rsi,
                'volume_ma20': volume_ma20,
                'is_golden_cross': is_golden_cross,
                'current_price': prices[0] if prices else 0
            }
            
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] 기술적 지표 계산 실패: {e}")
            return {}

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, min(period + 1, len(prices))):
                change = prices[i-1] - prices[i]  # 최신 데이터가 앞에 있음
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if not gains or not losses:
                return 50.0
            
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
        except Exception as e:
            logger.warning(f"⚠️ RSI 계산 실패: {e}")
            return 50.0

    # ===================================================================
    # 🔄 대체 데이터 소스 활용 (DART API 보완)
    # ===================================================================
    
    async def get_alternative_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        🔄 대체 데이터 소스를 활용한 펀더멘털 데이터 수집
        - FinanceDataReader 활용
        - 공개 데이터 소스 활용
        - DART API 보완
        """
        logger.info(f"🔄 [{symbol}] 대체 데이터 소스 활용 시작...")
        
        try:
            alternative_data = {
                'symbol': symbol,
                'collection_timestamp': datetime.now().isoformat(),
                'data_sources': []
            }
            
            # 1. FinanceDataReader로 기본 정보 수집
            try:
                import FinanceDataReader as fdr
                
                # 주식 기본 정보
                stock_info = fdr.StockListing('KRX')
                stock_row = stock_info[stock_info['Code'] == symbol]
                
                if not stock_row.empty:
                    stock_data = stock_row.iloc[0]
                    alternative_data['company_name'] = stock_data.get('Name', '')
                    alternative_data['market'] = stock_data.get('Market', '')
                    alternative_data['sector'] = stock_data.get('Sector', '')
                    alternative_data['industry'] = stock_data.get('Industry', '')
                    alternative_data['data_sources'].append('FinanceDataReader')
                
                # 가격 데이터로 기본 분석
                price_data = fdr.DataReader(symbol, start='2023-01-01')
                if not price_data.empty:
                    recent_price = price_data.iloc[-1]['Close']
                    year_high = price_data['High'].max()
                    year_low = price_data['Low'].min()
                    
                    alternative_data['price_analysis'] = {
                        'current_price': recent_price,
                        '52week_high': year_high,
                        '52week_low': year_low,
                        'high_ratio': (recent_price / year_high) * 100,
                        'low_ratio': (recent_price / year_low) * 100
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ [{symbol}] FinanceDataReader 데이터 수집 실패: {e}")
            
            # 2. KIS API로 보완 데이터 수집
            try:
                current_price_info = await self.trader.get_current_price(symbol)
                if current_price_info:
                    alternative_data['kis_current_data'] = current_price_info
                    alternative_data['data_sources'].append('KIS_API')
            except Exception as e:
                logger.warning(f"⚠️ [{symbol}] KIS API 보완 데이터 수집 실패: {e}")
            
            # 3. 간단한 분석 점수 계산
            analysis_score = self._calculate_alternative_analysis_score(alternative_data)
            alternative_data['alternative_analysis_score'] = analysis_score
            
            logger.info(f"✅ [{symbol}] 대체 데이터 수집 완료 (점수: {analysis_score:.1f})")
            return alternative_data
            
        except Exception as e:
            logger.error(f"❌ [{symbol}] 대체 데이터 수집 실패: {e}")
            return {'error': str(e)}

    def _calculate_alternative_analysis_score(self, data: Dict[str, Any]) -> float:
        """대체 데이터 기반 분석 점수 계산"""
        score = 50.0  # 기본 점수
        
        try:
            # 가격 분석 점수
            price_analysis = data.get('price_analysis', {})
            if price_analysis:
                high_ratio = price_analysis.get('high_ratio', 50)
                low_ratio = price_analysis.get('low_ratio', 150)
                
                # 52주 고점 대비 위치
                if high_ratio > 90:
                    score -= 10  # 고점 근처는 위험
                elif high_ratio > 70:
                    score += 5   # 상승 추세
                elif high_ratio < 30:
                    score += 15  # 저점 근처는 기회
                
                # 52주 저점 대비 위치
                if low_ratio > 200:
                    score += 10  # 저점 대비 많이 상승
                elif low_ratio < 120:
                    score -= 5   # 저점 근처
            
            # 시장/섹터 점수
            market = data.get('market', '')
            if market == 'KOSPI':
                score += 5  # 대형주 선호
            
            sector = data.get('sector', '')
            if sector in ['IT', '바이오', '전기전자']:
                score += 5  # 성장 섹터 선호
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"⚠️ 대체 분석 점수 계산 실패: {e}")
            return 50.0

    async def get_comprehensive_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        🎯 종합 주식 데이터 수집 (DART + 대체 소스)
        - DART API 우선 시도
        - 실패시 대체 데이터 소스 활용
        - 모든 가능한 데이터 통합
        """
        logger.info(f"🎯 [{symbol}] 종합 주식 데이터 수집 시작...")
        
        comprehensive_data = {
            'symbol': symbol,
            'collection_timestamp': datetime.now().isoformat(),
            'data_quality': 'HIGH'  # HIGH, MEDIUM, LOW
        }
        
        try:
            # 1. DART API 시도
            if self.dart_available:
                dart_data = await self.get_dart_comprehensive_data(symbol)
                if dart_data and 'error' not in dart_data:
                    comprehensive_data['dart_data'] = dart_data
                    comprehensive_data['has_dart_data'] = True
                    logger.info(f"✅ [{symbol}] DART 데이터 수집 성공")
                else:
                    comprehensive_data['has_dart_data'] = False
                    comprehensive_data['data_quality'] = 'MEDIUM'
                    logger.warning(f"⚠️ [{symbol}] DART 데이터 수집 실패, 대체 소스 활용")
            else:
                comprehensive_data['has_dart_data'] = False
                comprehensive_data['data_quality'] = 'MEDIUM'
            
            # 2. 대체 데이터 소스 활용
            alternative_data = await self.get_alternative_fundamental_data(symbol)
            if alternative_data and 'error' not in alternative_data:
                comprehensive_data['alternative_data'] = alternative_data
                comprehensive_data['has_alternative_data'] = True
                logger.info(f"✅ [{symbol}] 대체 데이터 수집 성공")
            else:
                comprehensive_data['has_alternative_data'] = False
                if comprehensive_data['data_quality'] != 'HIGH':
                    comprehensive_data['data_quality'] = 'LOW'
            
            # 3. KIS API 실시간 데이터
            kis_data = await self.get_kis_comprehensive_data(symbol)
            if kis_data:
                comprehensive_data['kis_data'] = kis_data
                comprehensive_data['has_kis_data'] = True
            
            # 4. 최종 분석 점수 계산
            final_score = self._calculate_comprehensive_score(comprehensive_data)
            comprehensive_data['comprehensive_score'] = final_score
            
            logger.info(f"✅ [{symbol}] 종합 데이터 수집 완료 (품질: {comprehensive_data['data_quality']}, 점수: {final_score:.1f})")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"❌ [{symbol}] 종합 데이터 수집 실패: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'data_quality': 'ERROR'
            }

    def _calculate_comprehensive_score(self, data: Dict[str, Any]) -> float:
        """종합 데이터 기반 최종 점수 계산"""
        score = 50.0
        
        try:
            # DART 데이터 점수 (가중치 40%)
            if data.get('has_dart_data'):
                dart_score = data['dart_data'].get('dart_analysis_score', 50.0)
                score += (dart_score - 50.0) * 0.4
            
            # 대체 데이터 점수 (가중치 30%)
            if data.get('has_alternative_data'):
                alt_score = data['alternative_data'].get('alternative_analysis_score', 50.0)
                score += (alt_score - 50.0) * 0.3
            
            # KIS 데이터 점수 (가중치 30%)
            if data.get('has_kis_data'):
                # KIS 데이터 기반 간단한 점수 계산
                kis_data = data['kis_data']
                current_price = kis_data.get('current_price', {})
                if current_price:
                    change_rate = float(current_price.get('prdy_ctrt', 0))
                    if change_rate > 3:
                        score += 10
                    elif change_rate > 0:
                        score += 5
                    elif change_rate < -5:
                        score -= 10
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.warning(f"⚠️ 종합 점수 계산 실패: {e}")
            return 50.0

class StockFilter:
    """
    시장 상황에 맞는 유망 종목을 발굴하는 지능형 필터
    - 다양한 필터링 조건 (시총, 거래량/대금) 적용
    - AI 기반 종합 점수 계산
    - 캐싱 및 병렬 처리를 통해 빠른 응답 속도 보장
    """
    def __init__(self, data_provider: 'AIDataCollector'):
        """
        초기화
        :param data_provider: 데이터 수집을 위임할 AIDataCollector 인스턴스
        """
        self.data_provider = data_provider
        self.criteria = FilterCriteria()
        self.cache = {'data': None, 'timestamp': 0}
        self.cache_ttl = 60 * 5  # 5분 캐시
        self.max_workers = 10 # 병렬 처리 작업자 수
        self.listed_stocks = None
        self._last_listed_stocks_update = 0

    def set_filter_criteria(self, criteria: FilterCriteria) -> None:
        """필터링 기준 설정"""
        self.criteria = criteria
        logger.info(f"📊 필터링 기준 업데이트: 시총 {criteria.min_market_cap}억 이상, "
                   f"거래량 {criteria.min_volume:,}주 이상")
    
    async def get_filtered_stocks(self, force_update: bool = False) -> List[StockInfo]:
        """필터링된 종목 목록 반환"""
        if not force_update and self._is_cache_valid():
            logger.info(f"📋 캐시된 필터링 결과 반환: {len(self.filtered_stocks)}개 종목")
            return self.filtered_stocks
        
        logger.info("🔍 종목 필터링 시작...")
        start_time = time.time()
        
        # 1. 기본 종목 데이터 수집 (AIDataCollector에 위임)
        all_stocks = await self._collect_stock_data()
        if not all_stocks:
            logger.error("❌ 종목 데이터 수집 실패")
            return []
        
        # 2. 기본 필터링 적용
        basic_filtered = self._apply_basic_filters(all_stocks)
        logger.info(f"✅ 기본 필터링 완료: {len(all_stocks)} → {len(basic_filtered)}개 종목")
        
        # 3. AI 점수 계산 및 순위 결정
        scored_stocks = await self._calculate_ai_scores(basic_filtered)
        
        # 4. 최종 순위로 정렬 및 제한
        final_stocks = sorted(scored_stocks, key=lambda x: x.score, reverse=True)
        final_stocks = final_stocks[:self.criteria.max_stocks]
        
        self.filtered_stocks = final_stocks
        self.last_update = datetime.now()
        
        elapsed = time.time() - start_time
        logger.info(f"🎯 종목 필터링 완료: {len(final_stocks)}개 종목 선별 (소요시간: {elapsed:.1f}초)")
        
        # 결과 요약 로깅
        self._log_filtering_summary(final_stocks)
        
        return final_stocks
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_update or not self.filtered_stocks:
            return False
        
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < self.cache_duration
    
    async def _collect_stock_data(self) -> List[StockInfo]:
        """모든 소스에서 종목 데이터를 수집하고 병합합니다."""
        logger.info("   - (1단계) 5가지 핵심 순위(상승률, 거래량, 거래대금, 기관/외국인 순매수) 병렬 조회 요청...")
        
        try:
            # fetch_ranking_data를 사용하여 병렬로 데이터 요청
            ranking_types = ["rise", "volume", "value", "institution_net_buy", "foreign_net_buy"]
            tasks = [self.data_provider.trader.fetch_ranking_data(rtype, limit=100) for rtype in ranking_types]
            results = await asyncio.gather(*tasks)
            
            # 모든 결과를 하나의 딕셔너리로 병합하여 중복 제거
            combined_stocks = {}
            for stock_list in results:
                if not stock_list: continue
                for item in stock_list:
                    code = item.get('mksc_shrn_iscd') or item.get('h_kor_isnm') # 투자자별 순위는 종목코드가 다른 키에 담겨있을 수 있음
                    if code and code not in combined_stocks:
                         # `mksc_shrn_iscd`가 없는 경우를 대비하여 name이라도 저장
                        combined_stocks[code] = {'code': code, 'name': item.get('hts_kor_isnm', 'N/A')}

            if not combined_stocks:
                logger.warning("   - KIS 순위 조회 결과가 없습니다.")
                return []
                
            logger.info(f"   - (2단계) {len(combined_stocks)}개 후보 종목 상세 정보 병렬 조회...")
            
            # 상세 정보를 병렬로 가져옴
            stock_codes = list(combined_stocks.keys())
            stock_details = await self.data_provider.get_stock_details_parallel(stock_codes)

            return [stock for stock in stock_details if stock]
            
        except Exception as e:
            logger.error(f"❌ 종목 데이터 수집 중 오류: {e}", exc_info=True)
            return []

    def _apply_basic_filters(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """기본 필터링(시총, 거래량, 거래대금)을 적용합니다."""
        filtered = []
        for stock in stocks:
            if (stock.market_cap >= self.criteria.min_market_cap and
                stock.volume >= self.criteria.min_volume and
                stock.volume_value >= self.criteria.min_volume_value and
                stock.market_type in self.criteria.market_types and
                stock.sector not in self.criteria.exclude_sectors):
                filtered.append(stock)
        return filtered

    async def _calculate_ai_scores(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """AI 점수 계산"""
        scored_stocks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calculate_single_score, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scored_stocks.append(result)
        return scored_stocks

    def _calculate_single_score(self, stock: StockInfo) -> Optional[StockInfo]:
        """단일 종목 점수 계산"""
        # 복잡한 스코어링 로직 (예시)
        score = 0
        # 모멘텀 점수
        # ...
        # 펀더멘털 점수
        # ...
        # 섹터 보너스
        score += self._get_sector_bonus(stock.sector)
        stock.score = round(score, 2)
        return stock
        
    def _get_sector_bonus(self, sector: str) -> float:
        """업종별 보너스 점수"""
        bonus_map = {
            'IT/반도체': 15.0,
            '배터리/화학': 12.0,
            '바이오': 10.0,
        }
        return bonus_map.get(sector, 0.0)

    def _log_filtering_summary(self, stocks: List[StockInfo]) -> None:
        """필터링 결과 요약 로깅"""
        logger.info("--- 필터링된 상위 5개 종목 ---")
        for i, stock in enumerate(stocks[:5]):
            logger.info(f"{i+1}. {stock.name}({stock.code}) - 점수: {stock.score}")
        logger.info("-----------------------------") 