"""
🏛️ DART Open API 데이터 제공자
- DART 공시 데이터 수집 및 캐싱을 담당합니다.
- OpenDartReader 라이브러리를 사용하여 API와 상호작용합니다.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache
import OpenDartReader

from core.config import DART_API_KEY

logger = logging.getLogger(__name__)

class DartProvider:
    """DART 데이터 제공 및 관리를 위한 클래스"""

    def __init__(self):
        """DART API 리더를 초기화하고 캐시를 설정합니다."""
        self.dart = None
        self.dart_available = False
        if DART_API_KEY:
            try:
                self.dart = OpenDartReader(DART_API_KEY)
                # 간단한 API 호출로 연결 테스트
                self.dart.list(corp_code='005930', start_dt='20240101', end_dt='20240102', page_count=1)
                logger.info("✅ DART API 리더가 성공적으로 초기화되었습니다.")
                self.dart_available = True
            except Exception as e:
                logger.error(f"❌ DART API 초기화 실패: {e}")
        else:
            logger.warning("⚠️ DART_API_KEY가 설정되지 않아 DART 공시 조회를 비활성화합니다.")

        # DART 전용 캐시
        self.company_cache = TTLCache(maxsize=1000, ttl=86400)  # 24시간
        self.financial_cache = TTLCache(maxsize=500, ttl=21600)  # 6시간
        self.disclosure_cache = TTLCache(maxsize=1000, ttl=3600) # 1시간
        self.corp_code_cache = TTLCache(maxsize=2000, ttl=86400) # 24시간

    async def _get_corp_code(self, symbol: str) -> Optional[str]:
        """종목코드로 DART 기업코드를 조회하고 캐시합니다."""
        if symbol in self.corp_code_cache:
            return self.corp_code_cache[symbol]
        
        try:
            # 주요 종목 하드코딩 (빠른 조회를 위해)
            symbol_to_corp_code = {
                '005930': '00126380', '000660': '00119397', '035420': '00164779',
                '207940': '00168099', '005380': '00126186', '051910': '00164593',
            }
            if symbol in symbol_to_corp_code:
                self.corp_code_cache[symbol] = symbol_to_corp_code[symbol]
                return symbol_to_corp_code[symbol]

            corp_list = await asyncio.to_thread(self.dart.company, symbol)
            if corp_list is not None and not corp_list.empty:
                corp_code = corp_list.iloc[0]['corp_code']
                self.corp_code_cache[symbol] = corp_code
                return corp_code
            
            logger.warning(f"⚠️ [{symbol}]에 대한 DART 기업코드를 찾을 수 없습니다.")
            return None
        except Exception as e:
            logger.error(f"❌ [{symbol}] 기업코드 조회 중 오류 발생: {e}")
            return None

    async def get_company_info(self, symbol: str) -> Optional[Dict]:
        """🏢 DART 기업 기본정보 조회"""
        cache_key = f"company_{symbol}"
        if cache_key in self.company_cache:
            return self.company_cache[cache_key]
        
        corp_code = await self._get_corp_code(symbol)
        if not corp_code:
            return None
            
        try:
            company_info = await asyncio.to_thread(self.dart.company, corp_code)
            if company_info is None or company_info.empty: return None
            info_dict = company_info.iloc[0].to_dict()
            
            corp_outline = await asyncio.to_thread(self.dart.company_outline, corp_code)
            
            result = {
                'corp_code': corp_code, 'corp_name': info_dict.get('corp_name'),
                'ceo_nm': info_dict.get('ceo_nm'), 'corp_cls': info_dict.get('corp_cls'),
                'adres': info_dict.get('adres'), 'hm_url': info_dict.get('hm_url'),
                'ir_url': info_dict.get('ir_url'), 'phn_no': info_dict.get('phn_no'),
                'est_dt': info_dict.get('est_dt'), 'acc_mt': info_dict.get('acc_mt'),
                'induty_code': info_dict.get('induty_code')
            }
            
            if corp_outline is not None and not corp_outline.empty:
                outline_dict = corp_outline.iloc[0].to_dict()
                result.update({
                    'employee_count': outline_dict.get('emp_co'),
                    'main_business': outline_dict.get('bsn_sumry_ctnt'),
                    'capital_stock': outline_dict.get('cptl_stck_co')
                })
            
            self.company_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 기업정보 조회 실패: {e}")
            return None

    async def get_financial_statements(self, symbol: str, years: int = 5) -> Optional[Dict]:
        """💰 DART 재무제표 조회 (최근 N년)"""
        cache_key = f"financial_{symbol}_{years}"
        if cache_key in self.financial_cache:
            return self.financial_cache[cache_key]
        
        corp_code = await self._get_corp_code(symbol)
        if not corp_code: return None
            
        try:
            current_year = datetime.now().year
            financial_data = {}
            
            for year in range(current_year - years, current_year):
                year_str = str(year)
                fs_annual = await asyncio.to_thread(self.dart.finstate, corp_code, bsns_year=year_str, reprt_code='11011')
                
                if fs_annual is not None and not fs_annual.empty:
                    fs_dict = {}
                    for _, row in fs_annual.iterrows():
                        account_name = row.get('account_nm', '')
                        current_amount = row.get('thstrm_amount', 0)
                        if any(k in account_name for k in ['매출액', '영업이익', '당기순이익', '자산총계', '부채총계', '자본총계']):
                            try:
                                fs_dict[account_name] = int(str(current_amount).replace(',', '')) if current_amount else 0
                            except (ValueError, TypeError):
                                fs_dict[account_name] = 0
                    if fs_dict:
                        financial_data[year_str] = fs_dict
            
            if not financial_data: return None
            
            self.financial_cache[cache_key] = financial_data
            return financial_data
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 재무제표 조회 실패: {e}")
            return None

    async def get_recent_disclosures(self, symbol: str, days: int = 30) -> List[Dict]:
        """📢 DART 최근 공시 조회"""
        cache_key = f"disclosures_{symbol}_{days}"
        if cache_key in self.disclosure_cache:
            return self.disclosure_cache[cache_key]
        
        corp_code = await self._get_corp_code(symbol)
        if not corp_code: return []
            
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            disclosure_list = await asyncio.to_thread(self.dart.list, corp_code=corp_code, start_dt=start_date, end_dt=end_date)
            
            if disclosure_list is None or disclosure_list.empty: return []
                
            important_disclosures = []
            for _, row in disclosure_list.iterrows():
                report_nm = row.get('report_nm', '')
                if any(k in report_nm for k in ['주요사항보고서', '증권발행', '합병', '분할', '영업양수도']):
                    important_disclosures.append(row.to_dict())
            
            self.disclosure_cache[cache_key] = important_disclosures
            return important_disclosures
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 공시 조회 실패: {e}")
            return []

    async def get_major_shareholders(self, symbol: str) -> Optional[Dict]:
        """👥 DART 주요주주 현황"""
        corp_code = await self._get_corp_code(symbol)
        if not corp_code: return None

        try:
            current_year = datetime.now().year - 1
            major_shareholders = await asyncio.to_thread(self.dart.major_shareholders, corp_code, bsns_year=str(current_year))
            
            if major_shareholders is None or major_shareholders.empty: return None

            return {
                'reference_date': f"{current_year}년 사업보고서 기준",
                'shareholders': major_shareholders.to_dict('records')
            }
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] DART 주요주주 조회 실패: {e}")
            return None

    # ... DART 관련 다른 모든 헬퍼/데이터 수집 함수들을 여기에 추가 ... 