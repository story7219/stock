# -*- coding: utf-8 -*-
"""
DART (금융감독원 전자공시시스템) API 핸들러 모듈
- 기업 개황 정보 조회
- 재무제표 (손익계산서, 재무상태표 등) 원본 데이터 조회
"""
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any

# config 모듈에서 API 키를 가져옵니다.
try:
    import config
    DART_API_KEY = config.DART_API_KEY
except (ImportError, AttributeError):
    print("⚠️ config.py 또는 DART_API_KEY를 찾을 수 없습니다. DART API 키를 수동으로 설정해주세요.")
    DART_API_KEY = None

logger = logging.getLogger(__name__)

class DartApiHandler:
    """
    DART API와의 통신을 관리하는 클래스입니다.
    """
    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self, api_key: Optional[str] = DART_API_KEY):
        """
        DART API 핸들러를 초기화합니다.

        :param api_key: DART API 키. 제공되지 않으면 config에서 가져옵니다.
        """
        if not api_key:
            raise ValueError("DART API 키가 필요합니다.")
        self.api_key = api_key
        self.session = requests.Session()
        self.corp_code_map = None
        self._load_corp_code_list()

    def _request_api(self, path: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        DART API에 GET 요청을 보냅니다.

        :param path: API 엔드포인트 경로
        :param params: 요청 파라미터
        :return: JSON 응답 또는 실패 시 None
        """
        url = f"{self.BASE_URL}/{path}"
        base_params = {'crtfc_key': self.api_key}
        if params:
            base_params.update(params)
        
        try:
            response = self.session.get(url, params=base_params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get('status') != '000':
                logger.error(f"❌ DART API 오류: {result.get('status')} - {result.get('message')}")
                return None
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ DART API 요청 실패: {e}")
        except json.JSONDecodeError:
            logger.error("❌ DART API 응답이 유효한 JSON이 아닙니다.")
        return None

    def _load_corp_code_list(self):
        """
        DART에 등록된 전체 기업의 고유번호를 다운로드하고 매핑을 생성합니다.
        """
        logger.info("DART 기업 고유번호 목록을 로드합니다...")
        path = "corpCode.xml"
        url = f"{self.BASE_URL}/{path}"
        params = {'crtfc_key': self.api_key}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open('CORPCODE.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    self.corp_code_map = {
                        item.find('stock_code').text.strip(): {
                            'corp_code': item.find('corp_code').text.strip(),
                            'corp_name': item.find('corp_name').text.strip(),
                        }
                        for item in root.findall('.//list') if item.find('stock_code').text.strip()
                    }
            logger.info(f"✅ DART 기업 고유번호 {len(self.corp_code_map)}개 로드 완료.")
        except Exception as e:
            logger.error(f"❌ DART 기업 고유번호 목록 로드 실패: {e}")
            self.corp_code_map = {}

    def get_corp_code(self, stock_code: str) -> Optional[str]:
        """
        종목 코드로 DART 기업 고유번호를 조회합니다.

        :param stock_code: 6자리 종목 코드 (예: "005930")
        :return: DART 기업 고유번호 또는 None
        """
        return self.corp_code_map.get(stock_code, {}).get('corp_code')

    def get_company_overview(self, stock_code: str) -> Optional[Dict[str, str]]:
        """
        특정 기업의 개황 정보를 조회합니다.

        :param stock_code: 6자리 종목 코드
        :return: 기업 개황 정보 딕셔너리
        """
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            logger.warning(f"종목코드({stock_code})에 해당하는 DART 고유번호를 찾을 수 없습니다.")
            return None
        
        path = "company.json"
        params = {'corp_code': corp_code}
        
        logger.info(f"DART API로 [{stock_code}] 기업 개황 정보 조회...")
        return self._request_api(path, params)

    def get_financial_statement(self, stock_code: str, year: int, reprt_code: str = "11011") -> Optional[Dict[str, Any]]:
        """
        특정 기업의 특정 연도 재무제표를 조회합니다.

        :param stock_code: 6자리 종목 코드
        :param year: 사업 연도 (예: 2023)
        :param reprt_code: 보고서 코드 ("11011": 사업보고서, "11012": 반기보고서, "11013": 1분기보고서, "11014": 3분기보고서)
        :return: 재무제표 데이터
        """
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            return None
            
        path = "fnlttSinglAcntAll.json" # 모든 재무제표
        params = {
            'corp_code': corp_code,
            'bsns_year': str(year),
            'reprt_code': reprt_code,
        }
        
        logger.info(f"DART API로 [{stock_code}] {year}년 재무제표 조회 (보고서 코드: {reprt_code})...")
        return self._request_api(path, params)

    def get_major_disclosures(self, stock_code: str, days: int = 90) -> Optional[List[Dict[str, Any]]]:
        """
        특정 기업의 주요 공시사항(정기공시 제외)을 조회합니다.

        :param stock_code: 6자리 종목 코드
        :param days: 조회할 기간(일)
        :return: 공시 목록 리스트
        """
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        path = "list.json"
        params = {
            'corp_code': corp_code,
            'bgn_de': start_date.strftime('%Y%m%d'),
            'end_de': end_date.strftime('%Y%m%d'),
            'pblntf_ty': 'A', # A: 주요사항보고서
            'page_no': 1,
            'page_count': 20 # 최근 20개
        }
        logger.info(f"DART API로 [{stock_code}] 최근 {days}일 주요 공시사항 조회...")
        response = self._request_api(path, params)
        return response.get('list') if response else None

    def get_financials_for_last_quarters(self, stock_code: str) -> Dict[str, Any]:
        """
        최근 4개 분기의 재무제표를 조회하여 하나로 합칩니다.

        :param stock_code: 6자리 종목 코드
        :return: 분기별 재무 데이터 딕셔너리
        """
        corp_code = self.get_corp_code(stock_code)
        if not corp_code:
            return {}
        
        today = datetime.now()
        quarters = []
        # 현재 연도부터 과거 2년까지 탐색
        for year in range(today.year, today.year - 2, -1):
            if len(quarters) >= 4: break
            # 3분기 보고서
            if not (year == today.year and today.month < 10):
                quarters.append({'year': year, 'reprt_code': '11014', 'name': f'{year} 3Q'})
            if len(quarters) >= 4: break
            # 반기 보고서
            if not (year == today.year and today.month < 7):
                quarters.append({'year': year, 'reprt_code': '11012', 'name': f'{year} 2Q'})
            if len(quarters) >= 4: break
            # 1분기 보고서
            if not (year == today.year and today.month < 4):
                 quarters.append({'year': year, 'reprt_code': '11013', 'name': f'{year} 1Q'})
            if len(quarters) >= 4: break
            # 사업 보고서 (전년도 실적)
            quarters.append({'year': year - 1, 'reprt_code': '11011', 'name': f'{year-1} 4Q'})

        all_financials = {}
        for q in quarters[:4]:
            fs = self.get_financial_statement(stock_code, q['year'], q['reprt_code'])
            if fs and fs.get('list'):
                all_financials[q['name']] = fs.get('list')
        
        return all_financials

if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트 코드
    if not DART_API_KEY:
        print("DART_API_KEY가 설정되지 않아 테스트를 진행할 수 없습니다.")
    else:
        handler = DartApiHandler(api_key=DART_API_KEY)
        
        # 삼성전자 (005930) 테스트
        stock_code = "005930"
        
        print(f"\n--- {stock_code} (삼성전자) DART API 테스트 ---")
        
        # 1. 기업 고유번호 조회
        corp_code = handler.get_corp_code(stock_code)
        print(f"종목코드 {stock_code}의 DART 고유번호: {corp_code}")
        
        # 2. 기업 개황 정보 조회
        overview = handler.get_company_overview(stock_code)
        if overview:
            print("\n[기업 개황 정보]")
            print(f"  회사명: {overview.get('corp_name')}")
            print(f"  대표자명: {overview.get('ceo_nm')}")
            print(f"  주소: {overview.get('adres')}")
            print(f"  홈페이지: {overview.get('hm_url')}")
        
        # 3. 2023년 사업보고서 기준 재무제표 조회
        fs_data = handler.get_financial_statement(stock_code, 2023)
        if fs_data and fs_data.get('list'):
            print("\n[2023년 재무제표 (일부)]")
            for item in fs_data['list'][:5]: # 상위 5개 항목만 출력
                print(f"  - 계정: {item.get('account_nm')}, 금액: {item.get('thstrm_amount', 'N/A')} 원")
        else:
            print("\n재무제표 데이터를 가져오지 못했습니다.")

        # 4. 주요 공시사항 조회
        disclosures = handler.get_major_disclosures(stock_code, days=90)
        if disclosures:
            print(f"\n[최근 90일 주요 공시 (최대 5개)]")
            for item in disclosures[:5]:
                print(f"  - [{item.get('rcept_dt')}] {item.get('report_nm')}")

        # 5. 최근 4분기 재무제표 조회
        quarterly_fs = handler.get_financials_for_last_quarters(stock_code)
        if quarterly_fs:
            print("\n[최근 4분기 재무 데이터 요약]")
            for quarter, data in quarterly_fs.items():
                sales = next((item['thstrm_amount'] for item in data if item['account_nm'] == '매출액'), 'N/A')
                profit = next((item['thstrm_amount'] for item in data if '영업이익' in item['account_nm']), 'N/A')
                print(f"  - {quarter}: 매출액={sales}, 영업이익={profit}") 