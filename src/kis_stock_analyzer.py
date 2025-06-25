#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔴 한국투자증권 API 기반 해외선물옵션 데이터 수집 시스템
================================================================
한투 공식 API 문서 스펙에 따른 정확한 구현
"""

import os
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class KISStockAnalyzer:
    """한국투자증권 API 기반 주식 분석기"""
    
    def __init__(self):
        # 한투 라이브 API 정보
        self.app_key = os.getenv('LIVE_KIS_APP_KEY', '')
        self.app_secret = os.getenv('LIVE_KIS_APP_SECRET', '')
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.mock_url = "https://openapivts.koreainvestment.com:29443"
        self.access_token = None
        self.token_expired_time = None
        
        # API 호출 제한 관리
        self.last_call_time = 0
        self.min_interval = 0.1  # 100ms 간격
        
        # 다양한 TR_ID 정의
        self.tr_ids = {
            # 해외선물 관련
            'overseas_futures_detail': 'HHDFS00000300',      # 해외선물 종목상세
            'overseas_futures_price': 'HDFFF020',            # 해외선물옵션 실시간시세
            'overseas_futures_current': 'HHDFS76240000',     # 해외선물 현재가
            'overseas_futures_chart': 'HHDFS76410000',       # 해외선물 차트
            
            # 해외옵션 관련  
            'overseas_options_detail': 'HHOPT00000300',      # 해외옵션 종목상세
            'overseas_options_price': 'HHOPT76240000',       # 해외옵션 현재가
            
            # 해외주식 관련
            'overseas_stock_price': 'HHDFS00000300',         # 해외주식 현재가
            'overseas_stock_detail': 'HHKDB67130300',        # 해외주식 종목상세
            'overseas_stock_chart': 'HHDFS76200200',         # 해외주식 차트
            
            # 국내주식 관련
            'domestic_stock_price': 'FHKST01010100',         # 국내주식 현재가
            'domestic_stock_detail': 'CTPF1002R',            # 국내주식 종목상세
            'domestic_stock_chart': 'FHKST03010100',         # 국내주식 차트
        }
        
        # 한투 계좌 정보
        self.account_no = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '').replace('-', '')
        
        if not all([self.app_key, self.app_secret, self.account_no]):
            raise ValueError("❌ 환경 변수에서 API 키를 찾을 수 없습니다. .env 파일을 확인하세요.")
        
        print("🔴 한국투자증권 API 기반 주식 분석기 초기화")
        print(f"📍 Base URL: {self.base_url}")
        
    def _rate_limit(self):
        """API 호출 속도 제한"""
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_access_token(self) -> str:
        """액세스 토큰 발급"""
        try:
            if self.access_token and self.token_expired_time:
                if datetime.now() < self.token_expired_time:
                    return self.access_token
            
            self._rate_limit()
            
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"Content-Type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                expires_in = result.get('expires_in', 86400)
                self.token_expired_time = datetime.now().timestamp() + expires_in
                logger.info("✅ 액세스 토큰 발급 성공")
                return self.access_token
            else:
                logger.error(f"❌ 토큰 발급 실패: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 토큰 발급 중 오류: {str(e)}")
            return None
    
    def make_api_call(self, tr_id: str, params: Dict = None, headers_extra: Dict = None) -> Optional[Dict]:
        """통합 API 호출 함수"""
        try:
            access_token = self.get_access_token()
            if not access_token:
                return None
            
            self._rate_limit()
            
            url = f"{self.base_url}/uapi/overseas-futureoption/v1/trading/inquire-present-balance"
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "authorization": f"Bearer {access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": tr_id,
                "custtype": "P",
            }
            
            if headers_extra:
                headers.update(headers_extra)
            
            # GET 요청
            response = requests.get(url, headers=headers, params=params)
            
            logger.info(f"📡 API 호출: {tr_id} - Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ API 호출 실패: {tr_id} - {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ API 호출 중 오류 ({tr_id}): {str(e)}")
            return None
    
    def test_all_tr_ids(self) -> Dict[str, Any]:
        """모든 TR_ID 테스트"""
        results = {}
        
        logger.info("🔍 모든 TR_ID 테스트 시작...")
        
        for name, tr_id in self.tr_ids.items():
            logger.info(f"📊 테스트 중: {name} ({tr_id})")
            
            # 기본 파라미터 설정
            params = {}
            headers_extra = {}
            
            if 'futures' in name:
                params = {
                    'CANO': self.account_no[:8],
                    'ACNT_PRDT_CD': self.account_no[8:],
                    'OVRS_EXCG_CD': 'CME',  # CME 거래소
                    'PDNO': 'ES2412',       # S&P500 E-mini 선물
                }
            elif 'options' in name:
                params = {
                    'CANO': self.account_no[:8],
                    'ACNT_PRDT_CD': self.account_no[8:],
                    'OVRS_EXCG_CD': 'CME',
                    'PDNO': 'ES2412C5600',  # S&P500 옵션
                }
            elif 'stock' in name:
                if 'overseas' in name:
                    params = {
                        'CANO': self.account_no[:8],
                        'ACNT_PRDT_CD': self.account_no[8:],
                        'OVRS_EXCG_CD': 'NASD',
                        'PDNO': 'AAPL',
                    }
                else:  # 국내주식
                    params = {
                        'CANO': self.account_no[:8],
                        'ACNT_PRDT_CD': self.account_no[8:],
                        'FID_COND_MRKT_DIV_CODE': 'J',
                        'FID_INPUT_ISCD': '005930',  # 삼성전자
                    }
            
            result = self.make_api_call(tr_id, params, headers_extra)
            results[name] = {
                'tr_id': tr_id,
                'success': result is not None,
                'data': result,
                'timestamp': datetime.now().isoformat()
            }
            
            # 성공/실패 로그
            if result:
                logger.info(f"✅ {name}: 성공")
                if 'output' in result and result['output']:
                    logger.info(f"   📈 데이터 필드 수: {len(result['output'])}")
            else:
                logger.warning(f"❌ {name}: 실패")
            
            time.sleep(0.2)  # 추가 대기
        
        return results
    
    def get_comprehensive_market_data(self) -> Dict[str, Any]:
        """종합적인 시장 데이터 수집"""
        logger.info("📊 종합 시장 데이터 수집 시작...")
        
        market_data = {
            'collection_time': datetime.now().isoformat(),
            'tr_id_test_results': {},
            'successful_data': {},
            'failed_requests': [],
            'summary': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0
            }
        }
        
        # 모든 TR_ID 테스트
        tr_results = self.test_all_tr_ids()
        market_data['tr_id_test_results'] = tr_results
        
        # 성공한 요청과 실패한 요청 분류
        for name, result in tr_results.items():
            market_data['summary']['total_requests'] += 1
            
            if result['success'] and result['data']:
                market_data['successful_data'][name] = result['data']
                market_data['summary']['successful_requests'] += 1
            else:
                market_data['failed_requests'].append({
                    'name': name,
                    'tr_id': result['tr_id'],
                    'reason': 'API 호출 실패 또는 빈 응답'
                })
                market_data['summary']['failed_requests'] += 1
        
        # 성공률 계산
        if market_data['summary']['total_requests'] > 0:
            market_data['summary']['success_rate'] = (
                market_data['summary']['successful_requests'] / 
                market_data['summary']['total_requests'] * 100
            )
        
        logger.info(f"📈 데이터 수집 완료: {market_data['summary']['successful_requests']}/{market_data['summary']['total_requests']} 성공 ({market_data['summary']['success_rate']:.1f}%)")
        
        return market_data
    
    def analyze_available_apis(self) -> Dict[str, Any]:
        """사용 가능한 API 분석"""
        logger.info("🔍 사용 가능한 API 분석 중...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'available_categories': [],
            'recommended_tr_ids': {},
            'data_coverage': {
                'kospi200': False,
                'nasdaq100': False,
                'sp500': False,
                'futures': False,
                'options': False
            },
            'next_steps': []
        }
        
        # 종합 데이터 수집
        market_data = self.get_comprehensive_market_data()
        
        # 성공한 API 분석
        successful_apis = market_data['successful_data']
        
        if successful_apis:
            analysis['available_categories'] = list(successful_apis.keys())
            
            # 각 카테고리별 추천 TR_ID
            for category, data in successful_apis.items():
                tr_id = market_data['tr_id_test_results'][category]['tr_id']
                analysis['recommended_tr_ids'][category] = tr_id
                
                # 데이터 커버리지 확인
                if 'futures' in category:
                    analysis['data_coverage']['futures'] = True
                    analysis['data_coverage']['sp500'] = True
                elif 'options' in category:
                    analysis['data_coverage']['options'] = True
                elif 'stock' in category:
                    if 'overseas' in category:
                        analysis['data_coverage']['nasdaq100'] = True
                    else:
                        analysis['data_coverage']['kospi200'] = True
        
        # 다음 단계 추천
        if analysis['data_coverage']['futures']:
            analysis['next_steps'].append("✅ 해외선물 데이터 수집 가능 - S&P500 E-mini 선물 활용")
        
        if analysis['data_coverage']['nasdaq100']:
            analysis['next_steps'].append("✅ 나스닥 주식 데이터 수집 가능 - 개별 종목 조회")
        
        if not any(analysis['data_coverage'].values()):
            analysis['next_steps'].extend([
                "❌ 현재 사용 가능한 API가 제한적입니다.",
                "💡 추가 TR_ID나 API 권한이 필요할 수 있습니다.",
                "📋 한투 API 문서에서 추가 TR_ID를 확인해보세요."
            ])
        
        return analysis

def main():
    """메인 실행 함수"""
    try:
        analyzer = KISStockAnalyzer()
        
        # API 분석 실행
        analysis_result = analyzer.analyze_available_apis()
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kis_api_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print("🔍 한투 API 분석 결과")
        print("="*80)
        
        print(f"\n📊 전체 API 테스트 결과:")
        print(f"   • 총 요청 수: {len(analyzer.tr_ids)}")
        
        if analysis_result['available_categories']:
            print(f"   • 성공한 API: {len(analysis_result['available_categories'])}")
            print(f"   • 사용 가능한 카테고리:")
            for category in analysis_result['available_categories']:
                tr_id = analysis_result['recommended_tr_ids'][category]
                print(f"     - {category}: {tr_id}")
        
        print(f"\n📈 데이터 커버리지:")
        for market, available in analysis_result['data_coverage'].items():
            status = "✅" if available else "❌"
            print(f"   {status} {market.upper()}")
        
        print(f"\n💡 다음 단계:")
        for step in analysis_result['next_steps']:
            print(f"   {step}")
        
        print(f"\n📄 상세 결과가 {filename}에 저장되었습니다.")
        print("="*80)
        
        # 추가 정보 제공
        if not analysis_result['available_categories']:
            print("\n🤔 현재 상황 분석:")
            print("   • 대부분의 TR_ID에서 데이터를 가져오지 못하고 있습니다.")
            print("   • 이는 다음 중 하나의 이유일 수 있습니다:")
            print("     1. API 권한 부족 (파생상품 권한 미신청)")
            print("     2. 잘못된 TR_ID 사용")
            print("     3. 파라미터 구조 오류")
            print("     4. 계좌 유형과 API 불일치")
            print("\n💡 해결 방법:")
            print("   • 한투 고객센터에 파생상품 권한 신청 확인")
            print("   • 한투 GitHub에서 정확한 예시 코드 확인")
            print("   • 모의투자와 실투자 API의 차이점 확인")
            
    except Exception as e:
        logger.error(f"❌ 메인 실행 중 오류: {str(e)}")
        raise

if __name__ == "__main__":
    main() 