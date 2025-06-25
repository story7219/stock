#!/usr/bin/env python3
"""
실패한 API들에 대한 대체 TR_ID 및 경로 테스트
"""

import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AlternativeAPITester:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('LIVE_KIS_APP_KEY')
        self.app_secret = os.getenv('LIVE_KIS_APP_SECRET')
        self.access_token = None
        
    def get_access_token(self) -> bool:
        """액세스 토큰 발급"""
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                print(f"🎫 토큰 발급: ✅ 성공")
                return True
            else:
                print(f"🎫 토큰 발급: ❌ 실패 ({response.status_code})")
                return False
                
        except Exception as e:
            print(f"🎫 토큰 발급: ❌ 예외 발생 - {str(e)}")
            return False
    
    def test_volume_ranking_alternatives(self):
        """거래량 순위 API 대체 방법들 테스트"""
        print("\n🔍 거래량 순위 API 대체 방법 테스트")
        print("-" * 50)
        
        alternatives = [
            {
                "name": "거래량순위_방법1",
                "tr_id": "FHPST01710000",
                "path": "/uapi/domestic-stock/v1/ranking/volume",
                "method": "GET"
            },
            {
                "name": "거래량순위_방법2", 
                "tr_id": "FHPST01710000",
                "path": "/uapi/domestic-stock/v1/quotations/volume-rank",
                "method": "GET"
            },
            {
                "name": "거래량순위_방법3",
                "tr_id": "FHKST01710000",  # 다른 TR_ID 시도
                "path": "/uapi/domestic-stock/v1/ranking/volume",
                "method": "GET"
            },
            {
                "name": "거래량순위_방법4_POST",
                "tr_id": "FHPST01710000",
                "path": "/uapi/domestic-stock/v1/ranking/volume",
                "method": "POST"
            }
        ]
        
        for alt in alternatives:
            self.test_single_api(alt)
            time.sleep(0.5)
    
    def test_sector_price_alternatives(self):
        """업종 현재가 API 대체 방법들 테스트"""
        print("\n🔍 업종 현재가 API 대체 방법 테스트")
        print("-" * 50)
        
        alternatives = [
            {
                "name": "업종현재가_방법1",
                "tr_id": "FHKUP03010100",
                "path": "/uapi/domestic-stock/v1/quotations/inquire-sector-price",
                "method": "GET"
            },
            {
                "name": "업종현재가_방법2",
                "tr_id": "FHKUP03010100", 
                "path": "/uapi/domestic-stock/v1/quotations/sector-price",
                "method": "GET"
            },
            {
                "name": "업종현재가_방법3",
                "tr_id": "FHKST03010100",  # 다른 TR_ID 시도
                "path": "/uapi/domestic-stock/v1/quotations/inquire-sector-price",
                "method": "GET"
            },
            {
                "name": "업종현재가_방법4_POST",
                "tr_id": "FHKUP03010100",
                "path": "/uapi/domestic-stock/v1/quotations/inquire-sector-price",
                "method": "POST"
            }
        ]
        
        for alt in alternatives:
            self.test_single_api(alt)
            time.sleep(0.5)
    
    def test_single_api(self, config):
        """단일 API 테스트"""
        try:
            url = f"{self.base_url}{config['path']}"
            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": config['tr_id']
            }
            
            # 거래량 순위용 파라미터
            if "거래량" in config['name']:
                params = {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_COND_SCR_DIV_CODE": "20170",
                    "FID_INPUT_ISCD": "0000",
                    "FID_DIV_CLS_CODE": "0",
                    "FID_BLNG_CLS_CODE": "0",
                    "FID_TRGT_CLS_CODE": "111111111",
                    "FID_TRGT_EXLS_CLS_CODE": "000000",
                    "FID_INPUT_PRICE_1": "",
                    "FID_INPUT_PRICE_2": "",
                    "FID_VOL_CNT": "",
                    "FID_INPUT_DATE_1": datetime.now().strftime("%Y%m%d")
                }
            # 업종 현재가용 파라미터
            else:
                params = {
                    "FID_COND_MRKT_DIV_CODE": "U",
                    "FID_INPUT_ISCD": "001"
                }
                headers["custtype"] = "P"
            
            if config['method'] == 'GET':
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.post(url, headers=headers, json=params)
            
            if response.status_code == 200:
                print(f"  ✅ {config['name']}: 성공 ({config['tr_id']})")
                try:
                    data = response.json()
                    if 'output' in data and isinstance(data['output'], list):
                        print(f"     📊 데이터: {len(data['output'])}개 레코드")
                    elif 'Output' in data and isinstance(data['Output'], list):
                        print(f"     📊 데이터: {len(data['Output'])}개 레코드")
                    else:
                        print(f"     📊 응답 키: {list(data.keys())}")
                except:
                    print(f"     📊 응답 크기: {len(response.text)} bytes")
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('msg1', 'Unknown error')
                    print(f"  ❌ {config['name']}: 실패 ({response.status_code}) - {error_msg}")
                except:
                    print(f"  ❌ {config['name']}: 실패 ({response.status_code}) - {response.text[:100]}")
                    
        except Exception as e:
            print(f"  ❌ {config['name']}: 예외 발생 - {str(e)}")
    
    def run_tests(self):
        """모든 대체 방법 테스트 실행"""
        print("=" * 60)
        print("🔧 실패한 API 대체 방법 테스트")
        print("=" * 60)
        
        if not self.get_access_token():
            print("토큰 발급 실패로 테스트 중단")
            return
        
        self.test_volume_ranking_alternatives()
        self.test_sector_price_alternatives()
        
        print("\n" + "=" * 60)
        print("테스트 완료")
        print("=" * 60)

if __name__ == "__main__":
    tester = AlternativeAPITester()
    tester.run_tests() 