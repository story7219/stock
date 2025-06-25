#!/usr/bin/env python3
"""
국내주식 대량체결건수 상위 API 테스트
사용자가 제공한 스펙을 기반으로 다양한 조합 시도
"""

import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LargeVolumeCountTester:
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
                if response.text:
                    try:
                        error_data = response.json()
                        print(f"   오류: {error_data}")
                    except:
                        print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"🎫 토큰 발급: ❌ 예외 발생 - {str(e)}")
            return False
    
    def test_large_volume_count_alternatives(self):
        """대량체결건수 상위 API 다양한 조합 테스트"""
        print("\n🔍 국내주식 대량체결건수 상위 API 테스트")
        print("-" * 60)
        
        alternatives = [
            {
                "name": "대량체결건수_방법1",
                "tr_id": "FHPST01760000",
                "path": "/uapi/domestic-stock/v1/ranking/large-volume-count"
            },
            {
                "name": "대량체결건수_방법2",
                "tr_id": "FHPST01760000", 
                "path": "/uapi/domestic-stock/v1/quotations/large-volume-count"
            },
            {
                "name": "대량체결건수_방법3",
                "tr_id": "FHPST01750000",  # 다른 TR_ID 시도
                "path": "/uapi/domestic-stock/v1/ranking/large-volume-count"
            },
            {
                "name": "대량체결건수_방법4",
                "tr_id": "FHPST01720000",  # 또 다른 TR_ID 시도
                "path": "/uapi/domestic-stock/v1/ranking/volume-count"
            },
            {
                "name": "대량체결건수_방법5",
                "tr_id": "FHPST01780000",  # 체결건수 관련 TR_ID
                "path": "/uapi/domestic-stock/v1/ranking/execution-count"
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
                "tr_id": config['tr_id'],
                "custtype": "P"  # 개인고객
            }
            
            # 사용자가 제공한 파라미터 스펙 사용
            params = {
                "fid_aply_rang_prc_2": "",
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20170",
                "fid_input_iscd": "0000",
                "fid_rank_sort_cls_code": "0",
                "fid_div_cls_code": "0",
                "fid_input_price_1": "",
                "fid_aply_rang_prc_1": "",
                "fid_input_iscd_2": "",
                "fid_trgt_exls_cls_code": "000000",
                "fid_trgt_cls_code": "111111111",
                "fid_vol_cnt": ""
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                print(f"  ✅ {config['name']}: 성공 ({config['tr_id']})")
                try:
                    data = response.json()
                    if 'output' in data and isinstance(data['output'], list):
                        print(f"     📊 데이터: {len(data['output'])}개 레코드")
                        if data['output']:
                            sample = data['output'][0]
                            print(f"     📋 샘플: {sample.get('hts_kor_isnm', '')} - 체결건수: {sample.get('shnu_cntg_csnu', '')}+{sample.get('seln_cntg_csnu', '')}")
                    elif 'Output' in data and isinstance(data['Output'], list):
                        print(f"     📊 데이터: {len(data['Output'])}개 레코드")
                    else:
                        print(f"     📊 응답 키: {list(data.keys())}")
                        print(f"     📋 전체 응답: {json.dumps(data, ensure_ascii=False, indent=2)[:200]}...")
                except Exception as e:
                    print(f"     📊 응답 크기: {len(response.text)} bytes")
                    print(f"     ❌ 파싱 오류: {str(e)}")
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('msg1', error_data.get('error_description', 'Unknown error'))
                    print(f"  ❌ {config['name']}: 실패 ({response.status_code}) - {error_msg}")
                except:
                    print(f"  ❌ {config['name']}: 실패 ({response.status_code}) - {response.text[:100]}")
                    
        except Exception as e:
            print(f"  ❌ {config['name']}: 예외 발생 - {str(e)}")
    
    def run_tests(self):
        """모든 테스트 실행"""
        print("=" * 60)
        print("🔧 국내주식 대량체결건수 상위 API 테스트")
        print("=" * 60)
        
        if not self.get_access_token():
            print("토큰 발급 실패로 테스트 중단")
            return
        
        self.test_large_volume_count_alternatives()
        
        print("\n" + "=" * 60)
        print("테스트 완료")
        print("=" * 60)

if __name__ == "__main__":
    tester = LargeVolumeCountTester()
    tester.run_tests() 