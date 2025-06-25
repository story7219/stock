#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔴 한투 API 국내주식 기본 테스트
================================
사용자 제공 스크린샷 기반 구현
"""

import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import time

# 환경 변수 로드
load_dotenv()

class KISDomesticTest:
    """한투 국내주식 기본 테스트"""
    
    def __init__(self):
        self.app_key = os.getenv('LIVE_KIS_APP_KEY', '')
        self.app_secret = os.getenv('LIVE_KIS_APP_SECRET', '')
        self.account_no = os.getenv('LIVE_KIS_ACCOUNT_NUMBER', '').replace('-', '')
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.access_token = None
        
        print(f"🔑 API Key: {self.app_key[:10]}...")
        print(f"📱 Account: {self.account_no}")
        
    def get_access_token(self):
        """액세스 토큰 발급"""
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"Content-Type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            response = requests.post(url, headers=headers, json=data)
            print(f"🎫 토큰 요청: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                print("✅ 토큰 발급 성공")
                return True
            else:
                print(f"❌ 토큰 발급 실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 토큰 발급 오류: {str(e)}")
            return False
    
    def test_domestic_stock_price(self, stock_code="005930"):
        """국내주식 현재가 조회 (삼성전자)"""
        try:
            if not self.access_token:
                if not self.get_access_token():
                    return None
            
            # 스크린샷에서 확인한 정확한 스펙
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "FHKST01010100",
                "custtype": "P",
            }
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code
            }
            
            response = requests.get(url, headers=headers, params=params)
            print(f"📈 현재가 조회 ({stock_code}): {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('rt_cd') == '0':
                    output = data.get('output', {})
                    print(f"✅ {stock_code} 현재가: {output.get('stck_prpr', 'N/A')}원")
                    print(f"📊 전일대비: {output.get('prdy_vrss', 'N/A')}원")
                    print(f"📈 등락률: {output.get('prdy_ctrt', 'N/A')}%")
                    return data
                else:
                    print(f"⚠️ API 응답 오류: {data.get('msg1', 'Unknown')}")
            else:
                print(f"❌ API 호출 실패: {response.text}")
                
        except Exception as e:
            print(f"❌ 현재가 조회 오류: {str(e)}")
        
        return None
    
    def test_domestic_stock_orderbook(self, stock_code="005930"):
        """국내주식 호가 조회"""
        try:
            if not self.access_token:
                if not self.get_access_token():
                    return None
            
            # 스크린샷에서 확인한 호가 조회 스펙
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-asking-price"
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self.access_token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "FHKST01010200",
                "custtype": "P",
            }
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code
            }
            
            response = requests.get(url, headers=headers, params=params)
            print(f"📊 호가 조회 ({stock_code}): {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('rt_cd') == '0':
                    output = data.get('output', {})
                    print(f"✅ 매도1호가: {output.get('askp1', 'N/A')}원")
                    print(f"✅ 매수1호가: {output.get('bidp1', 'N/A')}원")
                    return data
                else:
                    print(f"⚠️ API 응답 오류: {data.get('msg1', 'Unknown')}")
            else:
                print(f"❌ API 호출 실패: {response.text}")
                
        except Exception as e:
            print(f"❌ 호가 조회 오류: {str(e)}")
        
        return None

def main():
    """메인 테스트 함수"""
    print("🔴 한투 API 국내주식 기본 테스트 시작")
    print("="*50)
    
    # 테스트 객체 생성
    tester = KISDomesticTest()
    
    # 1. 토큰 발급 테스트
    print("\n1️⃣ 토큰 발급 테스트")
    token_success = tester.get_access_token()
    
    if token_success:
        # 2. 삼성전자 현재가 조회
        print("\n2️⃣ 삼성전자 현재가 조회")
        price_data = tester.test_domestic_stock_price("005930")
        
        # 3. 삼성전자 호가 조회  
        print("\n3️⃣ 삼성전자 호가 조회")
        orderbook_data = tester.test_domestic_stock_orderbook("005930")
        
        # 4. 다른 종목도 테스트 (SK하이닉스)
        print("\n4️⃣ SK하이닉스 현재가 조회")
        sk_data = tester.test_domestic_stock_price("000660")
        
        # 결과 요약
        print("\n" + "="*50)
        print("📊 테스트 결과 요약")
        print(f"🎫 토큰 발급: {'✅' if token_success else '❌'}")
        print(f"📈 현재가 조회: {'✅' if price_data else '❌'}")
        print(f"📊 호가 조회: {'✅' if orderbook_data else '❌'}")
        print(f"🔄 다종목 조회: {'✅' if sk_data else '❌'}")
        
    else:
        print("❌ 토큰 발급 실패로 테스트 중단")

if __name__ == "__main__":
    main() 