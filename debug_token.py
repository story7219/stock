#!/usr/bin/env python3
import os
import asyncio
import aiohttp
import json

async def test_token_issuance():
    """토큰 발급 테스트"""
    app_key = os.getenv('MOCK_KIS_APP_KEY')
    app_secret = os.getenv('MOCK_KIS_APP_SECRET')
    
    print(f"APP_KEY: {app_key}")
    print(f"APP_SECRET: {app_secret}")
    
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    
    headers = {
        "content-type": "application/json"
    }
    
    data = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }
    
    print(f"요청 URL: {url}")
    print(f"요청 헤더: {headers}")
    print(f"요청 데이터: {data}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            print(f"응답 상태: {response.status}")
            response_text = await response.text()
            print(f"응답 내용: {response_text}")
            
            if response.status == 200:
                result = json.loads(response_text)
                print(f"토큰 발급 성공: {result.get('access_token', 'N/A')[:20]}...")
            else:
                print(f"토큰 발급 실패: {response.status}")

if __name__ == "__main__":
    asyncio.run(test_token_issuance()) 