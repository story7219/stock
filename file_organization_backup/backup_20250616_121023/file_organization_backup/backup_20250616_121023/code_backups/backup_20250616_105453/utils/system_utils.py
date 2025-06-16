"""
시스템 유틸리티
"""
import requests
from utils.logger import log_event

def get_public_ip():
    """공인 IP 주소 조회"""
    try:
        response = requests.get('https://httpbin.org/ip', timeout=10)
        return response.json().get('origin')
    except:
        return None 