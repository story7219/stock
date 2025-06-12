import requests
from utils.logger import log_event

def get_public_ip() -> str | None:
    """
    현재 시스템의 공인 IP 주소를 반환합니다.
    """
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        response.raise_for_status()
        ip = response.json().get("ip")
        return ip
    except requests.RequestException as e:
        log_event("WARNING", f"공인 IP 주소를 가져오는 데 실패했습니다: {e}")
        return None 