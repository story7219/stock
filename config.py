# -*- coding: utf-8 -*-
"""
시스템 설정 파일
"""

import os
from typing import Dict, Any

# API 설정
API_CONFIG = {
    'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
    'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'IEX_CLOUD_API_KEY': os.getenv('IEX_CLOUD_API_KEY'),
    'REQUEST_TIMEOUT': int(os.getenv('REQUEST_TIMEOUT', '30')),
    'MAX_RETRIES': 3
}

# 데이터 수집 설정
DATA_CONFIG = {
    'MAX_CONCURRENT_REQUESTS': int(os.getenv('MAX_CONCURRENT_REQUESTS', '20')),
    'CACHE_TTL': int(os.getenv('CACHE_TTL', '300')),  # 5분
    'DATA_SOURCES': ['yahoo', 'alpha_vantage', 'iex_cloud'],
    'BACKUP_DATA_ENABLED': True
}

# 투자 전략 설정
STRATEGY_CONFIG = {
    'ENABLED_STRATEGIES': [
        'warren_buffett', 'benjamin_graham', 'peter_lynch',
        'john_templeton', 'philip_fisher', 'john_bogle',
        'david_dreman', 'joel_greenblatt', 'martin_zweig',
        'william_oneil', 'jesse_livermore', 'george_soros',
        'ray_dalio', 'carl_icahn', 'john_neff',
        'michael_price', 'mario_gabelli'
    ],
    'STRATEGY_WEIGHTS': {
        'warren_buffett': 15.0,
        'benjamin_graham': 12.0,
        'peter_lynch': 10.0,
        'john_templeton': 8.0,
        'philip_fisher': 8.0,
        'john_bogle': 7.0,
        'david_dreman': 6.0,
        'joel_greenblatt': 6.0,
        'martin_zweig': 5.0,
        'william_oneil': 5.0,
        'jesse_livermore': 4.0,
        'george_soros': 4.0,
        'ray_dalio': 3.0,
        'carl_icahn': 3.0,
        'john_neff': 2.0,
        'michael_price': 1.0,
        'mario_gabelli': 1.0
    }
}

# Gemini AI 설정
GEMINI_CONFIG = {
    'MODEL': 'gemini-1.5-pro',
    'TEMPERATURE': 0.3,
    'TOP_P': 0.8,
    'TOP_K': 40,
    'MAX_OUTPUT_TOKENS': 4096,
    'MAX_CONCURRENT_REQUESTS': 5,
    'CACHE_TTL': 1800  # 30분
}

# 기술적 분석 설정
TECHNICAL_CONFIG = {
    'INDICATORS': [
        'SMA_5', 'SMA_20', 'SMA_60',
        'EMA_12', 'EMA_26',
        'RSI', 'MACD',
        'BB_Upper', 'BB_Lower',
        'Stoch_K', 'Volume_SMA'
    ],
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'BOLLINGER_PERIOD': 20,
    'STOCHASTIC_PERIOD': 14
}

# 시장 설정
MARKET_CONFIG = {
    'MARKETS': ['KOSPI200', 'NASDAQ100', 'SP500'],
    'KOSPI200_SIZE': 200,
    'NASDAQ100_SIZE': 100,
    'SP500_SIZE': 500,
    'TOP_STOCKS_PER_MARKET': 5,
    'OVERALL_TOP_STOCKS': 10
}

# 로깅 설정
LOGGING_CONFIG = {
    'LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
    'FILE': os.getenv('LOG_FILE', 'stock_analysis.log'),
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'DATE_FORMAT': '%Y-%m-%d %H:%M:%S'
}

# 출력 설정
OUTPUT_CONFIG = {
    'RESULT_DIR': 'results',
    'SAVE_JSON': True,
    'SAVE_CSV': True,
    'SAVE_HTML': True,
    'INCLUDE_CHARTS': False  # 차트 생성 비활성화 (선택적)
}

# 알림 설정
NOTIFICATION_CONFIG = {
    'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
    'ENABLE_NOTIFICATIONS': False  # 기본적으로 비활성화
}

# 전체 설정 통합
CONFIG: Dict[str, Any] = {
    'api': API_CONFIG,
    'data': DATA_CONFIG,
    'strategy': STRATEGY_CONFIG,
    'gemini': GEMINI_CONFIG,
    'technical': TECHNICAL_CONFIG,
    'market': MARKET_CONFIG,
    'logging': LOGGING_CONFIG,
    'output': OUTPUT_CONFIG,
    'notification': NOTIFICATION_CONFIG
}


def get_config(section: str = None) -> Dict[str, Any]:
    """
    설정 조회
    
    Args:
        section: 설정 섹션명 (None일 경우 전체 설정 반환)
        
    Returns:
        Dict[str, Any]: 설정 딕셔너리
    """
    if section:
        return CONFIG.get(section, {})
    return CONFIG


def validate_config() -> bool:
    """
    설정 유효성 검사
    
    Returns:
        bool: 설정이 유효한지 여부
    """
    try:
        # 필수 설정 확인
        required_configs = [
            'data.MAX_CONCURRENT_REQUESTS',
            'market.MARKETS',
            'strategy.ENABLED_STRATEGIES'
        ]
        
        for config_path in required_configs:
            sections = config_path.split('.')
            current = CONFIG
            
            for section in sections:
                if section not in current:
                    print(f"필수 설정 누락: {config_path}")
                    return False
                current = current[section]
        
        # 전략 가중치 합계 확인
        total_weight = sum(STRATEGY_CONFIG['STRATEGY_WEIGHTS'].values())
        if abs(total_weight - 100.0) > 0.1:
            print(f"전략 가중치 합계 오류: {total_weight}% (100%이어야 함)")
            return False
        
        return True
        
    except Exception as e:
        print(f"설정 유효성 검사 실패: {e}")
        return False


if __name__ == "__main__":
    # 설정 테스트
    print("=== 시스템 설정 ===")
    
    if validate_config():
        print("✓ 설정 유효성 검사 통과")
        
        print(f"\n활성화된 시장: {MARKET_CONFIG['MARKETS']}")
        print(f"활성화된 전략: {len(STRATEGY_CONFIG['ENABLED_STRATEGIES'])}개")
        print(f"Gemini AI 모델: {GEMINI_CONFIG['MODEL']}")
        print(f"최대 동시 요청: {DATA_CONFIG['MAX_CONCURRENT_REQUESTS']}")
        print(f"결과 저장 위치: {OUTPUT_CONFIG['RESULT_DIR']}")
        
        # API 키 확인
        if API_CONFIG['GEMINI_API_KEY']:
            print("✓ Gemini API 키 설정됨")
        else:
            print("⚠ Gemini API 키 없음 (Mock 모드로 실행됨)")
            
    else:
        print("✗ 설정 유효성 검사 실패") 