"""
🔧 도우미 함수들
"""

def format_currency(amount: float) -> str:
    """통화 포맷팅"""
    return f"{amount:,.0f}원"

def calculate_profit_rate(entry_price: float, current_price: float) -> float:
    """수익률 계산"""
    if entry_price <= 0:
        return 0.0
    return (current_price - entry_price) / entry_price * 100 