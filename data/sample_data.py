"""
테스트용 샘플 주식 데이터

실제 투자 추천 시스템 테스트를 위한 샘플 데이터를 제공합니다.
"""

import random
import numpy as np
from typing import List, Dict

def generate_mock_stock_data(symbol: str, name: str, sector: str = "Technology") -> Dict:
    """모의 주식 데이터 생성"""
    
    # 기본 정보
    stock = {
        'symbol': symbol,
        'name': name,
        'sector': sector,
        'market_cap': random.uniform(1000, 500000),  # 백만 달러
        'price': random.uniform(10, 500),
        'current_price': random.uniform(10, 500)
    }
    
    # 재무 지표
    stock.update({
        'per': random.uniform(5, 50),
        'pbr': random.uniform(0.5, 10),
        'roe': random.uniform(-20, 40),
        'roa': random.uniform(-10, 20),
        'debt_ratio': random.uniform(10, 80),
        'current_ratio': random.uniform(0.5, 5),
        'revenue_growth_rate': random.uniform(-20, 100),
        'earnings_growth_rate': random.uniform(-30, 80),
        'dividend_yield': random.uniform(0, 8),
        'payout_ratio': random.uniform(0, 100)
    })
    
    # 버핏 전략 관련
    stock.update({
        'competitive_moat': random.uniform(0, 1),
        'management_quality': random.uniform(0, 1),
        'financial_strength': random.uniform(0, 1),
        'growth_prospects': random.uniform(0, 1),
        'valuation_attractiveness': random.uniform(0, 1),
        'brand_strength': random.uniform(0, 1),
        'pricing_power': random.uniform(0, 1),
        'market_leadership': random.uniform(0, 1)
    })
    
    # 린치 전략 관련
    stock.update({
        'peg_ratio': random.uniform(0.2, 3),
        'earnings_predictability': random.uniform(0, 1),
        'industry_growth': random.uniform(0, 1),
        'institutional_ownership': random.uniform(0, 1),
        'insider_buying': random.uniform(0, 1),
        'story_simplicity': random.uniform(0, 1)
    })
    
    # 그레이엄 전략 관련
    stock.update({
        'earnings_stability': random.uniform(0, 1),
        'intrinsic_value': stock['price'] * random.uniform(0.8, 1.5),
        'margin_of_safety': random.uniform(0, 1)
    })
    
    # 피셔 전략 관련
    stock.update({
        'research_development': random.uniform(0, 1),
        'sales_growth_potential': random.uniform(0, 1),
        'profit_margin_trend': random.uniform(0, 1),
        'cost_control': random.uniform(0, 1)
    })
    
    # 기술적 분석 관련
    stock.update({
        'rsi': random.uniform(20, 80),
        'macd_signal': random.choice([-1, 0, 1]),
        'volume_trend': random.uniform(0, 1),
        'price_momentum': random.uniform(-1, 1),
        'support_resistance_level': random.uniform(0, 1),
        'trend_strength': random.uniform(0, 1)
    })
    
    # 매크로 전략 관련
    stock.update({
        'economic_sensitivity': random.uniform(0, 1),
        'currency_exposure': random.uniform(0, 1),
        'interest_rate_sensitivity': random.uniform(0, 1),
        'inflation_hedge': random.uniform(0, 1)
    })
    
    # 혁신/성장 관련
    stock.update({
        'innovation_score': random.uniform(0, 1),
        'market_disruption_potential': random.uniform(0, 1),
        'technology_adoption': random.uniform(0, 1),
        'scalability': random.uniform(0, 1)
    })
    
    # 퀀트 관련
    stock.update({
        'momentum_score': random.uniform(0, 1),
        'quality_score': random.uniform(0, 1),
        'value_score': random.uniform(0, 1),
        'volatility': random.uniform(0.1, 0.8)
    })
    
    # 우드 전략 관련 (파괴적 혁신)
    stock.update({
        'innovation_tech_adoption': random.uniform(0, 1),
        'industry_disruption_power': random.uniform(0, 1),
        'innovation_investment_ratio': random.uniform(0, 1),
        'patent_ip_strength': random.uniform(0, 1),
        'innovation_velocity': random.uniform(0, 1),
        'tech_maturity_position': random.uniform(0, 1),
        'market_adoption_acceleration': random.uniform(0, 1),
        'cost_curve_improvement': random.uniform(0, 1),
        'network_effect_threshold': random.uniform(0, 1),
        'regulatory_tailwind': random.uniform(0, 1),
        'total_addressable_market': random.uniform(1, 1000),
        'global_expansion_potential': random.uniform(0, 1),
        'market_penetration_rate': random.uniform(0, 100),
        'adjacent_market_opportunity': random.uniform(0, 1),
        'partnership_network_strength': random.uniform(0, 1),
        'talent_acquisition_capability': random.uniform(0, 1),
        'innovation_culture_score': random.uniform(0, 1),
        'ecosystem_influence': random.uniform(0, 1)
    })
    
    # 소로스 전략 관련 (반사성 이론)
    stock.update({
        'market_reflexivity': random.uniform(0, 1),
        'macro_imbalance': random.uniform(0, 1),
        'currency_dynamics': random.uniform(0, 1),
        'crisis_opportunity': random.uniform(0, 1),
        'extreme_psychology': random.uniform(0, 1),
        'regime_change_signal': random.uniform(0, 1),
        'policy_asymmetry': random.uniform(0, 1),
        'market_structure_flaw': random.uniform(0, 1),
        'feedback_loop_strength': random.uniform(0, 1),
        'crowd_behavior_extreme': random.uniform(0, 1)
    })
    
    return stock

def get_sample_stocks() -> List[Dict]:
    """샘플 주식 데이터 반환"""
    
    # 대형 기술주
    stocks = [
        generate_mock_stock_data("AAPL", "Apple Inc.", "Technology"),
        generate_mock_stock_data("MSFT", "Microsoft Corp.", "Technology"),
        generate_mock_stock_data("GOOGL", "Alphabet Inc.", "Technology"),
        generate_mock_stock_data("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
        generate_mock_stock_data("TSLA", "Tesla Inc.", "Consumer Discretionary"),
        
        # 금융주
        generate_mock_stock_data("BRK.B", "Berkshire Hathaway", "Financial Services"),
        generate_mock_stock_data("JPM", "JPMorgan Chase", "Financial Services"),
        generate_mock_stock_data("BAC", "Bank of America", "Financial Services"),
        
        # 헬스케어
        generate_mock_stock_data("JNJ", "Johnson & Johnson", "Healthcare"),
        generate_mock_stock_data("PFE", "Pfizer Inc.", "Healthcare"),
        generate_mock_stock_data("UNH", "UnitedHealth Group", "Healthcare"),
        
        # 소비재
        generate_mock_stock_data("KO", "Coca-Cola Company", "Consumer Staples"),
        generate_mock_stock_data("PG", "Procter & Gamble", "Consumer Staples"),
        generate_mock_stock_data("WMT", "Walmart Inc.", "Consumer Staples"),
        
        # 산업재
        generate_mock_stock_data("CAT", "Caterpillar Inc.", "Industrials"),
        generate_mock_stock_data("BA", "Boeing Company", "Industrials"),
        
        # 에너지
        generate_mock_stock_data("XOM", "Exxon Mobil Corp.", "Energy"),
        generate_mock_stock_data("CVX", "Chevron Corp.", "Energy"),
        
        # 유틸리티
        generate_mock_stock_data("NEE", "NextEra Energy", "Utilities"),
        
        # 부동산
        generate_mock_stock_data("AMT", "American Tower Corp.", "Real Estate")
    ]
    
    # 일부 주식에 특별한 특성 부여
    
    # Apple - 버핏이 좋아할 만한 특성
    stocks[0].update({
        'competitive_moat': 0.9,
        'brand_strength': 0.95,
        'pricing_power': 0.85,
        'financial_strength': 0.9,
        'per': 25,
        'pbr': 5.2,
        'roe': 28
    })
    
    # Tesla - 우드가 좋아할 만한 혁신 특성
    stocks[4].update({
        'innovation_tech_adoption': 0.95,
        'industry_disruption_power': 0.9,
        'innovation_velocity': 0.85,
        'total_addressable_market': 500,
        'market_penetration_rate': 5,
        'per': 45,
        'revenue_growth_rate': 80
    })
    
    # Berkshire - 그레이엄/버핏 스타일
    stocks[5].update({
        'per': 12,
        'pbr': 1.2,
        'debt_ratio': 25,
        'current_ratio': 2.5,
        'earnings_stability': 0.9,
        'competitive_moat': 0.85
    })
    
    # Johnson & Johnson - 안정적 배당주
    stocks[8].update({
        'dividend_yield': 2.8,
        'payout_ratio': 55,
        'earnings_stability': 0.9,
        'debt_ratio': 35,
        'per': 16
    })
    
    return stocks

def get_test_portfolio() -> List[Dict]:
    """테스트용 소규모 포트폴리오"""
    return get_sample_stocks()[:10] 