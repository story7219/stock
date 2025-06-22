"""
주식 분석용 샘플 데이터 생성기
한국(20개) + 미국(20개) = 총 40개 종목
PBR, EPS 증가율, EBITDA, 거래대금 지표 포함 (PER 제외)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def create_enhanced_stock_data():
    """향상된 주식 데이터 생성 (PBR, EPS증가율, EBITDA, 거래대금 포함)"""
    
    # 한국 주식 20개 (실제 종목명)
    korean_stocks = [
        {'Ticker': '005930', 'Name': '삼성전자', 'Sector': '반도체'},
        {'Ticker': '000660', 'Name': 'SK하이닉스', 'Sector': '반도체'},
        {'Ticker': '035420', 'Name': 'NAVER', 'Sector': '인터넷'},
        {'Ticker': '035720', 'Name': '카카오', 'Sector': '인터넷'},
        {'Ticker': '005490', 'Name': 'POSCO홀딩스', 'Sector': '철강'},
        {'Ticker': '006400', 'Name': '삼성SDI', 'Sector': '배터리'},
        {'Ticker': '051910', 'Name': 'LG화학', 'Sector': '화학'},
        {'Ticker': '373220', 'Name': 'LG에너지솔루션', 'Sector': '배터리'},
        {'Ticker': '207940', 'Name': '삼성바이오로직스', 'Sector': '바이오'},
        {'Ticker': '105560', 'Name': 'KB금융', 'Sector': '금융'},
        {'Ticker': '055550', 'Name': '신한지주', 'Sector': '금융'},
        {'Ticker': '086790', 'Name': '하나금융지주', 'Sector': '금융'},
        {'Ticker': '352820', 'Name': '하이브', 'Sector': '엔터테인먼트'},
        {'Ticker': '018260', 'Name': '삼성에스디에스', 'Sector': 'IT서비스'},
        {'Ticker': '028260', 'Name': '삼성물산', 'Sector': '건설'},
        {'Ticker': '009150', 'Name': '삼성전기', 'Sector': '전자부품'},
        {'Ticker': '012330', 'Name': '현대모비스', 'Sector': '자동차부품'},
        {'Ticker': '066570', 'Name': 'LG전자', 'Sector': '가전'},
        {'Ticker': '003550', 'Name': 'LG', 'Sector': '지주회사'},
        {'Ticker': '017670', 'Name': 'SK텔레콤', 'Sector': '통신'}
    ]
    
    # 미국 주식 20개 (실제 종목명)
    us_stocks = [
        {'Ticker': 'AAPL', 'Name': 'Apple Inc', 'Sector': 'Technology'},
        {'Ticker': 'MSFT', 'Name': 'Microsoft Corp', 'Sector': 'Technology'},
        {'Ticker': 'GOOGL', 'Name': 'Alphabet Inc', 'Sector': 'Technology'},
        {'Ticker': 'AMZN', 'Name': 'Amazon.com Inc', 'Sector': 'Consumer Discretionary'},
        {'Ticker': 'TSLA', 'Name': 'Tesla Inc', 'Sector': 'Automotive'},
        {'Ticker': 'META', 'Name': 'Meta Platforms', 'Sector': 'Technology'},
        {'Ticker': 'NVDA', 'Name': 'NVIDIA Corp', 'Sector': 'Technology'},
        {'Ticker': 'NFLX', 'Name': 'Netflix Inc', 'Sector': 'Media'},
        {'Ticker': 'CRM', 'Name': 'Salesforce Inc', 'Sector': 'Technology'},
        {'Ticker': 'INTC', 'Name': 'Intel Corp', 'Sector': 'Technology'},
        {'Ticker': 'AMD', 'Name': 'Advanced Micro Devices', 'Sector': 'Technology'},
        {'Ticker': 'ORCL', 'Name': 'Oracle Corp', 'Sector': 'Technology'},
        {'Ticker': 'ADBE', 'Name': 'Adobe Inc', 'Sector': 'Technology'},
        {'Ticker': 'PYPL', 'Name': 'PayPal Holdings', 'Sector': 'Financial Services'},
        {'Ticker': 'SNOW', 'Name': 'Snowflake Inc', 'Sector': 'Technology'},
        {'Ticker': 'RBLX', 'Name': 'Roblox Corp', 'Sector': 'Gaming'},
        {'Ticker': 'UBER', 'Name': 'Uber Technologies', 'Sector': 'Transportation'},
        {'Ticker': 'ABNB', 'Name': 'Airbnb Inc', 'Sector': 'Travel'},
        {'Ticker': 'SHOP', 'Name': 'Shopify Inc', 'Sector': 'E-commerce'},
        {'Ticker': 'ZM', 'Name': 'Zoom Video', 'Sector': 'Technology'}
    ]
    
    all_data = []
    
    # 한국 주식 데이터 생성
    for stock in korean_stocks:
        # 기본 가격 데이터
        base_price = random.uniform(30000, 500000)  # 3만원 ~ 50만원
        
        # 핵심 밸류에이션 지표 (PER 제외)
        pbr = random.uniform(0.3, 4.0)  # PBR: 0.3 ~ 4.0
        eps_growth = random.uniform(-40, 80)  # EPS 증가율: -40% ~ 80%
        roe = random.uniform(3, 35)  # ROE: 3% ~ 35%
        ebitda = random.uniform(200, 80000)  # EBITDA (억원): 200억 ~ 8조
        
        # 거래대금 (억원) - 유동성 지표
        trading_value = random.uniform(50, 5000)  # 거래대금: 50억 ~ 5000억
        
        # 수익률 데이터
        returns_3m = random.uniform(-0.35, 0.6)  # 3개월 수익률: -35% ~ 60%
        returns_6m = random.uniform(-0.45, 0.8)  # 6개월 수익률: -45% ~ 80%
        returns_1y = random.uniform(-0.6, 1.5)  # 1년 수익률: -60% ~ 150%
        
        # 리스크 지표
        volatility = random.uniform(0.12, 0.9)  # 변동성: 12% ~ 90%
        beta = random.uniform(0.4, 2.2)  # 베타: 0.4 ~ 2.2
        
        # 시가총액 (조원 단위)
        market_cap = random.uniform(0.5e12, 600e12)  # 0.5조 ~ 600조
        
        # 거래량
        volume = random.uniform(50000, 15000000)  # 5만주 ~ 1500만주
        
        # 재무 건전성
        debt_ratio = random.uniform(5, 85)  # 부채비율: 5% ~ 85%
        current_ratio = random.uniform(70, 350)  # 유동비율: 70% ~ 350%
        
        # 배당 정보
        dividend_yield = random.uniform(0, 6)  # 배당수익률: 0% ~ 6%
        
        # 추가 지표들 (새로운 투자 대가용)
        price_momentum = random.uniform(-0.3, 0.5)  # 가격 모멘텀: -30% ~ 50%
        revenue_growth = random.uniform(-15, 40)  # 매출 성장률: -15% ~ 40%
        profit_growth = random.uniform(-25, 60)  # 이익 성장률: -25% ~ 60%
        rsi = random.uniform(20, 80)  # RSI: 20 ~ 80
        price_strength = random.uniform(0.2, 0.9)  # 가격 강도: 0.2 ~ 0.9
        
        stock_data = {
            'Ticker': stock['Ticker'],
            'Name': stock['Name'],
            'Market': 'KR',
            'Sector': stock['Sector'],
            'Close': base_price,
            'PBR': pbr,
            'EPS_Growth': eps_growth,
            'ROE': roe,
            'EBITDA': ebitda,
            'TradingValue': trading_value,  # 거래대금 추가
            '3M_Return': returns_3m,
            '6M_Return': returns_6m,
            '1Y_Return': returns_1y,
            'Volatility': volatility,
            'Beta': beta,
            'MarketCap': market_cap,
            'Volume': volume,
            'DebtRatio': debt_ratio,
            'CurrentRatio': current_ratio,
            'DividendYield': dividend_yield,
            'PriceMomentum': price_momentum,  # 가격 모멘텀
            'RevenueGrowth': revenue_growth,  # 매출 성장률
            'ProfitGrowth': profit_growth,  # 이익 성장률
            'RSI': rsi,  # RSI
            'PriceStrength': price_strength,  # 가격 강도
            'Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        all_data.append(stock_data)
    
    # 미국 주식 데이터 생성
    for stock in us_stocks:
        # 기본 가격 데이터 (달러)
        base_price = random.uniform(15, 1000)  # $15 ~ $1000
        
        # 핵심 밸류에이션 지표 (PER 제외)
        pbr = random.uniform(0.8, 20.0)  # PBR: 0.8 ~ 20.0
        eps_growth = random.uniform(-30, 150)  # EPS 증가율: -30% ~ 150%
        roe = random.uniform(5, 45)  # ROE: 5% ~ 45%
        ebitda = random.uniform(50, 150000)  # EBITDA (백만달러): 5천만 ~ 1500억
        
        # 거래대금 (백만달러) - 유동성 지표
        trading_value = random.uniform(100, 50000)  # 거래대금: 1억 ~ 500억 달러
        
        # 수익률 데이터
        returns_3m = random.uniform(-0.5, 1.0)  # 3개월 수익률: -50% ~ 100%
        returns_6m = random.uniform(-0.6, 1.5)  # 6개월 수익률: -60% ~ 150%
        returns_1y = random.uniform(-0.8, 3.0)  # 1년 수익률: -80% ~ 300%
        
        # 리스크 지표
        volatility = random.uniform(0.15, 1.5)  # 변동성: 15% ~ 150%
        beta = random.uniform(0.2, 3.0)  # 베타: 0.2 ~ 3.0
        
        # 시가총액 (달러 단위)
        market_cap = random.uniform(0.5e9, 4000e9)  # 5억 ~ 4조 달러
        
        # 거래량
        volume = random.uniform(500000, 200000000)  # 50만주 ~ 2억주
        
        # 재무 건전성
        debt_ratio = random.uniform(2, 70)  # 부채비율: 2% ~ 70%
        current_ratio = random.uniform(90, 500)  # 유동비율: 90% ~ 500%
        
        # 배당 정보
        dividend_yield = random.uniform(0, 5)  # 배당수익률: 0% ~ 5%
        
        # 추가 지표들 (새로운 투자 대가용)
        price_momentum = random.uniform(-0.3, 0.5)  # 가격 모멘텀: -30% ~ 50%
        revenue_growth = random.uniform(-15, 40)  # 매출 성장률: -15% ~ 40%
        profit_growth = random.uniform(-25, 60)  # 이익 성장률: -25% ~ 60%
        rsi = random.uniform(20, 80)  # RSI: 20 ~ 80
        price_strength = random.uniform(0.2, 0.9)  # 가격 강도: 0.2 ~ 0.9
        
        stock_data = {
            'Ticker': stock['Ticker'],
            'Name': stock['Name'],
            'Market': 'US',
            'Sector': stock['Sector'],
            'Close': base_price,
            'PBR': pbr,
            'EPS_Growth': eps_growth,
            'ROE': roe,
            'EBITDA': ebitda,
            'TradingValue': trading_value,  # 거래대금 추가
            '3M_Return': returns_3m,
            '6M_Return': returns_6m,
            '1Y_Return': returns_1y,
            'Volatility': volatility,
            'Beta': beta,
            'MarketCap': market_cap,
            'Volume': volume,
            'DebtRatio': debt_ratio,
            'CurrentRatio': current_ratio,
            'DividendYield': dividend_yield,
            'PriceMomentum': price_momentum,  # 가격 모멘텀
            'RevenueGrowth': revenue_growth,  # 매출 성장률
            'ProfitGrowth': profit_growth,  # 이익 성장률
            'RSI': rsi,  # RSI
            'PriceStrength': price_strength,  # 가격 강도
            'Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        all_data.append(stock_data)
    
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    # 데이터 생성
    print("📊 향상된 주식 데이터 생성 중 (PBR + EPS증가율 + EBITDA + 거래대금)...")
    data = create_enhanced_stock_data()
    
    # 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    
    # CSV 저장
    data.to_csv('data/stock_data.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ 데이터 생성 완료!")
    print(f"📁 저장 위치: data/stock_data.csv")
    print(f"📊 총 종목 수: {len(data)}개")
    print(f"🇰🇷 한국 종목: {len(data[data['Market']=='KR'])}개")
    print(f"🇺🇸 미국 종목: {len(data[data['Market']=='US'])}개")
    
    # 거래대금 순위 미리보기
    print(f"\n💰 거래대금 상위 5개 종목:")
    top_trading = data.nlargest(5, 'TradingValue')[['Ticker', 'Name', 'Market', 'TradingValue']]
    for _, row in top_trading.iterrows():
        unit = '억원' if row['Market'] == 'KR' else '백만달러'
        print(f"  • {row['Ticker']} ({row['Name'][:10]}): {row['TradingValue']:,.0f} {unit}")
    
    # 데이터 컬럼 정보
    print(f"\n📋 주요 데이터 컬럼:")
    key_columns = ['Ticker', 'Name', 'Market', 'Close', 'PBR', 'EPS_Growth', 'ROE', 'EBITDA', 'TradingValue']
    for col in key_columns:
        print(f"  • {col}")
    
    print(f"\n🔍 샘플 데이터 (상위 3개):")
    print(data.head(3)[key_columns].to_string(index=False)) 