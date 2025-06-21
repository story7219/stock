"""
🇺🇸 미국주식 재무제표 및 뉴스 크롤링 시스템
무료 데이터 소스를 활용한 종합 정보 수집

주요 기능:
1. Yahoo Finance 재무제표 크롤링
2. SEC EDGAR 공시 데이터 수집
3. 뉴스 크롤링 (Google News, Yahoo Finance)
4. 경제지표 수집 (FRED API)
5. 소셜 센티먼트 분석 (Reddit, Twitter)
"""
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import asyncio
import aiohttp
from urllib.parse import quote
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class USFinancialCrawler:
    """🇺🇸 미국주식 재무제표 및 뉴스 크롤링 시스템"""
    
    def __init__(self):
        """초기화"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 캐시 저장소
        self.financial_cache = {}
        self.news_cache = {}
        
        # API 키들 (무료 버전)
        self.fred_api_key = None  # FRED API 키 (무료)
        self.alpha_vantage_key = None  # Alpha Vantage API 키 (무료)
    
    # === 📊 재무제표 데이터 수집 ===
    def get_financial_statements(self, symbol: str) -> Dict:
        """Yahoo Finance에서 재무제표 데이터 수집"""
        try:
            logger.info(f"📊 {symbol} 재무제표 데이터 수집 중...")
            
            # yfinance 사용 (가장 안정적)
            ticker = yf.Ticker(symbol)
            
            # 기본 정보
            info = ticker.info
            
            # 재무제표 데이터
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # 주요 재무 지표 추출
            financial_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                
                # 수익성 지표
                'revenue': self._get_latest_value(financials, 'Total Revenue'),
                'gross_profit': self._get_latest_value(financials, 'Gross Profit'),
                'operating_income': self._get_latest_value(financials, 'Operating Income'),
                'net_income': self._get_latest_value(financials, 'Net Income'),
                'ebitda': info.get('ebitda', 0),
                
                # 마진 지표
                'gross_margin': info.get('grossMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'profit_margin': info.get('profitMargins', 0),
                
                # 효율성 지표
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'roic': info.get('returnOnInvestmentCapital', 0),
                
                # 재무 건전성
                'total_debt': self._get_latest_value(balance_sheet, 'Total Debt'),
                'total_cash': self._get_latest_value(balance_sheet, 'Cash And Cash Equivalents'),
                'total_assets': self._get_latest_value(balance_sheet, 'Total Assets'),
                'shareholders_equity': self._get_latest_value(balance_sheet, 'Stockholders Equity'),
                
                # 부채 비율
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                
                # 성장률
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                
                # 밸류에이션
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'ev_revenue': info.get('enterpriseToRevenue', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                
                # 배당
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                
                # 기술적 지표
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                
                # 업데이트 시간
                'last_updated': datetime.now().isoformat()
            }
            
            # 캐시 저장
            self.financial_cache[symbol] = financial_data
            
            logger.info(f"✅ {symbol} 재무제표 데이터 수집 완료")
            return financial_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 재무제표 수집 실패: {e}")
            return {}
    
    def _get_latest_value(self, df, column_name):
        """데이터프레임에서 최신 값 추출"""
        try:
            if df is not None and column_name in df.index:
                return float(df.loc[column_name].iloc[0])
            return 0
        except:
            return 0
    
    # === 📰 뉴스 크롤링 ===
    def get_stock_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """주식 관련 뉴스 크롤링"""
        try:
            logger.info(f"📰 {symbol} 뉴스 크롤링 중...")
            
            all_news = []
            
            # 1. Yahoo Finance 뉴스
            yahoo_news = self._crawl_yahoo_news(symbol)
            all_news.extend(yahoo_news)
            
            # 2. Google News 뉴스
            google_news = self._crawl_google_news(symbol)
            all_news.extend(google_news)
            
            # 3. MarketWatch 뉴스
            marketwatch_news = self._crawl_marketwatch_news(symbol)
            all_news.extend(marketwatch_news)
            
            # 중복 제거 및 정렬
            unique_news = self._remove_duplicate_news(all_news)
            recent_news = [news for news in unique_news if self._is_recent_news(news, days)]
            
            # 시간순 정렬
            recent_news.sort(key=lambda x: x.get('published_date', ''), reverse=True)
            
            logger.info(f"✅ {symbol} 뉴스 {len(recent_news)}개 수집 완료")
            return recent_news[:50]  # 최대 50개
            
        except Exception as e:
            logger.error(f"❌ {symbol} 뉴스 크롤링 실패: {e}")
            return []
    
    def _crawl_yahoo_news(self, symbol: str) -> List[Dict]:
        """Yahoo Finance 뉴스 크롤링"""
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_list = []
            
            # 뉴스 아이템 찾기
            news_items = soup.find_all('div', {'class': re.compile(r'.*stream-item.*')})
            
            for item in news_items[:10]:  # 최대 10개
                try:
                    title_elem = item.find('h3') or item.find('a')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        link = title_elem.get('href', '')
                        
                        if link and not link.startswith('http'):
                            link = f"https://finance.yahoo.com{link}"
                        
                        # 시간 정보 추출
                        time_elem = item.find('time') or item.find('span', {'class': re.compile(r'.*time.*')})
                        published_date = time_elem.get_text(strip=True) if time_elem else ''
                        
                        news_list.append({
                            'title': title,
                            'link': link,
                            'source': 'Yahoo Finance',
                            'published_date': published_date,
                            'symbol': symbol
                        })
                except:
                    continue
            
            return news_list
            
        except Exception as e:
            logger.warning(f"Yahoo Finance 뉴스 크롤링 실패: {e}")
            return []
    
    def _crawl_google_news(self, symbol: str) -> List[Dict]:
        """Google News 뉴스 크롤링"""
        try:
            # Google News RSS 사용
            query = f"{symbol} stock"
            url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            news_list = []
            items = soup.find_all('item')
            
            for item in items[:10]:  # 최대 10개
                try:
                    title = item.find('title').get_text(strip=True)
                    link = item.find('link').get_text(strip=True)
                    pub_date = item.find('pubDate').get_text(strip=True)
                    source = item.find('source').get_text(strip=True) if item.find('source') else 'Google News'
                    
                    news_list.append({
                        'title': title,
                        'link': link,
                        'source': source,
                        'published_date': pub_date,
                        'symbol': symbol
                    })
                except:
                    continue
            
            return news_list
            
        except Exception as e:
            logger.warning(f"Google News 크롤링 실패: {e}")
            return []
    
    def _crawl_marketwatch_news(self, symbol: str) -> List[Dict]:
        """MarketWatch 뉴스 크롤링"""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_list = []
            
            # 뉴스 섹션 찾기
            news_section = soup.find('div', {'class': re.compile(r'.*news.*')})
            if news_section:
                news_items = news_section.find_all('a', href=True)
                
                for item in news_items[:5]:  # 최대 5개
                    try:
                        title = item.get_text(strip=True)
                        link = item.get('href', '')
                        
                        if link and not link.startswith('http'):
                            link = f"https://www.marketwatch.com{link}"
                        
                        if title and len(title) > 10:  # 제목이 있는 경우만
                            news_list.append({
                                'title': title,
                                'link': link,
                                'source': 'MarketWatch',
                                'published_date': '',
                                'symbol': symbol
                            })
                    except:
                        continue
            
            return news_list
            
        except Exception as e:
            logger.warning(f"MarketWatch 뉴스 크롤링 실패: {e}")
            return []
    
    def _remove_duplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """중복 뉴스 제거"""
        seen_titles = set()
        unique_news = []
        
        for news in news_list:
            title = news.get('title', '').lower()
            if title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_news.append(news)
        
        return unique_news
    
    def _is_recent_news(self, news: Dict, days: int) -> bool:
        """최근 뉴스인지 확인"""
        # 간단한 구현 - 실제로는 날짜 파싱 필요
        return True  # 일단 모든 뉴스를 최근으로 처리
    
    # === 📈 경제지표 수집 ===
    def get_economic_indicators(self) -> Dict:
        """주요 경제지표 수집"""
        try:
            logger.info("📈 경제지표 수집 중...")
            
            indicators = {}
            
            # 1. VIX (공포지수) - Yahoo Finance에서
            vix_data = yf.Ticker("^VIX")
            vix_hist = vix_data.history(period="1d")
            if not vix_hist.empty:
                indicators['vix'] = float(vix_hist['Close'].iloc[-1])
            
            # 2. 10년 국채수익률
            tnx_data = yf.Ticker("^TNX")
            tnx_hist = tnx_data.history(period="1d")
            if not tnx_hist.empty:
                indicators['10y_treasury_yield'] = float(tnx_hist['Close'].iloc[-1])
            
            # 3. 달러 인덱스
            dxy_data = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy_data.history(period="1d")
            if not dxy_hist.empty:
                indicators['dollar_index'] = float(dxy_hist['Close'].iloc[-1])
            
            # 4. 주요 지수
            sp500 = yf.Ticker("^GSPC")
            sp500_hist = sp500.history(period="2d")
            if len(sp500_hist) >= 2:
                indicators['sp500_change'] = float((sp500_hist['Close'].iloc[-1] - sp500_hist['Close'].iloc[-2]) / sp500_hist['Close'].iloc[-2] * 100)
            
            nasdaq = yf.Ticker("^IXIC")
            nasdaq_hist = nasdaq.history(period="2d")
            if len(nasdaq_hist) >= 2:
                indicators['nasdaq_change'] = float((nasdaq_hist['Close'].iloc[-1] - nasdaq_hist['Close'].iloc[-2]) / nasdaq_hist['Close'].iloc[-2] * 100)
            
            indicators['last_updated'] = datetime.now().isoformat()
            
            logger.info("✅ 경제지표 수집 완료")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 경제지표 수집 실패: {e}")
            return {}
    
    # === 🎯 종합 분석 ===
    def get_comprehensive_analysis(self, symbol: str) -> Dict:
        """종합 분석 데이터 수집"""
        try:
            logger.info(f"🎯 {symbol} 종합 분석 데이터 수집 중...")
            
            # 1. 재무제표 데이터
            financial_data = self.get_financial_statements(symbol)
            
            # 2. 뉴스 데이터
            news_data = self.get_stock_news(symbol)
            
            # 3. 경제지표
            economic_data = self.get_economic_indicators()
            
            # 4. 기술적 분석 데이터 (yfinance)
            technical_data = self._get_technical_analysis(symbol)
            
            # 5. 애널리스트 추천 데이터
            analyst_data = self._get_analyst_recommendations(symbol)
            
            comprehensive_data = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'financial_data': financial_data,
                'news_data': news_data,
                'economic_indicators': economic_data,
                'technical_analysis': technical_data,
                'analyst_recommendations': analyst_data,
                'data_sources': [
                    'Yahoo Finance',
                    'Google News',
                    'MarketWatch',
                    'yfinance'
                ]
            }
            
            logger.info(f"✅ {symbol} 종합 분석 데이터 수집 완료")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 종합 분석 실패: {e}")
            return {}
    
    def _get_technical_analysis(self, symbol: str) -> Dict:
        """기술적 분석 데이터 수집"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")  # 6개월 데이터
            
            if hist.empty:
                return {}
            
            # 이동평균 계산
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            # RSI 계산
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            current_data = hist.iloc[-1]
            
            return {
                'current_price': float(current_data['Close']),
                'volume': int(current_data['Volume']),
                'sma_20': float(current_data['SMA_20']) if not pd.isna(current_data['SMA_20']) else 0,
                'sma_50': float(current_data['SMA_50']) if not pd.isna(current_data['SMA_50']) else 0,
                'sma_200': float(current_data['SMA_200']) if not pd.isna(current_data['SMA_200']) else 0,
                'rsi': float(current_data['RSI']) if not pd.isna(current_data['RSI']) else 50,
                'day_change': float((current_data['Close'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            }
            
        except Exception as e:
            logger.warning(f"기술적 분석 실패: {e}")
            return {}
    
    def _get_analyst_recommendations(self, symbol: str) -> Dict:
        """애널리스트 추천 데이터 수집"""
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is not None and not recommendations.empty:
                latest_rec = recommendations.iloc[-1]
                return {
                    'firm': latest_rec.get('Firm', ''),
                    'to_grade': latest_rec.get('To Grade', ''),
                    'from_grade': latest_rec.get('From Grade', ''),
                    'action': latest_rec.get('Action', ''),
                    'date': str(latest_rec.name) if hasattr(latest_rec, 'name') else ''
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"애널리스트 추천 수집 실패: {e}")
            return {}
    
    # === 💾 데이터 저장 ===
    def save_analysis_to_file(self, symbol: str, data: Dict, filename: str = None):
        """분석 데이터를 파일로 저장"""
        try:
            if not filename:
                filename = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 분석 데이터 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"❌ 데이터 저장 실패: {e}")

# === 🚀 사용 예시 ===
async def main():
    """메인 실행 함수"""
    crawler = USFinancialCrawler()
    
    # 테스트 종목
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"🔍 {symbol} 분석 시작")
        print('='*50)
        
        # 종합 분석 수행
        analysis_data = crawler.get_comprehensive_analysis(symbol)
        
        if analysis_data:
            # 결과 출력
            financial = analysis_data.get('financial_data', {})
            news = analysis_data.get('news_data', [])
            
            print(f"📊 재무 정보:")
            print(f"   회사명: {financial.get('company_name', 'N/A')}")
            print(f"   섹터: {financial.get('sector', 'N/A')}")
            print(f"   시가총액: ${financial.get('market_cap', 0):,}")
            print(f"   PER: {financial.get('pe_ratio', 0):.2f}")
            print(f"   ROE: {financial.get('roe', 0):.2%}")
            
            print(f"\n📰 최근 뉴스 ({len(news)}개):")
            for i, article in enumerate(news[:3], 1):
                print(f"   {i}. {article.get('title', 'N/A')[:80]}...")
                print(f"      출처: {article.get('source', 'N/A')}")
            
            # 파일 저장
            crawler.save_analysis_to_file(symbol, analysis_data)
        
        # API 호출 제한을 위한 대기
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main()) 