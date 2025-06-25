"""
실시간 뉴스 데이터 수집기
- 다양한 무료 뉴스 소스에서 실시간 금융 뉴스 수집
- RSS 피드, 웹 스크래핑, 무료 API 활용
- 뉴스 감정 분석 및 주식 관련성 분석
"""

import asyncio
import aiohttp
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import time
import json
from urllib.parse import urljoin, urlparse
import hashlib

@dataclass
class NewsItem:
    """뉴스 아이템 데이터 클래스"""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    sentiment_score: float
    relevance_score: float
    keywords: List[str]
    stock_symbols: List[str]
    category: str
    language: str = 'ko'

class NewsCollector:
    """실시간 뉴스 수집기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 뉴스 소스 설정
        self.news_sources = {
            'korean': {
                'naver_finance': 'https://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258',
                'hankyung': 'https://www.hankyung.com/feed/economy',
                'mk': 'https://www.mk.co.kr/rss/30000042/',
                'edaily': 'https://www.edaily.co.kr/rss/rss_economy.xml',
                'mt': 'https://www.mt.co.kr/rss/mt_economy.xml',
                'seoul_finance': 'https://www.sedaily.com/RSS/S11.xml',
                'fn_news': 'https://www.fnnews.com/rss/fn_realestate_stock.xml'
            },
            'global': {
                'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
                'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
                'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069',
                'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
                'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
                'investing_com': 'https://kr.investing.com/rss/news.rss'
            }
        }
        
        # 캐시 설정
        self.news_cache = {}
        self.cache_duration = 300  # 5분
        
    def get_cached_news(self, source_key: str) -> Optional[List[NewsItem]]:
        """캐시된 뉴스 반환"""
        if source_key in self.news_cache:
            cached_time, news_items = self.news_cache[source_key]
            if time.time() - cached_time < self.cache_duration:
                return news_items
        return None
    
    def cache_news(self, source_key: str, news_items: List[NewsItem]):
        """뉴스 캐시 저장"""
        self.news_cache[source_key] = (time.time(), news_items)    
    def extract_stock_symbols(self, text: str) -> List[str]:
        """텍스트에서 주식 심볼 추출"""
        symbols = []
        
        # 한국 주식 패턴
        korean_patterns = [
            r'(\d{6})',  # 6자리 종목코드
            r'([가-힣]+)\s*\((\d{6})\)',  # 회사명(종목코드)
        ]
        
        # 미국 주식 패턴
        us_patterns = [
            r'\b([A-Z]{1,5})\b',  # 1-5자리 대문자 심볼
        ]
        
        for pattern in korean_patterns + us_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    symbols.extend(match)
                else:
                    symbols.append(match)
        
        # 유명 주식 심볼 필터링
        famous_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'IBM', 'CSCO'
        ]
        
        filtered_symbols = []
        for symbol in symbols:
            if symbol in famous_symbols or (symbol.isdigit() and len(symbol) == 6):
                filtered_symbols.append(symbol)
        
        return list(set(filtered_symbols))[:5]  # 중복 제거 후 최대 5개
    
    async def fetch_web_news(self, url: str, source_name: str) -> List[NewsItem]:
        """웹 스크래핑으로 뉴스 수집"""
        try:
            # 캐시 확인
            cached_news = self.get_cached_news(f"web_{source_name}")
            if cached_news:
                return cached_news
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            news_items = []
            
            # 네이버 금융 뉴스 파싱
            if 'naver' in source_name:
                articles = soup.find_all('tr', {'onmouseover': True})[:15]
                for article in articles:
                    try:
                        title_elem = article.find('a', class_='tit')
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text().strip()
                        link = urljoin(url, title_elem.get('href'))
                        
                        # 상세 내용 가져오기
                        content = await self.fetch_article_content(link)
                        
                        keywords = self.extract_keywords(title + " " + content)
                        stock_symbols = self.extract_stock_symbols(title + " " + content)
                        
                        news_item = NewsItem(
                            title=title,
                            content=content,
                            url=link,
                            source=source_name,
                            published_date=datetime.now(),
                            sentiment_score=0.0,
                            relevance_score=0.0,
                            keywords=keywords,
                            stock_symbols=stock_symbols,
                            category='finance',
                            language='ko'
                        )
                        
                        news_items.append(news_item)
                        
                    except Exception as e:
                        self.logger.warning(f"네이버 뉴스 파싱 오류: {e}")
                        continue
            
            # 캐시 저장
            self.cache_news(f"web_{source_name}", news_items)
            return news_items
            
        except Exception as e:
            self.logger.error(f"웹 뉴스 수집 오류 ({source_name}): {e}")
            return []
    def get_top_keywords(self, news_items: List[NewsItem], top_n: int = 10) -> List[Tuple[str, int]]:
        """가장 많이 언급된 키워드 반환"""
        all_keywords = [keyword for item in news_items for keyword in item.keywords]
        from collections import Counter
        return Counter(all_keywords).most_common(top_n)

    def get_mentioned_symbols(self, news_items: List[NewsItem], top_n: int = 10) -> List[Tuple[str, int]]:
        """가장 많이 언급된 주식 심볼 반환"""
        all_symbols = [symbol for item in news_items for symbol in item.stock_symbols]
        from collections import Counter
        return Counter(all_symbols).most_common(top_n)

# 비동기 실행을 위한 헬퍼 함수
async def main():
    collector = NewsCollector()
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("실시간 뉴스 수집 시작...")
    # 특정 종목 관련 뉴스 필터링 예시
    # target_symbols = ['AAPL', 'TSLA', '005930'] # Apple, Tesla, 삼성전자
    # news = await collector.collect_all_news(target_symbols=target_symbols)
    
    # 전체 금융 뉴스 수집
    news = await collector.collect_all_news()
    
    news_summary = collector.get_news_summary(news, limit=15)
    
    print("\n--- 뉴스 요약 ---")
    print(json.dumps(news_summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    # Windows에서 asyncio 실행 시 ProactorEventLoop 사용
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
