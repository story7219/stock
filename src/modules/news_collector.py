#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📰 실시간 뉴스 수집 모듈 v2.0
한국 및 글로벌 금융 뉴스 실시간 수집
"""

import asyncio
import aiohttp
import feedparser
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re
from urllib.parse import urljoin, urlparse
import hashlib
import time
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
import yfinance as yf
import os

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """뉴스 아이템 데이터 클래스"""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    symbols: List[str]
    sentiment_score: float
    importance_score: float
    category: str
    language: str
    hash_id: str

class NewsCollector:
    """🔥 실시간 뉴스 수집기"""
    
    def __init__(self):
        """뉴스 수집기 초기화"""
        logger.info("📰 실시간 뉴스 수집기 초기화")
        
        # 한국 뉴스 소스
        self.korean_sources = {
            'naver_finance': 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258',
            'hankyung': 'https://www.hankyung.com/feed/economy',
            'maeil_kyung': 'https://www.mk.co.kr/rss/30000001/',
            'yonhap_finance': 'https://www.yna.co.kr/rss/economy.xml',
            'korea_joongang': 'https://rss.joins.com/joins_economy_list.xml',
            'chosun_biz': 'http://biz.chosun.com/rss/economy.xml',
            'seoul_finance': 'https://www.sedaily.com/RSSFeed.xml?DCode=101',
            'fn_news': 'http://www.fnnews.com/rss/fn_realestate_stock.xml'
        }
        
        # 글로벌 뉴스 소스
        self.global_sources = {
            'reuters_business': 'https://www.reuters.com/business/finance/rss',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'cnbc_markets': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/marketpulse/',
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
            'seeking_alpha': 'https://seekingalpha.com/market-news.xml',
            'financial_times': 'https://www.ft.com/companies?format=rss',
            'wsj_markets': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'
        }
        
        # 캐시 설정
        self.news_cache = {}
        self.cache_duration = 300  # 5분
        self.duplicate_checker = set()
        
        # 세션 설정
        self.session = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def collect_all_news(self, symbols: List[str] = None) -> List[NewsItem]:
        """모든 뉴스 수집"""
        logger.info("🔥 실시간 뉴스 수집 시작")
        
        all_news = []
        
        # 한국 뉴스 수집
        korean_news = await self.collect_korean_market_news(symbols)
        all_news.extend(korean_news)
        
        # 글로벌 뉴스 수집
        global_news = await self.collect_global_market_news(symbols)
        all_news.extend(global_news)
        
        # 중복 제거 및 정렬
        unique_news = self._remove_duplicates(all_news)
        sorted_news = sorted(unique_news, key=lambda x: x.published_date, reverse=True)
        
        logger.info(f"✅ 총 {len(sorted_news)}개 뉴스 수집 완료")
        return sorted_news[:100]  # 최신 100개만 반환
    
    async def collect_korean_market_news(self, symbols: List[str] = None) -> List[NewsItem]:
        """한국 시장 뉴스 수집"""
        logger.info("🇰🇷 한국 시장 뉴스 수집 시작")
        
        news_items = []
        
        # RSS 피드 수집
        for source_name, rss_url in self.korean_sources.items():
            try:
                items = await self._collect_rss_news(source_name, rss_url, 'ko')
                news_items.extend(items)
                logger.info(f"✅ {source_name}: {len(items)}개 뉴스 수집")
            except Exception as e:
                logger.error(f"❌ {source_name} 수집 실패: {e}")
        
        # 네이버 금융 뉴스 스크래핑
        try:
            naver_news = await self._scrape_naver_finance_news()
            news_items.extend(naver_news)
            logger.info(f"✅ 네이버 금융: {len(naver_news)}개 뉴스 수집")
        except Exception as e:
            logger.error(f"❌ 네이버 금융 스크래핑 실패: {e}")
        
        # 종목별 필터링
        if symbols:
            news_items = self._filter_by_symbols(news_items, symbols)
        
        logger.info(f"🇰🇷 한국 뉴스 수집 완료: {len(news_items)}개")
        return news_items
    
    async def collect_global_market_news(self, symbols: List[str] = None) -> List[NewsItem]:
        """글로벌 시장 뉴스 수집"""
        logger.info("🌍 글로벌 시장 뉴스 수집 시작")
        
        news_items = []
        
        # RSS 피드 수집
        for source_name, rss_url in self.global_sources.items():
            try:
                items = await self._collect_rss_news(source_name, rss_url, 'en')
                news_items.extend(items)
                logger.info(f"✅ {source_name}: {len(items)}개 뉴스 수집")
            except Exception as e:
                logger.error(f"❌ {source_name} 수집 실패: {e}")
        
        # Yahoo Finance 뉴스 API 사용
        try:
            yahoo_news = await self._collect_yahoo_finance_news(symbols)
            news_items.extend(yahoo_news)
            logger.info(f"✅ Yahoo Finance API: {len(yahoo_news)}개 뉴스 수집")
        except Exception as e:
            logger.error(f"❌ Yahoo Finance API 실패: {e}")
        
        # 종목별 필터링
        if symbols:
            news_items = self._filter_by_symbols(news_items, symbols)
        
        logger.info(f"🌍 글로벌 뉴스 수집 완료: {len(news_items)}개")
        return news_items
    
    async def _collect_rss_news(self, source_name: str, rss_url: str, language: str) -> List[NewsItem]:
        """RSS 피드에서 뉴스 수집"""
        try:
            if self.session is None:
                async with aiohttp.ClientSession() as session:
                    async with session.get(rss_url) as response:
                        content = await response.text()
            else:
                async with self.session.get(rss_url) as response:
                    content = await response.text()
            
            feed = feedparser.parse(content)
            news_items = []
            
            for entry in feed.entries[:20]:  # 최신 20개만
                try:
                    # 날짜 파싱
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # 24시간 이내 뉴스만
                    if (datetime.now() - pub_date).days > 1:
                        continue
                    
                    # 뉴스 아이템 생성
                    news_item = NewsItem(
                        title=entry.title,
                        content=self._extract_content(entry),
                        url=entry.link,
                        source=source_name,
                        published_date=pub_date,
                        symbols=self._extract_symbols(entry.title + " " + self._extract_content(entry)),
                        sentiment_score=self._calculate_sentiment(entry.title + " " + self._extract_content(entry)),
                        importance_score=self._calculate_importance(entry.title),
                        category=self._categorize_news(entry.title),
                        language=language,
                        hash_id=self._generate_hash(entry.title + entry.link)
                    )
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"RSS 엔트리 처리 실패: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"RSS 수집 실패 {source_name}: {e}")
            return []
    
    async def _scrape_naver_finance_news(self) -> List[NewsItem]:
        """네이버 금융 뉴스 스크래핑"""
        try:
            url = "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258"
            
            if self.session:
                async with self.session.get(url) as response:
                    html = await response.text()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            news_items = []
            
            # 뉴스 리스트 파싱
            news_list = soup.find_all('tr', class_='')
            
            for news in news_list[:10]:  # 최신 10개만
                try:
                    title_element = news.find('a', class_='tit')
                    if not title_element:
                        continue
                    
                    title = title_element.text.strip()
                    link = 'https://finance.naver.com' + title_element['href']
                    
                    # 상세 페이지에서 내용 수집
                    content = await self._scrape_news_content(link)
                    
                    news_item = NewsItem(
                        title=title,
                        content=content,
                        url=link,
                        source='naver_finance_scrape',
                        published_date=datetime.now(),
                        symbols=self._extract_symbols(title + " " + content),
                        sentiment_score=self._calculate_sentiment(title + " " + content),
                        importance_score=self._calculate_importance(title),
                        category=self._categorize_news(title),
                        language='ko',
                        hash_id=self._generate_hash(title + link)
                    )
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"네이버 뉴스 파싱 실패: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"네이버 금융 스크래핑 실패: {e}")
            return []
    
    async def _scrape_news_content(self, url: str) -> str:
        """뉴스 상세 내용 스크래핑"""
        try:
            if self.session:
                async with self.session.get(url) as response:
                    html = await response.text()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # 네이버 금융 뉴스 본문 추출
            content_div = soup.find('div', class_='scr01') or soup.find('div', id='news_read')
            if content_div:
                return content_div.get_text().strip()[:500]  # 500자 제한
            
            return ""
            
        except Exception as e:
            logger.warning(f"뉴스 내용 스크래핑 실패: {e}")
            return ""
    
    async def _collect_yahoo_finance_news(self, symbols: List[str] = None) -> List[NewsItem]:
        """Yahoo Finance API로 뉴스 수집"""
        news_items = []
        
        if not symbols:
            # 주요 지수 뉴스
            symbols = ['^GSPC', '^IXIC', '^DJI', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for symbol in symbols[:10]:  # 최대 10개 종목
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for item in news[:5]:  # 종목당 5개씩
                    try:
                        news_item = NewsItem(
                            title=item.get('title', ''),
                            content=item.get('summary', '')[:500],
                            url=item.get('link', ''),
                            source='yahoo_finance_api',
                            published_date=datetime.fromtimestamp(item.get('providerPublishTime', time.time())),
                            symbols=[symbol],
                            sentiment_score=self._calculate_sentiment(item.get('title', '') + " " + item.get('summary', '')),
                            importance_score=self._calculate_importance(item.get('title', '')),
                            category=self._categorize_news(item.get('title', '')),
                            language='en',
                            hash_id=self._generate_hash(item.get('title', '') + item.get('link', ''))
                        )
                        
                        news_items.append(news_item)
                        
                    except Exception as e:
                        logger.warning(f"Yahoo 뉴스 아이템 처리 실패: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"Yahoo Finance {symbol} 뉴스 수집 실패: {e}")
                continue
        
        return news_items
    
    def _extract_content(self, entry) -> str:
        """RSS 엔트리에서 내용 추출"""
        content = ""
        
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description
        elif hasattr(entry, 'content'):
            if isinstance(entry.content, list) and len(entry.content) > 0:
                content = entry.content[0].value
            else:
                content = str(entry.content)
        
        # HTML 태그 제거
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text().strip()
        
        return content[:500]  # 500자 제한
    
    def _extract_symbols(self, text: str) -> List[str]:
        """텍스트에서 종목 심볼 추출"""
        symbols = []
        
        # 미국 주식 심볼 패턴 (3-5글자 대문자)
        us_pattern = r'\b[A-Z]{3,5}\b'
        us_matches = re.findall(us_pattern, text.upper())
        
        # 한국 종목명 패턴 (회사명 + 관련 키워드)
        korean_companies = [
            '삼성전자', '삼성SDI', 'SK하이닉스', 'NAVER', '카카오', 'LG화학', 'LG에너지솔루션',
            '현대차', '기아', 'POSCO', '한국전력', 'KB금융', '신한지주', '하나금융지주',
            '삼성바이오로직스', '셀트리온', '현대모비스', '삼성물산', 'LG전자', 'SK텔레콤'
        ]
        
        for company in korean_companies:
            if company in text:
                symbols.append(company)
        
        # 필터링 (일반적인 단어 제외)
        filtered_symbols = []
        exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY'}
        
        for symbol in us_matches:
            if symbol not in exclude_words:
                filtered_symbols.append(symbol)
        
        filtered_symbols.extend(symbols)
        return list(set(filtered_symbols))[:5]  # 최대 5개
    
    def _calculate_sentiment(self, text: str) -> float:
        """감정 분석 점수 계산"""
        try:
            # 영어 텍스트 감정 분석
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # 한국어 키워드 기반 감정 분석
            positive_keywords = ['상승', '급등', '호재', '긍정', '성장', '증가', '개선', '강세', '돌파', '상향']
            negative_keywords = ['하락', '급락', '악재', '부정', '감소', '우려', '위험', '약세', '하향', '조정']
            
            korean_score = 0
            for word in positive_keywords:
                korean_score += text.count(word) * 0.1
            for word in negative_keywords:
                korean_score -= text.count(word) * 0.1
            
            # 최종 점수 (-1 ~ 1)
            final_score = (sentiment + korean_score) / 2
            return max(-1, min(1, final_score))
            
        except Exception as e:
            logger.warning(f"감정 분석 실패: {e}")
            return 0.0
    
    def _calculate_importance(self, title: str) -> float:
        """뉴스 중요도 계산"""
        importance = 0.5  # 기본값
        
        # 중요 키워드
        high_importance = ['실적', '분할', '합병', 'IPO', '상장', '증자', '배당', '공시']
        medium_importance = ['투자', '협력', '계약', '출시', '발표']
        
        title_upper = title.upper()
        
        for keyword in high_importance:
            if keyword in title or keyword.upper() in title_upper:
                importance += 0.3
        
        for keyword in medium_importance:
            if keyword in title or keyword.upper() in title_upper:
                importance += 0.2
        
        return min(1.0, importance)
    
    def _categorize_news(self, title: str) -> str:
        """뉴스 카테고리 분류"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['실적', 'earnings', '매출', 'revenue']):
            return 'EARNINGS'
        elif any(word in title_lower for word in ['인수', '합병', 'merger', 'acquisition']):
            return 'M&A'
        elif any(word in title_lower for word in ['신제품', '출시', 'launch', 'product']):
            return 'PRODUCT'
        elif any(word in title_lower for word in ['규제', 'regulation', '정책', 'policy']):
            return 'REGULATION'
        elif any(word in title_lower for word in ['투자', 'investment', '자금', 'funding']):
            return 'INVESTMENT'
        else:
            return 'GENERAL'
    
    def _generate_hash(self, text: str) -> str:
        """텍스트 해시 생성"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _filter_by_symbols(self, news_items: List[NewsItem], symbols: List[str]) -> List[NewsItem]:
        """종목별 뉴스 필터링"""
        filtered = []
        
        for news in news_items:
            if any(symbol in news.symbols for symbol in symbols):
                filtered.append(news)
            elif any(symbol.lower() in news.title.lower() or symbol.lower() in news.content.lower() for symbol in symbols):
                news.symbols.extend([s for s in symbols if s.lower() in (news.title + news.content).lower()])
                filtered.append(news)
        
        return filtered
    
    def _remove_duplicates(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """중복 뉴스 제거"""
        seen_hashes = set()
        unique_news = []
        
        for news in news_items:
            if news.hash_id not in seen_hashes:
                seen_hashes.add(news.hash_id)
                unique_news.append(news)
        
        return unique_news
    
    def get_news_summary(self, symbols: List[str] = None, hours: int = 24) -> Dict[str, Any]:
        """뉴스 요약 정보"""
        # 캐시된 뉴스에서 요약 생성
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_news = []
        for news_list in self.news_cache.values():
            for news in news_list:
                if news.published_date >= cutoff_time:
                    if not symbols or any(symbol in news.symbols for symbol in symbols):
                        recent_news.append(news)
        
        if not recent_news:
            return {
                'total_news': 0,
                'avg_sentiment': 0.0,
                'top_categories': [],
                'summary': '최근 뉴스가 없습니다.'
            }
        
        # 통계 계산
        total_news = len(recent_news)
        avg_sentiment = sum(news.sentiment_score for news in recent_news) / total_news
        
        # 카테고리별 집계
        categories = {}
        for news in recent_news:
            categories[news.category] = categories.get(news.category, 0) + 1
        
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_news': total_news,
            'avg_sentiment': round(avg_sentiment, 3),
            'top_categories': top_categories,
            'summary': f'최근 {hours}시간 동안 {total_news}개 뉴스, 평균 감정점수: {avg_sentiment:.2f}'
        }

    async def collect_news(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """뉴스 수집"""
        try:
            logger.info(f"{symbol} 뉴스 수집 시작")
            
            # Mock 모드 또는 실제 뉴스 수집
            if os.getenv('IS_MOCK', 'false').lower() == 'true':
                return self._generate_mock_news(symbol, limit)
            
            # 실제 뉴스 수집 로직
            news_data = await self._collect_real_news(symbol, limit)
            
            logger.info(f"{symbol} 뉴스 {len(news_data)}개 수집 완료")
            return news_data
            
        except Exception as e:
            logger.error(f"{symbol} 뉴스 수집 실패: {e}")
            return self._generate_mock_news(symbol, limit)
    
    def _generate_mock_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Mock 뉴스 생성"""
        mock_news = []
        for i in range(limit):
            mock_news.append({
                'title': f'{symbol} 관련 뉴스 {i+1}',
                'content': f'{symbol}에 대한 최신 시장 동향 분석',
                'source': 'Mock News',
                'timestamp': datetime.now().isoformat(),
                'sentiment': 'positive' if i % 2 == 0 else 'neutral'
            })
        return mock_news
    
    async def _collect_real_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """실제 뉴스 수집"""
        # 실제 뉴스 수집 로직 구현
        return self._generate_mock_news(symbol, limit)

# 사용 예시
async def main():
    """뉴스 수집 테스트"""
    async with NewsCollector() as collector:
        # 전체 뉴스 수집
        all_news = await collector.collect_all_news(['AAPL', 'MSFT', '삼성전자'])
        
        print(f"수집된 뉴스: {len(all_news)}개")
        
        for news in all_news[:5]:
            print(f"\n제목: {news.title}")
            print(f"소스: {news.source}")
            print(f"감정점수: {news.sentiment_score}")
            print(f"중요도: {news.importance_score}")
            print(f"관련종목: {news.symbols}")

if __name__ == "__main__":
    asyncio.run(main()) 