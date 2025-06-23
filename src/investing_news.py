"""
투자 뉴스 수집 모듈
실시간 금융 뉴스 수집 및 감정 분석
🚀 Gemini AI 최적화를 위한 고품질 뉴스 데이터 가공 시스템
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import aiohttp
import feedparser
import re
import json
from urllib.parse import urljoin, urlparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class NewsQualityMetrics:
    """뉴스 품질 지표"""
    relevance_score: float = 0.0      # 관련성 점수 (0-100)
    credibility_score: float = 0.0    # 신뢰도 점수 (0-100)
    freshness_score: float = 0.0      # 신선도 점수 (0-100)
    sentiment_confidence: float = 0.0  # 감정 분석 신뢰도 (0-100)
    overall_quality: float = 0.0      # 전체 품질 점수 (0-100)
    word_count: int = 0               # 단어 수
    has_financial_keywords: bool = False  # 금융 키워드 포함 여부

@dataclass
class NewsData:
    """뉴스 데이터 클래스 - Gemini AI 최적화"""
    # 기본 정보
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    
    # 분류 및 태그
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    related_stocks: List[str] = field(default_factory=list)
    market_sector: Optional[str] = None
    
    # 감정 분석
    sentiment_score: float = 0.0      # -1(부정) ~ 1(긍정)
    sentiment_label: str = "neutral"   # positive, negative, neutral
    confidence_score: float = 0.0     # 신뢰도 점수
    
    # 중요도 및 영향도
    importance_score: float = 0.0     # 0-100
    market_impact_score: float = 0.0  # 시장 영향도 0-100
    
    # 키워드 및 엔티티
    financial_keywords: List[str] = field(default_factory=list)
    company_mentions: List[str] = field(default_factory=list)
    numeric_data: Dict[str, float] = field(default_factory=dict)
    
    # 품질 및 메타데이터
    quality_metrics: NewsQualityMetrics = field(default_factory=NewsQualityMetrics)
    timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = "web_scraping"
    
    def calculate_quality_score(self) -> float:
        """뉴스 품질 점수 계산"""
        try:
            # 기본 품질 요소들
            title_quality = min(100, len(self.title) * 2) if self.title else 0
            content_quality = min(100, len(self.content) / 10) if self.content else 0
            freshness = max(0, 100 - (datetime.now() - self.published_date).days * 10)
            
            # 금융 관련성
            financial_relevance = 50 if self.financial_keywords else 0
            if self.related_stocks:
                financial_relevance += 30
            if self.company_mentions:
                financial_relevance += 20
            
            # 전체 품질 점수
            overall_quality = (
                title_quality * 0.2 +
                content_quality * 0.3 +
                freshness * 0.2 +
                financial_relevance * 0.3
            )
            
            self.quality_metrics.overall_quality = min(100, overall_quality)
            self.quality_metrics.relevance_score = financial_relevance
            self.quality_metrics.freshness_score = freshness
            self.quality_metrics.word_count = len(self.content.split()) if self.content else 0
            self.quality_metrics.has_financial_keywords = bool(self.financial_keywords)
            
            return self.quality_metrics.overall_quality
            
        except Exception as e:
            logger.warning(f"뉴스 품질 점수 계산 실패: {e}")
            return 0.0

class KoreanNewsCollector:
    """한국 금융 뉴스 수집기"""
    
    def __init__(self):
        self.sources = {
            "네이버금융": {
                "url": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258",
                "rss": "https://finance.naver.com/news/rss.naver?section_id=101&section_id2=258"
            },
            "한국경제": {
                "url": "https://www.hankyung.com/finance",
                "rss": "https://www.hankyung.com/feed/finance"
            },
            "매일경제": {
                "url": "https://www.mk.co.kr/news/stock/",
                "rss": "https://www.mk.co.kr/rss/40300001/"
            },
            "이데일리": {
                "url": "https://www.edaily.co.kr/news/newsList.asp?newsType=&sub_cd=AA01",
                "rss": "https://www.edaily.co.kr/rss/rss.asp?sub_cd=AA01"
            }
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3'
        })
        
        # 금융 키워드 사전
        self.financial_keywords = {
            "시장": ["코스피", "코스닥", "나스닥", "다우", "S&P", "증시", "주가", "지수"],
            "기업": ["삼성", "LG", "현대", "SK", "포스코", "네이버", "카카오"],
            "경제": ["금리", "환율", "인플레이션", "GDP", "수출", "수입", "무역"],
            "투자": ["투자", "펀드", "채권", "주식", "배당", "상장", "IPO"],
            "정책": ["한국은행", "금통위", "기준금리", "정부", "정책", "규제"]
        }
    
    async def collect_rss_news(self, source_name: str, rss_url: str, limit: int = 20) -> List[NewsData]:
        """RSS 피드에서 뉴스 수집"""
        news_list = []
        try:
            logger.info(f"{source_name} RSS 뉴스 수집 시작: {rss_url}")
            
            # RSS 피드 파싱
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:limit]:
                try:
                    # 기본 정보 추출
                    title = entry.get('title', '').strip()
                    link = entry.get('link', '')
                    summary = entry.get('summary', '').strip()
                    
                    # 발행일 파싱
                    published_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    
                    # 상세 내용 수집
                    content = await self._fetch_article_content(link)
                    if not content:
                        content = summary
                    
                    # 뉴스 데이터 생성
                    news_data = NewsData(
                        title=title,
                        content=content,
                        url=link,
                        source=source_name,
                        published_date=published_date,
                        category="finance",
                        data_source="rss"
                    )
                    
                    # 키워드 및 감정 분석
                    self._analyze_news_content(news_data)
                    
                    # 품질 점수 계산
                    quality_score = news_data.calculate_quality_score()
                    if quality_score > 30:  # 최소 품질 기준
                        news_list.append(news_data)
                        
                except Exception as e:
                    logger.warning(f"RSS 항목 처리 실패 ({source_name}): {e}")
                    continue
            
            logger.info(f"{source_name} RSS 뉴스 {len(news_list)}개 수집 완료")
            return news_list
            
        except Exception as e:
            logger.error(f"{source_name} RSS 수집 실패: {e}")
            return []
    
    async def _fetch_article_content(self, url: str) -> str:
        """기사 본문 내용 추출"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 일반적인 기사 본문 선택자들
                        content_selectors = [
                            '.article_body',
                            '.news_content',
                            '.article-content',
                            '.content',
                            '#articleBodyContents',
                            '.article_txt'
                        ]
                        
                        for selector in content_selectors:
                            content_elem = soup.select_one(selector)
                            if content_elem:
                                # 불필요한 태그 제거
                                for tag in content_elem.find_all(['script', 'style', 'iframe', 'img']):
                                    tag.decompose()
                                
                                text = content_elem.get_text(strip=True)
                                if len(text) > 100:  # 최소 길이 체크
                                    return text
                        
                        # 기본 텍스트 추출
                        return soup.get_text()[:1000]  # 최대 1000자
                        
        except Exception as e:
            logger.warning(f"기사 본문 추출 실패 {url}: {e}")
            return ""
    
    def _analyze_news_content(self, news_data: NewsData):
        """뉴스 내용 분석 (키워드, 감정, 관련 주식)"""
        try:
            full_text = f"{news_data.title} {news_data.content}".lower()
            
            # 금융 키워드 추출
            found_keywords = []
            for category, keywords in self.financial_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        found_keywords.append(keyword)
            
            news_data.financial_keywords = list(set(found_keywords))
            
            # 회사명 추출
            company_patterns = [
                r'([가-힣]+)(?:전자|화학|물산|건설|그룹|홀딩스?)',
                r'(삼성|LG|현대|SK|포스코|네이버|카카오|셀트리온)'
            ]
            
            companies = []
            for pattern in company_patterns:
                matches = re.findall(pattern, full_text)
                companies.extend(matches)
            
            news_data.company_mentions = list(set(companies))
            
            # 숫자 데이터 추출 (주가, 거래량 등)
            numeric_patterns = {
                'price': r'(\d+(?:,\d+)*)\s*원',
                'percentage': r'(\d+\.?\d*)\s*%',
                'volume': r'(\d+(?:,\d+)*)\s*주'
            }
            
            numeric_data = {}
            for data_type, pattern in numeric_patterns.items():
                matches = re.findall(pattern, full_text)
                if matches:
                    try:
                        # 첫 번째 매치 값을 숫자로 변환
                        value = float(matches[0].replace(',', ''))
                        numeric_data[data_type] = value
                    except ValueError:
                        pass
            
            news_data.numeric_data = numeric_data
            
            # 간단한 감정 분석 (키워드 기반)
            positive_words = ['상승', '증가', '호조', '성장', '개선', '확대', '신고가']
            negative_words = ['하락', '감소', '부진', '악화', '위기', '손실', '신저가']
            
            positive_count = sum(1 for word in positive_words if word in full_text)
            negative_count = sum(1 for word in negative_words if word in full_text)
            
            if positive_count > negative_count:
                news_data.sentiment_label = "positive"
                news_data.sentiment_score = min(1.0, (positive_count - negative_count) / 10)
            elif negative_count > positive_count:
                news_data.sentiment_label = "negative"
                news_data.sentiment_score = max(-1.0, (positive_count - negative_count) / 10)
            else:
                news_data.sentiment_label = "neutral"
                news_data.sentiment_score = 0.0
            
            # 중요도 점수 계산
            importance_factors = [
                len(news_data.financial_keywords) * 5,
                len(news_data.company_mentions) * 10,
                len(news_data.numeric_data) * 15,
                min(50, len(news_data.content) / 20)
            ]
            
            news_data.importance_score = min(100, sum(importance_factors))
            
        except Exception as e:
            logger.warning(f"뉴스 내용 분석 실패: {e}")
    
    async def collect_all_korean_news(self, limit_per_source: int = 20) -> List[NewsData]:
        """모든 한국 뉴스 소스에서 뉴스 수집"""
        all_news = []
        
        tasks = []
        for source_name, source_info in self.sources.items():
            if 'rss' in source_info:
                task = self.collect_rss_news(source_name, source_info['rss'], limit_per_source)
                tasks.append(task)
        
        # 비동기 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"뉴스 수집 중 오류: {result}")
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_news = []
        for news in all_news:
            if news.url not in seen_urls:
                seen_urls.add(news.url)
                unique_news.append(news)
        
        # 품질 점수 기준 정렬
        unique_news.sort(key=lambda x: x.quality_metrics.overall_quality, reverse=True)
        
        logger.info(f"한국 뉴스 총 {len(unique_news)}개 수집 완료")
        return unique_news

class GlobalNewsCollector:
    """글로벌 금융 뉴스 수집기"""
    
    def __init__(self):
        self.sources = {
            "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "MarketWatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
            "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
            "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def collect_global_news(self, limit_per_source: int = 15) -> List[NewsData]:
        """글로벌 뉴스 수집"""
        all_news = []
        
        for source_name, rss_url in self.sources.items():
            try:
                logger.info(f"{source_name} 글로벌 뉴스 수집 시작")
                
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:limit_per_source]:
                    try:
                        title = entry.get('title', '').strip()
                        link = entry.get('link', '')
                        summary = entry.get('summary', '').strip()
                        
                        # 발행일 파싱
                        published_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6])
                        
                        news_data = NewsData(
                            title=title,
                            content=summary,
                            url=link,
                            source=source_name,
                            published_date=published_date,
                            category="global_finance",
                            data_source="global_rss"
                        )
                        
                        # 영문 키워드 분석
                        self._analyze_global_content(news_data)
                        
                        if news_data.calculate_quality_score() > 25:
                            all_news.append(news_data)
                            
                    except Exception as e:
                        logger.warning(f"글로벌 뉴스 항목 처리 실패: {e}")
                        continue
                
                logger.info(f"{source_name} 뉴스 수집 완료")
                
            except Exception as e:
                logger.error(f"{source_name} 수집 실패: {e}")
                continue
        
        return all_news
    
    def _analyze_global_content(self, news_data: NewsData):
        """글로벌 뉴스 내용 분석"""
        try:
            full_text = f"{news_data.title} {news_data.content}".lower()
            
            # 글로벌 금융 키워드
            global_keywords = {
                "markets": ["nasdaq", "dow", "s&p", "ftse", "nikkei", "dax"],
                "companies": ["apple", "microsoft", "amazon", "google", "tesla", "nvidia"],
                "economics": ["fed", "interest rate", "inflation", "gdp", "unemployment"],
                "crypto": ["bitcoin", "ethereum", "cryptocurrency", "blockchain"]
            }
            
            found_keywords = []
            for category, keywords in global_keywords.items():
                for keyword in keywords:
                    if keyword in full_text:
                        found_keywords.append(keyword)
            
            news_data.financial_keywords = found_keywords
            
            # 간단한 영문 감정 분석
            positive_words = ['gain', 'rise', 'surge', 'bull', 'growth', 'profit']
            negative_words = ['fall', 'drop', 'crash', 'bear', 'loss', 'decline']
            
            positive_count = sum(1 for word in positive_words if word in full_text)
            negative_count = sum(1 for word in negative_words if word in full_text)
            
            if positive_count > negative_count:
                news_data.sentiment_label = "positive"
                news_data.sentiment_score = min(1.0, (positive_count - negative_count) / 5)
            elif negative_count > positive_count:
                news_data.sentiment_label = "negative"
                news_data.sentiment_score = max(-1.0, (positive_count - negative_count) / 5)
            
        except Exception as e:
            logger.warning(f"글로벌 뉴스 분석 실패: {e}")

class InvestingNewsCollector:
    """통합 투자 뉴스 수집기 - Gemini AI 최적화"""
    
    def __init__(self):
        self.korean_collector = KoreanNewsCollector()
        self.global_collector = GlobalNewsCollector()
        self.news_cache: List[NewsData] = []
        self.last_update = datetime.now() - timedelta(hours=1)
        
        # 설정값들
        self.update_interval = int(os.getenv('NEWS_UPDATE_INTERVAL', 5))  # 분
        self.news_limit = int(os.getenv('NEWS_LIMIT', 50))
        
        logger.info(f"InvestingNewsCollector 초기화 완료 (업데이트 간격: {self.update_interval}분)")
    
    async def collect_all_news(self, force_update: bool = False) -> List[NewsData]:
        """모든 뉴스 소스에서 뉴스 수집"""
        try:
            # 캐시 확인
            if not force_update and self._is_cache_valid():
                logger.info("캐시된 뉴스 데이터 반환")
                return self.news_cache[:self.news_limit]
            
            logger.info("새로운 뉴스 데이터 수집 시작")
            
            # 병렬로 한국 및 글로벌 뉴스 수집
            korean_task = self.korean_collector.collect_all_korean_news(20)
            global_task = self.global_collector.collect_global_news(15)
            
            korean_news, global_news = await asyncio.gather(
                korean_task, global_task, return_exceptions=True
            )
            
            # 결과 합치기
            all_news = []
            if isinstance(korean_news, list):
                all_news.extend(korean_news)
            if isinstance(global_news, list):
                all_news.extend(global_news)
            
            # 중복 제거 및 정렬
            all_news = self._deduplicate_and_sort(all_news)
            
            # 캐시 업데이트
            self.news_cache = all_news
            self.last_update = datetime.now()
            
            logger.info(f"총 {len(all_news)}개 뉴스 수집 완료")
            return all_news[:self.news_limit]
            
        except Exception as e:
            logger.error(f"뉴스 수집 중 오류: {e}")
            return self.news_cache[:self.news_limit] if self.news_cache else []
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        time_diff = datetime.now() - self.last_update
        return time_diff.total_seconds() < (self.update_interval * 60) and bool(self.news_cache)
    
    def _deduplicate_and_sort(self, news_list: List[NewsData]) -> List[NewsData]:
        """중복 제거 및 정렬"""
        # URL 기준 중복 제거
        seen_urls = set()
        unique_news = []
        
        for news in news_list:
            if news.url not in seen_urls:
                seen_urls.add(news.url)
                unique_news.append(news)
        
        # 품질 점수와 중요도 기준 정렬
        unique_news.sort(
            key=lambda x: (x.quality_metrics.overall_quality, x.importance_score), 
            reverse=True
        )
        
        return unique_news
    
    def get_market_sentiment(self) -> Dict[str, any]:
        """시장 감정 분석 결과"""
        if not self.news_cache:
            return {"sentiment": "neutral", "confidence": 0.0, "news_count": 0}
        
        try:
            sentiments = [news.sentiment_score for news in self.news_cache if news.sentiment_score != 0]
            
            if not sentiments:
                return {"sentiment": "neutral", "confidence": 0.0, "news_count": len(self.news_cache)}
            
            avg_sentiment = np.mean(sentiments)
            confidence = min(100, len(sentiments) * 2)
            
            if avg_sentiment > 0.1:
                sentiment_label = "positive"
            elif avg_sentiment < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "sentiment": sentiment_label,
                "score": float(avg_sentiment),
                "confidence": float(confidence),
                "news_count": len(self.news_cache),
                "positive_news": len([n for n in self.news_cache if n.sentiment_score > 0]),
                "negative_news": len([n for n in self.news_cache if n.sentiment_score < 0]),
                "neutral_news": len([n for n in self.news_cache if n.sentiment_score == 0])
            }
            
        except Exception as e:
            logger.error(f"시장 감정 분석 실패: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "news_count": 0}
    
    def prepare_gemini_news_dataset(self) -> Dict[str, any]:
        """Gemini AI를 위한 뉴스 데이터셋 준비"""
        try:
            if not self.news_cache:
                return {"error": "뉴스 데이터가 없습니다"}
            
            # 고품질 뉴스만 선별 (품질 점수 50 이상)
            high_quality_news = [
                news for news in self.news_cache 
                if news.quality_metrics.overall_quality >= 50
            ]
            
            # Gemini AI 형식으로 변환
            gemini_dataset = {
                "market_sentiment": self.get_market_sentiment(),
                "news_summary": {
                    "total_news": len(self.news_cache),
                    "high_quality_news": len(high_quality_news),
                    "korean_news": len([n for n in self.news_cache if n.category == "finance"]),
                    "global_news": len([n for n in self.news_cache if n.category == "global_finance"]),
                    "last_update": self.last_update.isoformat()
                },
                "top_news": [
                    {
                        "title": news.title,
                        "source": news.source,
                        "sentiment": news.sentiment_label,
                        "sentiment_score": news.sentiment_score,
                        "importance": news.importance_score,
                        "keywords": news.financial_keywords,
                        "companies": news.company_mentions,
                        "published": news.published_date.isoformat(),
                        "url": news.url
                    }
                    for news in high_quality_news[:20]  # 상위 20개
                ],
                "keyword_analysis": self._analyze_trending_keywords(),
                "company_mentions": self._analyze_company_mentions()
            }
            
            return gemini_dataset
            
        except Exception as e:
            logger.error(f"Gemini 데이터셋 준비 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_trending_keywords(self) -> Dict[str, int]:
        """트렌딩 키워드 분석"""
        try:
            keyword_counts = {}
            
            for news in self.news_cache:
                for keyword in news.financial_keywords:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            # 상위 15개 키워드
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_keywords[:15])
            
        except Exception as e:
            logger.error(f"키워드 분석 실패: {e}")
            return {}
    
    def _analyze_company_mentions(self) -> Dict[str, int]:
        """기업 언급 분석"""
        try:
            company_counts = {}
            
            for news in self.news_cache:
                for company in news.company_mentions:
                    company_counts[company] = company_counts.get(company, 0) + 1
            
            # 상위 10개 기업
            sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_companies[:10])
            
        except Exception as e:
            logger.error(f"기업 언급 분석 실패: {e}")
            return {}
    
    def get_news_by_category(self, category: str) -> List[NewsData]:
        """카테고리별 뉴스 조회"""
        return [news for news in self.news_cache if news.category == category]
    
    def get_news_by_sentiment(self, sentiment: str) -> List[NewsData]:
        """감정별 뉴스 조회"""
        return [news for news in self.news_cache if news.sentiment_label == sentiment]
    
    def search_news(self, keyword: str) -> List[NewsData]:
        """키워드로 뉴스 검색"""
        keyword_lower = keyword.lower()
        results = []
        
        for news in self.news_cache:
            if (keyword_lower in news.title.lower() or 
                keyword_lower in news.content.lower() or
                keyword_lower in [k.lower() for k in news.financial_keywords]):
                results.append(news)
        
        return sorted(results, key=lambda x: x.quality_metrics.overall_quality, reverse=True)

# 편의 함수들
async def collect_latest_news(limit: int = 50) -> List[NewsData]:
    """최신 뉴스 수집 (편의 함수)"""
    collector = InvestingNewsCollector()
    return await collector.collect_all_news()

def get_market_sentiment_summary() -> Dict[str, any]:
    """시장 감정 요약 (편의 함수)"""
    collector = InvestingNewsCollector()
    return collector.get_market_sentiment()

if __name__ == "__main__":
    # 테스트 코드
    async def test_news_collection():
        collector = InvestingNewsCollector()
        news_list = await collector.collect_all_news()
        
        print(f"수집된 뉴스: {len(news_list)}개")
        
        if news_list:
            print("\n=== 상위 5개 뉴스 ===")
            for i, news in enumerate(news_list[:5], 1):
                print(f"{i}. {news.title}")
                print(f"   출처: {news.source}")
                print(f"   감정: {news.sentiment_label} ({news.sentiment_score:.2f})")
                print(f"   품질: {news.quality_metrics.overall_quality:.1f}")
                print(f"   키워드: {', '.join(news.financial_keywords[:5])}")
                print()
        
        # 시장 감정 분석
        sentiment = collector.get_market_sentiment()
        print("=== 시장 감정 분석 ===")
        print(f"전체 감정: {sentiment['sentiment']}")
        print(f"신뢰도: {sentiment['confidence']:.1f}%")
        print(f"긍정 뉴스: {sentiment.get('positive_news', 0)}개")
        print(f"부정 뉴스: {sentiment.get('negative_news', 0)}개")
    
    # asyncio.run(test_news_collection()) 