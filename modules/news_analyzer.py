#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📰 실시간 뉴스 수집 및 감성분석 엔진
뉴스, 소셜미디어, 공시정보를 종합하여 시장 심리를 분석
Gemini AI 최적화 뉴스 인텔리전스 시스템
"""

import os
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import feedparser
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """뉴스 아이템"""
    title: str
    content: str
    source: str
    published_date: datetime
    url: str
    sentiment_score: float  # -1 ~ 1
    relevance_score: float  # 0 ~ 1
    symbols_mentioned: List[str]

@dataclass
class MarketSentiment:
    """시장 감성 분석 결과"""
    overall_sentiment: float
    positive_news_count: int
    negative_news_count: int
    neutral_news_count: int
    key_topics: List[str]
    sentiment_trend: str  # "IMPROVING", "DECLINING", "STABLE"
    confidence: float

class NewsCollector:
    """📰 뉴스 수집기"""
    
    def __init__(self):
        """초기화"""
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # RSS 피드 소스들
        self.rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.marketwatch.com/rss/topstories',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://rss.cnn.com/rss/money_latest.rss'
        ]
        
        # 한국 뉴스 소스
        self.korean_feeds = [
            'https://news.naver.com/main/rss/section.nhn?sid1=101',  # 경제
            'https://rss.hankyung.com/new/economy.xml',  # 한경 경제
            'https://www.mk.co.kr/rss/40300001/'  # 매경 증권
        ]
        
        logger.info("📰 뉴스 수집기 초기화 완료")
    
    def collect_news(self, symbols: List[str], hours_back: int = 24) -> List[NewsItem]:
        """뉴스 수집"""
        all_news = []
        
        try:
            # 1. RSS 피드에서 수집
            rss_news = self._collect_from_rss()
            all_news.extend(rss_news)
            
            # 2. News API에서 수집 (키가 있는 경우)
            if self.news_api_key:
                api_news = self._collect_from_news_api(symbols)
                all_news.extend(api_news)
            
            # 3. Alpha Vantage 뉴스 (키가 있는 경우)
            if self.alpha_vantage_key:
                av_news = self._collect_from_alpha_vantage(symbols)
                all_news.extend(av_news)
            
            # 4. 중복 제거 및 관련도 필터링
            filtered_news = self._filter_and_deduplicate(all_news, symbols)
            
            logger.info(f"📰 총 {len(filtered_news)}개 뉴스 수집 완료")
            return filtered_news
            
        except Exception as e:
            logger.error(f"❌ 뉴스 수집 실패: {e}")
            return []
    
    def _collect_from_rss(self) -> List[NewsItem]:
        """RSS 피드에서 뉴스 수집"""
        news_items = []
        
        all_feeds = self.rss_feeds + self.korean_feeds
        
        for feed_url in all_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # 최근 10개만
                    try:
                        # 발행일 파싱
                        published_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6])
                        
                        # 뉴스 아이템 생성
                        news_item = NewsItem(
                            title=entry.title,
                            content=getattr(entry, 'summary', entry.title),
                            source=feed.feed.title if hasattr(feed.feed, 'title') else feed_url,
                            published_date=published_date,
                            url=entry.link,
                            sentiment_score=0.0,  # 나중에 계산
                            relevance_score=0.0,  # 나중에 계산
                            symbols_mentioned=[]
                        )
                        
                        news_items.append(news_item)
                        
                    except Exception as e:
                        logger.warning(f"RSS 아이템 파싱 실패: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"RSS 피드 파싱 실패 ({feed_url}): {e}")
                continue
        
        return news_items
    
    def _collect_from_news_api(self, symbols: List[str]) -> List[NewsItem]:
        """News API에서 뉴스 수집"""
        news_items = []
        
        try:
            # 주요 금융 키워드들
            keywords = ["stock market", "economy", "finance", "nasdaq", "kospi"] + symbols[:5]
            
            for keyword in keywords:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        news_item = NewsItem(
                            title=article.get('title', ''),
                            content=article.get('description', ''),
                            source=article.get('source', {}).get('name', 'Unknown'),
                            published_date=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                            url=article.get('url', ''),
                            sentiment_score=0.0,
                            relevance_score=0.0,
                            symbols_mentioned=[]
                        )
                        
                        news_items.append(news_item)
                
        except Exception as e:
            logger.warning(f"News API 수집 실패: {e}")
        
        return news_items
    
    def _collect_from_alpha_vantage(self, symbols: List[str]) -> List[NewsItem]:
        """Alpha Vantage에서 뉴스 수집"""
        news_items = []
        
        try:
            for symbol in symbols[:5]:  # 상위 5개 종목만
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.alpha_vantage_key,
                    'limit': 50
                }
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('feed', []):
                        news_item = NewsItem(
                            title=item.get('title', ''),
                            content=item.get('summary', ''),
                            source=item.get('source', 'Alpha Vantage'),
                            published_date=datetime.fromisoformat(item.get('time_published', '')[:8] + 'T' + item.get('time_published', '')[9:15]),
                            url=item.get('url', ''),
                            sentiment_score=float(item.get('overall_sentiment_score', 0)),
                            relevance_score=float(item.get('relevance_score', 0)),
                            symbols_mentioned=[symbol]
                        )
                        
                        news_items.append(news_item)
                
        except Exception as e:
            logger.warning(f"Alpha Vantage 뉴스 수집 실패: {e}")
        
        return news_items
    
    def _filter_and_deduplicate(self, news_items: List[NewsItem], symbols: List[str]) -> List[NewsItem]:
        """중복 제거 및 관련도 필터링"""
        
        # 제목 기준 중복 제거
        seen_titles = set()
        unique_news = []
        
        for item in news_items:
            # 제목 정규화
            normalized_title = item.title.lower().strip()
            
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                
                # 관련 종목 찾기
                item.symbols_mentioned = self._find_mentioned_symbols(item.title + " " + item.content, symbols)
                
                # 관련도 점수 계산
                item.relevance_score = self._calculate_relevance(item, symbols)
                
                # 감성 점수 계산 (기존에 없는 경우)
                if item.sentiment_score == 0.0:
                    item.sentiment_score = self._calculate_sentiment(item.title + " " + item.content)
                
                # 관련도가 높은 뉴스만 포함
                if item.relevance_score > 0.3 or item.symbols_mentioned:
                    unique_news.append(item)
        
        # 발행일 기준 정렬 (최신순)
        unique_news.sort(key=lambda x: x.published_date, reverse=True)
        
        return unique_news[:50]  # 최신 50개만
    
    def _find_mentioned_symbols(self, text: str, symbols: List[str]) -> List[str]:
        """텍스트에서 언급된 종목 찾기"""
        mentioned = []
        text_lower = text.lower()
        
        for symbol in symbols:
            # 종목 코드 검색
            if symbol.lower() in text_lower:
                mentioned.append(symbol)
            
            # 회사명 검색 (간단한 매핑)
            company_names = {
                'AAPL': 'apple',
                'TSLA': 'tesla',
                'MSFT': 'microsoft',
                'GOOGL': 'google',
                'AMZN': 'amazon',
                '005930.KS': '삼성전자',
                '000660.KS': 'sk하이닉스',
                '035420.KS': 'naver'
            }
            
            company_name = company_names.get(symbol)
            if company_name and company_name in text_lower:
                if symbol not in mentioned:
                    mentioned.append(symbol)
        
        return mentioned
    
    def _calculate_relevance(self, news_item: NewsItem, symbols: List[str]) -> float:
        """뉴스 관련도 계산"""
        score = 0.0
        text = (news_item.title + " " + news_item.content).lower()
        
        # 금융 키워드 점수
        financial_keywords = [
            'stock', 'market', 'trading', 'investment', 'earnings', 'revenue',
            'profit', 'loss', 'nasdaq', 'nyse', 'kospi', 'kosdaq',
            '주식', '시장', '투자', '수익', '실적', '매출'
        ]
        
        for keyword in financial_keywords:
            if keyword in text:
                score += 0.1
        
        # 종목 언급 점수
        score += len(news_item.symbols_mentioned) * 0.3
        
        # 소스 신뢰도 점수
        trusted_sources = ['bloomberg', 'reuters', 'wall street journal', 'financial times', '한국경제', '매일경제']
        for source in trusted_sources:
            if source in news_item.source.lower():
                score += 0.2
                break
        
        return min(1.0, score)
    
    def _calculate_sentiment(self, text: str) -> float:
        """감성 분석"""
        try:
            # TextBlob을 사용한 기본 감성 분석
            blob = TextBlob(text)
            
            # 극성 점수 (-1 ~ 1)
            polarity = blob.sentiment.polarity
            
            # 주관성 고려한 조정
            subjectivity = blob.sentiment.subjectivity
            adjusted_sentiment = polarity * subjectivity
            
            return adjusted_sentiment
            
        except Exception as e:
            logger.warning(f"감성 분석 실패: {e}")
            return 0.0

class SentimentAnalyzer:
    """💭 감성 분석 엔진"""
    
    def __init__(self):
        """초기화"""
        self.positive_keywords = [
            'bull', 'bullish', 'rise', 'rising', 'up', 'gain', 'growth', 'profit',
            'beat', 'exceed', 'strong', 'robust', 'boom', 'surge', 'rally',
            '상승', '급등', '호재', '긍정', '강세', '호황'
        ]
        
        self.negative_keywords = [
            'bear', 'bearish', 'fall', 'falling', 'down', 'loss', 'decline', 'drop',
            'crash', 'plunge', 'weak', 'poor', 'recession', 'crisis',
            '하락', '급락', '악재', '부정', '약세', '불황'
        ]
        
        logger.info("💭 감성 분석 엔진 초기화 완료")
    
    def analyze_market_sentiment(self, news_items: List[NewsItem]) -> MarketSentiment:
        """시장 감성 종합 분석"""
        
        if not news_items:
            return MarketSentiment(
                overall_sentiment=0.0,
                positive_news_count=0,
                negative_news_count=0,
                neutral_news_count=0,
                key_topics=[],
                sentiment_trend="STABLE",
                confidence=0.0
            )
        
        # 감성 점수별 분류
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        sentiment_scores = []
        
        for news in news_items:
            if news.sentiment_score > 0.1:
                positive_count += 1
            elif news.sentiment_score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
            
            sentiment_scores.append(news.sentiment_score)
        
        # 전체 감성 점수 계산 (가중평균)
        if sentiment_scores:
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        else:
            overall_sentiment = 0.0
        
        # 주요 토픽 추출
        key_topics = self._extract_key_topics(news_items)
        
        # 감성 트렌드 분석
        sentiment_trend = self._analyze_sentiment_trend(news_items)
        
        # 신뢰도 계산
        confidence = min(1.0, len(news_items) / 20.0)  # 뉴스 개수 기반
        
        return MarketSentiment(
            overall_sentiment=overall_sentiment,
            positive_news_count=positive_count,
            negative_news_count=negative_count,
            neutral_news_count=neutral_count,
            key_topics=key_topics,
            sentiment_trend=sentiment_trend,
            confidence=confidence
        )
    
    def _extract_key_topics(self, news_items: List[NewsItem]) -> List[str]:
        """주요 토픽 추출"""
        
        # 간단한 키워드 빈도 분석
        word_counts = {}
        
        for news in news_items:
            words = (news.title + " " + news.content).lower().split()
            
            for word in words:
                if len(word) > 3 and word.isalpha():
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # 상위 키워드 선정
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [word for word, count in top_keywords if count >= 2]
    
    def _analyze_sentiment_trend(self, news_items: List[NewsItem]) -> str:
        """감성 트렌드 분석"""
        
        if len(news_items) < 10:
            return "STABLE"
        
        # 최근 뉴스와 이전 뉴스 비교
        sorted_news = sorted(news_items, key=lambda x: x.published_date)
        
        recent_news = sorted_news[-len(sorted_news)//2:]  # 최근 절반
        older_news = sorted_news[:len(sorted_news)//2]    # 이전 절반
        
        recent_sentiment = sum(news.sentiment_score for news in recent_news) / len(recent_news)
        older_sentiment = sum(news.sentiment_score for news in older_news) / len(older_news)
        
        sentiment_change = recent_sentiment - older_sentiment
        
        if sentiment_change > 0.1:
            return "IMPROVING"
        elif sentiment_change < -0.1:
            return "DECLINING"
        else:
            return "STABLE"

class NewsAnalyzer:
    """📊 통합 뉴스 분석 엔진"""
    
    def __init__(self):
        """초기화"""
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        logger.info("📊 뉴스 분석 엔진 초기화 완료")
    
    def analyze_stocks_news(self, symbols: List[str]) -> Dict[str, Any]:
        """종목별 뉴스 분석"""
        
        logger.info(f"📰 {len(symbols)}개 종목 뉴스 분석 시작")
        
        try:
            # 1. 뉴스 수집
            news_items = self.news_collector.collect_news(symbols)
            
            # 2. 전체 시장 감성 분석
            market_sentiment = self.sentiment_analyzer.analyze_market_sentiment(news_items)
            
            # 3. 종목별 감성 분석
            stock_sentiments = {}
            
            for symbol in symbols:
                # 해당 종목 관련 뉴스 필터링
                stock_news = [news for news in news_items if symbol in news.symbols_mentioned]
                
                if stock_news:
                    stock_sentiment = self.sentiment_analyzer.analyze_market_sentiment(stock_news)
                    stock_sentiments[symbol] = {
                        'sentiment_score': stock_sentiment.overall_sentiment,
                        'news_count': len(stock_news),
                        'positive_ratio': stock_sentiment.positive_news_count / len(stock_news),
                        'recent_news': [
                            {
                                'title': news.title,
                                'sentiment': news.sentiment_score,
                                'source': news.source,
                                'url': news.url
                            } for news in stock_news[:3]  # 최신 3개
                        ]
                    }
            
            result = {
                'market_sentiment': {
                    'overall_score': market_sentiment.overall_sentiment,
                    'positive_count': market_sentiment.positive_news_count,
                    'negative_count': market_sentiment.negative_news_count,
                    'neutral_count': market_sentiment.neutral_news_count,
                    'trend': market_sentiment.sentiment_trend,
                    'confidence': market_sentiment.confidence,
                    'key_topics': market_sentiment.key_topics
                },
                'stock_sentiments': stock_sentiments,
                'total_news_count': len(news_items),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ 뉴스 분석 완료 - 총 {len(news_items)}개 뉴스 처리")
            return result
            
        except Exception as e:
            logger.error(f"❌ 뉴스 분석 실패: {e}")
            return {
                'market_sentiment': {'overall_score': 0.0, 'trend': 'STABLE'},
                'stock_sentiments': {},
                'total_news_count': 0,
                'analysis_timestamp': datetime.now().isoformat()
            }

if __name__ == "__main__":
    print("📰 실시간 뉴스 분석 엔진 v1.0")
    print("=" * 50)
    
    # 테스트
    analyzer = NewsAnalyzer()
    
    test_symbols = ["AAPL", "TSLA", "005930.KS"]
    result = analyzer.analyze_stocks_news(test_symbols)
    
    print(f"\n📊 분석 결과:")
    print(f"  • 전체 뉴스: {result['total_news_count']}개")
    print(f"  • 시장 감성: {result['market_sentiment']['overall_score']:.3f}")
    print(f"  • 감성 트렌드: {result['market_sentiment']['trend']}")
    print(f"  • 종목별 분석: {len(result['stock_sentiments'])}개")
    
    print("\n✅ 뉴스 분석 엔진 테스트 완료!") 