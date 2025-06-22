#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📰 실시간 뉴스 데이터 처리 시스템
Gemini 1.5 Flash 모델 전용 뉴스 분석 및 가공

Features:
- 인베스팅닷컴 실시간 뉴스 수집
- Gemini 1.5 Flash 최적화 데이터 가공
- 주식 관련 뉴스 필터링 및 분류
- 감정 분석 및 영향도 평가
- 한국어 번역 및 요약
"""

import asyncio
import json
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from bs4 import BeautifulSoup
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv
import os

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCategory(Enum):
    """뉴스 카테고리"""
    MARKET = "시장"
    COMPANY = "기업"
    ECONOMIC = "경제"
    TECHNOLOGY = "기술"
    POLICY = "정책"
    GLOBAL = "해외"
    OTHER = "기타"

class SentimentType(Enum):
    """감정 분석 타입"""
    VERY_POSITIVE = "매우긍정"
    POSITIVE = "긍정"
    NEUTRAL = "중립"
    NEGATIVE = "부정"
    VERY_NEGATIVE = "매우부정"

@dataclass
class NewsData:
    """뉴스 데이터 구조"""
    title: str
    content: str
    url: str
    published_time: datetime
    source: str
    category: NewsCategory
    sentiment: SentimentType
    impact_score: float  # 0-100
    related_stocks: List[str]
    keywords: List[str]
    summary: str
    translated_title: str = ""
    translated_content: str = ""

class InvestingNewsCollector:
    """인베스팅닷컴 뉴스 수집기"""
    
    def __init__(self):
        self.base_url = "https://www.investing.com"
        self.rss_feeds = {
            "general": "https://www.investing.com/rss/news.rss",
            "stock_news": "https://www.investing.com/rss/news_285.rss",
            "economic": "https://www.investing.com/rss/news_95.rss",
            "forex": "https://www.investing.com/rss/news_1.rss"
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    async def collect_latest_news(self, hours_back: int = 24, max_articles: int = 50) -> List[Dict[str, Any]]:
        """최신 뉴스 수집"""
        logger.info(f"📰 인베스팅닷컴에서 최근 {hours_back}시간 뉴스 수집 시작")
        
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for feed_name, rss_url in self.rss_feeds.items():
            try:
                logger.info(f"📡 {feed_name} RSS 피드 수집 중...")
                feed_news = await self._parse_rss_feed(rss_url, cutoff_time)
                all_news.extend(feed_news)
                
                # API 호출 제한 방지
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"RSS 피드 수집 실패 ({feed_name}): {e}")
        
        # 중복 제거 및 시간순 정렬
        unique_news = self._remove_duplicates(all_news)
        sorted_news = sorted(unique_news, key=lambda x: x['published_time'], reverse=True)
        
        # 최대 기사 수 제한
        final_news = sorted_news[:max_articles]
        
        logger.info(f"✅ 총 {len(final_news)}개 뉴스 수집 완료")
        return final_news
    
    async def _parse_rss_feed(self, rss_url: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """RSS 피드 파싱"""
        try:
            # RSS 피드 파싱
            feed = await asyncio.to_thread(feedparser.parse, rss_url)
            
            if not feed.entries:
                return []
            
            news_list = []
            
            for entry in feed.entries:
                try:
                    # 발행 시간 파싱
                    pub_time = self._parse_publish_time(entry)
                    
                    # 시간 필터링
                    if pub_time < cutoff_time:
                        continue
                    
                    # 뉴스 데이터 구성
                    news_item = {
                        'title': entry.title,
                        'url': entry.link,
                        'published_time': pub_time,
                        'source': 'Investing.com',
                        'description': getattr(entry, 'description', ''),
                        'content': await self._extract_full_content(entry.link)
                    }
                    
                    news_list.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"뉴스 항목 처리 실패: {e}")
                    continue
            
            return news_list
            
        except Exception as e:
            logger.error(f"RSS 피드 파싱 실패: {e}")
            return []
    
    async def _extract_full_content(self, url: str) -> str:
        """전체 기사 내용 추출"""
        try:
            response = await asyncio.to_thread(
                self.session.get, 
                url, 
                timeout=10
            )
            
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 인베스팅닷컴 기사 본문 선택자
            content_selectors = [
                '.articlePage p',
                '.WYSIWYG p',
                '.InstrumentPageContentContainer p',
                'article p'
            ]
            
            content_text = ""
            for selector in content_selectors:
                paragraphs = soup.select(selector)
                if paragraphs:
                    content_text = ' '.join([p.get_text().strip() for p in paragraphs])
                    break
            
            return content_text[:2000]  # 내용 길이 제한
            
        except Exception as e:
            logger.warning(f"기사 내용 추출 실패 ({url}): {e}")
            return ""
    
    def _parse_publish_time(self, entry) -> datetime:
        """발행 시간 파싱"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'published'):
                # 다양한 시간 형식 처리
                time_str = entry.published
                return self._parse_time_string(time_str)
            else:
                return datetime.now()
        except:
            return datetime.now()
    
    def _parse_time_string(self, time_str: str) -> datetime:
        """시간 문자열 파싱"""
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except:
                continue
        
        return datetime.now()
    
    def _remove_duplicates(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 뉴스 제거"""
        seen_titles = set()
        unique_news = []
        
        for news in news_list:
            title_key = news['title'].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)
        
        return unique_news

class GeminiNewsProcessor:
    """Gemini 1.5 Flash 뉴스 처리기"""
    
    def __init__(self):
        # Gemini 1.5 Flash 모델로 고정
        self.model_name = "gemini-1.5-flash"
        self.api_key = os.getenv('GEMINI_API_KEY', '')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")
        
        # Gemini 설정
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # 일관된 분석을 위해 낮은 온도
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
                candidate_count=1
            )
        )
        
        logger.info(f"🤖 Gemini {self.model_name} 모델 초기화 완료")
    
    async def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[NewsData]:
        """뉴스 배치 처리"""
        logger.info(f"🔄 {len(news_list)}개 뉴스 Gemini 1.5 Flash로 처리 시작")
        
        processed_news = []
        
        for i, news_item in enumerate(news_list, 1):
            try:
                logger.info(f"📰 뉴스 처리 중... ({i}/{len(news_list)})")
                
                processed = await self._process_single_news(news_item)
                if processed:
                    processed_news.append(processed)
                
                # API 호출 제한 방지
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"뉴스 처리 실패: {e}")
                continue
        
        logger.info(f"✅ {len(processed_news)}개 뉴스 처리 완료")
        return processed_news
    
    async def _process_single_news(self, news_item: Dict[str, Any]) -> Optional[NewsData]:
        """단일 뉴스 처리"""
        try:
            # Gemini에게 최적화된 프롬프트 생성
            prompt = self._create_analysis_prompt(news_item)
            
            # Gemini 1.5 Flash 분석 실행
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            if not response or not response.text:
                return None
            
            # 응답 파싱
            analysis_result = self._parse_gemini_response(response.text)
            
            # NewsData 객체 생성
            return self._create_news_data(news_item, analysis_result)
            
        except Exception as e:
            logger.error(f"단일 뉴스 처리 실패: {e}")
            return None
    
    def _create_analysis_prompt(self, news_item: Dict[str, Any]) -> str:
        """Gemini 1.5 Flash 최적화 분석 프롬프트"""
        title = news_item.get('title', '')
        content = news_item.get('content', news_item.get('description', ''))
        
        return f"""
🤖 **Gemini 1.5 Flash 전문 뉴스 분석**

다음 금융 뉴스를 전문 애널리스트 관점에서 분석해주세요:

**제목:** {title}
**내용:** {content}

**분석 요구사항:**
1. 뉴스 카테고리 분류 (시장/기업/경제/기술/정책/해외/기타)
2. 감정 분석 (매우긍정/긍정/중립/부정/매우부정)
3. 시장 영향도 점수 (0-100점)
4. 관련 주식 종목 추출 (티커 심볼)
5. 핵심 키워드 5개
6. 한국어 요약 (100자 이내)
7. 제목 한국어 번역
8. 투자 시사점

**응답 형식 (JSON):**
```json
{{
    "category": "카테고리",
    "sentiment": "감정분석결과",
    "impact_score": 85,
    "related_stocks": ["AAPL", "MSFT"],
    "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
    "korean_summary": "한국어 요약",
    "korean_title": "한국어 제목",
    "investment_insight": "투자 시사점",
    "confidence": 0.9
}}
```

반드시 JSON 형식으로만 응답해주세요.
"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Gemini 응답 파싱"""
        try:
            # JSON 추출
            import re
            
            # JSON 패턴 찾기
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if matches:
                    try:
                        result = json.loads(matches[0])
                        return self._validate_analysis_result(result)
                    except json.JSONDecodeError:
                        continue
            
            # JSON 파싱 실패 시 기본값 반환
            return self._create_fallback_analysis()
            
        except Exception as e:
            logger.warning(f"Gemini 응답 파싱 실패: {e}")
            return self._create_fallback_analysis()
    
    def _validate_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 검증"""
        # 기본값 설정
        defaults = {
            "category": "기타",
            "sentiment": "중립",
            "impact_score": 50,
            "related_stocks": [],
            "keywords": [],
            "korean_summary": "요약 생성 중...",
            "korean_title": "제목 번역 중...",
            "investment_insight": "분석 중...",
            "confidence": 0.7
        }
        
        # 누락된 필드 보완
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
        
        # 데이터 타입 검증
        try:
            result['impact_score'] = max(0, min(100, int(float(result.get('impact_score', 50)))))
            result['confidence'] = max(0.0, min(1.0, float(result.get('confidence', 0.7))))
        except (ValueError, TypeError):
            result['impact_score'] = 50
            result['confidence'] = 0.7
        
        return result
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """폴백 분석 결과"""
        return {
            "category": "기타",
            "sentiment": "중립",
            "impact_score": 50,
            "related_stocks": [],
            "keywords": ["뉴스", "분석"],
            "korean_summary": "AI 분석 처리 중...",
            "korean_title": "제목 처리 중...",
            "investment_insight": "추가 분석 필요",
            "confidence": 0.5
        }
    
    def _create_news_data(self, news_item: Dict[str, Any], analysis: Dict[str, Any]) -> NewsData:
        """NewsData 객체 생성"""
        # 카테고리 매핑
        category_map = {
            "시장": NewsCategory.MARKET,
            "기업": NewsCategory.COMPANY,
            "경제": NewsCategory.ECONOMIC,
            "기술": NewsCategory.TECHNOLOGY,
            "정책": NewsCategory.POLICY,
            "해외": NewsCategory.GLOBAL,
            "기타": NewsCategory.OTHER
        }
        
        # 감정 매핑
        sentiment_map = {
            "매우긍정": SentimentType.VERY_POSITIVE,
            "긍정": SentimentType.POSITIVE,
            "중립": SentimentType.NEUTRAL,
            "부정": SentimentType.NEGATIVE,
            "매우부정": SentimentType.VERY_NEGATIVE
        }
        
        return NewsData(
            title=news_item.get('title', ''),
            content=news_item.get('content', news_item.get('description', '')),
            url=news_item.get('url', ''),
            published_time=news_item.get('published_time', datetime.now()),
            source=news_item.get('source', 'Investing.com'),
            category=category_map.get(analysis.get('category', '기타'), NewsCategory.OTHER),
            sentiment=sentiment_map.get(analysis.get('sentiment', '중립'), SentimentType.NEUTRAL),
            impact_score=analysis.get('impact_score', 50),
            related_stocks=analysis.get('related_stocks', []),
            keywords=analysis.get('keywords', []),
            summary=analysis.get('korean_summary', ''),
            translated_title=analysis.get('korean_title', ''),
            translated_content=analysis.get('investment_insight', '')
        )

class NewsAnalysisSystem:
    """통합 뉴스 분석 시스템"""
    
    def __init__(self):
        self.collector = InvestingNewsCollector()
        self.processor = GeminiNewsProcessor()
        self.cache = {}
        
    async def analyze_latest_news(self, hours_back: int = 6, max_articles: int = 20) -> List[NewsData]:
        """최신 뉴스 분석"""
        logger.info(f"📰 최근 {hours_back}시간 뉴스 분석 시작")
        
        try:
            # 1. 뉴스 수집
            raw_news = await self.collector.collect_latest_news(hours_back, max_articles)
            
            if not raw_news:
                logger.warning("수집된 뉴스가 없습니다")
                return []
            
            # 2. Gemini 1.5 Flash로 분석
            processed_news = await self.processor.process_news_batch(raw_news)
            
            # 3. 중요도순 정렬
            sorted_news = sorted(
                processed_news, 
                key=lambda x: x.impact_score, 
                reverse=True
            )
            
            logger.info(f"✅ 총 {len(sorted_news)}개 뉴스 분석 완료")
            return sorted_news
            
        except Exception as e:
            logger.error(f"뉴스 분석 시스템 오류: {e}")
            return []
    
    def get_news_by_category(self, news_list: List[NewsData], category: NewsCategory) -> List[NewsData]:
        """카테고리별 뉴스 필터링"""
        return [news for news in news_list if news.category == category]
    
    def get_news_by_sentiment(self, news_list: List[NewsData], sentiment: SentimentType) -> List[NewsData]:
        """감정별 뉴스 필터링"""
        return [news for news in news_list if news.sentiment == sentiment]
    
    def get_high_impact_news(self, news_list: List[NewsData], min_score: float = 70) -> List[NewsData]:
        """고영향도 뉴스 필터링"""
        return [news for news in news_list if news.impact_score >= min_score]
    
    def get_stock_related_news(self, news_list: List[NewsData], stock_symbol: str) -> List[NewsData]:
        """특정 주식 관련 뉴스"""
        return [
            news for news in news_list 
            if stock_symbol.upper() in [s.upper() for s in news.related_stocks]
        ]
    
    def create_news_summary(self, news_list: List[NewsData]) -> Dict[str, Any]:
        """뉴스 요약 리포트"""
        if not news_list:
            return {"error": "분석할 뉴스가 없습니다"}
        
        # 카테고리별 통계
        category_stats = {}
        for category in NewsCategory:
            count = len(self.get_news_by_category(news_list, category))
            if count > 0:
                category_stats[category.value] = count
        
        # 감정 분석 통계
        sentiment_stats = {}
        for sentiment in SentimentType:
            count = len(self.get_news_by_sentiment(news_list, sentiment))
            if count > 0:
                sentiment_stats[sentiment.value] = count
        
        # 평균 영향도
        avg_impact = sum(news.impact_score for news in news_list) / len(news_list)
        
        # 상위 뉴스
        top_news = sorted(news_list, key=lambda x: x.impact_score, reverse=True)[:5]
        
        return {
            "총_뉴스_수": len(news_list),
            "평균_영향도": round(avg_impact, 1),
            "카테고리별_통계": category_stats,
            "감정_분석_통계": sentiment_stats,
            "상위_5개_뉴스": [
                {
                    "제목": news.translated_title or news.title,
                    "영향도": news.impact_score,
                    "감정": news.sentiment.value,
                    "카테고리": news.category.value,
                    "관련주식": news.related_stocks
                }
                for news in top_news
            ],
            "분석_시간": datetime.now().isoformat()
        }

# 테스트 함수
async def test_news_system():
    """뉴스 분석 시스템 테스트"""
    print("📰 Gemini 1.5 Flash 뉴스 분석 시스템 테스트")
    print("=" * 60)
    
    try:
        system = NewsAnalysisSystem()
        
        # 최신 뉴스 분석
        print("🔄 최신 뉴스 수집 및 분석 중...")
        news_list = await system.analyze_latest_news(hours_back=12, max_articles=10)
        
        if not news_list:
            print("❌ 분석할 뉴스가 없습니다")
            return
        
        print(f"✅ {len(news_list)}개 뉴스 분석 완료\n")
        
        # 뉴스 요약 리포트
        summary = system.create_news_summary(news_list)
        print("📊 뉴스 분석 요약:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 60)
        print("🏆 상위 3개 뉴스 상세:")
        
        for i, news in enumerate(news_list[:3], 1):
            print(f"\n{i}. {news.translated_title or news.title}")
            print(f"   📈 영향도: {news.impact_score}점")
            print(f"   😊 감정: {news.sentiment.value}")
            print(f"   📂 카테고리: {news.category.value}")
            print(f"   🏢 관련주식: {', '.join(news.related_stocks) if news.related_stocks else '없음'}")
            print(f"   📝 요약: {news.summary}")
            print(f"   🔗 URL: {news.url}")
        
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """메인 실행"""
    await test_news_system()

if __name__ == "__main__":
    asyncio.run(main()) 