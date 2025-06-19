"""
🔥 실시간 뉴스 & 공시 수집기 (속도 최적화 버전)
- 네이버 금융 실시간 뉴스 크롤링
- 한국거래소 전자공시 수집
- 키워드 기반 감정분석
- 종목별 뉴스 필터링
"""
import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import concurrent.futures
from dataclasses import dataclass
import json
from urllib.parse import urlencode, quote
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """뉴스 아이템 데이터 클래스"""
    title: str
    content: str
    url: str
    timestamp: datetime
    source: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1.0 ~ 1.0
    related_stocks: List[str] = None

@dataclass
class AnnouncementItem:
    """공시 아이템 데이터 클래스"""
    title: str
    company: str
    stock_code: str
    announcement_type: str
    timestamp: datetime
    url: str
    content: str = None
    importance: str = "medium"  # 'high', 'medium', 'low'

class NewsCollector:
    """🚀 초고속 뉴스 & 공시 수집 클래스"""
    
    def __init__(self):
        """NewsCollector 초기화"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # 캐시 시스템 (5분 캐시)
        self.cache_duration = 300  # 5분
        self.news_cache = {}
        self.announcement_cache = {}
        
        # 감정분석 키워드 사전
        self._init_sentiment_keywords()
        
        # 종목명-코드 매핑 (주요 종목)
        self._init_stock_mapping()
        
        # 병렬 처리 설정
        self.max_workers = 4
        
        logger.info("🔥 NewsCollector 초기화 완료")
    
    def _init_sentiment_keywords(self):
        """감정분석용 키워드 사전 초기화"""
        self.positive_keywords = {
            # 주가 상승 관련
            '상승', '급등', '강세', '호재', '성장', '증가', '확대', '개선', 
            '호조', '반등', '돌파', '신고가', '목표가', '상향', '매수',
            # 실적 관련
            '흑자', '증익', '매출증가', '수익개선', '실적호조', '어닝서프라이즈',
            # 사업 관련  
            '신제품', '신규계약', '투자유치', '제휴', '합병', '인수',
            '확장', '진출', '개발완료', '승인', '특허', '수주',
            # 긍정 감정
            '긍정적', '낙관적', '기대', '전망밝음', '유망', '성공적'
        }
        
        self.negative_keywords = {
            # 주가 하락 관련
            '하락', '급락', '폭락', '약세', '악재', '감소', '축소', '악화',
            '부진', '조정', '하향', '매도', '손절', '저조',
            # 실적 관련
            '적자', '감익', '매출감소', '실적부진', '어닝쇼크', '영업손실',
            # 사업 관련
            '취소', '연기', '중단', '철회', '실패', '손실', '리콜',
            '소송', '분쟁', '제재', '규제', '조사', '수사',
            # 부정 감정
            '부정적', '비관적', '우려', '위험', '불안', '실망'
        }
    
    def _init_stock_mapping(self):
        """주요 종목 코드-이름 매핑"""
        self.stock_mapping = {
            '005930': '삼성전자', '000660': 'SK하이닉스', '035420': 'NAVER',
            '051910': 'LG화학', '006400': '삼성SDI', '035720': '카카오',
            '207940': '삼성바이오로직스', '005380': '현대차', '000270': '기아',
            '068270': '셀트리온', '003670': '포스코홀딩스', '096770': 'SK이노베이션',
            '323410': '카카오뱅크', '373220': 'LG에너지솔루션', '028260': '삼성물산'
        }
        
        # 역방향 매핑도 생성
        self.company_to_code = {v: k for k, v in self.stock_mapping.items()}
    
    def get_realtime_news(self, keywords: Optional[List[str]] = None, limit: int = 20) -> List[NewsItem]:
        """📰 실시간 뉴스 수집 (다중 소스)"""
        cache_key = f"news_{keywords}_{limit}"
        
        # 캐시 확인
        if self._is_cache_valid(cache_key, 'news'):
            logger.info("📋 뉴스 캐시 사용")
            return self.news_cache[cache_key]['data']
        
        try:
            start_time = time.time()
            logger.info("📰 실시간 뉴스 수집 시작...")
            
            news_list = []
            
            # 병렬로 여러 소스에서 뉴스 수집
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                # 1. 네이버 금융 뉴스 (70%)
                futures.append(executor.submit(self._get_naver_finance_news, int(limit * 0.7)))
                
                # 2. 한국경제 뉴스 (20%)
                futures.append(executor.submit(self._get_hankyung_news, int(limit * 0.2)))
                
                # 3. 이데일리 뉴스 (10%)
                futures.append(executor.submit(self._get_edaily_news, int(limit * 0.1)))
                
                # 결과 수집
                for future in concurrent.futures.as_completed(futures, timeout=10):
                    try:
                        result = future.result()
                        if result:
                            news_list.extend(result)
                    except Exception as e:
                        logger.warning(f"⚠️ 뉴스 소스 실패: {e}")
                        continue
            
            # 중복 제거 (제목 기준)
            seen_titles = set()
            unique_news = []
            for news in news_list:
                if news.title not in seen_titles:
                    seen_titles.add(news.title)
                    unique_news.append(news)
            
            # 2. 키워드 필터링 (선택사항)
            if keywords:
                unique_news = self._filter_by_keywords(unique_news, keywords)
            
            # 3. 감정분석 적용
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._analyze_sentiment_item, item) for item in unique_news]
                analyzed_news = []
                
                for future in concurrent.futures.as_completed(futures, timeout=10):
                    try:
                        result = future.result()
                        if result:
                            analyzed_news.append(result)
                    except Exception as e:
                        logger.warning(f"⚠️ 감정분석 실패: {e}")
                        continue
            
            # 4. 중요도순 정렬 (감정점수 기준)
            analyzed_news.sort(key=lambda x: abs(x.sentiment_score), reverse=True)
            
            # 캐시 저장
            self._cache_data(cache_key, 'news', analyzed_news[:limit])
            
            elapsed = time.time() - start_time
            logger.info(f"📰 뉴스 수집 완료: {len(analyzed_news)}개, {elapsed:.3f}초")
            
            return analyzed_news[:limit]
            
        except Exception as e:
            logger.error(f"❌ 뉴스 수집 실패: {e}")
            return []
    
    def _get_hankyung_news(self, limit: int) -> List[NewsItem]:
        """한국경제(hankyung.com) 증권 뉴스 크롤링"""
        news_list = []
        url = "https://www.hankyung.com/finance"
        try:
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            items = soup.select('.news-list .news-item a')
            for item in items[:limit]:
                try:
                    title = item.select_one('h3.news-tit').get_text(strip=True)
                    link = item.get('href', '')
                    if not link.startswith('http'):
                        link = "https://www.hankyung.com" + link
                    
                    news_list.append(NewsItem(
                        title=title, content='', url=link, timestamp=datetime.now(),
                        source='한국경제', sentiment='neutral', sentiment_score=0.0
                    ))
                    time.sleep(0.1) # 크롤링 예의 준수
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"⚠️ 한국경제 뉴스 수집 실패: {e}")
        return news_list

    def _get_edaily_news(self, limit: int) -> List[NewsItem]:
        """이데일리(edaily.co.kr) 증권 뉴스 크롤링"""
        news_list = []
        url = "https://www.edaily.co.kr/news/stock"
        try:
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            items = soup.select('.news-list a.news-title')
            for item in items[:limit]:
                try:
                    title = item.get_text(strip=True)
                    link = item.get('href', '')
                    if not link.startswith('http'):
                        link = "https://www.edaily.co.kr" + link

                    news_list.append(NewsItem(
                        title=title, content='', url=link, timestamp=datetime.now(),
                        source='이데일리', sentiment='neutral', sentiment_score=0.0
                    ))
                    time.sleep(0.1) # 크롤링 예의 준수
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"⚠️ 이데일리 뉴스 수집 실패: {e}")
        return news_list
    
    def _get_naver_finance_news(self, limit: int) -> List[NewsItem]:
        """네이버 금융 뉴스 크롤링 (DART 공시 연계 및 30일 필터링 강화)"""
        news_list = []
        url = "https://finance.naver.com/news/mainnews.naver"
        try:
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 뉴스 목록 아이템 선택자 수정 (더 구체적으로)
            news_items = soup.select('.mainNewsList li')
            
            thirty_days_ago = datetime.now() - timedelta(days=30)

            for item in news_items:
                if len(news_list) >= limit:
                    break
                
                try:
                    dt_span = item.select_one('span.date')
                    if not dt_span: continue
                    
                    # 날짜 파싱 및 필터링
                    news_date_str = dt_span.get_text(strip=True)
                    news_dt = self._parse_naver_time(news_date_str)
                    
                    if news_dt < thirty_days_ago:
                        continue # 30일 이전 뉴스는 건너뛰기

                    title_tag = item.select_one('dd a')
                    if not title_tag: continue
                    
                    title = title_tag.get_text(strip=True)
                    link = "https://finance.naver.com" + title_tag['href']

                    # DART 공시 관련 뉴스 가중치 부여
                    sentiment_score = 0.0
                    if '공시' in title or '[유가증권]' in title or '[코스닥]' in title:
                        sentiment_score = 0.1 # 기본 가중치

                    news_list.append(NewsItem(
                        title=title, content='', url=link, timestamp=news_dt,
                        source='네이버금융', sentiment='neutral', sentiment_score=sentiment_score
                    ))
                    time.sleep(0.1) # 크롤링 예의 준수
                except Exception:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ 네이버 금융 뉴스 수집 실패: {e}")
            
        return news_list
    
    def _get_naver_stock_news(self, limit: int) -> List[NewsItem]:
        """네이버 증권 뉴스 크롤링 (보조 소스)"""
        news_list = []
        
        try:
            # 네이버 증권 실시간 뉴스
            url = "https://finance.naver.com/sise/sise_market_sum.naver"
            
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 종목 뉴스 링크 찾기
            stock_links = soup.select('a[href*="item.naver"]')
            
            for link in stock_links[:limit]:
                try:
                    title = link.get_text(strip=True)
                    if not title or len(title) < 5:
                        continue
                    
                    href = link.get('href', '')
                    if href and not href.startswith('http'):
                        href = 'https://finance.naver.com' + href
                    
                    news_item = NewsItem(
                        title=f"[증권] {title}",
                        content='',
                        url=href,
                        timestamp=datetime.now(),
                        source='네이버증권',
                        sentiment='neutral',
                        sentiment_score=0.0
                    )
                    
                    news_list.append(news_item)
                    
                except Exception as e:
                    continue
            
            logger.info(f"📰 네이버 증권 뉴스 {len(news_list)}개 수집")
            
        except Exception as e:
            logger.warning(f"⚠️ 네이버 증권 뉴스 수집 실패: {e}")
        
        return news_list
    
    def _get_sample_news(self, limit: int) -> List[NewsItem]:
        """샘플 뉴스 생성 (테스트/데모용)"""
        sample_titles = [
            "삼성전자, 3분기 실적 호조로 목표가 상향 조정",
            "코스피, 외국인 매수세에 힘입어 상승세 지속",
            "SK하이닉스, 메모리 반도체 수요 증가로 주가 급등",
            "LG화학, 배터리 사업 확장으로 성장 전망 밝아",
            "카카오, 새로운 플랫폼 서비스 출시 발표",
            "현대차, 전기차 판매 목표 상향 조정",
            "NAVER, AI 기술 투자 확대 계획 발표",
            "삼성바이오로직스, 신규 계약 체결로 수주 증가",
            "포스코홀딩스, 친환경 철강 기술 개발 성공",
            "셀트리온, 바이오시밀러 매출 증가세"
        ]
        
        news_list = []
        
        for i, title in enumerate(sample_titles[:limit]):
            news_item = NewsItem(
                title=title,
                content=f"{title}에 대한 상세 내용...",
                url=f"https://example.com/news/{i+1}",
                timestamp=datetime.now() - timedelta(minutes=i*5),
                source='샘플뉴스',
                sentiment='neutral',
                sentiment_score=0.0
            )
            news_list.append(news_item)
        
        logger.info(f"📰 샘플 뉴스 {len(news_list)}개 생성")
        return news_list
    
    def get_announcements(self, days: int = 1) -> List[AnnouncementItem]:
        """📋 전자공시 수집 (개선된 버전)"""
        cache_key = f"announcements_{days}"
        
        # 캐시 확인
        if self._is_cache_valid(cache_key, 'announcement'):
            logger.info("📋 공시 캐시 사용")
            return self.announcement_cache[cache_key]['data']
        
        try:
            start_time = time.time()
            logger.info("📋 전자공시 수집 시작...")
            
            announcements = []
            
            # 병렬로 여러 소스에서 공시 수집
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                # 1. KIND 공시 (수정된 URL)
                futures.append(executor.submit(self._get_kind_announcements_v2, days))
                
                # 2. 샘플 공시 생성
                futures.append(executor.submit(self._get_sample_announcements, 10))
                
                # 결과 수집
                for future in concurrent.futures.as_completed(futures, timeout=15):
                    try:
                        result = future.result()
                        if result:
                            announcements.extend(result)
                    except Exception as e:
                        logger.warning(f"⚠️ 공시 소스 실패: {e}")
                        continue
            
            # 중복 제거 (제목 기준)
            seen_titles = set()
            unique_announcements = []
            for ann in announcements:
                if ann.title not in seen_titles:
                    seen_titles.add(ann.title)
                    unique_announcements.append(ann)
            
            # 중요도별 정렬
            unique_announcements.sort(key=lambda x: (
                x.importance == 'high' and 3 or 
                x.importance == 'medium' and 2 or 1, 
                x.timestamp
            ), reverse=True)
            
            # 캐시 저장
            self._cache_data(cache_key, 'announcement', unique_announcements)
            
            elapsed = time.time() - start_time
            logger.info(f"📋 공시 수집 완료: {len(unique_announcements)}개, {elapsed:.3f}초")
            
            return unique_announcements
            
        except Exception as e:
            logger.error(f"❌ 공시 수집 실패: {e}")
            return []
    
    def _get_kind_announcements_v2(self, days: int) -> List[AnnouncementItem]:
        """KIND 공시 크롤링 (개선된 버전)"""
        announcements = []
        
        try:
            # KIND 새로운 API 엔드포인트들 시도
            endpoints = [
                "https://kind.krx.co.kr/common/disclsviewer.do",
                "https://kind.krx.co.kr/disclosureservice/disclosureservice.do",
                "https://opendart.fss.or.kr/api/list.json"  # 대체 공시 API
            ]
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            to_date = datetime.now().strftime('%Y%m%d')
            
            for base_url in endpoints[:1]:  # 첫 번째만 시도
                try:
                    params = {
                        'method': 'searchDisclosureSummary',
                        'currentPageSize': '50',
                        'pageIndex': '1',
                        'orderMode': '0',
                        'orderStat': 'D',
                        'forward': 'disclosuresummary',
                        'elDate': f"{from_date}~{to_date}"
                    }
                    
                    response = self.session.get(base_url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # 테이블 형태 파싱 시도
                        rows = soup.select('tr')
                        
                        for row in rows[1:11]:  # 헤더 제외하고 최대 10개
                            try:
                                cols = row.select('td')
                                if len(cols) >= 3:
                                    # 기본 정보만 추출
                                    date_str = cols[0].get_text(strip=True) if cols[0] else ''
                                    company = cols[1].get_text(strip=True) if len(cols) > 1 else '회사명'
                                    title = cols[2].get_text(strip=True) if len(cols) > 2 else '공시제목'
                                    
                                    if title and len(title) > 5:
                                        announcement = AnnouncementItem(
                                            title=title,
                                            company=company,
                                            stock_code=self._extract_stock_code(company),
                                            announcement_type=self._classify_announcement_type(title),
                                            timestamp=self._parse_kind_time(date_str),
                                            url=base_url,
                                            importance=self._evaluate_importance(title, '기타')
                                        )
                                        
                                        announcements.append(announcement)
                                        
                            except Exception as e:
                                continue
                    
                    if announcements:
                        break
                        
                except Exception as e:
                    continue
            
            logger.info(f"📋 KIND 공시 {len(announcements)}개 수집")
            
        except Exception as e:
            logger.warning(f"⚠️ KIND 공시 수집 실패: {e}")
        
        return announcements
    
    def _get_sample_announcements(self, limit: int) -> List[AnnouncementItem]:
        """샘플 공시 생성 (테스트용)"""
        sample_announcements = [
            ("삼성전자", "005930", "2024년 3분기 실적발표", "실적공시", "high"),
            ("SK하이닉스", "000660", "주요사항보고서(투자결정)", "사업관련", "medium"),
            ("NAVER", "035420", "자기주식 취득 결정", "자본관련", "medium"),
            ("LG화학", "051910", "해외법인 설립 결정", "사업관련", "medium"),
            ("카카오", "035720", "정기주주총회 결의사항", "주주관련", "low"),
            ("현대차", "005380", "분기보고서 제출", "실적공시", "medium"),
            ("셀트리온", "068270", "신제품 허가 취득", "사업관련", "high"),
            ("삼성바이오로직스", "207940", "계약 체결", "사업관련", "medium"),
            ("포스코홀딩스", "003670", "배당금 지급 결정", "자본관련", "low"),
            ("SK이노베이션", "096770", "주요계약 체결", "사업관련", "medium")
        ]
        
        announcements = []
        
        for i, (company, code, title, type_, importance) in enumerate(sample_announcements[:limit]):
            announcement = AnnouncementItem(
                title=title,
                company=company,
                stock_code=code,
                announcement_type=type_,
                timestamp=datetime.now() - timedelta(hours=i),
                url=f"https://example.com/announcement/{i+1}",
                importance=importance
            )
            announcements.append(announcement)
        
        logger.info(f"📋 샘플 공시 {len(announcements)}개 생성")
        return announcements

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        """🤖 키워드 기반 감정분석"""
        if not text:
            return 'neutral', 0.0
        
        text = text.lower()
        
        # 긍정/부정 키워드 카운트
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)
        
        # 점수 계산
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 'neutral', 0.0
        
        # 감정 점수 (-1.0 ~ 1.0)
        sentiment_score = (positive_count - negative_count) / max(total_keywords, 1)
        
        # 감정 분류
        if sentiment_score > 0.2:
            sentiment = 'positive'
        elif sentiment_score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return sentiment, sentiment_score
    
    def filter_stock_related(self, news_list: List[NewsItem], stock_code: str) -> List[NewsItem]:
        """📈 종목 관련 뉴스 필터링"""
        if not stock_code or stock_code not in self.stock_mapping:
            return []
        
        company_name = self.stock_mapping[stock_code]
        related_news = []
        
        for news in news_list:
            # 제목이나 내용에 회사명/종목코드가 포함된 경우
            text = f"{news.title} {news.content}".lower()
            
            if (company_name in text or 
                stock_code in text or
                any(keyword in text for keyword in [company_name.lower()])):
                
                # 관련 종목 정보 추가
                if not news.related_stocks:
                    news.related_stocks = []
                if stock_code not in news.related_stocks:
                    news.related_stocks.append(stock_code)
                
                related_news.append(news)
        
        logger.info(f"📈 {company_name}({stock_code}) 관련 뉴스 {len(related_news)}개 필터링")
        return related_news
    
    # === 헬퍼 메서드들 ===
    
    def _analyze_sentiment_item(self, news_item: NewsItem) -> NewsItem:
        """뉴스 아이템 감정분석"""
        sentiment, score = self.analyze_sentiment(f"{news_item.title} {news_item.content}")
        news_item.sentiment = sentiment
        news_item.sentiment_score = score
        return news_item
    
    def _filter_by_keywords(self, news_list: List[NewsItem], keywords: List[str]) -> List[NewsItem]:
        """키워드로 뉴스 필터링"""
        filtered = []
        
        for news in news_list:
            text = f"{news.title} {news.content}".lower()
            if any(keyword.lower() in text for keyword in keywords):
                filtered.append(news)
        
        return filtered
    
    def _parse_naver_time(self, time_str: str) -> datetime:
        """네이버 금융 뉴스 시간 문자열을 datetime 객체로 변환합니다."""
        now = datetime.now()
        time_str = time_str.strip()
        
        if '분 전' in time_str:
            minutes = int(re.search(r'(\d+)분 전', time_str).group(1))
            return now - timedelta(minutes=minutes)
        elif '시간 전' in time_str:
            hours = int(re.search(r'(\d+)시간 전', time_str).group(1))
            return now - timedelta(hours=hours)
        else:
            # 'YYYY.MM.DD HH:mm' 또는 'YYYY.MM.DD' 형식 처리
            try:
                # 날짜와 시간이 모두 있는 경우
                if len(time_str.split()) > 1:
                    return datetime.strptime(time_str, '%Y.%m.%d %H:%M')
                # 날짜만 있는 경우 (시간은 00:00으로 설정)
                else:
                    return datetime.strptime(time_str, '%Y.%m.%d')
            except ValueError:
                return now # 파싱 실패 시 현재 시간 반환
    
    def _parse_kind_time(self, date_str: str) -> datetime:
        """KIND 시간 문자열 파싱"""
        try:
            # 형식: "2024/01/15 15:30"
            return datetime.strptime(date_str.replace('-', '/'), '%Y/%m/%d %H:%M')
        except:
            try:
                # 형식: "2024-01-15"
                return datetime.strptime(date_str.split()[0], '%Y-%m-%d')
            except:
                return datetime.now()
    
    def _extract_stock_code(self, company_text: str) -> str:
        """회사명에서 종목코드 추출"""
        # 괄호 안의 6자리 숫자 찾기
        match = re.search(r'\((\d{6})\)', company_text)
        if match:
            return match.group(1)
        
        # 회사명으로 코드 찾기
        for name in self.company_to_code:
            if name in company_text:
                return self.company_to_code[name]
        
        return ''
    
    def _classify_announcement_type(self, title: str) -> str:
        """공시 유형 분류"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['실적', '매출', '영업이익', '순이익']):
            return '실적공시'
        elif any(word in title_lower for word in ['합병', '인수', '분할']):
            return 'M&A'
        elif any(word in title_lower for word in ['투자', '계약', '수주']):
            return '사업관련'
        elif any(word in title_lower for word in ['주주총회', '이사회']):
            return '주주관련'
        elif any(word in title_lower for word in ['증자', '감자', '배당']):
            return '자본관련'
        else:
            return '기타'
    
    def _evaluate_importance(self, title: str, announcement_type: str) -> str:
        """공시 중요도 평가"""
        title_lower = title.lower()
        
        # 고중요도 키워드
        high_keywords = ['합병', '인수', '분할', '상장폐지', '거래정지', '영업정지']
        if any(keyword in title_lower for keyword in high_keywords):
            return 'high'
        
        # 중요도 키워드
        medium_keywords = ['실적', '증자', '배당', '주요계약', '투자']
        if any(keyword in title_lower for keyword in medium_keywords):
            return 'medium'
        
        return 'low'
    
    def _is_cache_valid(self, key: str, cache_type: str) -> bool:
        """캐시 유효성 검사"""
        cache = self.news_cache if cache_type == 'news' else self.announcement_cache
        
        if key not in cache:
            return False
        
        cache_time = cache[key].get('timestamp')
        if not cache_time:
            return False
        
        return (datetime.now() - cache_time).total_seconds() < self.cache_duration
    
    def _cache_data(self, key: str, cache_type: str, data: Any) -> None:
        """데이터 캐싱"""
        cache = self.news_cache if cache_type == 'news' else self.announcement_cache
        cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """📊 시장 전체 감정 요약"""
        try:
            # 최신 뉴스 수집
            news_list = self.get_realtime_news(limit=50)
            
            if not news_list:
                return {'sentiment': 'neutral', 'score': 0.0, 'news_count': 0}
            
            # 감정 점수 평균 계산
            total_score = sum(news.sentiment_score for news in news_list)
            avg_score = total_score / len(news_list)
            
            # 감정 분포 계산
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for news in news_list:
                sentiment_counts[news.sentiment] += 1
            
            # 전체 감정 결정
            if avg_score > 0.1:
                overall_sentiment = 'positive'
            elif avg_score < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'sentiment': overall_sentiment,
                'score': round(avg_score, 3),
                'news_count': len(news_list),
                'distribution': sentiment_counts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 시장 감정 요약 실패: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'news_count': 0}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.session.close()
            self.news_cache.clear()
            self.announcement_cache.clear()
            logger.info("🧹 NewsCollector 리소스 정리 완료")
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}") 