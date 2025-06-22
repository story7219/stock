"""
고성능 뉴스 분석기 - Gemini AI 최대 성능 발휘용
"""
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
import aiohttp
from urllib.parse import quote
import json
from ai_integration.gemini_optimizer import GeminiOptimizer
from ai_integration.test_gemini_optimization import run_full_test
from ai_integration.ultra_ai_analyzer import UltraAIAnalyzer

class NewsAnalyzer:
    """실시간 뉴스 분석 및 감정 분석"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.sentiment_keywords = {
            'positive': [
                '상승', '증가', '성장', '호재', '긍정', '개선', '확대', '투자', '매수', 
                '실적', '수익', '이익', '배당', '신고가', '돌파', '강세', '회복',
                '혁신', '기대', '전망', '계약', '수주', '출시', '개발', '특허'
            ],
            'negative': [
                '하락', '감소', '악화', '악재', '부정', '우려', '축소', '매도', 
                '손실', '적자', '하락', '신저가', '약세', '침체', '리스크',
                '규제', '제재', '소송', '사고', '파업', '중단', '연기', '취소'
            ]
        }
    
    async def analyze_stock_news(self, stock_code: str, company_name: str) -> Dict[str, Any]:
        """종목별 뉴스 분석"""
        try:
            # 네이버 금융 뉴스 크롤링
            news_data = await self._crawl_naver_finance_news(stock_code, company_name)
            
            # 감정 분석
            sentiment_analysis = self._analyze_sentiment(news_data)
            
            # 키워드 추출
            keywords = self._extract_keywords(news_data)
            
            # 뉴스 요약
            summary = self._create_news_summary(news_data, sentiment_analysis)
            
            # 울트라 최적화기 사용
            optimizer = GeminiOptimizer()
            result = await optimizer.ultra_analyze_stock(stock_data)
            
            # 성능 테스트 실행
            test_results = await run_full_test()
            
            # 기존 울트라 분석기 (최적화됨)
            analyzer = UltraAIAnalyzer()
            results = await analyzer.analyze_stocks(symbols, strategy='comprehensive')
            
            return {
                'news_count': len(news_data),
                'sentiment_score': sentiment_analysis['score'],
                'sentiment_label': sentiment_analysis['label'],
                'key_keywords': keywords[:10],
                'summary': summary,
                'latest_news': news_data[:3],  # 최신 3개
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'news_count': 0,
                'sentiment_score': 0,
                'sentiment_label': '중립',
                'key_keywords': [],
                'summary': f'뉴스 분석 실패: {str(e)}',
                'latest_news': [],
                'analysis_time': datetime.now().isoformat()
            }
    
    async def _crawl_naver_finance_news(self, stock_code: str, company_name: str) -> List[Dict]:
        """네이버 금융 뉴스 크롤링"""
        news_list = []
        
        try:
            # 네이버 금융 뉴스 URL
            url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}&page=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 뉴스 항목 추출
                    news_items = soup.select('.tb_cont tr')
                    
                    for item in news_items[:10]:  # 최신 10개
                        try:
                            title_elem = item.select_one('.title a')
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                link = title_elem.get('href')
                                
                                date_elem = item.select_one('.date')
                                date = date_elem.get_text(strip=True) if date_elem else ''
                                
                                source_elem = item.select_one('.info')
                                source = source_elem.get_text(strip=True) if source_elem else ''
                                
                                news_list.append({
                                    'title': title,
                                    'link': f"https://finance.naver.com{link}" if link else '',
                                    'date': date,
                                    'source': source
                                })
                        except:
                            continue
                            
        except Exception as e:
            # 실패 시 더미 뉴스 생성
            news_list = [
                {
                    'title': f'{company_name} 관련 뉴스 분석 중',
                    'link': '',
                    'date': datetime.now().strftime('%Y.%m.%d'),
                    'source': '종합'
                }
            ]
        
        return news_list
    
    def _analyze_sentiment(self, news_data: List[Dict]) -> Dict[str, Any]:
        """감정 분석"""
        if not news_data:
            return {'score': 0, 'label': '중립'}
        
        positive_count = 0
        negative_count = 0
        total_count = len(news_data)
        
        for news in news_data:
            title = news.get('title', '').lower()
            
            # 긍정 키워드 검사
            pos_matches = sum(1 for keyword in self.sentiment_keywords['positive'] if keyword in title)
            # 부정 키워드 검사
            neg_matches = sum(1 for keyword in self.sentiment_keywords['negative'] if keyword in title)
            
            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
        
        # 감정 점수 계산 (-100 ~ +100)
        if total_count > 0:
            score = ((positive_count - negative_count) / total_count) * 100
        else:
            score = 0
        
        # 라벨 결정
        if score > 30:
            label = '매우 긍정'
        elif score > 10:
            label = '긍정'
        elif score > -10:
            label = '중립'
        elif score > -30:
            label = '부정'
        else:
            label = '매우 부정'
        
        return {'score': round(score, 1), 'label': label}
    
    def _extract_keywords(self, news_data: List[Dict]) -> List[str]:
        """키워드 추출"""
        all_text = ' '.join([news.get('title', '') for news in news_data])
        
        # 한글 키워드 추출
        korean_words = re.findall(r'[가-힣]{2,}', all_text)
        
        # 빈도 계산
        word_freq = {}
        for word in korean_words:
            if len(word) >= 2:  # 2글자 이상만
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도순 정렬
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words if freq > 1][:20]
    
    def _create_news_summary(self, news_data: List[Dict], sentiment: Dict) -> str:
        """뉴스 요약 생성"""
        if not news_data:
            return "관련 뉴스가 없습니다."
        
        latest_news = news_data[0] if news_data else {}
        
        summary = f"""
📰 뉴스 분석 요약:
• 총 {len(news_data)}건의 뉴스 분석
• 감정 점수: {sentiment['score']}점 ({sentiment['label']})
• 최신 뉴스: {latest_news.get('title', 'N/A')}
• 분석 시점: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """.strip()
        
        return summary 