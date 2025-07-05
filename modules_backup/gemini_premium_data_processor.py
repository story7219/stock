"""
Gemini AI 프리미엄 데이터 처리 시스템
실시간 뉴스 + 차트 이미지 + 고품질 데이터를 Gemini AI가 100% 이해할 수 있도록 가공
"""
import os
import json
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup
import io
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

@dataclass
class NewsData:
    """뉴스 데이터 클래스"""
    title: str
    content: str
    source: str
    published_time: datetime
    url: str
    sentiment: float
    relevance_score: float
    keywords: List[str]

@dataclass 
class ChartData:
    """차트 데이터 클래스"""
    symbol: str
    image_base64: str
    technical_indicators: Dict[str, float]
    price_data: Dict[str, float]
    volume_data: Dict[str, float]
    chart_analysis: str

@dataclass
class ProcessedData:
    """처리된 데이터 클래스"""
    symbol: str
    news_summary: str
    chart_analysis: str
    technical_data: Dict[str, Any]
    market_sentiment: str
    risk_factors: List[str]
    opportunities: List[str]
    gemini_prompt: str

class GeminiPremiumDataProcessor:
    """Gemini AI 프리미엄 데이터 처리기"""
    
    def __init__(self):
        """초기화"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.is_mock = os.getenv('IS_MOCK', 'false').lower() == 'true'
        self._setup_gemini()
        self._setup_matplotlib()
        
        # 무료 뉴스 소스 설정
        self.news_sources = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'investing_com': 'https://www.investing.com/rss/news.rss',
            'finviz': 'https://finviz.com/news.ashx'
        }
        
    def _setup_gemini(self):
        """Gemini AI 설정"""
        if not self.is_mock and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                logger.info("Gemini AI 프리미엄 데이터 처리기 초기화 완료")
            except Exception as e:
                logger.warning(f"Gemini AI 초기화 실패: {e}")
                self.is_mock = True
        else:
            self.model = None
            logger.info("Mock 모드로 실행")
    
    def _setup_matplotlib(self):
        """Matplotlib 한글 폰트 설정"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        
    async def process_stock_data(self, symbol: str) -> ProcessedData:
        """주식 데이터 종합 처리"""
        try:
            logger.info(f"{symbol} 프리미엄 데이터 처리 시작")
            
            # 병렬로 데이터 수집
            tasks = [
                self._collect_real_time_news(symbol),
                self._generate_chart_image(symbol),
                self._collect_technical_data(symbol)
            ]
            
            news_data, chart_data, technical_data = await asyncio.gather(*tasks)
            
            # Gemini AI가 이해할 수 있는 형태로 데이터 가공
            processed_data = await self._process_for_gemini(
                symbol, news_data, chart_data, technical_data
            )
            
            logger.info(f"{symbol} 프리미엄 데이터 처리 완료")
            return processed_data
            
        except Exception as e:
            logger.error(f"{symbol} 데이터 처리 중 오류: {e}")
            return self._create_fallback_data(symbol)
    
    async def _collect_real_time_news(self, symbol: str) -> List[NewsData]:
        """실시간 뉴스 수집 (최고품질 무료 소스)"""
        news_list = []
        
        try:
            # Yahoo Finance 뉴스
            yahoo_news = await self._fetch_yahoo_finance_news(symbol)
            news_list.extend(yahoo_news)
            
            # MarketWatch 뉴스
            marketwatch_news = await self._fetch_marketwatch_news(symbol)
            news_list.extend(marketwatch_news)
            
            # Reuters 비즈니스 뉴스
            reuters_news = await self._fetch_reuters_news(symbol)
            news_list.extend(reuters_news)
            
            # 뉴스 품질 검증 및 정렬
            verified_news = self._verify_news_quality(news_list)
            
            logger.info(f"{symbol} 실시간 뉴스 {len(verified_news)}개 수집 완료")
            return verified_news[:10]  # 상위 10개만 반환
            
        except Exception as e:
            logger.error(f"뉴스 수집 중 오류: {e}")
            return []
    
    async def _fetch_yahoo_finance_news(self, symbol: str) -> List[NewsData]:
        """Yahoo Finance 뉴스 수집"""
        news_list = []
        try:
            # Yahoo Finance RSS 피드
            feed = feedparser.parse(f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}')
            
            for entry in feed.entries[:5]:
                news_data = NewsData(
                    title=entry.title,
                    content=BeautifulSoup(entry.summary, 'html.parser').get_text(),
                    source='Yahoo Finance',
                    published_time=datetime(*entry.published_parsed[:6]),
                    url=entry.link,
                    sentiment=0.0,  # 나중에 분석
                    relevance_score=0.8,
                    keywords=[]
                )
                news_list.append(news_data)
                
        except Exception as e:
            logger.warning(f"Yahoo Finance 뉴스 수집 실패: {e}")
            
        return news_list
    
    async def _fetch_marketwatch_news(self, symbol: str) -> List[NewsData]:
        """MarketWatch 뉴스 수집"""
        news_list = []
        try:
            async with aiohttp.ClientSession() as session:
                url = f'https://www.marketwatch.com/investing/stock/{symbol.lower()}/news'
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    articles = soup.find_all('div', class_='article__content')[:3]
                    for article in articles:
                        title_elem = article.find('h3') or article.find('h2')
                        if title_elem:
                            news_data = NewsData(
                                title=title_elem.get_text().strip(),
                                content=article.get_text().strip()[:500],
                                source='MarketWatch',
                                published_time=datetime.now(),
                                url=url,
                                sentiment=0.0,
                                relevance_score=0.7,
                                keywords=[]
                            )
                            news_list.append(news_data)
                            
        except Exception as e:
            logger.warning(f"MarketWatch 뉴스 수집 실패: {e}")
            
        return news_list
    
    async def _fetch_reuters_news(self, symbol: str) -> List[NewsData]:
        """Reuters 뉴스 수집"""
        news_list = []
        try:
            feed = feedparser.parse('https://feeds.reuters.com/reuters/businessNews')
            
            for entry in feed.entries[:3]:
                if symbol.upper() in entry.title.upper() or symbol.upper() in entry.summary.upper():
                    news_data = NewsData(
                        title=entry.title,
                        content=BeautifulSoup(entry.summary, 'html.parser').get_text(),
                        source='Reuters',
                        published_time=datetime(*entry.published_parsed[:6]),
                        url=entry.link,
                        sentiment=0.0,
                        relevance_score=0.9,
                        keywords=[]
                    )
                    news_list.append(news_data)
                    
        except Exception as e:
            logger.warning(f"Reuters 뉴스 수집 실패: {e}")
            
        return news_list
    
    def _verify_news_quality(self, news_list: List[NewsData]) -> List[NewsData]:
        """뉴스 품질 검증"""
        verified_news = []
        
        for news in news_list:
            # 품질 검증 기준
            if (len(news.title) > 10 and 
                len(news.content) > 50 and
                news.published_time > datetime.now() - timedelta(days=7)):
                
                # 관련성 점수 계산
                news.relevance_score = self._calculate_relevance_score(news)
                verified_news.append(news)
        
        # 관련성 점수로 정렬
        return sorted(verified_news, key=lambda x: x.relevance_score, reverse=True)
    
    def _calculate_relevance_score(self, news: NewsData) -> float:
        """뉴스 관련성 점수 계산"""
        score = 0.0
        
        # 소스별 가중치
        source_weights = {
            'Reuters': 0.9,
            'Yahoo Finance': 0.8,
            'MarketWatch': 0.7
        }
        score += source_weights.get(news.source, 0.5)
        
        # 시간 가중치 (최신일수록 높은 점수)
        hours_ago = (datetime.now() - news.published_time).total_seconds() / 3600
        time_weight = max(0.1, 1.0 - (hours_ago / 168))  # 1주일 기준
        score *= time_weight
        
        return min(1.0, score)
    
    async def _generate_chart_image(self, symbol: str) -> ChartData:
        """차트 이미지 생성"""
        try:
            # yfinance로 데이터 수집
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                return self._create_fallback_chart(symbol)
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # 가격 차트
            ax1.plot(hist.index, hist['Close'], linewidth=2, label='종가')
            ax1.plot(hist.index, hist['Close'].rolling(20).mean(), 
                    linewidth=1, alpha=0.7, label='20일 이평선')
            ax1.plot(hist.index, hist['Close'].rolling(60).mean(), 
                    linewidth=1, alpha=0.7, label='60일 이평선')
            ax1.set_title(f'{symbol} 주가 차트 (3개월)', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 거래량 차트
            ax2.bar(hist.index, hist['Volume'], alpha=0.6, color='orange')
            ax2.set_title('거래량', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 이미지를 base64로 변환
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # 기술적 지표 계산
            technical_indicators = self._calculate_technical_indicators(hist)
            
            chart_data = ChartData(
                symbol=symbol,
                image_base64=image_base64,
                technical_indicators=technical_indicators,
                price_data={
                    'current': float(hist['Close'].iloc[-1]),
                    'change_pct': float(((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100),
                    'high_52w': float(hist['High'].max()),
                    'low_52w': float(hist['Low'].min())
                },
                volume_data={
                    'current': float(hist['Volume'].iloc[-1]),
                    'avg_volume': float(hist['Volume'].mean())
                },
                chart_analysis=self._analyze_chart_pattern(hist)
            )
            
            return chart_data
            
        except Exception as e:
            logger.error(f"차트 생성 중 오류: {e}")
            return self._create_fallback_chart(symbol)
    
    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, float]:
        """기술적 지표 계산"""
        indicators = {}
        
        try:
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # MACD
            exp1 = hist['Close'].ewm(span=12).mean()
            exp2 = hist['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            indicators['macd'] = float(macd.iloc[-1])
            
            # 볼린저 밴드
            bb_middle = hist['Close'].rolling(20).mean()
            bb_std = hist['Close'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            current_price = hist['Close'].iloc[-1]
            indicators['bb_position'] = float((current_price - bb_lower.iloc[-1]) / 
                                            (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
            
        except Exception as e:
            logger.warning(f"기술적 지표 계산 실패: {e}")
            
        return indicators
    
    def _analyze_chart_pattern(self, hist: pd.DataFrame) -> str:
        """차트 패턴 분석"""
        try:
            recent_prices = hist['Close'].tail(20)
            
            # 추세 분석
            if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                trend = "상승 추세"
            elif recent_prices.iloc[-1] < recent_prices.iloc[0]:
                trend = "하락 추세"
            else:
                trend = "횡보 추세"
            
            # 변동성 분석
            volatility = recent_prices.std() / recent_prices.mean() * 100
            
            if volatility > 5:
                volatility_desc = "높은 변동성"
            elif volatility > 2:
                volatility_desc = "보통 변동성"
            else:
                volatility_desc = "낮은 변동성"
            
            return f"{trend}, {volatility_desc} (변동성: {volatility:.2f}%)"
            
        except Exception as e:
            return "차트 패턴 분석 불가"
    
    async def _collect_technical_data(self, symbol: str) -> Dict[str, Any]:
        """기술적 데이터 수집"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            technical_data = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            return technical_data
            
        except Exception as e:
            logger.warning(f"기술적 데이터 수집 실패: {e}")
            return {}
    
    async def _process_for_gemini(
        self, 
        symbol: str, 
        news_data: List[NewsData], 
        chart_data: ChartData, 
        technical_data: Dict[str, Any]
    ) -> ProcessedData:
        """Gemini AI가 이해할 수 있는 형태로 데이터 가공"""
        
        # 뉴스 요약
        news_summary = self._summarize_news(news_data)
        
        # 시장 심리 분석
        market_sentiment = self._analyze_market_sentiment(news_data, chart_data)
        
        # 리스크 요인 추출
        risk_factors = self._extract_risk_factors(news_data, technical_data)
        
        # 기회 요인 추출
        opportunities = self._extract_opportunities(news_data, chart_data, technical_data)
        
        # Gemini AI 프롬프트 생성
        gemini_prompt = self._create_gemini_prompt(
            symbol, news_summary, chart_data, technical_data, 
            market_sentiment, risk_factors, opportunities
        )
        
        return ProcessedData(
            symbol=symbol,
            news_summary=news_summary,
            chart_analysis=chart_data.chart_analysis,
            technical_data=technical_data,
            market_sentiment=market_sentiment,
            risk_factors=risk_factors,
            opportunities=opportunities,
            gemini_prompt=gemini_prompt
        )
    
    def _summarize_news(self, news_data: List[NewsData]) -> str:
        """뉴스 요약"""
        if not news_data:
            return "관련 뉴스 없음"
        
        summaries = []
        for news in news_data[:5]:  # 상위 5개만
            summary = f"[{news.source}] {news.title}: {news.content[:100]}..."
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _analyze_market_sentiment(self, news_data: List[NewsData], chart_data: ChartData) -> str:
        """시장 심리 분석"""
        # 간단한 감정 분석 (키워드 기반)
        positive_keywords = ['상승', '증가', '성장', '호재', '긍정', 'up', 'rise', 'growth', 'positive']
        negative_keywords = ['하락', '감소', '악재', '부정', '리스크', 'down', 'fall', 'negative', 'risk']
        
        positive_count = 0
        negative_count = 0
        
        for news in news_data:
            text = (news.title + " " + news.content).lower()
            positive_count += sum(1 for keyword in positive_keywords if keyword in text)
            negative_count += sum(1 for keyword in negative_keywords if keyword in text)
        
        # 차트 데이터도 고려
        if chart_data.price_data.get('change_pct', 0) > 0:
            positive_count += 2
        else:
            negative_count += 2
        
        if positive_count > negative_count:
            return "긍정적"
        elif negative_count > positive_count:
            return "부정적"
        else:
            return "중립적"
    
    def _extract_risk_factors(self, news_data: List[NewsData], technical_data: Dict[str, Any]) -> List[str]:
        """리스크 요인 추출"""
        risk_factors = []
        
        # 기술적 리스크
        pe_ratio = technical_data.get('pe_ratio', 0)
        if pe_ratio > 30:
            risk_factors.append(f"높은 PER ({pe_ratio:.1f}) - 고평가 우려")
        
        beta = technical_data.get('beta', 1.0)
        if beta > 1.5:
            risk_factors.append(f"높은 베타 ({beta:.2f}) - 시장 변동성에 민감")
        
        # 뉴스 기반 리스크
        risk_keywords = ['소송', '규제', '조사', '적자', '부채', 'lawsuit', 'regulation', 'debt']
        for news in news_data:
            text = (news.title + " " + news.content).lower()
            for keyword in risk_keywords:
                if keyword in text:
                    risk_factors.append(f"뉴스 리스크: {news.title[:50]}...")
                    break
        
        return risk_factors[:5]  # 상위 5개만
    
    def _extract_opportunities(
        self, 
        news_data: List[NewsData], 
        chart_data: ChartData, 
        technical_data: Dict[str, Any]
    ) -> List[str]:
        """기회 요인 추출"""
        opportunities = []
        
        # 기술적 기회
        rsi = chart_data.technical_indicators.get('rsi', 50)
        if rsi < 30:
            opportunities.append(f"RSI 과매도 구간 ({rsi:.1f}) - 반등 가능성")
        elif rsi > 70:
            opportunities.append(f"RSI 과매수 구간 ({rsi:.1f}) - 조정 후 재상승 가능")
        
        # 뉴스 기반 기회
        opportunity_keywords = ['신제품', '계약', '투자', '확장', '성장', 'new product', 'contract', 'expansion']
        for news in news_data:
            text = (news.title + " " + news.content).lower()
            for keyword in opportunity_keywords:
                if keyword in text:
                    opportunities.append(f"성장 기회: {news.title[:50]}...")
                    break
        
        return opportunities[:5]  # 상위 5개만
    
    def _create_gemini_prompt(
        self,
        symbol: str,
        news_summary: str,
        chart_data: ChartData,
        technical_data: Dict[str, Any],
        market_sentiment: str,
        risk_factors: List[str],
        opportunities: List[str]
    ) -> str:
        """Gemini AI를 위한 완벽한 프롬프트 생성"""
        
        prompt = f"""
# {symbol} 종목 투자 분석 요청

## 기본 정보
- 종목: {symbol}
- 현재가: {chart_data.price_data.get('current', 0):,.0f}
- 등락률: {chart_data.price_data.get('change_pct', 0):+.2f}%
- 섹터: {technical_data.get('sector', 'Unknown')}
- 산업: {technical_data.get('industry', 'Unknown')}

## 최신 뉴스 분석
{news_summary}

## 차트 기술적 분석
- 차트 패턴: {chart_data.chart_analysis}
- RSI: {chart_data.technical_indicators.get('rsi', 0):.1f}
- MACD: {chart_data.technical_indicators.get('macd', 0):.3f}
- 볼린저밴드 위치: {chart_data.technical_indicators.get('bb_position', 0.5):.2f}

## 재무 지표
- PER: {technical_data.get('pe_ratio', 0):.1f}
- PBR: {technical_data.get('pb_ratio', 0):.1f}
- 베타: {technical_data.get('beta', 1.0):.2f}
- 배당수익률: {technical_data.get('dividend_yield', 0)*100:.2f}%

## 시장 심리
현재 시장 심리: {market_sentiment}

## 리스크 요인
{chr(10).join(f"- {risk}" for risk in risk_factors)}

## 기회 요인  
{chr(10).join(f"- {opportunity}" for opportunity in opportunities)}

## 분석 요청사항
위의 모든 정보를 종합하여 다음 항목에 대해 전문적인 투자 분석을 제공해주세요:

1. **투자 추천도**: 매수/보유/매도 중 하나를 선택하고 그 이유
2. **목표가**: 구체적인 목표 주가와 근거
3. **투자 기간**: 단기/중기/장기 중 추천 기간
4. **리스크 레벨**: 1-5단계 중 위험도 평가
5. **핵심 투자 포인트**: 3가지 주요 투자 근거
6. **주의사항**: 투자 시 반드시 고려해야 할 리스크

분석은 객관적이고 구체적인 데이터에 기반하여 작성해주세요.
"""
        
        return prompt
    
    def _create_fallback_data(self, symbol: str) -> ProcessedData:
        """대체 데이터 생성"""
        return ProcessedData(
            symbol=symbol,
            news_summary="데이터 수집 실패",
            chart_analysis="차트 분석 불가",
            technical_data={},
            market_sentiment="중립적",
            risk_factors=["데이터 부족"],
            opportunities=["분석 불가"],
            gemini_prompt=f"{symbol} 데이터 처리 실패"
        )
    
    def _create_fallback_chart(self, symbol: str) -> ChartData:
        """대체 차트 데이터 생성"""
        return ChartData(
            symbol=symbol,
            image_base64="",
            technical_indicators={},
            price_data={},
            volume_data={},
            chart_analysis="차트 생성 실패"
        )

    async def send_to_gemini(self, processed_data: ProcessedData) -> Dict[str, Any]:
        """Gemini AI에게 데이터 전송 및 분석 결과 수신"""
        if self.is_mock or not self.model:
            return self._create_mock_gemini_response(processed_data)
        
        try:
            response = await self.model.generate_content_async(processed_data.gemini_prompt)
            
            analysis_result = {
                'symbol': processed_data.symbol,
                'gemini_analysis': response.text,
                'recommendation': self._extract_recommendation(response.text),
                'target_price': self._extract_target_price(response.text),
                'risk_level': self._extract_risk_level(response.text),
                'investment_period': self._extract_investment_period(response.text),
                'confidence_score': 0.85,
                'timestamp': datetime.now()
            }
            
            logger.info(f"{processed_data.symbol} Gemini AI 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Gemini AI 분석 실패: {e}")
            return self._create_mock_gemini_response(processed_data)
    
    def _create_mock_gemini_response(self, processed_data: ProcessedData) -> Dict[str, Any]:
        """Mock Gemini 응답 생성"""
        return {
            'symbol': processed_data.symbol,
            'gemini_analysis': f"{processed_data.symbol} Mock 분석 결과입니다.",
            'recommendation': 'HOLD',
            'target_price': 0,
            'risk_level': 'MEDIUM',
            'investment_period': 'MEDIUM_TERM',
            'confidence_score': 0.75,
            'timestamp': datetime.now()
        }
    
    def _extract_recommendation(self, text: str) -> str:
        """추천도 추출"""
        if '매수' in text or 'BUY' in text.upper():
            return 'BUY'
        elif '매도' in text or 'SELL' in text.upper():
            return 'SELL'
        else:
            return 'HOLD'
    
    def _extract_target_price(self, text: str) -> float:
        """목표가 추출"""
        import re
        matches = re.findall(r'목표가[:\s]*([0-9,]+)', text)
        if matches:
            return float(matches[0].replace(',', ''))
        return 0.0
    
    def _extract_risk_level(self, text: str) -> str:
        """리스크 레벨 추출"""
        if '높은' in text or 'HIGH' in text.upper():
            return 'HIGH'
        elif '낮은' in text or 'LOW' in text.upper():
            return 'LOW'
        else:
            return 'MEDIUM'
    
    def _extract_investment_period(self, text: str) -> str:
        """투자 기간 추출"""
        if '단기' in text or 'SHORT' in text.upper():
            return 'SHORT_TERM'
        elif '장기' in text or 'LONG' in text.upper():
            return 'LONG_TERM'
        else:
            return 'MEDIUM_TERM'

# 테스트 함수
async def test_premium_data_processor():
    """프리미엄 데이터 처리기 테스트"""
    processor = GeminiPremiumDataProcessor()
    
    # 테스트 종목들
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in test_symbols:
        logger.info(f"{symbol} 테스트 시작")
        
        # 데이터 처리
        processed_data = await processor.process_stock_data(symbol)
        
        # Gemini AI 분석
        gemini_result = await processor.send_to_gemini(processed_data)
        
        logger.info(f"{symbol} 테스트 완료: {gemini_result['recommendation']}")

if __name__ == "__main__":
    asyncio.run(test_premium_data_processor()) 