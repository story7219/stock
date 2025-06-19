"""
AI 기반 시장 분석 엔진 (시장 주도주 스크리닝 기능 탑재)
- 네이버 금융에서 거래량/상승률 상위 종목을 자동으로 스크리닝
- 스크리닝된 데이터를 기반으로 Gemini가 투자 의견 제시
- 실시간 뉴스 및 공시 크롤링 기능 추가
"""
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from PIL import Image
import logging
import config
from datetime import datetime, timedelta
import time
import re

logger = logging.getLogger(__name__)

try:
    if config.GEMINI_API_KEY and config.GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY':
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("✅ Google Gemini API가 설정되었습니다.")
    else:
        logger.warning("⚠️ Google Gemini API 키가 설정되지 않았습니다. config.py를 확인해주세요.")
except Exception as e:
    logger.error(f"❌ Google Gemini API 키 설정 중 오류 발생: {e}")

def fetch_market_leaders():
    """네이버 금융에서 시장 주도주 데이터를 스크래핑합니다."""
    logger.info("📈 시장 주도주 데이터 스크리닝 시작...")
    try:
        url = "https://finance.naver.com/sise/sise_market_sum.naver"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='type_2')
        
        market_data = "## 📊 오늘의 시장 주도주 (거래상위) ##\n"
        for row in table.find_all('tr')[2:12]: # 상위 10개
            cols = row.find_all('td')
            if len(cols) < 5: continue
            
            rank = cols[0].text.strip()
            name = cols[1].text.strip()
            price = cols[2].text.strip()
            change_rate = cols[4].text.strip()
            volume = cols[6].text.strip()
            
            market_data += f"- {rank}위 {name}: {price}원 ({change_rate}, 거래량: {volume})\n"
        
        logger.info("✅ 시장 주도주 데이터 스크리닝 완료")
        return market_data
    except Exception as e:
        logger.error(f"❌ 시장 주도주 스크리닝 실패: {e}")
        return "시장 데이터 수집에 실패했습니다."

def fetch_market_news():
    """📰 실시간 시장 뉴스 크롤링"""
    logger.info("📰 실시간 시장 뉴스 수집 시작...")
    try:
        news_data = []
        
        # 네이버 금융 뉴스
        naver_url = "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(naver_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_list = soup.find_all('dd', class_='articleSubject')[:10]  # 상위 10개 뉴스
        
        for i, news in enumerate(news_list, 1):
            try:
                link_tag = news.find('a')
                if link_tag:
                    title = link_tag.text.strip()
                    # 투자 관련 키워드 필터링
                    if any(keyword in title for keyword in ['주식', '상승', '하락', '급등', '급락', '매수', '매도', '투자', '실적', '영업익']):
                        news_data.append(f"{i}. {title}")
            except:
                continue
        
        # 증권 전문 뉴스 추가 (한국경제 증권)
        try:
            hankyung_url = "https://stock.hankyung.com/news/list"
            response = requests.get(hankyung_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stock_news = soup.find_all('dt', class_='tit')[:5]  # 상위 5개 추가
            for news in stock_news:
                try:
                    link_tag = news.find('a')
                    if link_tag:
                        title = link_tag.text.strip()
                        if len(title) > 10:  # 의미있는 제목만
                            news_data.append(f"📈 {title}")
                except:
                    continue
        except:
            logger.warning("⚠️ 한국경제 뉴스 수집 부분 실패")
        
        if news_data:
            news_text = "## 📰 실시간 주요 뉴스 ##\n" + "\n".join(news_data[:15])
            logger.info(f"✅ 뉴스 {len(news_data[:15])}개 수집 완료")
            return news_text
        else:
            return "## 📰 뉴스 수집 결과 없음 ##"
            
    except Exception as e:
        logger.error(f"❌ 뉴스 크롤링 실패: {e}")
        return "뉴스 데이터 수집에 실패했습니다."

def fetch_company_disclosures(symbol_list=None):
    """📋 기업 공시 정보 크롤링"""
    logger.info("📋 기업 공시 정보 수집 시작...")
    try:
        disclosures = []
        
        # DART 공시시스템의 최신 공시 (일반적인 공시 정보)
        dart_url = "http://kind.krx.co.kr/disclosure/todaydisclosure.do"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(dart_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 공시 테이블 찾기
            disclosure_table = soup.find('table')
            if disclosure_table:
                rows = disclosure_table.find_all('tr')[1:11]  # 상위 10개
                
                for row in rows:
                    try:
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            company = cols[1].text.strip()
                            disclosure_type = cols[2].text.strip()
                            time_str = cols[0].text.strip()
                            
                            # 중요한 공시만 필터링
                            important_keywords = ['실적', '배당', '합병', '분할', '유증', '감자', '영업', '투자']
                            if any(keyword in disclosure_type for keyword in important_keywords):
                                disclosures.append(f"🏢 {company}: {disclosure_type} ({time_str})")
                    except:
                        continue
        
        # 특정 종목 관련 공시 (symbol_list가 제공된 경우)
        if symbol_list and len(symbol_list) > 0:
            logger.info(f"특정 종목 {symbol_list} 공시 검색...")
            # 여기서는 간단히 일반 공시에서 종목명 매칭
            # 실제로는 각 종목의 종목코드로 상세 검색 가능
        
        if disclosures:
            disclosure_text = "## 📋 주요 공시 정보 ##\n" + "\n".join(disclosures[:10])
            logger.info(f"✅ 공시 {len(disclosures[:10])}개 수집 완료")
            return disclosure_text
        else:
            return "## 📋 중요한 공시 없음 ##"
            
    except Exception as e:
        logger.error(f"❌ 공시 크롤링 실패: {e}")
        return "공시 데이터 수집에 실패했습니다."

def get_comprehensive_market_data(symbol_list=None):
    """🌐 종합적인 시장 데이터 수집 (뉴스 + 공시 + 주도주)"""
    logger.info("🌐 종합 시장 데이터 수집 시작...")
    
    # 병렬로 데이터 수집 (시간 단축)
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 3개 작업을 동시에 실행
        future_leaders = executor.submit(fetch_market_leaders)
        future_news = executor.submit(fetch_market_news)
        future_disclosures = executor.submit(fetch_company_disclosures, symbol_list)
        
        # 결과 수집
        leaders_data = future_leaders.result()
        news_data = future_news.result()
        disclosure_data = future_disclosures.result()
    
    # 전체 데이터 통합
    comprehensive_data = f"""
{leaders_data}

{news_data}

{disclosure_data}

## ⏰ 데이터 수집 시간 ##
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    logger.info("✅ 종합 시장 데이터 수집 완료")
    return comprehensive_data

class MarketAnalyzer:
    """Gemini를 활용한 시장 데이터 및 차트 분석기"""
    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        # Gemini API 키 설정 상태 확인
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
            raise ValueError("Gemini API 키가 없어 MarketAnalyzer를 초기화할 수 없습니다.")
        self.model = genai.GenerativeModel(model_name)

    def get_trading_insights(self, chart_image_path: str) -> str:
        # 시장 데이터를 자동으로 가져옴
        market_data_text = fetch_market_leaders()
        
        logger.info(f"차트({chart_image_path})와 시장 데이터 분석을 시작합니다...")
        try:
            img = Image.open(chart_image_path)
        except Exception as e:
            return f"오류: 이미지 파일({chart_image_path})을 열 수 없습니다. {e}"

        prompt = f"""
        당신은 20년 경력의 베테랑 펀드매니저입니다. 아래의 최신 시장 주도주 데이터와 첨부된 차트 이미지를 종합 분석하여, 단기(1~3일) 트레이딩 관점에서 투자 전략을 제시하세요.

        **분석 가이드라인:**
        1. **종합 분석:** 모든 데이터(등락률, 거래량, 수급)와 차트를 함께 고려하세요.
        2. **단기 관점:** 1~3일 내의 움직임에 초점을 맞춥니다.
        3. **명확한 판단:** '매수', '매도', '관망' 중 하나로 명확히 제시하세요.
        4. **핵심 근거:** 판단의 핵심 이유를 1~2가지로 간결하게 요약합니다.

        **입력 데이터 (시장 주도주 현황):**
        ---
        {market_data_text}
        ---

        **분석 요청:**
        위 데이터와 첨부된 차트 이미지를 바탕으로, 아래 형식에 맞춰 주요 종목에 대한 투자 판단을 표로 정리해주세요.

        | 종목명 | 판단(매수/매도/관망) | 근거 요약 |
        |---|---|---|
        """
        try:
            response = self.model.generate_content([prompt, img])
            logger.info("✅ AI 분석이 완료되었습니다.")
            return response.text
        except Exception as e:
            return f"오류: AI 분석 중 문제가 발생했습니다. ({e})"
    
    def get_comprehensive_analysis(self, chart_image_path: str = None, symbol_list: list = None) -> str:
        """🎯 종합적인 시장 분석 (뉴스 + 공시 + 차트)"""
        logger.info("🎯 종합 시장 분석 시작...")
        
        # 종합 시장 데이터 수집
        market_data = get_comprehensive_market_data(symbol_list)
        
        # 차트가 있으면 차트도 함께 분석
        if chart_image_path:
            try:
                img = Image.open(chart_image_path)
                
                prompt = f"""
                당신은 AI 금융 분석가입니다. 아래의 실시간 시장 데이터(뉴스, 공시, 주도주)를 종합 분석하여 투자 전략을 제시하세요.

                **분석 데이터:**
                {market_data}

                **분석 요청:**
                1. 현재 시장 상황 요약 (3줄)
                2. 주목할 만한 뉴스/공시와 그 영향도
                3. 투자 추천 종목 Top 3 (근거와 함께)
                4. 리스크 요인 및 주의사항
                5. 오늘의 투자 전략 한 줄 요약
                """
                
                response = self.model.generate_content([prompt, img])
                return response.text
                
            except Exception as e:
                logger.warning(f"⚠️ 차트 분석 실패, 텍스트만 분석: {e}")
        
        # 텍스트만 분석
        prompt = f"""
        당신은 AI 금융 분석가입니다. 아래의 실시간 시장 데이터를 종합 분석하여 투자 전략을 제시하세요.

        **분석 데이터:**
        {market_data}

        **분석 요청:**
        1. 현재 시장 상황 요약 (3줄)
        2. 주목할 만한 뉴스/공시와 그 영향도  
        3. 투자 추천 종목 Top 3 (근거와 함께)
        4. 리스크 요인 및 주의사항
        5. 오늘의 투자 전략 한 줄 요약
        """
        
        try:
            response = self.model.generate_content(prompt)
            logger.info("✅ 종합 AI 분석 완료")
            return response.text
        except Exception as e:
            return f"오류: 종합 분석 중 문제가 발생했습니다. ({e})" 