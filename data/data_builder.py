"""
🤖 제미나이를 위한 양질의 투자 데이터 가공 시스템
- 7단계 데이터 구축 파이프라인 + 다중 소스 통합
- KIS, 네이버금융, 야후파이낸스, 인베스팅닷컴 연동
- AI 친화적 데이터 포맷 생성
- 자동 분석 및 인사이트 추출
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
import warnings
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import time
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class MultiSourceDataCollector:
    """다중 소스 데이터 수집기 - KIS, 네이버, 야후, 인베스팅닷컴"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # KIS API 설정 (기존 연동 활용)
        self.kis_token = self.load_kis_token()
        
        # 데이터 소스 우선순위
        self.data_sources = ['kis', 'naver', 'yahoo', 'investing']
    
    def load_kis_token(self) -> Optional[str]:
        """KIS 토큰 로드"""
        try:
            if os.path.exists('kis_token.json'):
                with open('kis_token.json', 'r') as f:
                    token_data = json.load(f)
                    return token_data.get('access_token')
        except Exception as e:
            logging.warning(f"KIS 토큰 로드 실패: {e}")
        return None
    
    def get_unified_stock_data(self, symbol: str, market: str = 'KR') -> Dict[str, Any]:
        """
        통합 주식 데이터 수집 - 여러 소스에서 데이터 수집하여 통합
        
        Args:
            symbol: 종목코드 (한국) 또는 티커 (미국)
            market: 'KR' 또는 'US'
            
        Returns:
            통합된 주식 데이터 딕셔너리
        """
        unified_data = {
            'symbol': symbol,
            'market': market,
            'name': None,
            'price': None,
            'change_rate': None,
            'volume': None,
            'market_cap': None,
            'per': None,
            'pbr': None,
            'roe': None,
            'debt_ratio': None,
            'dividend_yield': None,
            'sector': None,
            'data_sources': [],
            'data_quality': 0
        }
        
        # 각 소스에서 순차적으로 데이터 수집
        for source in self.data_sources:
            try:
                if source == 'kis' and self.kis_token and market == 'KR':
                    data = self.get_kis_data(symbol)
                elif source == 'naver' and market == 'KR':
                    data = self.get_naver_data(symbol)
                elif source == 'yahoo':
                    ticker = f"{symbol}.KS" if market == 'KR' and len(symbol) == 6 else symbol
                    data = self.get_yahoo_data(ticker)
                elif source == 'investing':
                    data = self.get_investing_data(symbol, market)
                else:
                    continue
                
                if data:
                    # 데이터 병합 (None이 아닌 값만 업데이트)
                    for key, value in data.items():
                        if value is not None and unified_data.get(key) is None:
                            unified_data[key] = value
                    
                    unified_data['data_sources'].append(source)
                    
            except Exception as e:
                logging.warning(f"{source} 데이터 수집 실패 ({symbol}): {e}")
                continue
        
        # 데이터 품질 점수 계산
        unified_data['data_quality'] = self.calculate_data_quality(unified_data)
        
        return unified_data
    
    def get_kis_data(self, symbol: str) -> Dict[str, Any]:
        """KIS API에서 데이터 수집"""
        if not self.kis_token:
            return {}
        
        try:
            # KIS API 호출 (기존 연동 활용)
            # 실제 KIS API 구현은 기존 코드 활용
            return {
                'source': 'kis',
                'price': None,  # KIS에서 가져온 현재가
                'volume': None,  # 거래량
                'market_cap': None,  # 시가총액
                # 추가 KIS 데이터...
            }
        except Exception as e:
            logging.error(f"KIS 데이터 수집 오류: {e}")
            return {}
    
    def get_naver_data(self, symbol: str) -> Dict[str, Any]:
        """네이버 금융에서 데이터 수집"""
        try:
            url = f"https://finance.naver.com/item/main.naver?code={symbol}"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            data = {'source': 'naver'}
            
            # 종목명
            name_elem = soup.select_one('.wrap_company h2 a')
            if name_elem:
                data['name'] = name_elem.text.strip()
            
            # 현재가
            price_elem = soup.select_one('.no_today .blind')
            if price_elem:
                price_text = price_elem.text.replace(',', '')
                try:
                    data['price'] = float(price_text)
                except ValueError:
                    pass
            
            # 등락률
            rate_elem = soup.select_one('.no_exday .blind')
            if rate_elem:
                rate_text = rate_elem.text.replace('%', '').replace('+', '')
                try:
                    data['change_rate'] = float(rate_text)
                except ValueError:
                    pass
            
            # 거래량
            volume_elem = soup.select_one('table.no_info tr:nth-child(1) td:nth-child(4)')
            if volume_elem:
                volume_text = volume_elem.text.replace(',', '').replace('주', '')
                try:
                    data['volume'] = int(volume_text)
                except ValueError:
                    pass
            
            # 시가총액
            market_cap_elem = soup.select_one('table.tb_type1 tr:nth-child(3) td:nth-child(2)')
            if market_cap_elem:
                cap_text = market_cap_elem.text.replace(',', '').replace('억원', '')
                try:
                    data['market_cap'] = float(cap_text) * 100000000  # 억원을 원으로
                except ValueError:
                    pass
            
            # PER, PBR 등 재무지표
            ratio_rows = soup.select('table.tb_type1 tr')
            for row in ratio_rows:
                cells = row.select('td')
                if len(cells) >= 2:
                    label = cells[0].text.strip()
                    value_text = cells[1].text.strip()
                    
                    try:
                        if 'PER' in label:
                            data['per'] = float(value_text.replace(',', ''))
                        elif 'PBR' in label:
                            data['pbr'] = float(value_text.replace(',', ''))
                        elif 'ROE' in label:
                            data['roe'] = float(value_text.replace('%', ''))
                    except (ValueError, AttributeError):
                        continue
            
            return data
            
        except Exception as e:
            logging.error(f"네이버 데이터 수집 오류 ({symbol}): {e}")
            return {}
    
    def get_yahoo_data(self, ticker: str) -> Dict[str, Any]:
        """야후 파이낸스에서 데이터 수집"""
        try:
            stock_obj = yf.Ticker(ticker)
            info = stock_obj.info
            hist = stock_obj.history(period="1d")
            
            data = {'source': 'yahoo'}
            
            # 기본 정보
            if 'longName' in info:
                data['name'] = info['longName']
            elif 'shortName' in info:
                data['name'] = info['shortName']
            
            # 가격 정보
            if not hist.empty:
                data['price'] = float(hist['Close'].iloc[-1])
                
            if 'regularMarketPrice' in info:
                data['price'] = info['regularMarketPrice']
            
            # 시가총액
            if 'marketCap' in info:
                data['market_cap'] = info['marketCap']
            
            # 재무지표
            financial_metrics = {
                'per': ['trailingPE', 'forwardPE'],
                'pbr': ['priceToBook'],
                'roe': ['returnOnEquity'],
                'debt_ratio': ['debtToEquity'],
                'dividend_yield': ['dividendYield']
            }
            
            for metric, keys in financial_metrics.items():
                for key in keys:
                    if key in info and info[key] is not None:
                        value = info[key]
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            if metric in ['roe', 'dividend_yield']:
                                data[metric] = value * 100  # 퍼센트로 변환
                            else:
                                data[metric] = value
                        break
            
            # 섹터 정보
            if 'sector' in info:
                data['sector'] = info['sector']
            
            return data
            
        except Exception as e:
            logging.error(f"야후 데이터 수집 오류 ({ticker}): {e}")
            return {}
    
    def get_investing_data(self, symbol: str, market: str) -> Dict[str, Any]:
        """인베스팅닷컴에서 데이터 수집"""
        try:
            # 인베스팅닷컴은 복잡한 스크래핑이 필요하므로 기본 구조만 제공
            # 실제 구현시에는 더 정교한 파싱 필요
            
            if market == 'KR':
                # 한국 주식 URL 패턴 (예시)
                search_url = f"https://www.investing.com/search/?q={symbol}"
            else:
                # 미국 주식 URL 패턴 (예시)
                search_url = f"https://www.investing.com/search/?q={symbol}"
            
            response = self.session.get(search_url, timeout=10)
            
            # 간단한 데이터 추출 (실제로는 더 복잡한 파싱 필요)
            data = {'source': 'investing'}
            
            # 기본적인 정보만 추출 (예시)
            if response.status_code == 200:
                # 실제 파싱 로직 구현 필요
                pass
            
            return data
            
        except Exception as e:
            logging.error(f"인베스팅닷컴 데이터 수집 오류 ({symbol}): {e}")
            return {}
    
    def calculate_data_quality(self, data: Dict[str, Any]) -> float:
        """데이터 품질 점수 계산 (0-100)"""
        essential_fields = ['name', 'price', 'market_cap', 'per', 'pbr']
        optional_fields = ['roe', 'debt_ratio', 'dividend_yield', 'sector']
        
        essential_score = sum(1 for field in essential_fields if data.get(field) is not None)
        optional_score = sum(1 for field in optional_fields if data.get(field) is not None)
        
        # 필수 필드 70%, 선택 필드 30%
        quality_score = (essential_score / len(essential_fields)) * 70 + (optional_score / len(optional_fields)) * 30
        
        return round(quality_score, 1)

class GeminiDataProcessor:
    """제미나이를 위한 스마트 데이터 프로세서 - 다중 소스 통합"""
    
    def __init__(self):
        self.setup_logging()
        self.data_dir = "./data"
        self.ensure_data_directory()
        
        # 다중 소스 데이터 수집기 초기화
        self.data_collector = MultiSourceDataCollector()
        
        # 🎯 제미나이 친화적 컬럼 정의 (확장)
        self.required_columns = [
            'Date', 'Ticker', 'Market', 'Name', 'Sector',
            'Close', 'PER', 'PBR', 'ROE', 'ROIC', 'EPS', 
            'MarketCap', 'Return_3M', 'Return_6M', 'Volatility',
            'Quality_Score', 'Value_Score', 'Momentum_Score', 'Final_Score',
            'Data_Sources', 'Data_Quality'  # 새로 추가
        ]
        
        # 📊 투자 지표 기준값 (실전 검증된 수치)
        self.investment_criteria = {
            'value': {'PER': (0, 15), 'PBR': (0, 1.5)},
            'quality': {'ROE': (15, 100), 'ROIC': (10, 100)},
            'momentum': {'Return_3M': (0.05, 1), 'Return_6M': (0.1, 2)},
            'volatility': {'Volatility': (0, 0.3)}
        }
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gemini_data_processor.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def ensure_data_directory(self):
        """데이터 디렉토리 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.info(f"📁 데이터 디렉토리 생성: {self.data_dir}")
    
    def get_kospi200_tickers(self) -> List[str]:
        """한국 KOSPI200 종목 리스트 수집 (1단계)"""
        try:
            self.logger.info("한국 KOSPI200 종목 리스트 수집 중...")
            tickers = stock.get_index_portfolio_deposit_file("1028")  # KOSPI200
            return tickers[:50]  # 상위 50개 종목으로 제한
        except Exception as e:
            self.logger.error(f"KOSPI200 종목 수집 실패: {e}")
            return []
    
    def get_nasdaq100_tickers(self) -> List[str]:
        """미국 NASDAQ100 종목 리스트 수집 (1단계)"""
        try:
            self.logger.info("미국 NASDAQ100 종목 리스트 수집 중...")
            # 주요 NASDAQ100 종목들
            nasdaq_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'ADBE', 'CRM', 'PYPL', 'INTC', 'CMCSA', 'AVGO', 'TXN', 'QCOM',
                'COST', 'SBUX', 'GILD', 'AMGN', 'MDLZ', 'ISRG', 'BKNG', 'ADP',
                'REGN', 'VRTX', 'LRCX', 'ATVI', 'MU', 'AMAT', 'FISV', 'CSX',
                'ORLY', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'FTNT', 'ADSK', 'NXPI',
                'WDAY', 'TEAM', 'DXCM', 'ILMN', 'BIIB', 'KDP', 'XEL', 'EXC', 'DLTR', 'FAST'
            ]
            return nasdaq_tickers
        except Exception as e:
            self.logger.error(f"NASDAQ100 종목 수집 실패: {e}")
            return []
    
    def collect_korean_stock_data(self, ticker: str, end_date: str) -> Dict:
        """🇰🇷 한국 주식 데이터 수집 (2-4단계)"""
        try:
            # 주가 데이터 수집
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y%m%d')
            end_date_kr = end_date.replace('-', '')
            
            # 주가 정보
            price_data = stock.get_market_ohlcv_by_date(start_date, end_date_kr, ticker)
            if price_data.empty:
                return None
            
            # 재무 정보
            fundamental = stock.get_market_fundamental_by_date(start_date, end_date_kr, ticker)
            if fundamental.empty:
                return None
            
            # 기업 정보
            company_info = stock.get_market_ticker_name(ticker)
            
            # 최신 데이터 추출
            latest_price = price_data.iloc[-1]
            latest_fundamental = fundamental.iloc[-1]
            
            # 수익률 계산
            prices = price_data['종가'].values
            return_3m = self.calculate_return(prices, 63)  # 약 3개월
            return_6m = self.calculate_return(prices, 126)  # 약 6개월
            volatility = self.calculate_volatility(prices)
            
            return {
                'Ticker': ticker,
                'Market': 'KR',
                'Name': company_info,
                'Sector': '한국주식',  # 섹터 정보 추가 필요시 별도 API 사용
                'Close': latest_price['종가'],
                'PER': latest_fundamental['PER'] if 'PER' in latest_fundamental else np.nan,
                'PBR': latest_fundamental['PBR'] if 'PBR' in latest_fundamental else np.nan,
                'EPS': latest_fundamental['EPS'] if 'EPS' in latest_fundamental else np.nan,
                'MarketCap': latest_price['종가'] * latest_price['거래량'] * 1000,  # 근사치
                'Return_3M': return_3m,
                'Return_6M': return_6m,
                'Volatility': volatility,
                'ROE': np.nan,  # 별도 API 필요
                'ROIC': np.nan  # 별도 API 필요
            }
            
        except Exception as e:
            self.logger.error(f"한국 주식 {ticker} 데이터 수집 실패: {e}")
            return None
    
    def collect_us_stock_data(self, ticker: str, end_date: str) -> Dict:
        """🇺🇸 미국 주식 데이터 수집 (2-4단계)"""
        try:
            stock_obj = yf.Ticker(ticker)
            
            # 주가 데이터
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            hist = stock_obj.history(start=start_date, end=end_date)
            if hist.empty:
                return None
            
            # 기업 정보
            info = stock_obj.info
            
            # 재무 정보
            financials = stock_obj.financials
            balance_sheet = stock_obj.balance_sheet
            
            # 최신 가격
            latest_price = hist['Close'].iloc[-1]
            
            # 수익률 계산
            prices = hist['Close'].values
            return_3m = self.calculate_return(prices, 63)
            return_6m = self.calculate_return(prices, 126)
            volatility = self.calculate_volatility(prices)
            
            # 재무 지표 계산
            per = info.get('trailingPE', np.nan)
            pbr = info.get('priceToBook', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            roic = self.calculate_roic(info, financials, balance_sheet)
            
            return {
                'Ticker': ticker,
                'Market': 'US',
                'Name': info.get('longName', ticker),
                'Sector': info.get('sector', '기타'),
                'Close': latest_price,
                'PER': per,
                'PBR': pbr,
                'ROE': roe * 100 if roe else np.nan,  # 백분율로 변환
                'ROIC': roic,
                'EPS': info.get('trailingEps', np.nan),
                'MarketCap': info.get('marketCap', np.nan),
                'Return_3M': return_3m,
                'Return_6M': return_6m,
                'Volatility': volatility
            }
            
        except Exception as e:
            self.logger.error(f"미국 주식 {ticker} 데이터 수집 실패: {e}")
            return None
    
    def calculate_return(self, prices: np.array, period: int) -> float:
        """수익률 계산"""
        if len(prices) < period:
            return np.nan
        return (prices[-1] / prices[-period] - 1)
    
    def calculate_volatility(self, prices: np.array, period: int = 252) -> float:
        """변동성 계산 (연환산)"""
        if len(prices) < 2:
            return np.nan
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(period)
    
    def calculate_roic(self, info: Dict, financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """ROIC 계산"""
        try:
            if financials.empty or balance_sheet.empty:
                return np.nan
            
            # NOPAT (Net Operating Profit After Tax)
            operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else np.nan
            tax_rate = info.get('taxRate', 0.25)  # 기본 25% 세율
            nopat = operating_income * (1 - tax_rate)
            
            # Invested Capital
            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            shareholders_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else np.nan
            invested_capital = total_debt + shareholders_equity
            
            if invested_capital and not np.isnan(invested_capital) and invested_capital != 0:
                return (nopat / invested_capital) * 100
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """투자 점수 계산 (5단계)"""
        self.logger.info("투자 점수 계산 중...")
        
        # 정규화 함수
        def normalize_score(series, ascending=True):
            if ascending:
                return (series.rank(pct=True) * 100).fillna(50)
            else:
                return ((1 - series.rank(pct=True)) * 100).fillna(50)
        
        # Value Score (PER↓, PBR↓가 좋음)
        df['Value_Score'] = (
            normalize_score(df['PER'], ascending=False) * 0.6 +
            normalize_score(df['PBR'], ascending=False) * 0.4
        )
        
        # Quality Score (ROE↑, ROIC↑가 좋음)
        df['Quality_Score'] = (
            normalize_score(df['ROE'], ascending=True) * 0.5 +
            normalize_score(df['ROIC'], ascending=True) * 0.5
        )
        
        # Momentum Score (수익률↑이 좋음)
        df['Momentum_Score'] = (
            normalize_score(df['Return_3M'], ascending=True) * 0.4 +
            normalize_score(df['Return_6M'], ascending=True) * 0.6
        )
        
        # 변동성 점수 (변동성↓이 좋음)
        df['Volatility_Score'] = normalize_score(df['Volatility'], ascending=False)
        
        # 최종 종합 점수
        df['Final_Score'] = (
            df['Value_Score'] * 0.3 +
            df['Quality_Score'] * 0.3 +
            df['Momentum_Score'] * 0.3 +
            df['Volatility_Score'] * 0.1
        )
        
        return df
    
    def clean_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제 및 정규화 (6단계)"""
        self.logger.info("데이터 정제 및 정규화 중...")
        
        # 결측치가 너무 많은 행 제거 (50% 이상 결측치)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        # 수치형 컬럼 정제
        numeric_columns = ['Close', 'PER', 'PBR', 'ROE', 'ROIC', 'EPS', 'MarketCap', 
                          'Return_3M', 'Return_6M', 'Volatility']
        
        for col in numeric_columns:
            if col in df.columns:
                # 극단값 제거 (상하위 1% 제거)
                Q1 = df[col].quantile(0.01)
                Q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=Q1, upper=Q99)
                
                # 음수 PER, PBR 제거
                if col in ['PER', 'PBR']:
                    df = df[df[col] > 0]
        
        # 결측치 처리
        df = df.fillna({
            'ROE': df['ROE'].median(),
            'ROIC': df['ROIC'].median(),
            'Sector': '기타'
        })
        
        return df
    
    def create_gemini_friendly_summary(self, df: pd.DataFrame) -> Dict:
        """🤖 제미나이 친화적 요약 생성"""
        summary = {
            "데이터_개요": {
                "총_종목수": len(df),
                "한국_종목수": len(df[df['Market'] == 'KR']),
                "미국_종목수": len(df[df['Market'] == 'US']),
                "수집_날짜": datetime.now().strftime('%Y-%m-%d')
            },
            "투자_기회_분석": {
                "저평가_고품질_종목": df[(df['Value_Score'] > 70) & (df['Quality_Score'] > 70)]['Ticker'].tolist()[:5],
                "고모멘텀_종목": df[df['Momentum_Score'] > 80]['Ticker'].tolist()[:5],
                "안정성_우수_종목": df.nsmallest(5, 'Volatility')['Ticker'].tolist()  # 변동성이 낮은 종목 = 안정적
            },
            "시장별_평균지표": {
                "한국시장": {
                    "평균_PER": round(df[df['Market'] == 'KR']['PER'].mean(), 2),
                    "평균_ROE": round(df[df['Market'] == 'KR']['ROE'].mean(), 2),
                    "평균_6개월수익률": round(df[df['Market'] == 'KR']['Return_6M'].mean() * 100, 2)
                },
                "미국시장": {
                    "평균_PER": round(df[df['Market'] == 'US']['PER'].mean(), 2),
                    "평균_ROE": round(df[df['Market'] == 'US']['ROE'].mean(), 2),
                    "평균_6개월수익률": round(df[df['Market'] == 'US']['Return_6M'].mean() * 100, 2)
                }
            },
            "투자_추천_종목_TOP10": df.nlargest(10, 'Final_Score')[['Ticker', 'Name', 'Market', 'Final_Score']].to_dict('records')
        }
        return summary
    
    def build_complete_dataset(self, target_date: str = None) -> Tuple[pd.DataFrame, Dict]:
        """완전한 데이터셋 구축 (전체 파이프라인)"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"{target_date} 기준 완전한 데이터셋 구축 시작")
        
        all_data = []
        
        # 1단계: 종목 리스트 수집
        kr_tickers = self.get_kospi200_tickers()
        us_tickers = self.get_nasdaq100_tickers()
        
        self.logger.info(f"수집 대상: 한국 {len(kr_tickers)}개, 미국 {len(us_tickers)}개 종목")
        
        # 2-4단계: 한국 주식 데이터 수집
        self.logger.info("한국 주식 데이터 수집 중...")
        for ticker in kr_tickers[:20]:  # 테스트용으로 20개만
            data = self.collect_korean_stock_data(ticker, target_date)
            if data:
                data['Date'] = target_date
                all_data.append(data)
        
        # 2-4단계: 미국 주식 데이터 수집
        self.logger.info("미국 주식 데이터 수집 중...")
        for ticker in us_tickers[:20]:  # 테스트용으로 20개만
            data = self.collect_us_stock_data(ticker, target_date)
            if data:
                data['Date'] = target_date
                all_data.append(data)
        
        if not all_data:
            self.logger.error("데이터 수집 실패")
            return pd.DataFrame(), {}
        
        # 5단계: DataFrame 생성 및 통합
        df = pd.DataFrame(all_data)
        self.logger.info(f"총 {len(df)}개 종목 데이터 수집 완료")
        
        # 투자 점수 계산을 먼저 실행 (데이터 정제 전)
        df = self.calculate_scores(df)
        
        # 6단계: 데이터 정제 및 정규화 (점수 계산 후)
        df = self.clean_and_normalize_data(df)
        
        # 컬럼 순서 정리
        df = df.reindex(columns=self.required_columns, fill_value=np.nan)
        
        # 7단계: CSV 저장
        filename = f"stock_data_{target_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        self.logger.info(f"데이터 저장 완료: {filepath}")
        
        # 제미나이 친화적 요약 생성
        summary = self.create_gemini_friendly_summary(df)
        
        # 요약 JSON 저장
        summary_filepath = os.path.join(self.data_dir, f"summary_{target_date}.json")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return df, summary
    
    def generate_gemini_prompt(self, df: pd.DataFrame, summary: Dict) -> str:
        """🤖 제미나이를 위한 최적화된 프롬프트 생성"""
        prompt = f"""
# 🤖 제미나이 투자 분석 요청

## 📊 데이터 개요
- 총 종목 수: {summary['데이터_개요']['총_종목수']}개
- 한국 주식: {summary['데이터_개요']['한국_종목수']}개
- 미국 주식: {summary['데이터_개요']['미국_종목수']}개
- 분석 기준일: {summary['데이터_개요']['수집_날짜']}

## 🎯 투자 점수 기준
- **Value Score**: PER↓ + PBR↓ (저평가 우선)
- **Quality Score**: ROE↑ + ROIC↑ (수익성 우선)
- **Momentum Score**: 3M/6M 수익률↑ (상승 추세)
- **Final Score**: 종합 점수 (0-100점)

## 📈 현재 시장 상황
### 한국 시장 평균
- PER: {summary['시장별_평균지표']['한국시장']['평균_PER']}배
- ROE: {summary['시장별_평균지표']['한국시장']['평균_ROE']}%
- 6개월 수익률: {summary['시장별_평균지표']['한국시장']['평균_6개월수익률']}%

### 미국 시장 평균
- PER: {summary['시장별_평균지표']['미국시장']['평균_PER']}배
- ROE: {summary['시장별_평균지표']['미국시장']['평균_ROE']}%
- 6개월 수익률: {summary['시장별_평균지표']['미국시장']['평균_6개월수익률']}%

## 🏆 TOP 10 추천 종목
{chr(10).join([f"{i+1}. {stock['Ticker']} ({stock['Name']}) - {stock['Market']} - 점수: {stock['Final_Score']:.1f}" 
               for i, stock in enumerate(summary['투자_추천_종목_TOP10'])])}

## ❓ 분석 요청사항
1. 위 데이터를 바탕으로 현재 시장 상황을 분석해주세요
2. TOP 10 종목 중 가장 매력적인 5개 종목을 선별하고 이유를 설명해주세요
3. 한국과 미국 시장 중 어느 쪽이 더 투자 기회가 많은지 판단해주세요
4. 향후 3-6개월 투자 전략을 제시해주세요

## 📋 답변 형식
- 구체적인 수치와 근거를 제시해주세요
- 리스크 요인도 함께 언급해주세요
- 초보자도 이해할 수 있도록 쉽게 설명해주세요
"""
        return prompt
    
    def create_data_quality_visualizations(self, data: pd.DataFrame):
        """📊 데이터 품질 시각화 - 선 그래프 중심"""
        
        # 결과 디렉토리 생성
        viz_dir = "./data_quality_charts"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. 시장별 종목 수 추세
            plt.figure(figsize=(15, 10))
            
            # 서브플롯 1: 시장별 종목 수 비교
            market_counts = data['Market'].value_counts()
            plt.plot(market_counts.index, market_counts.values, 'o-', 
                    linewidth=3, markersize=10, color='blue')
            
            for i, (market, count) in enumerate(market_counts.items()):
                plt.annotate(f'{count}개', (i, count), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
            
            plt.title('🌍 시장별 종목 수', fontsize=14, fontweight='bold')
            plt.ylabel('종목 수')
            plt.grid(True, alpha=0.3)
            
            # 서브플롯 2: PER 분포 비교 (선 그래프)
            plt.subplot(2, 2, 2)
            kr_data = data[data['Market'] == 'KR']['PER'].dropna()
            us_data = data[data['Market'] == 'US']['PER'].dropna()
            
            # 히스토그램을 선 그래프로 변환
            kr_hist, kr_bins = np.histogram(kr_data, bins=20, density=True)
            us_hist, us_bins = np.histogram(us_data, bins=20, density=True)
            
            kr_centers = (kr_bins[:-1] + kr_bins[1:]) / 2
            us_centers = (us_bins[:-1] + us_bins[1:]) / 2
            
            plt.plot(kr_centers, kr_hist, 'o-', label='한국 PER', 
                    linewidth=2, markersize=6, color='red')
            plt.plot(us_centers, us_hist, 's-', label='미국 PER', 
                    linewidth=2, markersize=6, color='blue')
            
            plt.title('📊 PER 분포 비교', fontsize=14, fontweight='bold')
            plt.xlabel('PER (배)')
            plt.ylabel('밀도')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 서브플롯 3: ROE 분포 비교 (선 그래프)
            plt.subplot(2, 2, 3)
            kr_roe = data[data['Market'] == 'KR']['ROE'].dropna()
            us_roe = data[data['Market'] == 'US']['ROE'].dropna()
            
            kr_roe_hist, kr_roe_bins = np.histogram(kr_roe, bins=20, density=True)
            us_roe_hist, us_roe_bins = np.histogram(us_roe, bins=20, density=True)
            
            kr_roe_centers = (kr_roe_bins[:-1] + kr_roe_bins[1:]) / 2
            us_roe_centers = (us_roe_bins[:-1] + us_roe_bins[1:]) / 2
            
            plt.plot(kr_roe_centers, kr_roe_hist, 'o-', label='한국 ROE', 
                    linewidth=2, markersize=6, color='green')
            plt.plot(us_roe_centers, us_roe_hist, 's-', label='미국 ROE', 
                    linewidth=2, markersize=6, color='orange')
            
            plt.title('🏆 ROE 분포 비교', fontsize=14, fontweight='bold')
            plt.xlabel('ROE (%)')
            plt.ylabel('밀도')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 서브플롯 4: 데이터 완성도 (선 그래프)
            plt.subplot(2, 2, 4)
            completeness = []
            columns = ['PER', 'ROE', '6개월수익률', '변동성']
            
            for col in columns:
                complete_ratio = (data[col].notna().sum() / len(data)) * 100
                completeness.append(complete_ratio)
            
            plt.plot(columns, completeness, 'o-', linewidth=3, markersize=10, color='purple')
            
            for i, (col, ratio) in enumerate(zip(columns, completeness)):
                plt.annotate(f'{ratio:.1f}%', (i, ratio), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
            
            plt.title('📈 데이터 완성도', fontsize=14, fontweight='bold')
            plt.ylabel('완성도 (%)')
            plt.ylim(0, 110)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/data_quality_overview_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. 시장별 평균 지표 비교 (선 그래프)
            self.create_market_indicators_comparison(data, viz_dir, timestamp)
            
            # 3. 상위 종목 분석 (선 그래프)
            self.create_top_stocks_analysis(data, viz_dir, timestamp)
            
            print(f"📊 데이터 품질 시각화 완료: {viz_dir}/")
            
        except Exception as e:
            print(f"❌ 시각화 생성 오류: {e}")
    
    def create_market_indicators_comparison(self, data: pd.DataFrame, viz_dir: str, timestamp: str):
        """시장별 지표 비교 선 그래프"""
        plt.figure(figsize=(14, 8))
        
        # 시장별 평균 지표 계산
        indicators = ['PER', 'ROE', '6개월수익률', '변동성']
        kr_data = data[data['Market'] == 'KR']
        us_data = data[data['Market'] == 'US']
        
        kr_means = [kr_data[ind].mean() for ind in indicators]
        us_means = [us_data[ind].mean() for ind in indicators]
        
        # 정규화 (0-100 스케일)
        kr_normalized = []
        us_normalized = []
        
        for i, (kr_val, us_val) in enumerate(zip(kr_means, us_means)):
            if indicators[i] in ['PER', '변동성']:  # 낮을수록 좋음
                max_val = max(kr_val, us_val)
                kr_normalized.append((max_val - kr_val) / max_val * 100)
                us_normalized.append((max_val - us_val) / max_val * 100)
            else:  # 높을수록 좋음
                max_val = max(kr_val, us_val)
                kr_normalized.append(kr_val / max_val * 100 if max_val > 0 else 0)
                us_normalized.append(us_val / max_val * 100 if max_val > 0 else 0)
        
        # 선 그래프 생성
        x_pos = range(len(indicators))
        plt.plot(x_pos, kr_normalized, 'o-', label='🇰🇷 한국 시장', 
                linewidth=4, markersize=12, color='red')
        plt.plot(x_pos, us_normalized, 's-', label='🇺🇸 미국 시장', 
                linewidth=4, markersize=12, color='blue')
        
        # 수치 표시
        for i, (kr_val, us_val) in enumerate(zip(kr_normalized, us_normalized)):
            plt.annotate(f'{kr_val:.1f}', (i, kr_val), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontweight='bold', color='red', fontsize=11)
            plt.annotate(f'{us_val:.1f}', (i, us_val), textcoords="offset points", 
                        xytext=(0,-20), ha='center', fontweight='bold', color='blue', fontsize=11)
        
        plt.title('🌍 시장별 투자 지표 비교 (정규화 점수)', fontsize=16, fontweight='bold')
        plt.xlabel('투자 지표', fontsize=12)
        plt.ylabel('정규화 점수 (0-100)', fontsize=12)
        plt.xticks(x_pos, indicators)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/market_indicators_comparison_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_top_stocks_analysis(self, data: pd.DataFrame, viz_dir: str, timestamp: str):
        """상위 종목 분석 선 그래프"""
        
        # 간단한 종합점수 계산 (PER 낮음 + ROE 높음 + 수익률 높음)
        data_clean = data.dropna(subset=['PER', 'ROE', '6개월수익률'])
        
        # 정규화 점수 계산
        data_clean['PER_score'] = (100 - data_clean['PER']) / 100 * 100  # PER 역순
        data_clean['ROE_score'] = data_clean['ROE']
        data_clean['Return_score'] = data_clean['6개월수익률'] + 50  # 음수 보정
        
        # 종합점수
        data_clean['Total_score'] = (
            data_clean['PER_score'] * 0.3 + 
            data_clean['ROE_score'] * 0.4 + 
            data_clean['Return_score'] * 0.3
        )
        
        # 상위 10개 종목
        top_stocks = data_clean.nlargest(10, 'Total_score')
        
        plt.figure(figsize=(16, 10))
        
        # 서브플롯 1: 종합점수 순위
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(top_stocks)+1), top_stocks['Total_score'], 'o-', 
                linewidth=3, markersize=8, color='purple')
        
        plt.title('🏆 상위 10개 종목 종합점수', fontsize=14, fontweight='bold')
        plt.xlabel('순위')
        plt.ylabel('종합점수')
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: PER 추세
        plt.subplot(2, 2, 2)
        kr_top = top_stocks[top_stocks['Market'] == 'KR']
        us_top = top_stocks[top_stocks['Market'] == 'US']
        
        if len(kr_top) > 0:
            plt.plot(range(1, len(kr_top)+1), kr_top['PER'], 'o-', 
                    label='한국', linewidth=2, markersize=6, color='red')
        if len(us_top) > 0:
            plt.plot(range(1, len(us_top)+1), us_top['PER'], 's-', 
                    label='미국', linewidth=2, markersize=6, color='blue')
        
        plt.title('💰 상위 종목 PER 분포', fontsize=14, fontweight='bold')
        plt.xlabel('순위')
        plt.ylabel('PER (배)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 3: ROE 추세
        plt.subplot(2, 2, 3)
        if len(kr_top) > 0:
            plt.plot(range(1, len(kr_top)+1), kr_top['ROE'], 'o-', 
                    label='한국', linewidth=2, markersize=6, color='green')
        if len(us_top) > 0:
            plt.plot(range(1, len(us_top)+1), us_top['ROE'], 's-', 
                    label='미국', linewidth=2, markersize=6, color='orange')
        
        plt.title('🏆 상위 종목 ROE 분포', fontsize=14, fontweight='bold')
        plt.xlabel('순위')
        plt.ylabel('ROE (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 4: 수익률 추세
        plt.subplot(2, 2, 4)
        if len(kr_top) > 0:
            plt.plot(range(1, len(kr_top)+1), kr_top['6개월수익률'], 'o-', 
                    label='한국', linewidth=2, markersize=6, color='navy')
        if len(us_top) > 0:
            plt.plot(range(1, len(us_top)+1), us_top['6개월수익률'], 's-', 
                    label='미국', linewidth=2, markersize=6, color='darkred')
        
        plt.title('📈 상위 종목 6개월 수익률', fontsize=14, fontweight='bold')
        plt.xlabel('순위')
        plt.ylabel('수익률 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/top_stocks_analysis_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 실행 함수"""
    print("제미나이를 위한 양질의 투자 데이터 가공 시스템")
    print("=" * 60)
    
    processor = GeminiDataProcessor()
    
    # 데이터 구축 실행
    df, summary = processor.build_complete_dataset()
    
    if df.empty:
        print("데이터 수집에 실패했습니다.")
        return
    
    print(f"\n데이터 구축 완료!")
    print(f"총 {len(df)}개 종목 데이터 수집")
    print(f"한국: {len(df[df['Market'] == 'KR'])}개")
    print(f"미국: {len(df[df['Market'] == 'US'])}개")
    
    # 📊 데이터 품질 시각화 생성 (선 그래프 중심)
    print("\n📊 데이터 품질 시각화 생성 중...")
    processor.create_data_quality_visualizations(df)
    
    # 제미나이 프롬프트 생성
    gemini_prompt = processor.generate_gemini_prompt(df, summary)
    
    # 프롬프트 저장
    prompt_filepath = os.path.join(processor.data_dir, f"gemini_prompt_{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(prompt_filepath, 'w', encoding='utf-8') as f:
        f.write(gemini_prompt)
    
    print(f"\n제미나이 프롬프트 생성 완료: {prompt_filepath}")
    print("\n" + "="*60)
    print("제미나이에게 다음 프롬프트를 입력하세요:")
    print("="*60)
    print(gemini_prompt[:1000] + "..." if len(gemini_prompt) > 1000 else gemini_prompt)

if __name__ == "__main__":
    main() 