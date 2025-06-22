"""
섹터/산업 비교 분석기 - 동종업계 대비 성과 분석
"""
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import quote

class SectorAnalyzer:
    """섹터/산업 비교 분석"""
    
    def __init__(self):
        # 주요 섹터별 대표 종목들
        self.sector_mapping = {
            '반도체': ['005930', '000660', '006400'],  # 삼성전자, SK하이닉스, 삼성SDI
            'IT서비스': ['035420', '035720'],  # 네이버, 카카오
            '화학': ['051910', '009830'],  # LG화학, 한화솔루션
            '금융': ['323410', '086790'],  # 카카오뱅크, 하나금융지주
            '바이오': ['068270', '196170'],  # 셀트리온, 알테오젠
            '건설': ['028260', '000720'],  # 삼성물산, 현대건설
            '자동차': ['005380', '012330'],  # 현대차, 현대모비스
            '철강': ['005490', '000810'],  # POSCO홀딩스, 삼성화재
            '에너지': ['096770', '010950'],  # SK이노베이션, S-Oil
            '통신': ['030200', '017670']   # KT, SKT
        }
        
        # 섹터별 평균 밸류에이션 (참고값)
        self.sector_averages = {
            '반도체': {'per': 15.5, 'pbr': 1.4, 'roe': 12.5},
            'IT서비스': {'per': 22.0, 'pbr': 2.8, 'roe': 14.2},
            '화학': {'per': 18.2, 'pbr': 1.6, 'roe': 9.8},
            '금융': {'per': 8.5, 'pbr': 0.8, 'roe': 8.5},
            '바이오': {'per': 28.5, 'pbr': 3.2, 'roe': 15.8},
            '건설': {'per': 12.8, 'pbr': 0.9, 'roe': 7.2},
            '자동차': {'per': 16.2, 'pbr': 1.2, 'roe': 8.8},
            '철강': {'per': 14.5, 'pbr': 1.1, 'roe': 6.5},
            '에너지': {'per': 13.8, 'pbr': 1.0, 'roe': 7.8},
            '통신': {'per': 11.2, 'pbr': 1.3, 'roe': 9.2}
        }
        
        # 인베스팅닷컴 섹터 URL 매핑 (실제 URL로 수정)
        self.investing_sector_urls = {
            '반도체': 'https://www.investing.com/indices/kospi-200',
            'IT서비스': 'https://www.investing.com/indices/kosdaq-150',
            '화학': 'https://www.investing.com/indices/kospi-200',
            '금융': 'https://www.investing.com/indices/kospi-200',
            '바이오': 'https://www.investing.com/indices/kosdaq-150',
            '건설': 'https://www.investing.com/indices/kospi-200',
            '자동차': 'https://www.investing.com/indices/kospi-200',
            '철강': 'https://www.investing.com/indices/kospi-200',
            '에너지': 'https://www.investing.com/indices/kospi-200',
            '통신': 'https://www.investing.com/indices/kospi-200'
        }
        
        # 대안 URL (야후 파이낸스 한국 섹터)
        self.alternative_urls = {
            '반도체': 'https://finance.yahoo.com/quote/%5EKS11',
            'IT서비스': 'https://finance.yahoo.com/quote/%5EKOSDAQ',
            '화학': 'https://finance.yahoo.com/quote/%5EKS11',
            '금융': 'https://finance.yahoo.com/quote/%5EKS11',
            '바이오': 'https://finance.yahoo.com/quote/%5EKOSDAQ'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def analyze_sector_comparison(self, stock_code: str, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """섹터 비교 분석"""
        try:
            # 종목의 섹터 찾기
            sector = self._find_sector(stock_code)
            
            if not sector:
                return self._create_default_sector_analysis()
            
            # 섹터 평균과 비교
            comparison = self._compare_with_sector(stock_data, sector)
            
            # 섹터 내 순위 계산
            ranking = self._calculate_sector_ranking(stock_data, sector)
            
            # 섹터 트렌드 분석
            trend = self._analyze_sector_trend(sector)
            
            return {
                'sector': sector,
                'sector_comparison': comparison,
                'sector_ranking': ranking,
                'sector_trend': trend,
                'competitive_advantage': self._analyze_competitive_advantage(stock_code, sector),
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._create_default_sector_analysis(f"분석 오류: {str(e)}")
    
    def _find_sector(self, stock_code: str) -> str:
        """종목의 섹터 찾기"""
        for sector, codes in self.sector_mapping.items():
            if stock_code in codes:
                return sector
        return None
    
    def _compare_with_sector(self, stock_data: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """섹터 평균과 비교"""
        sector_avg = self.sector_averages.get(sector, {'per': 15.0, 'pbr': 1.5, 'roe': 10.0})
        
        fundamental = stock_data.get('fundamental', {})
        stock_per = fundamental.get('per', 0)
        stock_pbr = fundamental.get('pbr', 0)
        stock_roe = fundamental.get('roe', 0)
        
        return {
            'per_vs_sector': {
                'stock': stock_per,
                'sector_avg': sector_avg['per'],
                'premium_discount': round(((stock_per / sector_avg['per'] - 1) * 100), 1) if sector_avg['per'] > 0 else 0,
                'evaluation': '할인' if stock_per < sector_avg['per'] else '프리미엄'
            },
            'pbr_vs_sector': {
                'stock': stock_pbr,
                'sector_avg': sector_avg['pbr'],
                'premium_discount': round(((stock_pbr / sector_avg['pbr'] - 1) * 100), 1) if sector_avg['pbr'] > 0 else 0,
                'evaluation': '할인' if stock_pbr < sector_avg['pbr'] else '프리미엄'
            },
            'roe_vs_sector': {
                'stock': stock_roe,
                'sector_avg': sector_avg['roe'],
                'outperformance': round((stock_roe - sector_avg['roe']), 1),
                'evaluation': '우수' if stock_roe > sector_avg['roe'] else '평균 이하'
            }
        }
    
    def _calculate_sector_ranking(self, stock_data: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """섹터 내 순위 계산 (추정)"""
        fundamental = stock_data.get('fundamental', {})
        roe = fundamental.get('roe', 0)
        per = fundamental.get('per', 999)
        
        # ROE 기준 순위 (높을수록 좋음)
        roe_rank = 1 if roe > 15 else 2 if roe > 10 else 3 if roe > 5 else 4
        
        # PER 기준 순위 (낮을수록 좋음, 단 0은 제외)
        if per > 0:
            per_rank = 1 if per < 10 else 2 if per < 15 else 3 if per < 20 else 4
        else:
            per_rank = 4
        
        # 종합 순위
        total_rank = min(round((roe_rank + per_rank) / 2), 4)
        
        rank_labels = {1: '상위 25%', 2: '상위 50%', 3: '하위 50%', 4: '하위 25%'}
        
        return {
            'roe_ranking': rank_labels[roe_rank],
            'valuation_ranking': rank_labels[per_rank],
            'overall_ranking': rank_labels[total_rank],
            'competitive_position': '리더' if total_rank <= 2 else '팔로워'
        }
    
    def _analyze_sector_trend(self, sector: str) -> Dict[str, Any]:
        """섹터 트렌드 분석 - 인베스팅닷컴 실시간 데이터 활용"""
        try:
            # 실시간 섹터 데이터 수집
            real_time_data = self._get_investing_sector_data(sector)
            
            if real_time_data:
                return self._create_trend_analysis(sector, real_time_data)
            else:
                # 실시간 데이터 실패시 기본 데이터 사용
                return self._get_default_sector_trend(sector)
                
        except Exception as e:
            print(f"섹터 트렌드 분석 오류: {e}")
            return self._get_default_sector_trend(sector)
    
    def _get_investing_sector_data(self, sector: str) -> Dict[str, Any]:
        """인베스팅닷컴에서 섹터 데이터 수집 (대안 포함)"""
        try:
            # 1차 시도: 인베스팅닷컴
            sector_data = self._try_investing_com(sector)
            if sector_data:
                return sector_data
            
            # 2차 시도: 야후 파이낸스
            sector_data = self._try_yahoo_finance(sector)
            if sector_data:
                return sector_data
            
            # 3차 시도: 네이버 금융 (한국 지수)
            sector_data = self._try_naver_finance(sector)
            if sector_data:
                return sector_data
                
            return None
            
        except Exception as e:
            print(f"전체 섹터 데이터 수집 오류 ({sector}): {e}")
            return None
    
    def _try_investing_com(self, sector: str) -> Dict[str, Any]:
        """인베스팅닷컴 시도"""
        try:
            sector_url = self.investing_sector_urls.get(sector)
            if not sector_url:
                return None
            
            session = requests.Session()
            session.headers.update(self.headers)
            
            response = session.get(sector_url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            sector_data = {'source': 'Investing.com'}
            
            # 다양한 셀렉터 시도
            price_selectors = [
                '[data-test="instrument-price-last"]',
                '.text-2xl',
                '.pid-169-last',
                '.last-price-value'
            ]
            
            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem:
                    sector_data['current_price'] = price_elem.get_text(strip=True)
                    break
            
            # 등락률 추출
            change_selectors = [
                '[data-test="instrument-price-change-percent"]',
                '.pid-169-pc',
                '.change-percent'
            ]
            
            for selector in change_selectors:
                change_elem = soup.select_one(selector)
                if change_elem:
                    change_text = change_elem.get_text(strip=True)
                    sector_data['change_percent'] = change_text
                    
                    if '+' in change_text:
                        sector_data['trend_direction'] = '상승'
                    elif '-' in change_text:
                        sector_data['trend_direction'] = '하락'
                    else:
                        sector_data['trend_direction'] = '보합'
                    break
            
            return sector_data if sector_data.get('current_price') else None
            
        except Exception as e:
            print(f"인베스팅닷컴 시도 실패 ({sector}): {e}")
            return None
    
    def _try_yahoo_finance(self, sector: str) -> Dict[str, Any]:
        """야후 파이낸스 시도"""
        try:
            import yfinance as yf
            
            # 섹터별 대표 티커
            tickers = {
                '반도체': '^KS11',  # 코스피200
                'IT서비스': '^KOSDAQ',  # 코스닥
                '화학': '^KS11',
                '금융': '^KS11',
                '바이오': '^KOSDAQ'
            }
            
            ticker = tickers.get(sector, '^KS11')
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return None
            
            sector_data = {
                'source': 'Yahoo Finance',
                'current_price': str(info.get('regularMarketPrice', 'N/A')),
                'change_percent': f"{info.get('regularMarketChangePercent', 0):.2f}%"
            }
            
            change_percent = info.get('regularMarketChangePercent', 0)
            if change_percent > 0:
                sector_data['trend_direction'] = '상승'
            elif change_percent < 0:
                sector_data['trend_direction'] = '하락'
            else:
                sector_data['trend_direction'] = '보합'
            
            return sector_data
            
        except Exception as e:
            print(f"야후 파이낸스 시도 실패 ({sector}): {e}")
            return None
    
    def _try_naver_finance(self, sector: str) -> Dict[str, Any]:
        """네이버 금융 시도"""
        try:
            # 코스피200 지수 정보
            url = "https://finance.naver.com/sise/sise_index.naver?code=KPI200"
            
            session = requests.Session()
            session.headers.update(self.headers)
            
            response = session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            sector_data = {'source': '네이버 금융'}
            
            # 현재가
            price_elem = soup.select_one('.no_today .blind')
            if price_elem:
                sector_data['current_price'] = price_elem.get_text(strip=True)
            
            # 등락률
            change_elem = soup.select_one('.no_exday .blind')
            if change_elem:
                change_text = change_elem.get_text(strip=True)
                sector_data['change_percent'] = change_text
                
                if '+' in change_text:
                    sector_data['trend_direction'] = '상승'
                elif '-' in change_text:
                    sector_data['trend_direction'] = '하락'
                else:
                    sector_data['trend_direction'] = '보합'
            
            return sector_data if sector_data.get('current_price') else None
            
        except Exception as e:
            print(f"네이버 금융 시도 실패 ({sector}): {e}")
            return None
    
    def _create_trend_analysis(self, sector: str, real_data: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 데이터 기반 트렌드 분석 생성"""
        try:
            change_percent = real_data.get('change_percent', '0%')
            trend_direction = real_data.get('trend_direction', '보합')
            
            # 등락률에 따른 트렌드 강도 판단
            if '%' in change_percent:
                percent_value = float(re.findall(r'[-+]?\d*\.?\d+', change_percent)[0])
                
                if percent_value >= 2.0:
                    trend_strength = '강세'
                elif percent_value >= 1.0:
                    trend_strength = '상승'
                elif percent_value >= -1.0:
                    trend_strength = '보합'
                elif percent_value >= -2.0:
                    trend_strength = '하락'
                else:
                    trend_strength = '약세'
            else:
                trend_strength = '보합'
            
            # 섹터별 맞춤 분석
            outlook = self._generate_sector_outlook(sector, trend_direction, trend_strength)
            risk_factors = self._get_sector_risks(sector, trend_direction)
            growth_drivers = self._get_growth_drivers(sector, trend_strength)
            
            return {
                'trend': trend_strength,
                'direction': trend_direction,
                'change_percent': change_percent,
                'outlook': outlook,
                'risk_factors': risk_factors,
                'growth_drivers': growth_drivers,
                'recent_news': real_data.get('recent_news', []),
                'data_source': 'Investing.com 실시간',
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"트렌드 분석 생성 오류: {e}")
            return self._get_default_sector_trend(sector)
    
    def _generate_sector_outlook(self, sector: str, direction: str, strength: str) -> str:
        """섹터별 맞춤 전망 생성"""
        base_outlooks = {
            '반도체': f"메모리 반도체 {direction} 추세, AI 수요 지속",
            'IT서비스': f"디지털 전환 {direction} 모멘텀, 클라우드 확산",
            '화학': f"원자재 가격 {direction}, 친환경 소재 수요 증가",
            '금융': f"금리 환경 변화로 {direction} 영향, 디지털 금융 성장",
            '바이오': f"신약 개발 {direction} 흐름, 고령화 수혜 지속"
        }
        
        base = base_outlooks.get(sector, f"{sector} 업종 {direction} 추세")
        return f"{base} ({strength} 신호)"
    
    def _get_sector_risks(self, sector: str, direction: str) -> List[str]:
        """섹터별 리스크 요인"""
        common_risks = {
            '반도체': ['중국 리스크', '재고 조정'],
            'IT서비스': ['규제 강화', '경쟁 심화'],
            '화학': ['유가 변동성', '환경 규제'],
            '금융': ['부실 우려', '핀테크 위협'],
            '바이오': ['규제 리스크', '개발 실패']
        }
        
        risks = common_risks.get(sector, ['거시경제 불확실성'])
        
        # 방향성에 따른 추가 리스크
        if direction == '하락':
            risks.append('단기 조정 위험')
        elif direction == '상승':
            risks.append('과열 우려')
            
        return risks
    
    def _get_growth_drivers(self, sector: str, strength: str) -> List[str]:
        """섹터별 성장 동력"""
        drivers = {
            '반도체': ['AI/HPC 수요', '서버 메모리'],
            'IT서비스': ['AI 서비스', '클라우드'],
            '화학': ['친환경 소재', '배터리 소재'],
            '금융': ['디지털 뱅킹', '자산관리'],
            '바이오': ['항체 치료제', '맞춤 의료']
        }
        
        base_drivers = drivers.get(sector, ['구조적 성장'])
        
        # 강도에 따른 추가 동력
        if strength in ['강세', '상승']:
            base_drivers.append('모멘텀 지속')
            
        return base_drivers
    
    def _get_default_sector_trend(self, sector: str) -> Dict[str, Any]:
        """기본 섹터 트렌드 (백업용)"""
        sector_trends = {
            '반도체': {
                'trend': '강세',
                'outlook': '메모리 반도체 슈퍼사이클 진입, AI 수요 급증',
                'risk_factors': ['중국 리스크', '재고 조정 우려'],
                'growth_drivers': ['AI/HPC 수요', '서버 메모리 확대']
            },
            'IT서비스': {
                'trend': '성장',
                'outlook': '디지털 전환 가속화, 클라우드 확산',
                'risk_factors': ['규제 강화', '경쟁 심화'],
                'growth_drivers': ['AI 서비스', '클라우드 전환']
            },
            '화학': {
                'trend': '회복',
                'outlook': '원자재 가격 안정화, 수요 회복',
                'risk_factors': ['유가 변동성', '환경 규제'],
                'growth_drivers': ['친환경 소재', '배터리 소재']
            },
            '금융': {
                'trend': '안정',
                'outlook': '금리 상승 수혜, 디지털 금융 확산',
                'risk_factors': ['부실 우려', '핀테크 위협'],
                'growth_drivers': ['디지털 뱅킹', '자산관리']
            },
            '바이오': {
                'trend': '성장',
                'outlook': '고령화 진행, 혁신 신약 개발',
                'risk_factors': ['규제 리스크', '개발 실패'],
                'growth_drivers': ['항체 치료제', '개인 맞춤 의료']
            }
        }
        
        default_trend = sector_trends.get(sector, {
            'trend': '중립',
            'outlook': '업종별 차별화 진행',
            'risk_factors': ['거시경제 불확실성'],
            'growth_drivers': ['구조적 성장 동력']
        })
        
        default_trend['data_source'] = '기본 데이터 (백업)'
        return default_trend
    
    def _analyze_competitive_advantage(self, stock_code: str, sector: str) -> Dict[str, Any]:
        """경쟁 우위 분석"""
        # 종목별 경쟁 우위 요소 (실제로는 더 정교한 분석 필요)
        competitive_map = {
            '005930': {  # 삼성전자
                'advantages': ['글로벌 1위 메모리', '수직계열화', '기술력'],
                'moat_strength': '매우 강함',
                'market_position': '절대 강자'
            },
            '000660': {  # SK하이닉스
                'advantages': ['메모리 2위', 'AI 메모리 특화', '기술 혁신'],
                'moat_strength': '강함',
                'market_position': '강력한 2위'
            },
            '035420': {  # 네이버
                'advantages': ['국내 검색 독점', '플랫폼 생태계', '데이터'],
                'moat_strength': '강함',
                'market_position': '국내 1위'
            }
        }
        
        return competitive_map.get(stock_code, {
            'advantages': ['업계 경험', '안정적 사업구조'],
            'moat_strength': '보통',
            'market_position': '중견 기업'
        })
    
    def _create_default_sector_analysis(self, error_msg: str = None) -> Dict[str, Any]:
        """기본 섹터 분석"""
        return {
            'sector': '기타',
            'sector_comparison': {
                'per_vs_sector': {'evaluation': '분석 불가'},
                'pbr_vs_sector': {'evaluation': '분석 불가'},
                'roe_vs_sector': {'evaluation': '분석 불가'}
            },
            'sector_ranking': {
                'overall_ranking': '분석 불가',
                'competitive_position': '분석 불가'
            },
            'sector_trend': {
                'trend': '중립',
                'outlook': error_msg or '섹터 분석 데이터 부족'
            },
            'competitive_advantage': {
                'moat_strength': '분석 불가'
            },
            'analysis_time': datetime.now().isoformat()
        } 