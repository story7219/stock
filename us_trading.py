"""
🚀 고급 AI 스캘핑 & 데이트레이딩 시스템 
깃허브 분석 결과를 바탕으로 한 고급 기법 적용

주요 기능:
1. ATR 기반 동적 스캘핑 전략 (Alpaca 스타일)
2. 멀티타임프레임 분석 (1분, 5분, 15분)
3. 실시간 모멘텀 스코어링
4. 고급 호가창 분석 (Depth Analysis)
5. 시장 마이크로스트럭처 분석
6. AI 기반 진입/청산 타이밍
7. 🇺🇸 미국주식 나스닥100, S&P500 TOP5 추천 시스템
"""
import asyncio
import logging
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional, Callable
import numpy as np
from collections import deque
import statistics
import yfinance as yf
import requests

logger = logging.getLogger(__name__)

class AdvancedScalpingAI:
    """🎯 고급 스캘핑 & 데이트레이딩 AI 시스템 (한국 + 미국주식 통합)"""
    
    def __init__(self, trader):
        """초기화 - CoreTrader와 연동"""
        self.trader = trader
        self.is_monitoring = False
        self.price_buffer = {}  # 실시간 가격 버퍼 (심볼별 최근 100개)
        self.volume_buffer = {}  # 거래량 버퍼
        self.momentum_scores = {}  # 모멘텀 점수 추적
        self.atr_values = {}  # ATR 값들
        
        # 콜백 함수들
        self.scalping_signal_callback = None
        self.risk_alert_callback = None
        
        # 🇺🇸 미국주식 지수 구성종목 캐시
        self.nasdaq100_symbols = []
        self.sp500_symbols = []
        self.us_stock_cache = {}
        
        # 미국주식 초기화
        self._initialize_us_indices()
    
    # === 🇺🇸 미국주식 지수 초기화 ===
    def _initialize_us_indices(self):
        """나스닥100, S&P500 구성종목 초기화"""
        try:
            logger.info("🇺🇸 미국주식 지수 구성종목 초기화 중...")
            
            # 나스닥100 주요 종목들 (실제로는 API에서 가져와야 함)
            self.nasdaq100_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 
                'COST', 'NFLX', 'ADBE', 'PEP', 'TMUS', 'CSCO', 'INTC', 'CMCSA',
                'TXN', 'QCOM', 'INTU', 'AMAT', 'AMD', 'ISRG', 'BKNG', 'HON',
                'AMGN', 'VRTX', 'ADP', 'GILD', 'ADI', 'SBUX', 'LRCX', 'MELI',
                'KLAC', 'REGN', 'SNPS', 'CDNS', 'MRVL', 'ORLY', 'CSX', 'FTNT',
                'NXPI', 'DXCM', 'ADSK', 'ABNB', 'CHTR', 'ASML', 'TEAM', 'RGEN',
                'WDAY', 'FANG', 'BIIB', 'ILMN', 'KDP', 'CRWD', 'MRNA', 'SIRI'
            ]
            
            # S&P500 주요 종목들 (나스닥과 중복 제외한 대표 종목들)
            self.sp500_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
                'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
                'LLY', 'BAC', 'KO', 'AVGO', 'PFE', 'TMO', 'COST', 'WMT', 'DIS',
                'ABT', 'CRM', 'ACN', 'VZ', 'ADBE', 'DHR', 'NEE', 'BMY', 'TXN',
                'RTX', 'QCOM', 'PM', 'SPGI', 'T', 'SCHW', 'HON', 'UPS', 'LOW',
                'INTU', 'AMGN', 'CAT', 'ISRG', 'AMAT', 'GS', 'DE', 'BKNG', 'AMD'
            ]
            
            logger.info(f"✅ 나스닥100: {len(self.nasdaq100_symbols)}개 종목")
            logger.info(f"✅ S&P500: {len(self.sp500_symbols)}개 종목")
            
        except Exception as e:
            logger.error(f"❌ 미국주식 지수 초기화 실패: {e}")
    
    # === 🇺🇸 미국주식 TOP5 추천 시스템 ===
    async def analyze_nasdaq100_top5(self, strategy_name: str = "blackrock") -> List[Dict]:
        """나스닥100 TOP5 종목 추천 (투자대가 방식)"""
        try:
            logger.info(f"🇺🇸 나스닥100 {strategy_name.upper()} 전략 TOP5 종목 분석 시작...")
            
            # 병렬로 모든 나스닥100 종목 분석
            tasks = []
            for symbol in self.nasdaq100_symbols[:20]:  # 처음 20개만 분석 (API 한도 고려)
                task = asyncio.create_task(self._analyze_us_stock(symbol, strategy_name))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 성공한 결과만 필터링
            valid_results = []
            for result in results:
                if isinstance(result, dict) and 'error' not in result:
                    valid_results.append(result)
            
            # 점수순 정렬 후 TOP5 선택
            sorted_results = sorted(valid_results, key=lambda x: x.get('score', 0), reverse=True)
            top5_results = sorted_results[:5]
            
            logger.info(f"✅ 나스닥100 {strategy_name.upper()} TOP5 분석 완료: {len(top5_results)}개 종목")
            return top5_results
            
        except Exception as e:
            logger.error(f"❌ 나스닥100 {strategy_name.upper()} TOP5 분석 실패: {e}")
            return []
    
    async def analyze_sp500_top5(self, strategy_name: str = "warren_buffett") -> List[Dict]:
        """S&P500 TOP5 종목 추천 (투자대가 방식)"""
        try:
            logger.info(f"🇺🇸 S&P500 {strategy_name.upper()} 전략 TOP5 종목 분석 시작...")
            
            # 병렬로 모든 S&P500 종목 분석
            tasks = []
            for symbol in self.sp500_symbols[:20]:  # 처음 20개만 분석 (API 한도 고려)
                task = asyncio.create_task(self._analyze_us_stock(symbol, strategy_name))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 성공한 결과만 필터링
            valid_results = []
            for result in results:
                if isinstance(result, dict) and 'error' not in result:
                    valid_results.append(result)
            
            # 점수순 정렬 후 TOP5 선택
            sorted_results = sorted(valid_results, key=lambda x: x.get('score', 0), reverse=True)
            top5_results = sorted_results[:5]
            
            logger.info(f"✅ S&P500 {strategy_name.upper()} TOP5 분석 완료: {len(top5_results)}개 종목")
            return top5_results
            
        except Exception as e:
            logger.error(f"❌ S&P500 {strategy_name.upper()} TOP5 분석 실패: {e}")
            return []
    
    async def _analyze_us_stock(self, symbol: str, strategy_name: str) -> Dict:
        """개별 미국주식 분석"""
        try:
            # 미국주식 데이터 수집
            stock_data = await self._get_us_stock_data(symbol)
            if not stock_data:
                return {'error': f'{symbol} 데이터 수집 실패'}
            
            # 전략별 분석 수행
            if strategy_name == "william_oneil":
                analysis_result = self._analyze_us_william_oneil_strategy(stock_data)
            elif strategy_name == "jesse_livermore":
                analysis_result = self._analyze_us_jesse_livermore_strategy(stock_data)
            elif strategy_name == "ichimoku":
                analysis_result = self._analyze_us_ichimoku_strategy(stock_data)
            elif strategy_name == "warren_buffett":
                analysis_result = self._analyze_us_warren_buffett_strategy(stock_data)
            elif strategy_name == "peter_lynch":
                analysis_result = self._analyze_us_peter_lynch_strategy(stock_data)
            elif strategy_name == "blackrock":
                analysis_result = self._analyze_us_blackrock_strategy(stock_data)
            else:
                analysis_result = self._analyze_us_comprehensive_strategy(stock_data)
            
            return {
                'symbol': symbol,
                'name': stock_data.get('name', symbol),
                'current_price': stock_data.get('current_price', 0),
                'score': analysis_result.get('score', 50),
                'recommendation': analysis_result.get('recommendation', 'HOLD'),
                'reason': analysis_result.get('reason', '종합 분석 결과'),
                'market_cap': stock_data.get('market_cap', 0),
                'volume': stock_data.get('volume', 0),
                'change_rate': stock_data.get('change_rate', 0),
                'strategy': strategy_name,
                'country': 'US',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} 미국주식 분석 실패: {e}")
            return {'error': str(e)}
    
    async def _get_us_stock_data(self, symbol: str) -> Optional[Dict]:
        """미국주식 데이터 수집 (yfinance 활용)"""
        try:
            # 캐시 확인
            cache_key = f"us_stock_{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.us_stock_cache:
                return self.us_stock_cache[cache_key]
            
            # yfinance로 데이터 수집
            ticker = yf.Ticker(symbol)
            
            # 기본 정보
            info = ticker.info
            hist = ticker.history(period="1mo")  # 1개월 데이터
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            # 기술적 지표 계산
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1]
            rsi = self._calculate_rsi(hist['Close'].values)
            
            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': round(current_price, 2),
                'volume': int(volume),
                'market_cap': info.get('marketCap', 0),
                'change_rate': round(((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100), 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'rsi': round(rsi, 2),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            # 캐시 저장
            self.us_stock_cache[cache_key] = stock_data
            
            return stock_data
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 미국주식 데이터 수집 실패: {e}")
            return None
    
    # === 🇺🇸 미국주식 전략별 분석 메서드들 ===
    def _analyze_us_william_oneil_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 윌리엄 오닐 전략 분석 (CAN SLIM)"""
        try:
            score = 50  # 기본 점수
            reasons = []
            
            # C - Current Earnings (현재 수익)
            profit_margin = stock_data.get('profit_margin', 0)
            if profit_margin > 0.20:  # 20% 이상
                score += 15
                reasons.append("높은 수익률 (C)")
            elif profit_margin > 0.10:
                score += 8
                reasons.append("양호한 수익률 (C)")
            
            # A - Annual Earnings (연간 수익 증가)
            revenue_growth = stock_data.get('revenue_growth', 0)
            if revenue_growth > 0.25:  # 25% 이상
                score += 15
                reasons.append("강한 매출 성장 (A)")
            elif revenue_growth > 0.15:
                score += 10
                reasons.append("양호한 매출 성장 (A)")
            
            # N - New (신제품, 신경영진, 신고가)
            current_price = stock_data.get('current_price', 0)
            change_rate = stock_data.get('change_rate', 0)
            if change_rate > 5:  # 5% 이상 상승
                score += 15
                reasons.append("강한 신고가 돌파 (N)")
            elif change_rate > 2:
                score += 8
                reasons.append("상승 모멘텀 (N)")
            
            # S - Supply and Demand (수급)
            volume = stock_data.get('volume', 0)
            if volume > 1000000:  # 높은 거래량
                score += 10
                reasons.append("활발한 거래량 (S)")
            
            # L - Leader or Laggard (선도주)
            sector = stock_data.get('sector', '')
            if 'Technology' in sector or 'Healthcare' in sector:
                score += 10
                reasons.append("선도 섹터 (L)")
            
            # I - Institutional Sponsorship (기관 후원)
            market_cap = stock_data.get('market_cap', 0)
            if market_cap > 50_000_000_000:  # 500억 달러 이상
                score += 10
                reasons.append("대형주 안정성 (I)")
            
            # M - Market Direction (시장 방향)
            rsi = stock_data.get('rsi', 50)
            if 50 < rsi < 70:
                score += 5
                reasons.append("시장 모멘텀 양호 (M)")
            
            score = max(0, min(100, score))
            
            if score >= 75:
                recommendation = "STRONG_BUY"
            elif score >= 60:
                recommendation = "BUY"
            elif score <= 40:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"윌리엄 오닐 CAN SLIM: {', '.join(reasons) if reasons else '중립적 신호'}"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}
    
    def _analyze_us_jesse_livermore_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 제시 리버모어 전략 분석 (추세 추종)"""
        try:
            score = 50  # 기본 점수
            reasons = []
            
            current_price = stock_data.get('current_price', 0)
            sma_20 = stock_data.get('sma_20', 0)
            sma_50 = stock_data.get('sma_50', 0)
            rsi = stock_data.get('rsi', 50)
            change_rate = stock_data.get('change_rate', 0)
            volume = stock_data.get('volume', 0)
            
            # 추세 분석 (가장 중요)
            if current_price > sma_20 > sma_50:
                score += 25
                reasons.append("강한 상승 추세")
            elif current_price > sma_20:
                score += 15
                reasons.append("단기 상승 추세")
            elif current_price < sma_20 < sma_50:
                score -= 15
                reasons.append("하락 추세")
            
            # 모멘텀 분석
            if change_rate > 3:
                score += 20
                reasons.append("강한 상승 모멘텀")
            elif change_rate > 1:
                score += 10
                reasons.append("상승 모멘텀")
            elif change_rate < -3:
                score -= 15
                reasons.append("하락 모멘텀")
            
            # 거래량 분석 (리버모어는 거래량 중시)
            if volume > 2000000:  # 높은 거래량
                score += 15
                reasons.append("높은 거래량 확인")
            elif volume > 1000000:
                score += 8
                reasons.append("적정 거래량")
            
            # RSI 모멘텀
            if 50 < rsi < 80:
                score += 10
                reasons.append("모멘텀 지속")
            elif rsi > 80:
                score -= 5
                reasons.append("과매수 주의")
            
            score = max(0, min(100, score))
            
            if score >= 75:
                recommendation = "STRONG_BUY"
            elif score >= 60:
                recommendation = "BUY"
            elif score <= 35:
                recommendation = "STRONG_SELL"
            elif score <= 45:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"제시 리버모어 추세추종: {', '.join(reasons) if reasons else '중립적 신호'}"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}
    
    def _analyze_us_ichimoku_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 일목산인 전략 분석 (일목균형표)"""
        try:
            score = 50  # 기본 점수
            reasons = []
            
            current_price = stock_data.get('current_price', 0)
            sma_20 = stock_data.get('sma_20', 0)
            sma_50 = stock_data.get('sma_50', 0)
            rsi = stock_data.get('rsi', 50)
            change_rate = stock_data.get('change_rate', 0)
            
            # 일목균형표 대용 분석 (단순화)
            # 전환선(9일) > 기준선(26일) 개념으로 SMA20 > SMA50 사용
            if current_price > sma_20 and sma_20 > sma_50:
                score += 20
                reasons.append("정배열 (구름대 위)")
            elif current_price > sma_20:
                score += 10
                reasons.append("전환선 돌파")
            
            # 후행스팬 분석 (현재가 vs 과거가 비교)
            if change_rate > 0:
                score += 10
                reasons.append("후행스팬 양호")
            
            # 선행스팬 분석 (구름대 두께 - 변동성으로 대체)
            price_volatility = abs(change_rate)
            if price_volatility < 2:  # 낮은 변동성 = 얇은 구름대
                score += 5
                reasons.append("구름대 얇음")
            elif price_volatility > 5:  # 높은 변동성 = 두꺼운 구름대
                score -= 5
                reasons.append("구름대 두꺼움")
            
            # 삼역전환 (강한 신호)
            if current_price > sma_20 and change_rate > 2 and 50 < rsi < 70:
                score += 15
                reasons.append("삼역전환 신호")
            
            # 시간론 (시간 사이클 - 간단히 RSI로 대체)
            if 30 < rsi < 70:
                score += 10
                reasons.append("시간론 균형")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "STRONG_BUY"
            elif score >= 55:
                recommendation = "BUY"
            elif score <= 40:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"일목산인 균형표: {', '.join(reasons) if reasons else '중립적 신호'}"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}
    
    def _analyze_us_warren_buffett_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 워렌 버핏 전략 분석 (가치투자)"""
        try:
            score = 50  # 기본 점수
            reasons = []
            
            pe_ratio = stock_data.get('pe_ratio', 0)
            pb_ratio = stock_data.get('pb_ratio', 0)
            roe = stock_data.get('roe', 0)
            debt_to_equity = stock_data.get('debt_to_equity', 0)
            profit_margin = stock_data.get('profit_margin', 0)
            market_cap = stock_data.get('market_cap', 0)
            
            # 경제적 해자 (브랜드 파워 - 대형주로 판단)
            if market_cap > 100_000_000_000:  # 1000억 달러 이상
                score += 15
                reasons.append("경제적 해자 (대형주)")
            elif market_cap > 50_000_000_000:
                score += 10
                reasons.append("준대형주 안정성")
            
            # ROE (자기자본이익률)
            if roe > 0.15:  # 15% 이상
                score += 20
                reasons.append("우수한 ROE")
            elif roe > 0.10:
                score += 10
                reasons.append("양호한 ROE")
            
            # 적정 PER (저평가 선호)
            if 5 <= pe_ratio <= 15:
                score += 15
                reasons.append("적정 PER")
            elif 15 < pe_ratio <= 25:
                score += 5
                reasons.append("보통 PER")
            elif pe_ratio > 30:
                score -= 10
                reasons.append("높은 PER")
            
            # 낮은 PBR
            if 0 < pb_ratio < 1.5:
                score += 10
                reasons.append("저평가 PBR")
            elif 1.5 <= pb_ratio < 3:
                score += 5
                reasons.append("적정 PBR")
            
            # 건전한 재무구조
            if debt_to_equity < 0.6:
                score += 15
                reasons.append("건전한 부채비율")
            elif debt_to_equity < 1.0:
                score += 5
                reasons.append("적정 부채비율")
            
            # 높은 이익률
            if profit_margin > 0.10:
                score += 10
                reasons.append("높은 이익률")
            elif profit_margin > 0.05:
                score += 5
                reasons.append("적정 이익률")
            
            score = max(0, min(100, score))
            
            if score >= 75:
                recommendation = "STRONG_BUY"
            elif score >= 60:
                recommendation = "BUY"
            elif score <= 40:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"워렌 버핏 가치투자: {', '.join(reasons) if reasons else '중립적 신호'}"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}
    
    def _analyze_us_peter_lynch_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 피터 린치 전략 분석 (성장주 투자)"""
        try:
            score = 50  # 기본 점수
            reasons = []
            
            revenue_growth = stock_data.get('revenue_growth', 0)
            profit_margin = stock_data.get('profit_margin', 0)
            pe_ratio = stock_data.get('pe_ratio', 0)
            sector = stock_data.get('sector', '')
            market_cap = stock_data.get('market_cap', 0)
            
            # PEG 비율 (PE/성장률) - 간단히 PE와 성장률로 판단
            if revenue_growth > 0 and pe_ratio > 0:
                peg_estimate = pe_ratio / (revenue_growth * 100)
                if peg_estimate < 1:
                    score += 20
                    reasons.append("우수한 PEG 비율")
                elif peg_estimate < 1.5:
                    score += 10
                    reasons.append("양호한 PEG 비율")
            
            # 성장률 분석
            if revenue_growth > 0.20:  # 20% 이상
                score += 15
                reasons.append("높은 성장률")
            elif revenue_growth > 0.15:
                score += 10
                reasons.append("양호한 성장률")
            elif revenue_growth > 0.10:
                score += 5
                reasons.append("안정적 성장")
            
            # 이익률 (수익성)
            if profit_margin > 0.15:
                score += 15
                reasons.append("높은 수익성")
            elif profit_margin > 0.10:
                score += 10
                reasons.append("양호한 수익성")
            
            # 섹터 분석 (린치가 선호하는 소비재, 기술)
            growth_sectors = ['Technology', 'Consumer Discretionary', 'Healthcare', 'Consumer Staples']
            if any(gs in sector for gs in growth_sectors):
                score += 10
                reasons.append("성장 섹터")
            
            # 적정 규모 (너무 크지 않은 기업 선호)
            if 10_000_000_000 <= market_cap <= 100_000_000_000:  # 100억~1000억 달러
                score += 10
                reasons.append("적정 규모")
            elif market_cap < 10_000_000_000:
                score += 5
                reasons.append("소형주 성장 가능성")
            
            # 적정 PER (성장주이므로 다소 높아도 OK)
            if 10 <= pe_ratio <= 30:
                score += 10
                reasons.append("성장주 적정 PER")
            elif pe_ratio <= 40:
                score += 5
                reasons.append("성장주 보통 PER")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "STRONG_BUY"
            elif score >= 55:
                recommendation = "BUY"
            elif score <= 40:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"피터 린치 성장주: {', '.join(reasons) if reasons else '중립적 신호'}"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}
    
    def _analyze_us_blackrock_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 블랙록 전략 분석 (기관투자자 관점)"""
        try:
            score = 50  # 기본 점수
            reasons = []
            
            market_cap = stock_data.get('market_cap', 0)
            pe_ratio = stock_data.get('pe_ratio', 0)
            pb_ratio = stock_data.get('pb_ratio', 0)
            roe = stock_data.get('roe', 0)
            debt_to_equity = stock_data.get('debt_to_equity', 0)
            profit_margin = stock_data.get('profit_margin', 0)
            revenue_growth = stock_data.get('revenue_growth', 0)
            sector = stock_data.get('sector', '')
            rsi = stock_data.get('rsi', 50)
            current_price = stock_data.get('current_price', 0)
            sma_20 = stock_data.get('sma_20', 0)
            
            # 대형주 선호 (유동성)
            if market_cap > 50_000_000_000:  # 500억 달러 이상
                score += 15
                reasons.append("대형주 유동성")
            elif market_cap > 20_000_000_000:
                score += 10
                reasons.append("중대형주")
            
            # 재무 건전성
            if roe > 0.12:  # 12% 이상
                score += 12
                reasons.append("우수한 ROE")
            elif roe > 0.08:
                score += 8
                reasons.append("양호한 ROE")
            
            # 적정 밸류에이션
            if 10 <= pe_ratio <= 25:
                score += 10
                reasons.append("적정 PER")
            elif pe_ratio <= 35:
                score += 5
                reasons.append("보통 PER")
            
            if 0.5 <= pb_ratio <= 3:
                score += 8
                reasons.append("적정 PBR")
            
            # 부채 관리
            if debt_to_equity < 0.6:
                score += 10
                reasons.append("건전한 부채비율")
            elif debt_to_equity < 1.0:
                score += 5
                reasons.append("적정 부채비율")
            
            # 수익성
            if profit_margin > 0.10:
                score += 10
                reasons.append("높은 수익성")
            elif profit_margin > 0.05:
                score += 5
                reasons.append("적정 수익성")
            
            # 성장성
            if revenue_growth > 0.10:
                score += 8
                reasons.append("양호한 성장")
            elif revenue_growth > 0.05:
                score += 5
                reasons.append("안정적 성장")
            
            # ESG 친화 섹터
            esg_sectors = ['Technology', 'Healthcare', 'Renewable Energy', 'Utilities']
            if any(esg in sector for esg in esg_sectors):
                score += 8
                reasons.append("ESG 친화 섹터")
            
            # 기술적 분석 (단순)
            if current_price > sma_20:
                score += 5
                reasons.append("기술적 상승")
            
            if 30 <= rsi <= 70:
                score += 5
                reasons.append("기술적 균형")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "STRONG_BUY"
            elif score >= 55:
                recommendation = "BUY"
            elif score <= 40:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"블랙록 기관투자: {', '.join(reasons) if reasons else '중립적 신호'}"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}
    
    def _analyze_us_comprehensive_strategy(self, stock_data: Dict) -> Dict:
        """🇺🇸 미국주식 종합 전략 분석 (6가지 투자대가 통합)"""
        try:
            # 각 전략별 점수 계산
            william_oneil_result = self._analyze_us_william_oneil_strategy(stock_data)
            jesse_livermore_result = self._analyze_us_jesse_livermore_strategy(stock_data)
            ichimoku_result = self._analyze_us_ichimoku_strategy(stock_data)
            warren_buffett_result = self._analyze_us_warren_buffett_strategy(stock_data)
            peter_lynch_result = self._analyze_us_peter_lynch_strategy(stock_data)
            blackrock_result = self._analyze_us_blackrock_strategy(stock_data)
            
            # 가중 평균 계산 (투자대가별 가중치)
            weights = {
                'william_oneil': 0.18,    # 윌리엄 오닐 (성장주)
                'jesse_livermore': 0.16,  # 제시 리버모어 (추세)
                'ichimoku': 0.15,         # 일목산인 (기술적)
                'warren_buffett': 0.20,   # 워렌 버핏 (가치)
                'peter_lynch': 0.16,      # 피터 린치 (성장)
                'blackrock': 0.15         # 블랙록 (기관)
            }
            
            total_score = (
                william_oneil_result['score'] * weights['william_oneil'] +
                jesse_livermore_result['score'] * weights['jesse_livermore'] +
                ichimoku_result['score'] * weights['ichimoku'] +
                warren_buffett_result['score'] * weights['warren_buffett'] +
                peter_lynch_result['score'] * weights['peter_lynch'] +
                blackrock_result['score'] * weights['blackrock']
            )
            
            score = round(total_score)
            
            # 최고 점수 전략 찾기
            strategy_scores = {
                '윌리엄오닐': william_oneil_result['score'],
                '제시리버모어': jesse_livermore_result['score'],
                '일목산인': ichimoku_result['score'],
                '워렌버핏': warren_buffett_result['score'],
                '피터린치': peter_lynch_result['score'],
                '블랙록': blackrock_result['score']
            }
            
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            
            if score >= 75:
                recommendation = "STRONG_BUY"
            elif score >= 60:
                recommendation = "BUY"
            elif score <= 40:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'score': score,
                'recommendation': recommendation,
                'reason': f"투자대가 통합분석 (최고점수: {best_strategy})"
            }
            
        except Exception as e:
            return {'score': 50, 'recommendation': 'HOLD', 'reason': f'분석 오류: {e}'}

    # === 🎯 ATR 기반 동적 스캘핑 ===
    def calculate_atr_scalping_signals(self, symbol: str, period: int = 14) -> Dict:
        """ATR 기반 동적 스캘핑 신호 생성 (깃허브 분석된 전략 적용)"""
        try:
            logger.info(f"🔥 {symbol} ATR 스캘핑 분석 시작...")
            
            # 1분봉 데이터 수집
            minute_data = self._get_minute_bars(symbol, count=period + 5)
            if not minute_data or len(minute_data) < period:
                return None
            
            # ATR 계산
            atr = self._calculate_atr(minute_data, period)
            current_price = minute_data[-1]['close']
            
            # 동적 진입/청산 레벨 계산 (Alpaca 스타일)
            entry_threshold = atr * 0.5  # ATR의 50%에서 진입
            stop_loss = atr * 1.5       # ATR의 150%에서 손절
            take_profit = atr * 2.0     # ATR의 200%에서 익절
            
            # 현재 변동성 상태 분석
            volatility_state = self._analyze_volatility_state(minute_data, atr)
            
            # 모멘텀 방향성 분석
            momentum_direction = self._calculate_momentum_direction(minute_data)
            
            # 스캘핑 적합성 점수 (0-100)
            scalping_suitability = self._calculate_scalping_suitability(
                atr, current_price, volatility_state, momentum_direction
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'atr': round(atr, 2),
                'entry_threshold': round(entry_threshold, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'volatility_state': volatility_state,  # high/medium/low
                'momentum_direction': momentum_direction,  # bullish/bearish/neutral
                'scalping_suitability': scalping_suitability,  # 0-100 점수
                'recommended_action': self._get_atr_recommendation(
                    scalping_suitability, momentum_direction, volatility_state
                )
            }
            
        except Exception as e:
            logger.error(f"❌ ATR 스캘핑 분석 실패 {symbol}: {e}")
            return None
    
    # === 📊 멀티타임프레임 분석 ===
    def multi_timeframe_analysis(self, symbol: str) -> Dict:
        """멀티타임프레임 종합 분석 (1분, 5분, 15분)"""
        try:
            logger.info(f"📊 {symbol} 멀티타임프레임 분석...")
            
            analysis_results = {}
            timeframes = [
                ('1min', 1, 20),   # 1분봉 20개
                ('5min', 5, 12),   # 5분봉 12개 (1시간)
                ('15min', 15, 8)   # 15분봉 8개 (2시간)
            ]
            
            for tf_name, interval, count in timeframes:
                try:
                    bars = self._get_minute_bars(symbol, interval, count)
                    if not bars or len(bars) < 5:
                        continue
                    
                    tf_analysis = {
                        'trend_direction': self._calculate_trend_direction(bars),
                        'momentum_strength': self._calculate_momentum_strength(bars),
                        'support_resistance': self._find_support_resistance(bars),
                        'volume_profile': self._analyze_volume_profile(bars),
                        'breakout_potential': self._assess_breakout_potential(bars)
                    }
                    
                    analysis_results[tf_name] = tf_analysis
                    
                except Exception as e:
                    logger.warning(f"⚠️ {tf_name} 분석 실패: {e}")
                    continue
            
            # 타임프레임 간 일치도 분석
            consistency_score = self._calculate_timeframe_consistency(analysis_results)
            
            # 종합 판단
            overall_signal = self._generate_multi_tf_signal(analysis_results, consistency_score)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe_analysis': analysis_results,
                'consistency_score': consistency_score,
                'overall_signal': overall_signal,
                'confidence_level': self._calculate_confidence_level(analysis_results, consistency_score)
            }
            
        except Exception as e:
            logger.error(f"❌ 멀티타임프레임 분석 실패 {symbol}: {e}")
            return None
    
    # === ⚡ 실시간 모멘텀 스코어링 ===
    def real_time_momentum_scoring(self, symbol: str) -> Dict:
        """실시간 모멘텀 점수 계산 및 추적"""
        try:
            current_price = self._get_current_price(symbol)
            if not current_price:
                return None
            
            # 가격 버퍼에 추가 (최근 100개 유지)
            if symbol not in self.price_buffer:
                self.price_buffer[symbol] = deque(maxlen=100)
            self.price_buffer[symbol].append({
                'price': current_price,
                'timestamp': datetime.now()
            })
            
            if len(self.price_buffer[symbol]) < 10:
                return None
            
            # 다양한 시간 단위 모멘텀 계산
            momentum_1min = self._calculate_price_momentum(symbol, minutes=1)
            momentum_5min = self._calculate_price_momentum(symbol, minutes=5)
            momentum_15min = self._calculate_price_momentum(symbol, minutes=15)
            
            # 가속도 계산 (모멘텀의 변화율)
            acceleration = self._calculate_momentum_acceleration(symbol)
            
            # 거래량 가중 모멘텀
            volume_weighted_momentum = self._calculate_volume_weighted_momentum(symbol)
            
            # 종합 모멘텀 스코어 (0-100)
            composite_momentum = self._calculate_composite_momentum(
                momentum_1min, momentum_5min, momentum_15min, 
                acceleration, volume_weighted_momentum
            )
            
            # 모멘텀 등급 결정
            momentum_grade = self._get_momentum_grade(composite_momentum)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'momentum_1min': momentum_1min,
                'momentum_5min': momentum_5min,
                'momentum_15min': momentum_15min,
                'acceleration': acceleration,
                'volume_weighted_momentum': volume_weighted_momentum,
                'composite_momentum': composite_momentum,
                'momentum_grade': momentum_grade,
                'trading_signal': self._get_momentum_trading_signal(composite_momentum, momentum_grade)
            }
            
        except Exception as e:
            logger.error(f"❌ 실시간 모멘텀 스코어링 실패 {symbol}: {e}")
            return None
    
    # === 🏗️ 고급 호가창 분석 ===
    def advanced_depth_analysis(self, symbol: str) -> Dict:
        """고급 호가창 심도 분석 (Level 2 데이터 활용)"""
        try:
            logger.info(f"🏗️ {symbol} 고급 호가창 분석...")
            
            # 전체 호가창 데이터 수집
            depth_data = self._get_full_orderbook(symbol)
            if not depth_data:
                return None
            
            # 호가창 불균형 분석
            imbalance_analysis = self._calculate_orderbook_imbalance(depth_data)
            
            # 큰 주문 감지 (Iceberg 주문 등)
            large_order_detection = self._detect_large_orders(depth_data)
            
            # 호가창 압력 분석
            pressure_analysis = self._analyze_order_pressure(depth_data)
            
            # 스프레드 분석
            spread_analysis = self._analyze_bid_ask_spread(depth_data)
            
            # 호가창 유동성 분석
            liquidity_analysis = self._analyze_market_liquidity(depth_data)
            
            # 프로 vs 아마추어 매매 패턴 감지
            smart_money_analysis = self._detect_smart_money_flow(depth_data)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'imbalance_analysis': imbalance_analysis,
                'large_order_detection': large_order_detection,
                'pressure_analysis': pressure_analysis,
                'spread_analysis': spread_analysis,
                'liquidity_analysis': liquidity_analysis,
                'smart_money_analysis': smart_money_analysis,
                'overall_depth_score': self._calculate_overall_depth_score(
                    imbalance_analysis, pressure_analysis, liquidity_analysis
                ),
                'recommended_action': self._get_depth_based_recommendation(
                    imbalance_analysis, pressure_analysis, smart_money_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"❌ 고급 호가창 분석 실패 {symbol}: {e}")
            return None
    
    # === 🎯 AI 기반 진입/청산 타이밍 ===
    def ai_entry_exit_timing(self, symbol: str) -> Dict:
        """AI 기반 최적 진입/청산 타이밍 분석"""
        try:
            logger.info(f"🎯 {symbol} AI 진입/청산 타이밍 분석...")
            
            # 여러 분석 결과 통합
            atr_signals = self.calculate_atr_scalping_signals(symbol)
            momentum_scores = self.real_time_momentum_scoring(symbol)
            depth_analysis = self.advanced_depth_analysis(symbol)
            
            if not all([atr_signals, momentum_scores, depth_analysis]):
                return None
            
            # 가중치 기반 종합 점수 계산
            entry_score = self._calculate_entry_score(
                atr_signals, momentum_scores, depth_analysis
            )
            
            exit_score = self._calculate_exit_score(
                atr_signals, momentum_scores, depth_analysis
            )
            
            # 리스크 조정 점수
            risk_adjusted_scores = self._apply_risk_adjustment(entry_score, exit_score, symbol)
            
            # 최종 AI 추천
            ai_recommendation = self._generate_ai_recommendation(
                risk_adjusted_scores, atr_signals, momentum_scores
            )
            
            # 확신도 계산
            confidence_level = self._calculate_ai_confidence(
                atr_signals, momentum_scores, depth_analysis
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'entry_score': entry_score,
                'exit_score': exit_score,
                'risk_adjusted_entry': risk_adjusted_scores['entry'],
                'risk_adjusted_exit': risk_adjusted_scores['exit'],
                'ai_recommendation': ai_recommendation,
                'confidence_level': confidence_level,
                'execution_priority': self._calculate_execution_priority(
                    entry_score, confidence_level, momentum_scores
                ),
                'optimal_position_size': self._calculate_optimal_position_size(
                    symbol, confidence_level, atr_signals
                )
            }
            
        except Exception as e:
            logger.error(f"❌ AI 진입/청산 타이밍 분석 실패 {symbol}: {e}")
            return None
    
    # === 🚀 실시간 멀티 심볼 모니터링 ===
    async def start_advanced_monitoring(self, symbols: List[str], 
                                      signal_callback: Callable = None,
                                      risk_callback: Callable = None):
        """고급 실시간 모니터링 시작"""
        try:
            logger.info(f"🚀 고급 실시간 모니터링 시작: {len(symbols)}개 종목")
            
            self.is_monitoring = True
            self.scalping_signal_callback = signal_callback
            self.risk_alert_callback = risk_callback
            
            # 각 심볼별 비동기 모니터링 태스크 생성
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._monitor_symbol(symbol))
                tasks.append(task)
            
            # 시장 전체 상황 모니터링 태스크
            market_task = asyncio.create_task(self._monitor_market_conditions())
            tasks.append(market_task)
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"❌ 고급 모니터링 실패: {e}")
            self.is_monitoring = False
    
    async def _monitor_symbol(self, symbol: str):
        """개별 심볼 모니터링"""
        while self.is_monitoring:
            try:
                # AI 분석 실행
                ai_analysis = self.ai_entry_exit_timing(symbol)
                
                if ai_analysis and self.scalping_signal_callback:
                    self.scalping_signal_callback(symbol, ai_analysis)
                
                # 리스크 체크
                risk_alert = self._check_risk_conditions(symbol, ai_analysis)
                if risk_alert and self.risk_alert_callback:
                    self.risk_alert_callback(symbol, risk_alert)
                
                # 2초 대기 (API 한도 고려)
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"⚠️ {symbol} 모니터링 오류: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_market_conditions(self):
        """시장 전체 상황 모니터링"""
        while self.is_monitoring:
            try:
                market_health = self._assess_market_health()
                
                if market_health['risk_level'] == 'HIGH':
                    logger.warning("🚨 시장 고위험 상황 감지!")
                    if self.risk_alert_callback:
                        self.risk_alert_callback('MARKET', market_health)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.warning(f"⚠️ 시장 상황 모니터링 오류: {e}")
                await asyncio.sleep(60)
    
    def stop_advanced_monitoring(self):
        """고급 모니터링 중지"""
        self.is_monitoring = False
        logger.info("🔴 고급 모니터링 중지")
    
    # === 🛠️ 핵심 헬퍼 메서드들 ===
    def _get_minute_bars(self, symbol: str, interval: int = 1, count: int = 20) -> List[Dict]:
        """분봉 데이터 수집"""
        try:
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice", 
                    headers={"tr_id": "FHKST03010200"}, 
                    params={
                        "fid_etc_cls_code": "",
                        "fid_cond_mrkt_div_code": "J",
                        "fid_input_iscd": symbol,
                        "fid_input_hour_1": str(interval),
                        "fid_pw_data_incu_yn": "Y"
                    }
                )
                
                if res and res.get('rt_cd') == '0':
                    items = res.get('output2', [])[:count]
                    
                    bars = []
                    for item in items:
                        bars.append({
                            'timestamp': item.get('stck_cntg_hour', ''),
                            'open': float(item.get('stck_oprc', 0)),
                            'high': float(item.get('stck_hgpr', 0)),
                            'low': float(item.get('stck_lwpr', 0)),
                            'close': float(item.get('stck_prpr', 0)),
                            'volume': int(item.get('cntg_vol', 0))
                        })
                    
                    return bars
                
                return []
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 분봉 데이터 수집 실패: {e}")
            return []
    
    def _calculate_atr(self, bars: List[Dict], period: int = 14) -> float:
        """Average True Range 계산"""
        if len(bars) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        else:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            price_info = self.trader.get_current_price(symbol)
            return float(price_info.get('price', 0)) if price_info else None
        except:
            return None
    
    def _get_full_orderbook(self, symbol: str) -> Optional[Dict]:
        """전체 호가창 데이터 수집"""
        try:
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", 
                    headers={"tr_id": "FHKST01010200"}, 
                    params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol}
                )
                
                if res and res.get('rt_cd') == '0':
                    return res.get('output1', {})
                return None
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 호가창 데이터 수집 실패: {e}")
            return None
    
    # === 🧮 분석 메서드들 (기본 구현) ===
    def _analyze_volatility_state(self, bars, atr):
        """변동성 상태 분석"""
        if atr > 2000:
            return "high"
        elif atr > 1000:
            return "medium"
        else:
            return "low"
    
    def _calculate_momentum_direction(self, bars):
        """모멘텀 방향성 계산"""
        if len(bars) < 3:
            return "neutral"
        
        recent_prices = [bar['close'] for bar in bars[-3:]]
        if recent_prices[-1] > recent_prices[0]:
            return "bullish"
        elif recent_prices[-1] < recent_prices[0]:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_scalping_suitability(self, atr, price, volatility, momentum):
        """스캘핑 적합성 점수"""
        score = 50  # 기본 점수
        
        # 변동성 점수
        if volatility == "high":
            score += 20
        elif volatility == "medium":
            score += 10
        
        # 모멘텀 점수
        if momentum in ["bullish", "bearish"]:
            score += 15
        
        # ATR 기반 점수
        if price > 0 and atr / price > 0.02:  # 2% 이상 변동성
            score += 15
        
        return min(100, max(0, score))
    
    def _get_atr_recommendation(self, suitability, momentum, volatility):
        """ATR 기반 추천"""
        if suitability > 70 and momentum == "bullish":
            return "STRONG_BUY"
        elif suitability > 70 and momentum == "bearish":
            return "STRONG_SELL"
        elif suitability > 50:
            if momentum == "bullish":
                return "BUY"
            elif momentum == "bearish":
                return "SELL"
            else:
                return "WATCH"
        else:
            return "HOLD"
    
    def _calculate_trend_direction(self, bars):
        """추세 방향 분석"""
        if len(bars) < 5:
            return "SIDEWAYS"
        
        closes = [bar['close'] for bar in bars]
        if closes[-1] > closes[0] * 1.02:
            return "UP"
        elif closes[-1] < closes[0] * 0.98:
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def _calculate_momentum_strength(self, bars):
        """모멘텀 강도 분석"""
        if len(bars) < 2:
            return 0.0
        
        price_change = (bars[-1]['close'] - bars[0]['close']) / bars[0]['close']
        return abs(price_change) * 100  # 백분율로 변환
    
    def _find_support_resistance(self, bars):
        """지지/저항 분석"""
        if len(bars) < 5:
            return {"support": 0, "resistance": 0}
        
        lows = [bar['low'] for bar in bars]
        highs = [bar['high'] for bar in bars]
        
        return {
            "support": min(lows),
            "resistance": max(highs)
        }
    
    def _analyze_volume_profile(self, bars):
        """거래량 프로파일 분석"""
        if len(bars) < 2:
            return 0.0
        
        volumes = [bar['volume'] for bar in bars]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1] if volumes else 0
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _assess_breakout_potential(self, bars):
        """브레이크아웃 잠재력 평가"""
        if len(bars) < 5:
            return 0.0
        
        # 최근 5개 봉의 범위 대비 현재 위치
        highs = [bar['high'] for bar in bars[-5:]]
        lows = [bar['low'] for bar in bars[-5:]]
        current = bars[-1]['close']
        
        high_range = max(highs)
        low_range = min(lows)
        
        if high_range == low_range:
            return 0.0
        
        # 0~100 범위로 정규화
        position = (current - low_range) / (high_range - low_range) * 100
        
        # 80% 이상이면 상향 돌파 가능성, 20% 이하면 하향 돌파 가능성
        if position > 80:
            return position
        elif position < 20:
            return -position
        else:
            return 0.0
    
    # === 📊 스텁 메서드들 (추후 확장 가능) ===
    def _calculate_price_momentum(self, symbol: str, minutes: int) -> float:
        """가격 모멘텀 계산 (현재는 간단 구현)"""
        if symbol not in self.price_buffer or len(self.price_buffer[symbol]) < 5:
            return 0.0
        
        # 간단한 모멘텀 계산 (최근 vs 이전 가격)
        recent_prices = list(self.price_buffer[symbol])[-5:]
        if len(recent_prices) < 2:
            return 0.0
        
        price_change = (recent_prices[-1]['price'] - recent_prices[0]['price']) / recent_prices[0]['price']
        return price_change * 100  # 백분율로 반환
    
    def _calculate_momentum_acceleration(self, symbol: str) -> float:
        return self._calculate_price_momentum(symbol, 1) * 0.5  # 간단 구현
    
    def _calculate_volume_weighted_momentum(self, symbol: str) -> float:
        return self._calculate_price_momentum(symbol, 5) * 0.8  # 간단 구현
    
    def _calculate_composite_momentum(self, m1, m5, m15, acc, vol_weighted):
        return (m1 + m5 + m15 + acc + vol_weighted) / 5
    
    def _get_momentum_grade(self, composite_momentum: float) -> str:
        if composite_momentum > 80:
            return "A+"
        elif composite_momentum > 60:
            return "A"
        elif composite_momentum > 40:
            return "B"
        else:
            return "C"
    
    def _get_momentum_trading_signal(self, composite_momentum: float, momentum_grade: str) -> str:
        if composite_momentum > 70:
            return "STRONG_BUY"
        elif composite_momentum > 30:
            return "BUY"
        elif composite_momentum < -30:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_orderbook_imbalance(self, depth_data):
        return 50.0  # 기본값
    
    def _detect_large_orders(self, depth_data):
        return False  # 기본값
    
    def _analyze_order_pressure(self, depth_data):
        return 50.0  # 기본값
    
    def _analyze_bid_ask_spread(self, depth_data):
        return 50.0  # 기본값
    
    def _analyze_market_liquidity(self, depth_data):
        return 50.0  # 기본값
    
    def _detect_smart_money_flow(self, depth_data):
        return False  # 기본값
    
    def _calculate_overall_depth_score(self, imbalance, pressure, liquidity):
        return (imbalance + pressure + liquidity) / 3
    
    def _get_depth_based_recommendation(self, imbalance, pressure, smart_money):
        return "HOLD"  # 기본값
    
    def _calculate_entry_score(self, atr_signals, momentum_scores, depth_analysis):
        # 간단한 가중치 계산
        atr_score = atr_signals.get('scalping_suitability', 50)
        momentum_score = momentum_scores.get('composite_momentum', 50)
        depth_score = depth_analysis.get('overall_depth_score', 50)
        
        return (atr_score * 0.4 + momentum_score * 0.4 + depth_score * 0.2)
    
    def _calculate_exit_score(self, atr_signals, momentum_scores, depth_analysis):
        # 진입 점수의 역순으로 계산
        entry_score = self._calculate_entry_score(atr_signals, momentum_scores, depth_analysis)
        return 100 - entry_score
    
    def _apply_risk_adjustment(self, entry_score, exit_score, symbol):
        # 간단한 리스크 조정 (10% 할인)
        return {
            'entry': entry_score * 0.9,
            'exit': exit_score * 0.9
        }
    
    def _generate_ai_recommendation(self, risk_adjusted_scores, atr_signals, momentum_scores):
        entry_score = risk_adjusted_scores['entry']
        
        if entry_score > 80:
            return "STRONG_BUY"
        elif entry_score > 60:
            return "BUY"
        elif entry_score < 20:
            return "SELL"
        elif entry_score < 40:
            return "WEAK_SELL"
        else:
            return "HOLD"
    
    def _calculate_ai_confidence(self, atr_signals, momentum_scores, depth_analysis):
        # 각 분석의 일관성을 기반으로 확신도 계산
        scores = [
            atr_signals.get('scalping_suitability', 50),
            momentum_scores.get('composite_momentum', 50),
            depth_analysis.get('overall_depth_score', 50)
        ]
        
        # 표준편차가 낮을수록 높은 확신도
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        confidence = max(0, 100 - std_dev * 2)
        
        return confidence
    
    def _calculate_execution_priority(self, entry_score, confidence_level, momentum_scores):
        return (entry_score + confidence_level) / 2
    
    def _calculate_optimal_position_size(self, symbol, confidence_level, atr_signals):
        # 확신도에 비례한 포지션 사이즈 (0-100%)
        return min(100, confidence_level) / 100
    
    def _calculate_timeframe_consistency(self, analysis_results):
        if not analysis_results:
            return 0.0
        
        # 간단한 일치도 계산
        trend_directions = []
        for tf_data in analysis_results.values():
            trend_directions.append(tf_data.get('trend_direction', 'SIDEWAYS'))
        
        # 동일한 방향성이 많을수록 높은 일치도
        if not trend_directions:
            return 0.0
        
        most_common = max(set(trend_directions), key=trend_directions.count)
        consistency = trend_directions.count(most_common) / len(trend_directions) * 100
        
        return consistency
    
    def _generate_multi_tf_signal(self, analysis_results, consistency_score):
        if consistency_score > 70:
            return "HIGH_CONFIDENCE"
        elif consistency_score > 50:
            return "MEDIUM_CONFIDENCE"
        else:
            return "LOW_CONFIDENCE"
    
    def _calculate_confidence_level(self, analysis_results, consistency_score):
        return consistency_score
    
    def _check_risk_conditions(self, symbol, ai_analysis):
        if not ai_analysis:
            return None
        
        confidence = ai_analysis.get('confidence_level', 0)
        if confidence < 30:
            return {
                'risk_type': 'LOW_CONFIDENCE',
                'message': f'{symbol} 분석 확신도 부족: {confidence}%'
            }
        
        return None
    
    def _assess_market_health(self):
        # 간단한 시장 건강도 체크
        return {'risk_level': 'LOW'}

    def _calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return 50.0 