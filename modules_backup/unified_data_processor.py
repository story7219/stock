#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 통합 데이터 처리 시스템 v3.0
====================================
- 코스피200·나스닥100·S&P500 데이터 수집
- 비동기 고속 병렬처리
- 멀티레벨 캐싱 적용
- Gemini AI 분석 통합
- 기술적 분석 자동화
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import ta
from pathlib import Path
import sqlite3
import google.generativeai as genai

from ..core.optimized_core import get_core, SystemConfig
from ..core.base_interfaces import StockData, MarketData, AnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class MarketConfig:
    """시장별 설정"""
    name: str
    symbols: List[str]
    suffix: str = ""
    timezone: str = "UTC"
    trading_hours: Dict[str, str] = None
    
    def __post_init__(self):
        if self.trading_hours is None:
            self.trading_hours = {"open": "09:30", "close": "16:00"}

class MarketDataManager:
    """시장 데이터 관리자"""
    
    def __init__(self):
        self.core = get_core()
        self.markets = {
            "KOSPI200": MarketConfig(
                name="코스피200",
                symbols=self._load_kospi200_symbols(),
                suffix=".KS",
                timezone="Asia/Seoul",
                trading_hours={"open": "09:00", "close": "15:30"}
            ),
            "NASDAQ100": MarketConfig(
                name="나스닥100",
                symbols=self._load_nasdaq100_symbols(),
                suffix="",
                timezone="America/New_York"
            ),
            "SP500": MarketConfig(
                name="S&P500",
                symbols=self._load_sp500_symbols(),
                suffix="",
                timezone="America/New_York"
            )
        }
        
        # 데이터베이스 초기화
        self._init_database()
    
    def _load_kospi200_symbols(self) -> List[str]:
        """코스피200 종목 코드 로드"""
        # 주요 코스피200 종목들 (실제 운영시에는 KRX API 사용)
        return [
            "005930",  # 삼성전자
            "000660",  # SK하이닉스
            "051910",  # LG화학
            "068270",  # 셀트리온
            "035420",  # NAVER
            "207940",  # 삼성바이오로직스
            "006400",  # 삼성SDI
            "035720",  # 카카오
            "028260",  # 삼성물산
            "066570",  # LG전자
            "323410",  # 카카오뱅크
            "000270",  # 기아
            "012330",  # 현대모비스
            "003670",  # 포스코홀딩스
            "096770",  # SK이노베이션
            "017670",  # SK텔레콤
            "030200",  # KT
            "009150",  # 삼성전기
            "018260",  # 삼성에스디에스
            "011200"   # HMM
        ]
    
    def _load_nasdaq100_symbols(self) -> List[str]:
        """나스닥100 종목 코드 로드"""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "ADBE", "CRM", "PYPL", "INTC", "CMCSA", "PEP", "COST", "TMUS",
            "AVGO", "TXN", "QCOM", "CHTR", "GILD", "FISV", "BKNG", "ADP",
            "SBUX", "MDLZ", "ISRG", "CSX", "REGN", "ATVI", "VRTX", "ILMN",
            "MU", "AMAT", "LRCX", "ADI", "MELI", "KLAC", "EXC", "BIIB",
            "JD", "KDP", "WDAY", "MRNA", "DXCM", "SGEN", "ROST", "CTSH",
            "FAST", "PAYX", "CSGP", "ODFL", "VRSK", "NTES", "DLTR", "XEL",
            "ORLY", "MNST", "LULU", "PCAR", "CPRT", "ANSS", "SIRI", "MCHP"
        ]
    
    def _load_sp500_symbols(self) -> List[str]:
        """S&P500 주요 종목 코드 로드"""
        return [
            "JPM", "JNJ", "V", "WMT", "PG", "UNH", "DIS", "HD", "MA", "BAC",
            "XOM", "ABBV", "KO", "LLY", "PFE", "TMO", "ABT", "CVX", "ACN", "NKE",
            "ORCL", "DHR", "VZ", "BMY", "PM", "T", "HON", "UPS", "IBM", "LOW",
            "SPGI", "GS", "CAT", "MMM", "AXP", "TJX", "BLK", "AMT", "RTX", "SCHW",
            "DE", "LMT", "ADP", "SYK", "BKNG", "GILD", "MDLZ", "CI", "TGT", "CVS",
            "MO", "USB", "BDX", "ANTM", "ISRG", "ZTS", "TFC", "MMC", "PNC", "AON"
        ]
    
    def _init_database(self):
        """데이터베이스 초기화"""
        db_path = Path("data/market_data.db")
        db_path.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                market TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                symbol TEXT,
                date DATE,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                volume_sma REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        ''')
        
        conn.commit()
        conn.close()

class AsyncDataCollector:
    """비동기 데이터 수집기"""
    
    def __init__(self, market_manager: MarketDataManager):
        self.core = get_core()
        self.market_manager = market_manager
        self.session_pool = {}
        
    async def collect_market_data(self, market: str, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """시장 데이터 수집"""
        if market not in self.market_manager.markets:
            raise ValueError(f"지원하지 않는 시장: {market}")
        
        market_config = self.market_manager.markets[market]
        cache_key = f"market_data_{market}_{period}"
        
        # 캐시 확인
        cached_data = await self.core.cache.get(cache_key)
        if cached_data:
            logger.info(f"캐시에서 {market} 데이터 로드")
            return cached_data
        
        logger.info(f"🔄 {market} 데이터 수집 시작 ({len(market_config.symbols)}개 종목)")
        
        # 병렬 데이터 수집
        tasks = []
        for symbol in market_config.symbols:
            full_symbol = f"{symbol}{market_config.suffix}"
            task = self._collect_single_stock(full_symbol, period, market)
            tasks.append(task)
        
        # 배치 처리로 메모리 최적화
        batch_size = 20
        all_data = {}
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await self.core.task_manager.gather_with_limit(*batch_tasks)
            
            for symbol, data in zip(market_config.symbols[i:i + batch_size], batch_results):
                if data is not None:
                    all_data[symbol] = data
            
            # 메모리 정리
            self.core.memory_optimizer.force_gc()
        
        # 캐시 저장 (1시간)
        await self.core.cache.set(cache_key, all_data, ttl=3600)
        
        logger.info(f"✅ {market} 데이터 수집 완료 ({len(all_data)}개 종목)")
        return all_data
    
    @get_core().performance_decorator("collect_single_stock")
    async def _collect_single_stock(self, symbol: str, period: str, market: str) -> Optional[pd.DataFrame]:
        """단일 종목 데이터 수집"""
        try:
            # yfinance를 스레드 풀에서 실행
            data = await self.core.task_manager.run_in_thread(
                self._fetch_yfinance_data, symbol, period
            )
            
            if data is not None and not data.empty:
                # 데이터베이스 저장
                await self._save_to_database(symbol, data, market)
                return data
            
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 데이터 수집 실패: {e}")
        
        return None
    
    def _fetch_yfinance_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """yfinance 데이터 가져오기"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                # 컬럼명 정규화
                data.columns = data.columns.str.lower()
                data.reset_index(inplace=True)
                data['symbol'] = symbol
                return data
            
        except Exception as e:
            logger.debug(f"yfinance 오류 {symbol}: {e}")
        
        return None
    
    async def _save_to_database(self, symbol: str, data: pd.DataFrame, market: str):
        """데이터베이스에 저장"""
        try:
            db_path = Path("data/market_data.db")
            
            def save_data():
                conn = sqlite3.connect(str(db_path))
                
                # 기존 데이터 삭제
                conn.execute("DELETE FROM stock_data WHERE symbol = ?", (symbol,))
                
                # 새 데이터 삽입
                for _, row in data.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO stock_data 
                        (symbol, date, open, high, low, close, volume, market)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        row['date'] if 'date' in row else row.name,
                        row.get('open', 0),
                        row.get('high', 0),
                        row.get('low', 0),
                        row.get('close', 0),
                        row.get('volume', 0),
                        market
                    ))
                
                conn.commit()
                conn.close()
            
            await self.core.task_manager.run_in_thread(save_data)
            
        except Exception as e:
            logger.warning(f"DB 저장 실패 {symbol}: {e}")

class TechnicalAnalyzer:
    """기술적 분석 엔진"""
    
    def __init__(self):
        self.core = get_core()
    
    async def analyze_batch(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """배치 기술적 분석"""
        logger.info(f"🔧 기술적 분석 시작 ({len(market_data)}개 종목)")
        
        # 병렬 분석
        tasks = []
        for symbol, data in market_data.items():
            task = self._analyze_single_stock(symbol, data)
            tasks.append(task)
        
        results = await self.core.task_manager.gather_with_limit(*tasks)
        
        analysis_results = {}
        for symbol, result in zip(market_data.keys(), results):
            if result:
                analysis_results[symbol] = result
        
        logger.info(f"✅ 기술적 분석 완료 ({len(analysis_results)}개 종목)")
        return analysis_results
    
    @get_core().performance_decorator("technical_analysis")
    async def _analyze_single_stock(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """단일 종목 기술적 분석"""
        try:
            if data.empty or len(data) < 50:
                return None
            
            # 스레드 풀에서 계산 집약적 작업 실행
            indicators = await self.core.task_manager.run_in_thread(
                self._calculate_indicators, data
            )
            
            # 신호 생성
            signals = self._generate_signals(indicators)
            
            # 점수 계산
            score = self._calculate_technical_score(indicators, signals)
            
            return {
                'symbol': symbol,
                'indicators': indicators,
                'signals': signals,
                'technical_score': score,
                'last_price': float(data['close'].iloc[-1]),
                'volume': float(data['volume'].iloc[-1]),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"기술적 분석 실패 {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기술적 지표 계산"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            indicators = {
                # 추세 지표
                'sma_20': ta.trend.sma_indicator(close, window=20).iloc[-1],
                'sma_50': ta.trend.sma_indicator(close, window=50).iloc[-1],
                'sma_200': ta.trend.sma_indicator(close, window=200).iloc[-1],
                'ema_12': ta.trend.ema_indicator(close, window=12).iloc[-1],
                'ema_26': ta.trend.ema_indicator(close, window=26).iloc[-1],
                
                # 모멘텀 지표
                'rsi': ta.momentum.rsi(close, window=14).iloc[-1],
                'macd': ta.trend.macd_diff(close).iloc[-1],
                'macd_signal': ta.trend.macd_signal(close).iloc[-1],
                'stoch_k': ta.momentum.stoch(high, low, close).iloc[-1],
                'stoch_d': ta.momentum.stoch_signal(high, low, close).iloc[-1],
                
                # 볼린저 밴드
                'bb_upper': ta.volatility.bollinger_hband(close).iloc[-1],
                'bb_middle': ta.volatility.bollinger_mavg(close).iloc[-1],
                'bb_lower': ta.volatility.bollinger_lband(close).iloc[-1],
                
                # 거래량 지표
                'volume_sma': ta.volume.volume_sma(close, volume, window=20).iloc[-1],
                'obv': ta.volume.on_balance_volume(close, volume).iloc[-1],
                
                # 변동성 지표
                'atr': ta.volatility.average_true_range(high, low, close).iloc[-1],
                
                # 현재 가격
                'current_price': float(close.iloc[-1])
            }
            
            # NaN 값 처리
            for key, value in indicators.items():
                if pd.isna(value):
                    indicators[key] = 0.0
                else:
                    indicators[key] = float(value)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"지표 계산 오류: {e}")
            return {}
    
    def _generate_signals(self, indicators: Dict[str, Any]) -> Dict[str, str]:
        """매매 신호 생성"""
        signals = {}
        
        try:
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            sma_200 = indicators.get('sma_200', 0)
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            # 추세 신호
            if current_price > sma_20 > sma_50 > sma_200:
                signals['trend'] = 'STRONG_UPTREND'
            elif current_price > sma_20 > sma_50:
                signals['trend'] = 'UPTREND'
            elif current_price < sma_20 < sma_50 < sma_200:
                signals['trend'] = 'STRONG_DOWNTREND'
            elif current_price < sma_20 < sma_50:
                signals['trend'] = 'DOWNTREND'
            else:
                signals['trend'] = 'SIDEWAYS'
            
            # RSI 신호
            if rsi < 30:
                signals['rsi'] = 'OVERSOLD'
            elif rsi > 70:
                signals['rsi'] = 'OVERBOUGHT'
            else:
                signals['rsi'] = 'NEUTRAL'
            
            # MACD 신호
            if macd > 0:
                signals['macd'] = 'BULLISH'
            else:
                signals['macd'] = 'BEARISH'
            
            # 볼린저 밴드 신호
            if current_price > bb_upper:
                signals['bollinger'] = 'OVERBOUGHT'
            elif current_price < bb_lower:
                signals['bollinger'] = 'OVERSOLD'
            else:
                signals['bollinger'] = 'NEUTRAL'
            
        except Exception as e:
            logger.warning(f"신호 생성 오류: {e}")
        
        return signals
    
    def _calculate_technical_score(self, indicators: Dict[str, Any], signals: Dict[str, str]) -> float:
        """기술적 점수 계산"""
        try:
            score = 50.0  # 기본 점수
            
            # 추세 점수
            trend_signal = signals.get('trend', 'SIDEWAYS')
            if trend_signal == 'STRONG_UPTREND':
                score += 20
            elif trend_signal == 'UPTREND':
                score += 10
            elif trend_signal == 'STRONG_DOWNTREND':
                score -= 20
            elif trend_signal == 'DOWNTREND':
                score -= 10
            
            # RSI 점수
            rsi_signal = signals.get('rsi', 'NEUTRAL')
            if rsi_signal == 'OVERSOLD':
                score += 10  # 매수 기회
            elif rsi_signal == 'OVERBOUGHT':
                score -= 10  # 매도 신호
            
            # MACD 점수
            if signals.get('macd') == 'BULLISH':
                score += 10
            else:
                score -= 5
            
            # 볼린저 밴드 점수
            bb_signal = signals.get('bollinger', 'NEUTRAL')
            if bb_signal == 'OVERSOLD':
                score += 5
            elif bb_signal == 'OVERBOUGHT':
                score -= 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.warning(f"점수 계산 오류: {e}")
            return 50.0

class GeminiAIAnalyzer:
    """Gemini AI 분석 엔진"""
    
    def __init__(self, api_key: str):
        self.core = get_core()
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            logger.warning("Gemini API 키가 없어 AI 분석 비활성화")
    
    async def analyze_top_candidates(self, 
                                   technical_results: Dict[str, Dict[str, Any]], 
                                   top_n: int = 20) -> Dict[str, Any]:
        """상위 후보 AI 분석"""
        if not self.model:
            return self._fallback_analysis(technical_results, top_n)
        
        # 기술적 점수 기준 상위 종목 선별
        sorted_stocks = sorted(
            technical_results.items(),
            key=lambda x: x[1].get('technical_score', 0),
            reverse=True
        )[:top_n]
        
        logger.info(f"🤖 Gemini AI 분석 시작 (상위 {len(sorted_stocks)}개 종목)")
        
        # AI 분석을 위한 데이터 준비
        analysis_data = self._prepare_analysis_data(sorted_stocks)
        
        try:
            # AI 분석 실행
            ai_response = await self.core.task_manager.run_in_thread(
                self._run_gemini_analysis, analysis_data
            )
            
            # 결과 파싱 및 처리
            final_recommendations = self._process_ai_response(ai_response, sorted_stocks)
            
            logger.info("✅ Gemini AI 분석 완료")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"AI 분석 오류: {e}")
            return self._fallback_analysis(technical_results, top_n)
    
    def _prepare_analysis_data(self, sorted_stocks: List[Tuple[str, Dict[str, Any]]]) -> str:
        """AI 분석용 데이터 준비"""
        analysis_prompt = """
다음은 기술적 분석을 통해 선별된 상위 투자 후보 종목들입니다.
각 종목의 기술적 지표와 신호를 분석하여 최종 Top 5 종목을 선정해주세요.

분석 기준:
1. 기술적 분석 점수
2. 추세의 강도와 지속성
3. 모멘텀 지표의 일관성
4. 리스크 대비 수익 잠재력
5. 시장 환경과의 적합성

종목 정보:
"""
        
        for symbol, data in sorted_stocks:
            indicators = data.get('indicators', {})
            signals = data.get('signals', {})
            
            analysis_prompt += f"""
종목: {symbol}
- 기술적 점수: {data.get('technical_score', 0):.1f}점
- 현재가: ${indicators.get('current_price', 0):.2f}
- 추세: {signals.get('trend', 'N/A')}
- RSI: {indicators.get('rsi', 0):.1f} ({signals.get('rsi', 'N/A')})
- MACD: {signals.get('macd', 'N/A')}
- 볼린저밴드: {signals.get('bollinger', 'N/A')}
- 20일 이평선: ${indicators.get('sma_20', 0):.2f}
- 50일 이평선: ${indicators.get('sma_50', 0):.2f}

"""
        
        analysis_prompt += """
위 정보를 종합하여 다음 형식으로 Top 5 종목을 선정하고 근거를 제시해주세요:

1. [종목명] - 선정 이유
2. [종목명] - 선정 이유
3. [종목명] - 선정 이유
4. [종목명] - 선정 이유
5. [종목명] - 선정 이유

전체 시장 분석과 투자 전략도 함께 제시해주세요.
"""
        
        return analysis_prompt
    
    def _run_gemini_analysis(self, prompt: str) -> str:
        """Gemini AI 분석 실행"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 오류: {e}")
            raise
    
    def _process_ai_response(self, ai_response: str, sorted_stocks: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """AI 응답 처리"""
        try:
            # 간단한 파싱 로직 (실제로는 더 정교한 파싱 필요)
            lines = ai_response.split('\n')
            top_5_symbols = []
            
            for line in lines:
                if any(char.isdigit() and char in '12345' for char in line[:3]):
                    # 종목 기호 추출 시도
                    for symbol, _ in sorted_stocks:
                        if symbol in line:
                            if symbol not in top_5_symbols:
                                top_5_symbols.append(symbol)
                            break
            
            # Top 5가 부족하면 기술적 점수 순으로 보완
            while len(top_5_symbols) < 5 and len(top_5_symbols) < len(sorted_stocks):
                for symbol, _ in sorted_stocks:
                    if symbol not in top_5_symbols:
                        top_5_symbols.append(symbol)
                        break
            
            # 최종 결과 구성
            final_recommendations = {
                'top_5_stocks': [],
                'ai_reasoning': ai_response,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_analyzed': len(sorted_stocks)
            }
            
            for i, symbol in enumerate(top_5_symbols[:5], 1):
                stock_data = next((data for s, data in sorted_stocks if s == symbol), {})
                final_recommendations['top_5_stocks'].append({
                    'rank': i,
                    'symbol': symbol,
                    'technical_score': stock_data.get('technical_score', 0),
                    'current_price': stock_data.get('indicators', {}).get('current_price', 0),
                    'signals': stock_data.get('signals', {}),
                    'recommendation_reason': f"AI 분석 결과 {i}위 선정"
                })
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"AI 응답 처리 오류: {e}")
            return self._fallback_analysis({s: d for s, d in sorted_stocks}, 5)
    
    def _fallback_analysis(self, technical_results: Dict[str, Dict[str, Any]], top_n: int) -> Dict[str, Any]:
        """폴백 분석 (AI 사용 불가시)"""
        sorted_stocks = sorted(
            technical_results.items(),
            key=lambda x: x[1].get('technical_score', 0),
            reverse=True
        )[:top_n]
        
        return {
            'top_5_stocks': [
                {
                    'rank': i + 1,
                    'symbol': symbol,
                    'technical_score': data.get('technical_score', 0),
                    'current_price': data.get('indicators', {}).get('current_price', 0),
                    'signals': data.get('signals', {}),
                    'recommendation_reason': f"기술적 분석 점수 {data.get('technical_score', 0):.1f}점"
                }
                for i, (symbol, data) in enumerate(sorted_stocks[:5])
            ],
            'ai_reasoning': "기술적 분석 기반 자동 선정 (AI 분석 미사용)",
            'analysis_timestamp': datetime.now().isoformat(),
            'total_analyzed': len(sorted_stocks)
        }

class UnifiedDataProcessor:
    """통합 데이터 처리 시스템"""
    
    def __init__(self, gemini_api_key: str = ""):
        self.core = get_core()
        self.market_manager = MarketDataManager()
        self.data_collector = AsyncDataCollector(self.market_manager)
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_analyzer = GeminiAIAnalyzer(gemini_api_key)
        
        logger.info("🚀 통합 데이터 처리 시스템 초기화 완료")
    
    async def run_complete_analysis(self, 
                                  markets: List[str] = None, 
                                  period: str = "1y") -> Dict[str, Any]:
        """완전 자동화 분석 실행"""
        markets = markets or ["NASDAQ100", "SP500"]  # 기본값
        
        logger.info(f"🔍 완전 분석 시작 - 시장: {markets}, 기간: {period}")
        
        timer_id = self.core.performance_monitor.start_timer("complete_analysis")
        
        try:
            # 1. 데이터 수집 (병렬)
            collection_tasks = [
                self.data_collector.collect_market_data(market, period)
                for market in markets
            ]
            
            market_data_results = await self.core.task_manager.gather_with_limit(*collection_tasks)
            
            # 데이터 통합
            all_market_data = {}
            for market, data in zip(markets, market_data_results):
                if data:
                    all_market_data.update(data)
            
            if not all_market_data:
                raise ValueError("수집된 데이터가 없습니다")
            
            logger.info(f"📊 총 {len(all_market_data)}개 종목 데이터 수집 완료")
            
            # 2. 기술적 분석
            technical_results = await self.technical_analyzer.analyze_batch(all_market_data)
            
            if not technical_results:
                raise ValueError("기술적 분석 결과가 없습니다")
            
            # 3. AI 분석
            final_analysis = await self.ai_analyzer.analyze_top_candidates(technical_results)
            
            # 4. 결과 정리
            duration = self.core.performance_monitor.end_timer(timer_id)
            
            complete_result = {
                'analysis_summary': {
                    'total_stocks_analyzed': len(all_market_data),
                    'technical_analysis_count': len(technical_results),
                    'markets_covered': markets,
                    'analysis_period': period,
                    'processing_time_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'top_recommendations': final_analysis,
                'technical_analysis_summary': self._summarize_technical_analysis(technical_results),
                'system_performance': await self.core.get_system_status()
            }
            
            logger.info(f"✅ 완전 분석 완료 ({duration:.2f}초)")
            return complete_result
            
        except Exception as e:
            self.core.performance_monitor.end_timer(timer_id)
            logger.error(f"❌ 완전 분석 실패: {e}")
            raise
    
    def _summarize_technical_analysis(self, technical_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """기술적 분석 요약"""
        if not technical_results:
            return {}
        
        scores = [data.get('technical_score', 0) for data in technical_results.values()]
        trend_signals = [data.get('signals', {}).get('trend', 'UNKNOWN') for data in technical_results.values()]
        
        from collections import Counter
        trend_counter = Counter(trend_signals)
        
        return {
            'total_analyzed': len(technical_results),
            'average_technical_score': sum(scores) / len(scores) if scores else 0,
            'max_technical_score': max(scores) if scores else 0,
            'min_technical_score': min(scores) if scores else 0,
            'trend_distribution': dict(trend_counter),
            'high_score_count': len([s for s in scores if s > 70]),
            'low_score_count': len([s for s in scores if s < 30])
        }

# 전역 인스턴스
_processor_instance = None

def get_processor(gemini_api_key: str = "") -> UnifiedDataProcessor:
    """전역 프로세서 인스턴스 반환"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = UnifiedDataProcessor(gemini_api_key)
    return _processor_instance 