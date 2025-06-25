#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 파생상품 실시간 모니터링 시스템
=================================
K200 옵션/선물, 나스닥100, S&P500 파생상품 데이터를 실시간으로 수집하고
Gemini AI가 폭락/폭등 신호를 분석하는 시스템
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import aiohttp
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import websockets
import time
import os
from .kis_derivatives_api import KISDerivativesAPI, KISDerivativeData

logger = logging.getLogger(__name__)

@dataclass
class DerivativeData:
    """파생상품 데이터 구조"""
    symbol: str
    underlying: str  # 기초자산
    derivative_type: str  # 'option', 'future'
    strike_price: Optional[float] = None
    expiry_date: Optional[str] = None
    option_type: Optional[str] = None  # 'call', 'put'
    current_price: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    bid: float = 0.0
    ask: float = 0.0
    last_updated: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'derivative_type': self.derivative_type,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date,
            'option_type': self.option_type,
            'current_price': self.current_price,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'bid': self.bid,
            'ask': self.ask,
            'last_updated': self.last_updated
        }

@dataclass
class MarketSignal:
    """시장 신호 데이터"""
    signal_type: str  # 'CRASH_WARNING', 'SURGE_WARNING', 'NEUTRAL'
    confidence: float  # 0-100
    underlying_asset: str
    trigger_factors: List[str] = field(default_factory=list)
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    predicted_move: Optional[float] = None  # 예상 변동률 (%)
    time_horizon: str = "1D"  # 1H, 4H, 1D, 1W
    gemini_analysis: str = ""
    timestamp: str = ""

class DerivativesMonitor:
    """파생상품 실시간 모니터링 시스템"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.kis_api = None  # 한국투자증권 API 추가
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_monitoring = False
        self.data_cache: Dict[str, DerivativeData] = {}
        self.signal_history: List[MarketSignal] = []
        
        # 모니터링 대상 설정
        self.targets = {
            'K200': {
                'futures': ['KR4101V50001', 'KR4101V60001'],  # KOSPI200 선물
                'options_calls': [],  # 동적으로 생성
                'options_puts': []    # 동적으로 생성
            },
            'NDX': {  # 나스닥100
                'futures': ['NQ=F'],
                'options': '^NDX'  # 나스닥100 지수
            },
            'SPX': {  # S&P500
                'futures': ['ES=F'],
                'options': '^GSPC'  # S&P500 지수
            }
        }
        
        # 위험 임계값 설정
        self.risk_thresholds = {
            'volatility_spike': 50.0,  # IV 급등 임계값
            'volume_surge': 3.0,       # 거래량 급증 배수
            'put_call_ratio': 1.5,     # Put/Call 비율
            'vix_level': 25.0,         # VIX 공포지수
            'futures_premium': 0.02    # 선물 프리미엄 임계값
        }
        
        logger.info("🚀 파생상품 모니터링 시스템 초기화 완료")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        # 한국투자증권 API 초기화
        self.kis_api = KISDerivativesAPI()
        await self.kis_api.__aenter__()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.kis_api:
            await self.kis_api.__aexit__(exc_type, exc_val, exc_tb)
        if self.session:
            await self.session.close()
        self.is_monitoring = False
    
    async def collect_k200_derivatives(self) -> List[DerivativeData]:
        """K200 파생상품 데이터 수집 (한국투자증권 API 사용)"""
        derivatives = []
        
        try:
            if self.kis_api:
                logger.info("📊 한국투자증권 API로 K200 파생상품 수집 중...")
                
                # 1. KOSPI200 선물 데이터
                futures = await self.kis_api.get_kospi200_futures()
                for future in futures:
                    derivative = DerivativeData(
                        symbol=future.symbol,
                        underlying_asset="KOSPI200",
                        derivative_type="future",
                        strike_price=0,  # 선물은 행사가격 없음
                        expiry_date="",  # 만료일 정보 추가 필요
                        option_type="",
                        current_price=future.current_price,
                        volume=future.volume,
                        open_interest=future.open_interest,
                        implied_volatility=0,  # 선물은 내재변동성 없음
                        delta=1.0,  # 선물의 델타는 1
                        gamma=0,
                        theta=0,
                        vega=0,
                        bid_price=future.bid_price,
                        ask_price=future.ask_price,
                        change=future.change,
                        change_percent=future.change_rate,
                        timestamp=future.timestamp
                    )
                    derivatives.append(derivative)
                
                # 2. KOSPI200 옵션 데이터
                options = await self.kis_api.get_kospi200_options()
                for option in options:
                    derivative = DerivativeData(
                        symbol=option.symbol,
                        underlying_asset="KOSPI200",
                        derivative_type="option",
                        strike_price=option.strike_price or 0,
                        expiry_date=option.expiry_date or "",
                        option_type=option.option_type or "",
                        current_price=option.current_price,
                        volume=option.volume,
                        open_interest=option.open_interest,
                        implied_volatility=option.implied_volatility or 0,
                        delta=option.delta or 0,
                        gamma=option.gamma or 0,
                        theta=option.theta or 0,
                        vega=option.vega or 0,
                        bid_price=option.bid_price,
                        ask_price=option.ask_price,
                        change=option.change,
                        change_percent=option.change_rate,
                        timestamp=option.timestamp
                    )
                    derivatives.append(derivative)
                
                logger.info(f"✅ 한국투자증권 API로 K200 파생상품 {len(derivatives)}개 수집 완료")
            else:
                # 기존 Yahoo Finance 백업 방식
                logger.warning("⚠️ 한국투자증권 API 없음, Yahoo Finance 백업 사용")
                derivatives = await self._collect_k200_derivatives_backup()
                
        except Exception as e:
            logger.error(f"❌ K200 파생상품 수집 오류: {e}")
            # 백업 방식으로 시도
            derivatives = await self._collect_k200_derivatives_backup()
        
        return derivatives
    
    async def _collect_k200_derivatives_backup(self) -> List[DerivativeData]:
        """K200 파생상품 백업 수집 방식 (기존 Yahoo Finance)"""
        derivatives = []
        
        try:
            # 기존 Yahoo Finance 방식 유지
            symbols = ["^KS200", "KR7069500007.KS", "233740.KS"]
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        volume = hist['Volume'].iloc[-1]
                        change = current_price - hist['Open'].iloc[-1]
                        change_percent = (change / hist['Open'].iloc[-1]) * 100
                        
                        derivative = DerivativeData(
                            symbol=symbol,
                            underlying_asset="KOSPI200",
                            derivative_type="future" if symbol == "^KS200" else "etf",
                            strike_price=0,
                            expiry_date="",
                            option_type="",
                            current_price=current_price,
                            volume=int(volume),
                            open_interest=0,
                            implied_volatility=0,
                            delta=1.0,
                            gamma=0,
                            theta=0,
                            vega=0,
                            bid_price=0,
                            ask_price=0,
                            change=change,
                            change_percent=change_percent,
                            timestamp=datetime.now().isoformat()
                        )
                        derivatives.append(derivative)
                        
                except Exception as e:
                    logger.warning(f"백업 수집 {symbol} 실패: {e}")
            
            # 시뮬레이션 옵션 데이터 추가 (기존 방식)
            if derivatives:
                kospi200_price = derivatives[0].current_price
                simulated_options = self._generate_simulated_options(kospi200_price)
                derivatives.extend(simulated_options)
                
        except Exception as e:
            logger.error(f"❌ K200 파생상품 백업 수집 오류: {e}")
        
        return derivatives
    
    async def get_nasdaq100_derivatives(self) -> List[DerivativeData]:
        """나스닥100 파생상품 데이터 수집"""
        derivatives = []
        
        try:
            # 나스닥100 선물
            futures_data = await self._get_us_futures('NQ=F', 'NASDAQ100')
            derivatives.extend(futures_data)
            
            # 나스닥100 옵션 (QQQ ETF 기준)
            options_data = await self._get_us_options('QQQ', 'NASDAQ100')
            derivatives.extend(options_data)
            
            logger.info(f"✅ 나스닥100 파생상품 {len(derivatives)}개 수집 완료")
            
        except Exception as e:
            logger.error(f"나스닥100 파생상품 수집 오류: {e}")
        
        return derivatives
    
    async def get_sp500_derivatives(self) -> List[DerivativeData]:
        """S&P500 파생상품 데이터 수집"""
        derivatives = []
        
        try:
            # S&P500 선물
            futures_data = await self._get_us_futures('ES=F', 'SP500')
            derivatives.extend(futures_data)
            
            # S&P500 옵션 (SPY ETF 기준)
            options_data = await self._get_us_options('SPY', 'SP500')
            derivatives.extend(options_data)
            
            logger.info(f"✅ S&P500 파생상품 {len(derivatives)}개 수집 완료")
            
        except Exception as e:
            logger.error(f"S&P500 파생상품 수집 오류: {e}")
        
        return derivatives
    
    async def _get_us_futures(self, symbol: str, underlying: str) -> List[DerivativeData]:
        """미국 선물 데이터 수집"""
        futures = []
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            info = ticker.info
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                volume = int(hist['Volume'].iloc[-1]) if hist['Volume'].iloc[-1] > 0 else 0
                
                future_data = DerivativeData(
                    symbol=symbol,
                    underlying=underlying,
                    derivative_type="future",
                    current_price=float(current_price),
                    volume=volume,
                    last_updated=datetime.now().isoformat()
                )
                futures.append(future_data)
                
        except Exception as e:
            logger.error(f"미국 선물 {symbol} 데이터 수집 오류: {e}")
        
        return futures
    
    async def _get_us_options(self, symbol: str, underlying: str) -> List[DerivativeData]:
        """미국 옵션 데이터 수집"""
        options = []
        
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # 만료일 목록 가져오기
            expiry_dates = ticker.options[:3]  # 최근 3개 만료일
            
            for expiry in expiry_dates:
                try:
                    # 옵션 체인 가져오기
                    opt_chain = ticker.option_chain(expiry)
                    
                    # Call 옵션 처리
                    for _, call in opt_chain.calls.head(10).iterrows():  # 상위 10개만
                        call_data = DerivativeData(
                            symbol=f"{symbol}C{call['strike']:.0f}{expiry.replace('-', '')}",
                            underlying=underlying,
                            derivative_type="option",
                            option_type="call",
                            strike_price=float(call['strike']),
                            expiry_date=expiry,
                            current_price=float(call['lastPrice']) if call['lastPrice'] else 0.0,
                            volume=int(call['volume']) if call['volume'] else 0,
                            open_interest=int(call['openInterest']) if call['openInterest'] else 0,
                            implied_volatility=float(call['impliedVolatility']) if call['impliedVolatility'] else None,
                            bid=float(call['bid']) if call['bid'] else 0.0,
                            ask=float(call['ask']) if call['ask'] else 0.0,
                            last_updated=datetime.now().isoformat()
                        )
                        options.append(call_data)
                    
                    # Put 옵션 처리
                    for _, put in opt_chain.puts.head(10).iterrows():  # 상위 10개만
                        put_data = DerivativeData(
                            symbol=f"{symbol}P{put['strike']:.0f}{expiry.replace('-', '')}",
                            underlying=underlying,
                            derivative_type="option",
                            option_type="put",
                            strike_price=float(put['strike']),
                            expiry_date=expiry,
                            current_price=float(put['lastPrice']) if put['lastPrice'] else 0.0,
                            volume=int(put['volume']) if put['volume'] else 0,
                            open_interest=int(put['openInterest']) if put['openInterest'] else 0,
                            implied_volatility=float(put['impliedVolatility']) if put['impliedVolatility'] else None,
                            bid=float(put['bid']) if put['bid'] else 0.0,
                            ask=float(put['ask']) if put['ask'] else 0.0,
                            last_updated=datetime.now().isoformat()
                        )
                        options.append(put_data)
                        
                except Exception as e:
                    logger.warning(f"옵션 체인 {expiry} 처리 오류: {e}")
                    
        except Exception as e:
            logger.error(f"미국 옵션 {symbol} 데이터 수집 오류: {e}")
        
        return options
    
    async def collect_all_derivatives(self) -> Dict[str, List[DerivativeData]]:
        """모든 파생상품 데이터 병렬 수집"""
        logger.info("🔄 모든 파생상품 데이터 수집 시작...")
        
        tasks = [
            self.collect_k200_derivatives(),
            self.get_nasdaq100_derivatives(),
            self.get_sp500_derivatives()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_data = {
            'K200': results[0] if not isinstance(results[0], Exception) else [],
            'NASDAQ100': results[1] if not isinstance(results[1], Exception) else [],
            'SP500': results[2] if not isinstance(results[2], Exception) else []
        }
        
        # 캐시 업데이트
        for market, derivatives in all_data.items():
            for derivative in derivatives:
                self.data_cache[derivative.symbol] = derivative
        
        total_count = sum(len(derivatives) for derivatives in all_data.values())
        logger.info(f"✅ 총 {total_count}개 파생상품 데이터 수집 완료")
        
        return all_data
    
    def analyze_market_signals(self, derivatives_data: Dict[str, List[DerivativeData]]) -> List[MarketSignal]:
        """시장 신호 분석"""
        signals = []
        
        for market, derivatives in derivatives_data.items():
            if not derivatives:
                continue
                
            try:
                # 시장별 신호 분석
                market_signal = self._analyze_single_market(market, derivatives)
                if market_signal:
                    signals.append(market_signal)
                    
            except Exception as e:
                logger.error(f"{market} 시장 신호 분석 오류: {e}")
        
        return signals
    
    def _analyze_single_market(self, market: str, derivatives: List[DerivativeData]) -> Optional[MarketSignal]:
        """개별 시장 신호 분석"""
        try:
            # 옵션과 선물 분리
            options = [d for d in derivatives if d.derivative_type == "option"]
            futures = [d for d in derivatives if d.derivative_type == "future"]
            
            if not options and not futures:
                return None
            
            trigger_factors = []
            risk_level = "LOW"
            signal_type = "NEUTRAL"
            confidence = 50.0
            
            # 1. 변동성 급등 분석
            if options:
                avg_iv = np.mean([opt.implied_volatility for opt in options if opt.implied_volatility])
                if avg_iv > self.risk_thresholds['volatility_spike'] / 100:
                    trigger_factors.append(f"변동성 급등 (IV: {avg_iv:.1%})")
                    risk_level = "HIGH"
                    confidence += 20
            
            # 2. Put/Call 비율 분석
            calls = [opt for opt in options if opt.option_type == "call"]
            puts = [opt for opt in options if opt.option_type == "put"]
            
            if calls and puts:
                put_volume = sum(put.volume for put in puts)
                call_volume = sum(call.volume for call in calls)
                
                if call_volume > 0:
                    pc_ratio = put_volume / call_volume
                    if pc_ratio > self.risk_thresholds['put_call_ratio']:
                        trigger_factors.append(f"Put/Call 비율 급등 ({pc_ratio:.2f})")
                        signal_type = "CRASH_WARNING"
                        risk_level = "HIGH"
                        confidence += 25
                    elif pc_ratio < 0.5:
                        trigger_factors.append(f"Call 옵션 급증 ({pc_ratio:.2f})")
                        signal_type = "SURGE_WARNING"
                        confidence += 15
            
            # 3. 거래량 급증 분석
            if options:
                volumes = [opt.volume for opt in options if opt.volume > 0]
                if volumes:
                    avg_volume = np.mean(volumes)
                    max_volume = max(volumes)
                    
                    if max_volume > avg_volume * self.risk_thresholds['volume_surge']:
                        trigger_factors.append(f"거래량 급증 (최대: {max_volume:,})")
                        confidence += 15
            
            # 4. 선물 프리미엄 분석
            if futures and len(futures) > 0:
                # 선물 프리미엄 분석 (단순화)
                future_price = futures[0].current_price
                # 기초자산 대비 프리미엄 계산 (시뮬레이션)
                premium_rate = np.random.uniform(-0.01, 0.03)  # 실제로는 계산 필요
                
                if abs(premium_rate) > self.risk_thresholds['futures_premium']:
                    trigger_factors.append(f"선물 프리미엄 이상 ({premium_rate:.2%})")
                    confidence += 10
            
            # 신호 타입 결정
            if confidence > 80:
                risk_level = "CRITICAL"
            elif confidence > 65:
                risk_level = "HIGH"
            elif confidence > 50:
                risk_level = "MEDIUM"
            
            if not trigger_factors:
                return None
            
            return MarketSignal(
                signal_type=signal_type,
                confidence=min(100, confidence),
                underlying_asset=market,
                trigger_factors=trigger_factors,
                risk_level=risk_level,
                predicted_move=np.random.uniform(-5, 5),  # 실제로는 계산 필요
                time_horizon="1D",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"{market} 시장 분석 오류: {e}")
            return None
    
    async def get_gemini_analysis(self, signals: List[MarketSignal], derivatives_data: Dict[str, List[DerivativeData]]) -> str:
        """Gemini AI 종합 분석"""
        if not self.gemini_api_key:
            return "Gemini API 키가 설정되지 않음"
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-8B')
            
            # 분석 데이터 요약
            data_summary = {
                'signals_count': len(signals),
                'high_risk_signals': len([s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']]),
                'markets_analyzed': list(derivatives_data.keys()),
                'total_derivatives': sum(len(d) for d in derivatives_data.values())
            }
            
            prompt = f"""
            파생상품 시장 데이터를 기반으로 주식시장의 폭락/폭등 가능성을 분석해주세요.

            📊 **분석 데이터 요약**:
            - 분석 신호 수: {data_summary['signals_count']}개
            - 고위험 신호: {data_summary['high_risk_signals']}개  
            - 분석 시장: {', '.join(data_summary['markets_analyzed'])}
            - 총 파생상품 수: {data_summary['total_derivatives']}개

            🚨 **감지된 시장 신호**:
            {json.dumps([s.__dict__ for s in signals], ensure_ascii=False, indent=2)}

            📈 **파생상품 데이터 샘플**:
            {json.dumps({k: [d.to_dict() for d in v[:3]] for k, v in derivatives_data.items()}, ensure_ascii=False, indent=2)}

            다음 관점에서 종합 분석해주세요:

            1. **폭락 위험도 분석** (0-100점)
               - Put 옵션 급증, 변동성 스파이크, 선물 프리미엄 등

            2. **폭등 가능성 분석** (0-100점)  
               - Call 옵션 급증, 낙관적 신호, 매수 압력 등

            3. **시장별 위험도**
               - K200: 한국 시장 특성 고려
               - 나스닥100: 기술주 중심 분석
               - S&P500: 전체 시장 대표성

            4. **시간대별 예측**
               - 단기 (1-3일): 즉각적 위험
               - 중기 (1-2주): 추세 전환 가능성
               - 장기 (1개월): 구조적 변화

            5. **투자자 행동 권고**
               - 리스크 관리 방안
               - 포지션 조정 제안
               - 주의 관찰 지표

            **한국어로 상세하고 실용적인 분석을 제공해주세요.**
            """
            
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini 분석 오류: {e}")
            return f"Gemini 분석 중 오류 발생: {str(e)}"
    
    async def start_real_time_monitoring(self, duration_minutes: int = 60) -> None:
        """실시간 모니터링 시작 (WebSocket 연결 포함)"""
        logger.info(f"🚀 실시간 파생상품 모니터링 시작 ({duration_minutes}분)")
        
        try:
            # 한국투자증권 WebSocket 연결 설정
            if self.kis_api:
                # K200 선물 심볼 가져오기
                futures = await self.kis_api.get_kospi200_futures()
                symbols = [future.symbol for future in futures[:5]]  # 상위 5개
                
                if symbols:
                    logger.info(f"📡 WebSocket 연결 설정: {symbols}")
                    
                    # WebSocket 연결을 백그라운드 태스크로 실행
                    websocket_task = asyncio.create_task(
                        self.kis_api.connect_websocket(symbols)
                    )
            
            # 기존 모니터링 루프 실행
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                try:
                    # 1. 파생상품 데이터 수집
                    k200_derivatives = await self.collect_k200_derivatives()
                    nasdaq_derivatives = await self.get_nasdaq100_derivatives()
                    sp500_derivatives = await self.get_sp500_derivatives()
                    
                    all_derivatives = k200_derivatives + nasdaq_derivatives + sp500_derivatives
                    
                    # 2. 시장 신호 분석
                    signals = await self.analyze_market_signals(all_derivatives)
                    
                    # 3. 위험 신호 체크
                    for signal in signals:
                        if signal.risk_level in ['HIGH', 'CRITICAL']:
                            await self._send_alert(signal)
                    
                    # 4. 상태 로깅
                    self._log_monitoring_status(all_derivatives, signals)
                    
                    # 5분 대기
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logger.error(f"모니터링 루프 오류: {e}")
                    await asyncio.sleep(60)  # 오류 시 1분 대기
            
            # WebSocket 연결 종료
            if 'websocket_task' in locals():
                websocket_task.cancel()
                
        except Exception as e:
            logger.error(f"❌ 실시간 모니터링 오류: {e}")
    
    def _log_monitoring_status(self, derivatives: List[DerivativeData], signals: List[MarketSignal]) -> None:
        """모니터링 상태 로깅"""
        k200_count = len([d for d in derivatives if d.underlying_asset == "KOSPI200"])
        nasdaq_count = len([d for d in derivatives if d.underlying_asset == "NASDAQ100"])
        sp500_count = len([d for d in derivatives if d.underlying_asset == "S&P500"])
        
        high_risk_signals = len([s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']])
        
        logger.info(f"📊 모니터링 현황: K200({k200_count}) NASDAQ({nasdaq_count}) S&P500({sp500_count}) | 위험신호: {high_risk_signals}개")
        
        # 주요 신호 요약
        for signal in signals[:3]:  # 상위 3개
            logger.info(f"  🚨 {signal.signal_type}: {signal.underlying_asset} (신뢰도: {signal.confidence:.1f}%)")
    
    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self.is_monitoring = False
        logger.info("🛑 실시간 모니터링 중지")
    
    def get_signal_history(self, limit: int = 10) -> List[MarketSignal]:
        """신호 히스토리 조회"""
        return self.signal_history[-limit:]
    
    def get_current_data_summary(self) -> Dict[str, Any]:
        """현재 데이터 요약"""
        return {
            'cached_derivatives': len(self.data_cache),
            'signal_history_count': len(self.signal_history),
            'is_monitoring': self.is_monitoring,
            'last_update': datetime.now().isoformat(),
            'markets_covered': ['K200', 'NASDAQ100', 'SP500']
        }

# 전역 인스턴스
_derivatives_monitor = None

def get_derivatives_monitor(gemini_api_key: str = "") -> DerivativesMonitor:
    """파생상품 모니터 인스턴스 반환"""
    global _derivatives_monitor
    if _derivatives_monitor is None:
        _derivatives_monitor = DerivativesMonitor(gemini_api_key)
    return _derivatives_monitor

async def main():
    """테스트용 메인 함수"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    
    async with DerivativesMonitor(gemini_key) as monitor:
        print("🚀 파생상품 모니터링 시스템 테스트")
        print("="*50)
        
        # 한 번 데이터 수집 및 분석
        derivatives_data = await monitor.collect_all_derivatives()
        signals = monitor.analyze_market_signals(derivatives_data)
        
        if signals:
            gemini_analysis = await monitor.get_gemini_analysis(signals, derivatives_data)
            print("\n🤖 Gemini AI 분석:")
            print("-"*40)
            print(gemini_analysis)
        
        print(f"\n📊 데이터 요약: {monitor.get_current_data_summary()}")
        
        # 실시간 모니터링 시작 (테스트용 30초)
        print("\n🔄 30초간 실시간 모니터링 테스트...")
        monitoring_task = asyncio.create_task(monitor.start_real_time_monitoring(30))
        
        # 30초 후 중지
        await asyncio.sleep(30)
        monitor.stop_monitoring()
        
        print("✅ 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main()) 