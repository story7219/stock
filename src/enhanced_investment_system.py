#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 향상된 AI 투자 분석 시스템 v4.0
=====================================
모든 환경 변수를 활용한 완전한 투자 시스템
- Gemini AI 고급 분석
- 텔레그램 실시간 알림
- 한국투자증권 API 연동
- DART 공시정보 활용
- Google 스프레드시트 연동
- 고급 기술적 분석
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# AI 및 API 라이브러리
import google.generativeai as genai
import yfinance as yf
import requests
from telegram import Bot
import gspread
from google.oauth2.service_account import Credentials

# 내부 모듈
from .core.optimized_core import get_core
from .modules.unified_data_processor import get_processor
from .modules.optimized_investment_strategies import get_strategy_engine
from .modules.notification_system import NotificationSystem
from .modules.technical_analysis import TechnicalAnalyzer
from .modules.derivatives_monitor import get_derivatives_monitor, DerivativesMonitor

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """환경 변수 설정"""
    # Gemini AI 설정
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash-8B"
    gemini_temperature: float = 0.03
    gemini_max_tokens: int = 8192
    
    # 텔레그램 설정
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # 한국투자증권 API
    kis_app_key: str = ""
    kis_app_secret: str = ""
    kis_account_number: str = ""
    
    # DART API
    dart_api_key: str = ""
    
    # Google 서비스
    google_service_account_file: str = ""
    google_spreadsheet_id: str = ""
    google_worksheet_name: str = ""
    
    # GitHub API
    github_api_token: str = ""
    
    # ZAPIER NLA API
    zapier_nla_api_key: str = ""
    
    # 기타 설정
    is_mock: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'EnvironmentConfig':
        """환경 변수에서 설정 로드"""
        return cls(
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-8B'),
            gemini_temperature=float(os.getenv('GEMINI_TEMPERATURE', '0.03')),
            gemini_max_tokens=int(os.getenv('GEMINI_MAX_TOKENS', '8192')),
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
            kis_app_key=os.getenv('LIVE_KIS_APP_KEY', ''),
            kis_app_secret=os.getenv('LIVE_KIS_APP_SECRET', ''),
            kis_account_number=os.getenv('LIVE_KIS_ACCOUNT_NUMBER', ''),
            dart_api_key=os.getenv('DART_API_KEY', ''),
            google_service_account_file=os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', ''),
            google_spreadsheet_id=os.getenv('GOOGLE_SPREADSHEET_ID', ''),
            google_worksheet_name=os.getenv('GOOGLE_WORKSHEET_NAME', ''),
            github_api_token=os.getenv('GITHUB_API_TOKEN', ''),
            zapier_nla_api_key=os.getenv('ZAPIER_NLA_API_KEY', ''),
            is_mock=os.getenv('IS_MOCK', 'true').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )

class EnhancedInvestmentSystem:
    """향상된 투자 시스템"""
    
    def __init__(self):
        self.config = EnvironmentConfig.from_env()
        self.core = get_core()
        self.data_processor = get_processor(self.config.gemini_api_key)
        self.strategy_engine = get_strategy_engine()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # 파생상품 모니터링 시스템 추가
        self.derivatives_monitor = get_derivatives_monitor(self.config.gemini_api_key)
        
        # 외부 서비스 초기화
        self._init_gemini_ai()
        self._init_telegram_bot()
        self._init_google_sheets()
        self._init_kis_api()
        
        # 알림 시스템
        self.notification_system = NotificationSystem({
            'telegram_enabled': bool(self.config.telegram_bot_token),
            'telegram_bot_token': self.config.telegram_bot_token,
            'telegram_chat_id': self.config.telegram_chat_id
        })
        
        logger.info("🚀 향상된 투자 시스템 초기화 완료 (파생상품 모니터링 포함)")
    
    def _init_gemini_ai(self):
        """Gemini AI 초기화"""
        if self.config.gemini_api_key:
            genai.configure(api_key=self.config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name=self.config.gemini_model,
                generation_config={
                    "temperature": self.config.gemini_temperature,
                    "max_output_tokens": self.config.gemini_max_tokens,
                }
            )
            logger.info(f"✅ Gemini AI 초기화: {self.config.gemini_model}")
        else:
            self.gemini_model = None
            logger.warning("⚠️ Gemini API 키가 설정되지 않음")
    
    def _init_telegram_bot(self):
        """텔레그램 봇 초기화"""
        if self.config.telegram_bot_token:
            self.telegram_bot = Bot(token=self.config.telegram_bot_token)
            logger.info("✅ 텔레그램 봇 초기화 완료")
        else:
            self.telegram_bot = None
            logger.warning("⚠️ 텔레그램 봇 토큰이 설정되지 않음")
    
    def _init_google_sheets(self):
        """Google 스프레드시트 초기화"""
        if self.config.google_service_account_file and os.path.exists(self.config.google_service_account_file):
            try:
                credentials = Credentials.from_service_account_file(
                    self.config.google_service_account_file,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                self.google_client = gspread.authorize(credentials)
                logger.info("✅ Google 스프레드시트 초기화 완료")
            except Exception as e:
                self.google_client = None
                logger.warning(f"⚠️ Google 스프레드시트 초기화 실패: {e}")
        else:
            self.google_client = None
            logger.warning("⚠️ Google 서비스 계정 파일이 설정되지 않음")
    
    def _init_kis_api(self):
        """한국투자증권 API 초기화"""
        if self.config.kis_app_key and self.config.kis_app_secret:
            self.kis_config = {
                'app_key': self.config.kis_app_key,
                'app_secret': self.config.kis_app_secret,
                'account_number': self.config.kis_account_number,
                'base_url': 'https://openapi.koreainvestment.com:9443' if not self.config.is_mock else 'https://openapivts.koreainvestment.com:29443'
            }
            logger.info("✅ 한국투자증권 API 초기화 완료")
        else:
            self.kis_config = None
            logger.warning("⚠️ 한국투자증권 API 키가 설정되지 않음")
    
    async def get_kis_access_token(self) -> Optional[str]:
        """KIS API 액세스 토큰 획득"""
        if not self.kis_config:
            return None
        
        try:
            url = f"{self.kis_config['base_url']}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.kis_config['app_key'],
                "appsecret": self.kis_config['app_secret']
            }
            
            async with self.core.get_session() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('access_token')
                    else:
                        logger.error(f"KIS 토큰 획득 실패: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"KIS 토큰 획득 오류: {e}")
            return None
    
    async def get_kospi_stocks(self) -> List[Dict[str, Any]]:
        """코스피 종목 리스트 획득"""
        try:
            # KIS API를 통한 코스피 종목 조회
            if self.kis_config:
                access_token = await self.get_kis_access_token()
                if access_token:
                    return await self._get_kospi_from_kis(access_token)
            
            # 대체 방법: yfinance를 통한 주요 종목
            kospi_symbols = [
                '005930.KS',  # 삼성전자
                '000660.KS',  # SK하이닉스
                '035420.KS',  # NAVER
                '051910.KS',  # LG화학
                '006400.KS',  # 삼성SDI
                '035720.KS',  # 카카오
                '207940.KS',  # 삼성바이오로직스
                '068270.KS',  # 셀트리온
                '323410.KS',  # 카카오뱅크
                '003670.KS'   # 포스코
            ]
            
            stocks = []
            for symbol in kospi_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    stocks.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'sector': info.get('sector', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'current_price': info.get('currentPrice', 0)
                    })
                except Exception as e:
                    logger.warning(f"종목 정보 획득 실패 {symbol}: {e}")
            
            return stocks
            
        except Exception as e:
            logger.error(f"코스피 종목 조회 오류: {e}")
            return []
    
    async def _get_kospi_from_kis(self, access_token: str) -> List[Dict[str, Any]]:
        """KIS API를 통한 코스피 종목 조회"""
        try:
            url = f"{self.kis_config['base_url']}/uapi/domestic-stock/v1/quotations/psearch-title"
            headers = {
                "authorization": f"Bearer {access_token}",
                "appkey": self.kis_config['app_key'],
                "appsecret": self.kis_config['app_secret'],
                "tr_id": "CTPF1002R"
            }
            
            params = {
                "prdt_type_cd": "300",  # 주식
                "pdno": "",
                "prdt_name": "",
                "start_dt": "",
                "end_dt": ""
            }
            
            async with self.core.get_session() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        stocks = []
                        for item in data.get('output', []):
                            stocks.append({
                                'symbol': item.get('pdno', ''),
                                'name': item.get('prdt_name', ''),
                                'market': 'KOSPI',
                                'current_price': float(item.get('stck_prpr', 0))
                            })
                        return stocks[:50]  # 상위 50개 종목
                    else:
                        logger.error(f"KIS 종목 조회 실패: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"KIS 종목 조회 오류: {e}")
            return []
    
    async def get_nasdaq_stocks(self) -> List[Dict[str, Any]]:
        """나스닥 주요 종목 리스트"""
        nasdaq_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
            'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'ORCL', 'CSCO', 'COST', 'PEP'
        ]
        
        stocks = []
        for symbol in nasdaq_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                stocks.append({
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', 0)
                })
            except Exception as e:
                logger.warning(f"나스닥 종목 정보 획득 실패 {symbol}: {e}")
        
        return stocks
    
    async def get_sp500_stocks(self) -> List[Dict[str, Any]]:
        """S&P 500 주요 종목 리스트"""
        sp500_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA',
            'BRK-B', 'UNH', 'JNJ', 'XOM', 'JPM',
            'V', 'PG', 'MA', 'HD', 'CVX',
            'LLY', 'ABBV', 'PFE', 'KO', 'AVGO'
        ]
        
        stocks = []
        for symbol in sp500_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                stocks.append({
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', 0)
                })
            except Exception as e:
                logger.warning(f"S&P 500 종목 정보 획득 실패 {symbol}: {e}")
        
        return stocks
    
    async def analyze_with_gemini(self, stock_data: Dict[str, Any], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini AI를 통한 고급 분석"""
        if not self.gemini_model:
            return {"error": "Gemini AI가 설정되지 않음"}
        
        try:
            prompt = f"""
            다음 종목을 투자 대가들의 관점에서 분석해주세요:
            
            종목 정보:
            - 심볼: {stock_data.get('symbol', 'N/A')}
            - 이름: {stock_data.get('name', 'N/A')}
            - 현재가: {stock_data.get('current_price', 0):,.0f}
            - 섹터: {stock_data.get('sector', 'N/A')}
            
            기술적 분석 결과:
            {json.dumps(technical_analysis, ensure_ascii=False, indent=2)}
            
            다음 관점에서 분석해주세요:
            1. 워런 버핏 관점 (가치투자)
            2. 피터 린치 관점 (성장투자)
            3. 벤저민 그레이엄 관점 (안전마진)
            4. 종합 투자 점수 (0-100점)
            5. 투자 추천 여부 및 이유
            
            한국어로 답변해주세요.
            """
            
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            
            return {
                "gemini_analysis": response.text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gemini 분석 오류: {e}")
            return {"error": str(e)}
    
    async def send_telegram_notification(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """텔레그램 알림 전송"""
        if not self.telegram_bot or not self.config.telegram_chat_id:
            return False
        
        try:
            await self.telegram_bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            logger.error(f"텔레그램 알림 전송 오류: {e}")
            return False
    
    async def save_to_google_sheets(self, data: List[Dict[str, Any]], sheet_name: str = "투자분석결과") -> bool:
        """Google 스프레드시트에 결과 저장"""
        if not self.google_client or not self.config.google_spreadsheet_id:
            return False
        
        try:
            spreadsheet = self.google_client.open_by_key(self.config.google_spreadsheet_id)
            
            # 워크시트 존재 확인 및 생성
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
            
            # 데이터 변환
            if data:
                df = pd.DataFrame(data)
                # 헤더 추가
                worksheet.clear()
                worksheet.append_row(df.columns.tolist())
                
                # 데이터 추가
                for _, row in df.iterrows():
                    worksheet.append_row(row.tolist())
                
                logger.info(f"✅ Google 스프레드시트에 {len(data)}개 데이터 저장 완료")
                return True
            
        except Exception as e:
            logger.error(f"Google 스프레드시트 저장 오류: {e}")
            return False
    
    async def run_comprehensive_analysis(self, markets: List[str] = None) -> Dict[str, Any]:
        """종합 투자 분석 실행"""
        if markets is None:
            markets = ['KOSPI', 'NASDAQ', 'SP500']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'markets_analyzed': markets,
            'top_recommendations': [],
            'detailed_analysis': {},
            'summary': {}
        }
        
        all_stocks = []
        
        # 시장별 종목 수집
        for market in markets:
            logger.info(f"📊 {market} 종목 수집 중...")
            
            if market == 'KOSPI':
                stocks = await self.get_kospi_stocks()
            elif market == 'NASDAQ':
                stocks = await self.get_nasdaq_stocks()
            elif market == 'SP500':
                stocks = await self.get_sp500_stocks()
            else:
                continue
            
            for stock in stocks:
                stock['market'] = market
                all_stocks.append(stock)
        
        logger.info(f"🔍 총 {len(all_stocks)}개 종목 분석 시작")
        
        # 병렬 분석 실행
        analysis_tasks = []
        for stock in all_stocks[:20]:  # 상위 20개 종목만 분석
            task = self._analyze_single_stock(stock)
            analysis_tasks.append(task)
        
        # 분석 결과 수집
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        valid_results = []
        for result in analysis_results:
            if isinstance(result, dict) and 'error' not in result:
                valid_results.append(result)
        
        # 결과 정렬 및 상위 5개 선정
        valid_results.sort(key=lambda x: x.get('total_score', 0), reverse=True)
        top_5 = valid_results[:5]
        
        results['top_recommendations'] = top_5
        results['detailed_analysis'] = {r['symbol']: r for r in valid_results}
        results['summary'] = {
            'total_analyzed': len(valid_results),
            'avg_score': sum(r.get('total_score', 0) for r in valid_results) / len(valid_results) if valid_results else 0,
            'top_sectors': self._get_top_sectors(valid_results)
        }
        
        # 결과 저장 및 알림
        await self._save_and_notify_results(results, top_5)
        
        return results
    
    async def _analyze_single_stock(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """개별 종목 분석"""
        try:
            symbol = stock['symbol']
            
            # 기술적 분석 - analyze 메서드 사용
            technical_data = await self.technical_analyzer.analyze(symbol)
            
            # 전략 분석
            strategy_results = await self.strategy_engine.analyze_with_all_strategies({symbol: stock})
            
            # Gemini AI 분석
            gemini_analysis = await self.analyze_with_gemini(stock, technical_data.__dict__ if hasattr(technical_data, '__dict__') else {})
            
            # 종합 점수 계산
            total_score = self._calculate_total_score(technical_data.__dict__ if hasattr(technical_data, '__dict__') else {}, strategy_results, gemini_analysis)
            
            return {
                'symbol': symbol,
                'name': stock.get('name', ''),
                'market': stock.get('market', ''),
                'current_price': stock.get('current_price', 0),
                'sector': stock.get('sector', ''),
                'technical_analysis': technical_data.__dict__ if hasattr(technical_data, '__dict__') else {},
                'strategy_analysis': strategy_results,
                'gemini_analysis': gemini_analysis,
                'total_score': total_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"종목 분석 오류 {stock.get('symbol', 'Unknown')}: {e}")
            return {'error': str(e), 'symbol': stock.get('symbol', 'Unknown')}
    
    def _calculate_total_score(self, technical: Dict, strategy: Dict, gemini: Dict) -> float:
        """종합 점수 계산"""
        score = 0.0
        
        # 기술적 분석 점수 (30%)
        if technical and 'signals' in technical:
            signals = technical['signals']
            trend_score = 50
            if signals.get('trend') in ['STRONG_UPTREND', 'UPTREND']:
                trend_score = 80
            elif signals.get('trend') in ['STRONG_DOWNTREND', 'DOWNTREND']:
                trend_score = 20
            score += trend_score * 0.3
        
        # 전략 분석 점수 (40%)
        if strategy:
            strategy_scores = []
            for symbol_results in strategy.values():
                for signal in symbol_results:
                    strategy_scores.append(signal.signal_strength)
            if strategy_scores:
                score += (sum(strategy_scores) / len(strategy_scores)) * 0.4
        
        # Gemini 분석 점수 (30%)
        if gemini and 'gemini_analysis' in gemini:
            # 간단한 감정 분석으로 점수 추정
            analysis_text = gemini['gemini_analysis'].lower()
            if '추천' in analysis_text or '매수' in analysis_text:
                score += 70 * 0.3
            elif '중립' in analysis_text:
                score += 50 * 0.3
            else:
                score += 30 * 0.3
        
        return min(100, max(0, score))
    
    def _get_top_sectors(self, results: List[Dict]) -> List[str]:
        """상위 섹터 추출"""
        sector_scores = {}
        for result in results:
            sector = result.get('sector', 'Unknown')
            score = result.get('total_score', 0)
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(score)
        
        # 섹터별 평균 점수 계산
        sector_avg = {sector: sum(scores)/len(scores) for sector, scores in sector_scores.items()}
        
        # 상위 5개 섹터 반환
        return sorted(sector_avg.keys(), key=lambda x: sector_avg[x], reverse=True)[:5]
    
    async def _save_and_notify_results(self, results: Dict, top_5: List[Dict]):
        """결과 저장 및 알림"""
        try:
            # JSON 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 분석 결과 저장: {filename}")
            
            # Google 스프레드시트에 저장
            if top_5:
                await self.save_to_google_sheets(top_5, f"투자분석_{timestamp}")
            
            # 텔레그램 알림
            if top_5:
                message = f"""
🚀 **AI 투자 분석 결과** 🚀
📅 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🏆 **TOP 5 추천 종목**:
"""
                for i, stock in enumerate(top_5, 1):
                    message += f"""
{i}. **{stock['name']}** ({stock['symbol']})
   💰 현재가: {stock['current_price']:,.0f}
   📊 종합점수: {stock['total_score']:.1f}/100
   🏢 섹터: {stock['sector']}
   🌍 시장: {stock['market']}
"""
                
                message += f"""
📈 **분석 요약**:
- 총 분석 종목: {results['summary']['total_analyzed']}개
- 평균 점수: {results['summary']['avg_score']:.1f}점
- 상위 섹터: {', '.join(results['summary']['top_sectors'][:3])}
"""
                
                await self.send_telegram_notification(message)
            
        except Exception as e:
            logger.error(f"결과 저장/알림 오류: {e}")
    
    async def monitor_derivatives_for_crash_signals(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """파생상품 모니터링을 통한 폭락/폭등 신호 감지"""
        logger.info(f"📊 파생상품 폭락/폭등 신호 모니터링 시작 ({duration_minutes}분)")
        
        crash_signals = []
        surge_signals = []
        monitoring_data = []
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            async with self.derivatives_monitor as monitor:
                while datetime.now() < end_time:
                    # 파생상품 데이터 수집
                    derivatives_data = await monitor.collect_all_derivatives()
                    
                    # 시장 신호 분석
                    signals = monitor.analyze_market_signals(derivatives_data)
                    
                    # 신호 분류
                    for signal in signals:
                        if signal.signal_type == "CRASH_WARNING" and signal.risk_level in ['HIGH', 'CRITICAL']:
                            crash_signals.append(signal)
                        elif signal.signal_type == "SURGE_WARNING" and signal.risk_level in ['HIGH', 'CRITICAL']:
                            surge_signals.append(signal)
                    
                    # 모니터링 데이터 저장
                    monitoring_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'derivatives_count': sum(len(d) for d in derivatives_data.values()),
                        'signals_count': len(signals),
                        'high_risk_count': len([s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']])
                    })
                    
                    # 고위험 신호 발견 시 즉시 Gemini 분석
                    high_risk_signals = [s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']]
                    if high_risk_signals:
                        logger.warning(f"🚨 고위험 신호 {len(high_risk_signals)}개 감지!")
                        
                        # Gemini AI 분석 요청
                        gemini_analysis = await monitor.get_gemini_analysis(high_risk_signals, derivatives_data)
                        
                        # 텔레그램 즉시 알림
                        await self._send_crash_alert(high_risk_signals, gemini_analysis)
                    
                    # 30초 대기
                    await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"파생상품 모니터링 오류: {e}")
        
        # 결과 정리
        results = {
            'monitoring_period': f"{duration_minutes}분",
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'crash_signals': [s.__dict__ for s in crash_signals],
            'surge_signals': [s.__dict__ for s in surge_signals],
            'total_crash_warnings': len(crash_signals),
            'total_surge_warnings': len(surge_signals),
            'monitoring_data': monitoring_data,
            'risk_assessment': self._assess_overall_risk(crash_signals, surge_signals)
        }
        
        logger.info(f"✅ 파생상품 모니터링 완료 - 폭락신호: {len(crash_signals)}개, 폭등신호: {len(surge_signals)}개")
        
        return results
    
    async def _send_crash_alert(self, signals: List, gemini_analysis: str):
        """폭락/폭등 경고 알림 전송"""
        try:
            crash_count = len([s for s in signals if s.signal_type == "CRASH_WARNING"])
            surge_count = len([s for s in signals if s.signal_type == "SURGE_WARNING"])
            
            alert_type = "🔴 폭락 경고" if crash_count > surge_count else "🟢 폭등 신호"
            
            message = f"""
🚨 **{alert_type}** 🚨
📅 감지 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ **위험 신호 요약**:
- 폭락 경고: {crash_count}개
- 폭등 신호: {surge_count}개
- 총 신호: {len(signals)}개

📊 **주요 신호들**:
"""
            
            for i, signal in enumerate(signals[:3], 1):  # 상위 3개만
                message += f"""
{i}. {signal.underlying_asset} - {signal.signal_type}
   신뢰도: {signal.confidence:.1f}% | 위험도: {signal.risk_level}
   요인: {', '.join(signal.trigger_factors[:2])}
"""
            
            message += f"""
🤖 **AI 분석 요약**:
{gemini_analysis[:300]}...

⚡ 즉시 포지션 점검을 권장합니다!
"""
            
            # 텔레그램 전송
            await self.send_telegram_notification(message)
            
            # 로그에도 기록
            logger.critical(f"🚨 {alert_type} 발송 완료 - 신호 {len(signals)}개")
            
        except Exception as e:
            logger.error(f"경고 알림 전송 오류: {e}")
    
    def _assess_overall_risk(self, crash_signals: List, surge_signals: List) -> Dict[str, Any]:
        """전체 위험도 평가"""
        crash_risk = 0
        surge_potential = 0
        
        # 폭락 위험도 계산
        if crash_signals:
            crash_confidences = [s.confidence for s in crash_signals]
            crash_risk = min(100, sum(crash_confidences) / len(crash_confidences))
        
        # 폭등 가능성 계산
        if surge_signals:
            surge_confidences = [s.confidence for s in surge_signals]
            surge_potential = min(100, sum(surge_confidences) / len(surge_confidences))
        
        # 전체 위험도 결정
        if crash_risk > 70:
            overall_risk = "CRITICAL"
            recommendation = "즉시 포지션 축소 및 현금 비중 확대"
        elif crash_risk > 50:
            overall_risk = "HIGH"
            recommendation = "리스크 관리 강화 및 손절 준비"
        elif surge_potential > 70:
            overall_risk = "OPPORTUNITY"
            recommendation = "매수 기회 포착 준비"
        else:
            overall_risk = "NORMAL"
            recommendation = "정상 시장 상태 - 기존 전략 유지"
        
        return {
            'crash_risk_score': crash_risk,
            'surge_potential_score': surge_potential,
            'overall_risk_level': overall_risk,
            'recommendation': recommendation,
            'dominant_signal': 'CRASH' if crash_risk > surge_potential else 'SURGE' if surge_potential > 30 else 'NEUTRAL'
        }
    
    async def get_real_time_market_pulse(self) -> Dict[str, Any]:
        """실시간 시장 맥박 체크 (파생상품 기반)"""
        logger.info("💓 실시간 시장 맥박 체크 시작...")
        
        try:
            async with self.derivatives_monitor as monitor:
                # 현재 파생상품 데이터 수집
                derivatives_data = await monitor.collect_all_derivatives()
                
                # 시장 신호 분석
                signals = monitor.analyze_market_signals(derivatives_data)
                
                # Gemini AI 분석
                gemini_analysis = ""
                if signals:
                    gemini_analysis = await monitor.get_gemini_analysis(signals, derivatives_data)
                
                # 시장별 상태 분석
                market_status = {}
                for market, derivatives in derivatives_data.items():
                    if derivatives:
                        market_signals = [s for s in signals if s.underlying_asset == market]
                        
                        # 옵션 데이터 분석
                        options = [d for d in derivatives if d.derivative_type == "option"]
                        calls = [opt for opt in options if opt.option_type == "call"]
                        puts = [opt for opt in options if opt.option_type == "put"]
                        
                        put_call_ratio = 0
                        if calls and puts:
                            put_volume = sum(put.volume for put in puts)
                            call_volume = sum(call.volume for call in calls)
                            put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
                        
                        avg_iv = 0
                        if options:
                            iv_values = [opt.implied_volatility for opt in options if opt.implied_volatility]
                            avg_iv = sum(iv_values) / len(iv_values) if iv_values else 0
                        
                        market_status[market] = {
                            'signal_count': len(market_signals),
                            'put_call_ratio': put_call_ratio,
                            'avg_implied_volatility': avg_iv,
                            'derivatives_count': len(derivatives),
                            'status': self._determine_market_status(put_call_ratio, avg_iv, market_signals)
                        }
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'overall_signals': len(signals),
                    'high_risk_signals': len([s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']]),
                    'market_status': market_status,
                    'gemini_pulse_analysis': gemini_analysis,
                    'risk_assessment': self._assess_overall_risk(
                        [s for s in signals if s.signal_type == "CRASH_WARNING"],
                        [s for s in signals if s.signal_type == "SURGE_WARNING"]
                    )
                }
                
        except Exception as e:
            logger.error(f"실시간 시장 맥박 체크 오류: {e}")
            return {'error': str(e)}
    
    def _determine_market_status(self, put_call_ratio: float, avg_iv: float, signals: List) -> str:
        """시장 상태 판단"""
        if put_call_ratio > 1.5 and avg_iv > 0.3:
            return "FEAR"  # 공포 상태
        elif put_call_ratio < 0.5 and avg_iv > 0.25:
            return "GREED"  # 탐욕 상태
        elif avg_iv > 0.4:
            return "VOLATILE"  # 고변동성
        elif len([s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']]) > 0:
            return "WARNING"  # 경고 상태
        else:
            return "NORMAL"  # 정상 상태

# 전역 인스턴스
_enhanced_system = None

def get_enhanced_system() -> EnhancedInvestmentSystem:
    """향상된 투자 시스템 인스턴스 반환"""
    global _enhanced_system
    if _enhanced_system is None:
        _enhanced_system = EnhancedInvestmentSystem()
    return _enhanced_system

async def main():
    """메인 실행 함수"""
    system = get_enhanced_system()
    
    print("🚀 향상된 AI 투자 분석 시스템 시작")
    print("=" * 50)
    
    # 1. 실시간 시장 맥박 체크
    print("\n💓 실시간 시장 맥박 체크...")
    market_pulse = await system.get_real_time_market_pulse()
    
    if 'error' not in market_pulse:
        print(f"📊 전체 신호: {market_pulse['overall_signals']}개")
        print(f"⚠️ 고위험 신호: {market_pulse['high_risk_signals']}개")
        
        # 시장별 상태 출력
        for market, status in market_pulse['market_status'].items():
            print(f"🌍 {market}: {status['status']} (PC비율: {status['put_call_ratio']:.2f}, IV: {status['avg_implied_volatility']:.1%})")
        
        # 위험도 평가
        risk_assessment = market_pulse['risk_assessment']
        print(f"\n🎯 위험도 평가:")
        print(f"   폭락 위험: {risk_assessment['crash_risk_score']:.1f}점")
        print(f"   폭등 가능성: {risk_assessment['surge_potential_score']:.1f}점")
        print(f"   전체 위험도: {risk_assessment['overall_risk_level']}")
        print(f"   권고사항: {risk_assessment['recommendation']}")
    
    # 2. 종합 분석 실행
    print("\n📈 종합 투자 분석 실행...")
    results = await system.run_comprehensive_analysis(['KOSPI', 'NASDAQ', 'SP500'])
    
    print("\n🏆 TOP 5 투자 추천 종목:")
    print("-" * 40)
    
    for i, stock in enumerate(results['top_recommendations'], 1):
        print(f"{i}. {stock['name']} ({stock['symbol']})")
        print(f"   💰 현재가: {stock['current_price']:,.0f}")
        print(f"   📊 종합점수: {stock['total_score']:.1f}/100")
        print(f"   🏢 섹터: {stock['sector']}")
        print(f"   🌍 시장: {stock['market']}")
        print()
    
    # 3. 파생상품 모니터링 (5분간 테스트)
    print("\n🔍 파생상품 폭락/폭등 신호 모니터링 (5분간)...")
    derivatives_results = await system.monitor_derivatives_for_crash_signals(5)
    
    print(f"📊 모니터링 결과:")
    print(f"   폭락 경고: {derivatives_results['total_crash_warnings']}개")
    print(f"   폭등 신호: {derivatives_results['total_surge_warnings']}개")
    print(f"   전체 위험도: {derivatives_results['risk_assessment']['overall_risk_level']}")
    print(f"   권고사항: {derivatives_results['risk_assessment']['recommendation']}")
    
    # 4. 최종 요약
    print(f"\n📊 최종 분석 요약:")
    print(f"   총 {results['summary']['total_analyzed']}개 종목 분석 완료")
    print(f"   평균 점수: {results['summary']['avg_score']:.1f}점")
    print(f"   상위 섹터: {', '.join(results['summary']['top_sectors'][:3])}")
    print(f"   파생상품 모니터링: {derivatives_results['monitoring_period']} 완료")
    print("\n✅ 전체 분석 완료! 텔레그램으로 상세 결과가 전송되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 