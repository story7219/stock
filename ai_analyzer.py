"""
🤖 AI 트레이딩 분석 시스템 (v2.0 - 모듈화)
- 시장 데이터, 뉴스, 차트 이미지 종합 분석
- 빠른 판단(척후병)과 깊은 분석(본대)을 위한 듀얼 모델 아키텍처
- ScoutStrategyManager, TradingEngine 등 다른 모듈에 분석 결과 제공
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re
import google.generativeai as genai
import asyncio

# rich 라이브러리 추가
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# --- Local Imports ---
from chart_manager import ChartManager # 이름 변경된 모듈
from market_data_provider import AIDataCollector # 의존성 추가
from core_trader import CoreTrader # 의존성 추가
import pandas as pd
import numpy as np
import config

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """시장 데이터 구조"""
    stock_code: str
    current_price: float
    price_change: float
    price_change_rate: float
    volume: int
    market_cap: float = None
    pbr: float = None
    per: float = None
    dividend_yield: float = None

@dataclass
class NewsData:
    """뉴스 데이터 구조"""
    headlines: List[str]
    sentiment_scores: List[float]  # -1(부정) ~ 1(긍정)
    relevance_scores: List[float]  # 0 ~ 1
    summary: str = ""

@dataclass
class TradingSignal:
    """매매 신호 구조"""
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 ~ 1.0
    position_size: float  # 0.0 ~ 1.0 (전체 자본 대비 비중)
    entry_price: float
    stop_loss: float
    target_price: float
    reasoning: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    time_horizon: str  # "SCALPING", "SWING", "POSITION"
    # 트레일링 스탑 관련 필드 추가
    trailing_activation_price: float  # 트레일링 활성화 가격
    trailing_stop_rate: float  # 트레일링 스탑 비율

@dataclass
class AnalysisResult:
    """종합 분석 결과"""
    technical_score: float  # 기술적 분석 점수 (0-100)
    fundamental_score: float  # 펀더멘털 점수 (0-100)
    sentiment_score: float  # 뉴스 감정 점수 (0-100)
    chart_pattern_score: float  # 차트 패턴 점수 (0-100)
    overall_score: float  # 종합 점수 (0-100)
    key_factors: List[str]  # 주요 결정 요인들
    risks: List[str]  # 리스크 요인들
    opportunities: List[str]  # 기회 요인들

class AIAnalyzer:
    """🤖 AI 기반 트레이딩 분석 시스템"""
    
    def __init__(self, trader: CoreTrader, data_provider: AIDataCollector):
        """AIAnalyzer 초기화"""
        self.gemini_api_key = config.GEMINI_API_KEY
        if not self.gemini_api_key:
            logger.error("❌ GEMINI_API_KEY가 설정되지 않았습니다. AI 분석기능을 사용할 수 없습니다.")
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되어야 합니다.")

        # 의존성 주입
        self.trader = trader
        self.data_provider = data_provider
        self.chart_manager = ChartManager(trader_instance=self.trader)
        
        # Gemini API 설정
        genai.configure(api_key=self.gemini_api_key)
        # 모든 모델을 Gemini 1.5 Flash로 통일하여 속도 및 비용 효율성 확보
        self.flash_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.pro_model = self.flash_model  # Pro 모델 호출 시에도 Flash 모델 사용
        
        # rich 콘솔 초기화
        self.console = Console()
        
        # 트레이딩 설정 (config 모듈에서 로드)
        self.max_position_size = config.AI_MAX_POSITION_SIZE
        self.min_confidence = config.AI_MIN_CONFIDENCE
        self.risk_tolerance = config.AI_RISK_TOLERANCE
        
        # 백테스트로 검증된 최적화 파라미터 (config 모듈에서 로드)
        self.optimized_params = {
            'initial_stop_loss': config.AI_INITIAL_STOP_LOSS,
            'trailing_activation': config.AI_TRAILING_ACTIVATION,
            'trailing_stop': config.AI_TRAILING_STOP
        }
        
        logger.info("🤖 AIAnalyzer 초기화 완료 (모델: Gemini 1.5 Flash, 백테스트 최적화 파라미터 적용)")

    # ===================================================================
    # ScoutStrategyManager를 위한 새로운 메서드들 (v2.1 - 배치 처리 및 심층 분석)
    # ===================================================================
    
    async def analyze_scout_candidates(self, stock_infos: List[Dict]) -> List[Dict]:
        """
        여러 척후병 후보 종목들을 배치로 받아 심층 분석 후 점수와 코멘트를 반환합니다.
        (gemini-1.5-flash 사용, 병렬 처리)
        """
        if not stock_infos:
            return []

        logger.info(f"🤖 {len(stock_infos)}개 후보 종목에 대한 AI 배치 분석 시작...")
        
        stock_codes = [s['code'] for s in stock_infos]
        
        # 1. 모든 후보 종목의 종합 분석 데이터를 한 번에 가져옵니다.
        holistic_data_list = await self.data_provider.get_batch_holistic_analysis(stock_codes)
        
        # 데이터를 종목 코드로 쉽게 찾을 수 있도록 맵으로 변환
        holistic_data_map = {data['symbol']: data for data in holistic_data_list if data}

        # 2. 각 종목에 대한 AI 분석 작업을 병렬로 실행합니다.
        tasks = []
        for stock_info in stock_infos:
            holistic_data = holistic_data_map.get(stock_info['code'])
            if holistic_data:
                tasks.append(self._generate_scout_decision(stock_info, holistic_data))
        
        ai_results = await asyncio.gather(*tasks)
        
        # None이 아닌 결과만 필터링하여 반환
        valid_results = [res for res in ai_results if res]
        logger.info(f"✅ AI 배치 분석 완료. {len(valid_results)}개의 유효한 분석 결과 확보.")
        
        return valid_results

    async def _generate_scout_decision(self, stock_info: Dict, holistic_data: Dict) -> Optional[Dict]:
        """한 종목에 대한 AI 분석 프롬프트를 생성하고 API를 호출합니다."""
        try:
            prompt = self._create_scout_prompt(stock_info, holistic_data)
            
            # Flash 모델을 명시적으로 사용
            response = await self.flash_model.generate_content_async(prompt)
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            decision = json.loads(cleaned_response)

            # 응답에 종목 코드 추가
            decision['symbol'] = stock_info['code']
            
            logger.info(f"💡 AI 분석 [{stock_info['code']}]: 점수={decision.get('score')}, 코멘트='{decision.get('comment')}'")
            return decision

        except Exception as e:
            logger.error(f"❌ AI 개별 분석 실패 ({stock_info.get('code')}): {e}", exc_info=True)
            return None

    def _create_scout_prompt(self, stock_info: Dict, holistic_data: Dict) -> str:
        """척후병 판단을 위한 상세 프롬프트를 생성합니다."""
        
        # 뉴스 요약 (있는 경우)
        news_summary = "\n".join([f"- {news['title']}" for news in holistic_data.get('news', [])[:3]])
        if not news_summary: news_summary = "최근 주요 뉴스 없음."

        # 수급 요약
        investor_trends = holistic_data.get('investor_trends', {})
        
        return f"""
        당신은 최고의 데이터 기반 단기 트레이딩 AI 분석가입니다. 제시된 모든 데이터를 종합하여, 다음 종목에 '척후병'을 투입할지 여부를 판단해주세요. 반드시 JSON 형식으로만 답변해야 합니다.

        **1. 기본 정보:**
        - 종목명: {stock_info.get('name', 'N/A')} ({stock_info.get('code', 'N/A')})
        - 현재가: {stock_info.get('current_price', 0):,}원
        - AI 기본 점수: {stock_info.get('score', 'N/A')} / 100

        **2. 기술적 분석 (차트):**
        - 일봉 추세: {holistic_data.get('daily_chart_summary', 'N/A')}
        - 분봉 추세 (단기): {holistic_data.get('minute_chart_summary', 'N/A')}

        **3. 뉴스 및 공시:**
        {news_summary}

        **4. 수급 동향 (개인/외국인/기관):**
        - 개인: {investor_trends.get('individual_net_buy', 0):,}억
        - 외국인: {investor_trends.get('foreign_net_buy', 0):,}억
        - 기관: {investor_trends.get('institution_net_buy', 0):,}억

        **[MISSION]**
        위 모든 정보를 종합적으로 고려하여, 이 종목의 **'단기 매수 매력도'**를 0점에서 100점 사이의 점수로 평가하고, 핵심적인 평가 이유를 한 문장으로 요약해주세요.

        **응답 형식 (JSON):**
        {{
          "score": <0-100 사이의 정수>,
          "comment": "점수를 매긴 핵심적인 이유 (예: '일봉상 정배열 초기이며, 기관 순매수가 유입되고 있어 긍정적.')"
        }}
        """

    # ===================================================================
    # 기존의 깊은 분석 기능들 (본대 투입 등 정교한 판단용)
    # ===================================================================

    def analyze_market_data(self, market_data: MarketData, news_data: NewsData, 
                          chart_period: str = "1M") -> AnalysisResult:
        """📊 종합 시장 데이터 분석"""
        try:
            # 1. 차트 이미지 및 기술적 지표 생성 (ChartManager 사용)
            chart_image = self.chart_manager.generate_chart_image(
                market_data.stock_code, chart_period
            )
            if not chart_image:
                logger.warning(f"차트 이미지를 생성할 수 없어, 텍스트 분석만 진행합니다: {market_data.stock_code}")

            chart_summary = self.chart_manager.get_chart_analysis_summary(
                market_data.stock_code, chart_period
            )
            
            # 2. Gemini API용 구조화된 프롬프트 생성
            prompt = self._create_analysis_prompt(market_data, news_data, chart_summary)
            
            # 3. Gemini API 호출 (Flash 모델 사용)
            response = self._call_gemini_api(prompt, chart_image)
            
            # 4. 응답 파싱 및 분석 결과 생성
            analysis_result = self._parse_gemini_response(response)
            
            logger.info(f"📊 {market_data.stock_code} 종합 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ 시장 분석 실패: {e}")
            # 기본 분석 결과 반환
            return AnalysisResult(
                technical_score=50.0,
                fundamental_score=50.0,
                sentiment_score=50.0,
                chart_pattern_score=50.0,
                overall_score=50.0,
                key_factors=["분석 오류 발생"],
                risks=["시스템 분석 불가"],
                opportunities=[]
            )
    
    def make_trading_decision(self, analysis_result: AnalysisResult, 
                            market_data: MarketData) -> TradingSignal:
        """🎯 매매 결정 생성"""
        try:
            # 1. 기본 매매 신호 결정
            action = self._determine_action(analysis_result)
            
            # 2. 신뢰도 계산
            confidence = self._calculate_confidence(analysis_result)
            
            # 3. 포지션 사이즈 계산
            position_size = self.calculate_position_size(
                market_data.stock_code, confidence
            )
            
            # 4. 손절/목표가 설정
            stop_loss, target_price = self.set_stop_loss_target(
                market_data.stock_code, market_data.current_price
            )
            
            # 5. 트레일링 관련 값 계산
            trailing_activation_price = market_data.current_price * (1 + self.optimized_params['trailing_activation'] / 100)
            trailing_stop_rate = self.optimized_params['trailing_stop'] / 100
            
            # 6. 리스크 레벨 결정
            risk_level = self._assess_risk_level(analysis_result, confidence)
            
            # 7. 투자 기간 설정
            time_horizon = self._determine_time_horizon(analysis_result)
            
            # 8. 매매 근거 생성
            reasoning = self._generate_reasoning(analysis_result, action, confidence)
            
            signal = TradingSignal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                entry_price=market_data.current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                reasoning=reasoning,
                risk_level=risk_level,
                time_horizon=time_horizon,
                trailing_activation_price=trailing_activation_price,
                trailing_stop_rate=trailing_stop_rate
            )
            
            logger.info(f"🎯 매매 결정: {action} (신뢰도: {confidence:.1%})")
            return signal
            
        except Exception as e:
            logger.error(f"❌ 매매 결정 실패: {e}")
            # 안전한 기본 신호 (HOLD)
            return TradingSignal(
                action="HOLD",
                confidence=0.3,
                position_size=0.0,
                entry_price=market_data.current_price,
                stop_loss=market_data.current_price * 0.96,
                target_price=market_data.current_price * 1.06,
                reasoning="시스템 오류로 인한 관망",
                risk_level="HIGH",
                time_horizon="POSITION",
                trailing_activation_price=market_data.current_price * 1.06,
                trailing_stop_rate=0.03
            )
    
    def calculate_position_size(self, stock_code: str, confidence: float) -> float:
        """💰 포지션 사이즈 계산"""
        try:
            # 1. 기본 신뢰도 기반 사이즈
            base_size = confidence * self.max_position_size
            
            # 2. 변동성 조정
            volatility_adjustment = self._get_volatility_adjustment(stock_code)
            adjusted_size = base_size * volatility_adjustment
            
            # 3. 최대/최소 제한 적용
            position_size = max(0.01, min(adjusted_size, self.max_position_size))
            
            logger.info(f"💰 {stock_code} 포지션 사이즈: {position_size:.1%}")
            return position_size
            
        except Exception as e:
            logger.error(f"❌ 포지션 사이즈 계산 실패: {e}")
            return 0.05  # 기본 5%
    
    def set_stop_loss_target(self, stock_code: str, entry_price: float) -> Tuple[float, float]:
        """🛡️ 백테스트 검증된 최적화 손절가/목표가 설정"""
        try:
            # 1. 백테스트 검증된 초기 손절가 설정
            initial_stop_loss = entry_price * (1 - self.optimized_params['initial_stop_loss'] / 100)
            
            # 2. 트레일링 활성화 시점 (6% 수익)
            trailing_activation_price = entry_price * (1 + self.optimized_params['trailing_activation'] / 100)
            
            # 3. 기본 목표가는 트레일링 활성화 지점으로 설정
            target_price = trailing_activation_price
            
            # 4. 차트 패턴 및 지지/저항선 기반 조정 (ChartManager 사용)
            try:
                support_resistance = self.chart_manager.get_support_resistance(stock_code, "1M")
                
                # 지지선 기반 손절가 조정 (더 보수적으로)
                if support_resistance.support_levels:
                    nearest_support = min(support_resistance.support_levels, 
                                        key=lambda x: abs(x - entry_price) if x < entry_price else float('inf'))
                    if nearest_support < entry_price:
                        # 지지선과 백테스트 손절가 중 더 보수적인 값 선택
                        chart_based_stop = nearest_support * 0.98  # 지지선 2% 아래
                        initial_stop_loss = max(initial_stop_loss, chart_based_stop)
                
                # 저항선 기반 목표가 조정
                if support_resistance.resistance_levels:
                    nearest_resistance = min(support_resistance.resistance_levels,
                                           key=lambda x: abs(x - entry_price) if x > entry_price else float('inf'))
                    if nearest_resistance > entry_price:
                        # 저항선이 트레일링 활성화 지점보다 낮으면 저항선을 1차 목표로
                        resistance_target = nearest_resistance * 0.98  # 저항선 2% 아래
                        if resistance_target < trailing_activation_price:
                            target_price = resistance_target
                        else:
                            # 저항선이 높으면 더 공격적인 목표가 설정
                            target_price = min(resistance_target, entry_price * 1.12)  # 최대 12% 수익
                            
            except Exception as e:
                logger.warning(f"⚠️ 차트 분석 기반 조정 실패, 백테스트 파라미터 사용: {e}")
            
            # 5. 최종 검증 및 제한
            stop_loss = max(initial_stop_loss, entry_price * 0.90)  # 최대 10% 손실로 제한
            target_price = min(target_price, entry_price * 1.15)    # 최대 15% 수익으로 제한
            
            # 6. 손절/목표가 비율 검증 (리스크 대비 수익 1:1.5 이상)
            loss_ratio = (entry_price - stop_loss) / entry_price
            profit_ratio = (target_price - entry_price) / entry_price
            
            if profit_ratio / loss_ratio < 1.5:  # 리스크 대비 수익이 1.5배 미만이면 조정
                target_price = entry_price + (entry_price - stop_loss) * 1.5
                logger.info(f"📊 리스크 대비 수익 비율 조정: 1:{profit_ratio/loss_ratio:.1f} → 1:1.5")
            
            logger.info(f"🛡️ 최적화 손절가: {stop_loss:,.0f}원 ({((stop_loss/entry_price-1)*100):+.1f}%)")
            logger.info(f"🎯 최적화 목표가: {target_price:,.0f}원 ({((target_price/entry_price-1)*100):+.1f}%)")
            logger.info(f"📊 트레일링 활성화: {trailing_activation_price:,.0f}원 (+{self.optimized_params['trailing_activation']:.1f}%)")
            
            return stop_loss, target_price
            
        except Exception as e:
            logger.error(f"❌ 최적화 손절/목표가 설정 실패: {e}")
            # 백테스트 검증된 기본값으로 폴백
            return (entry_price * (1 - self.optimized_params['initial_stop_loss'] / 100), 
                    entry_price * (1 + self.optimized_params['trailing_activation'] / 100))
    
    def _create_analysis_prompt(self, market_data: MarketData, news_data: NewsData, 
                              chart_summary: Dict) -> str:
        """Gemini API용 종합 분석 프롬프트 생성"""
        prompt = f"""
한국 주식시장 종목 분석 요청

## 종목 정보
- 종목코드: {market_data.stock_code}
- 현재가: {market_data.current_price:,}원
- 등락률: {market_data.price_change_rate:.2f}%
- 거래량: {market_data.volume:,}주

## 기술적 분석 데이터
- 현재 추세: {chart_summary.get('trend', 'N/A')}
- 감지된 패턴: {', '.join(chart_summary.get('detected_patterns', []))}
- 기술적 신호: {', '.join(chart_summary.get('technical_signals', []))}
- RSI: {chart_summary.get('rsi', 'N/A')}
- MACD: {chart_summary.get('macd', 'N/A')}

## 뉴스 감정 분석
- 주요 헤드라인: {news_data.headlines[:3] if news_data.headlines else ['뉴스 없음']}
- 평균 감정 점수: {np.mean(news_data.sentiment_scores) if news_data.sentiment_scores else 0:.2f}
- 뉴스 요약: {news_data.summary or '해당 없음'}

## 분석 요청사항
다음 형식으로 JSON 응답을 제공해주세요:

{{
    "technical_score": 0-100점,
    "fundamental_score": 0-100점, 
    "sentiment_score": 0-100점,
    "chart_pattern_score": 0-100점,
    "overall_score": 0-100점,
    "recommendation": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "key_factors": ["요인1", "요인2", "요인3"],
    "risks": ["리스크1", "리스크2"],
    "opportunities": ["기회1", "기회2"],
    "time_horizon": "SCALPING/SWING/POSITION",
    "reasoning": "상세한 분석 근거"
}}

특히 차트 이미지를 함께 분석하여 기술적 패턴과 추세를 정확히 파악해주세요.
"""
        return prompt
    
    def _call_gemini_api(self, prompt: str, chart_image: str) -> str:
        """Gemini API 호출 (Flash 모델 사용)"""
        try:
            if chart_image:
                contents = [prompt, chart_image]
            else:
                contents = [prompt]
            
            # Flash 모델 사용
            response = self.flash_model.generate_content(contents)
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API 호출 실패: {e}")
            # 오류 발생 시 기본 JSON 응답 반환
            return json.dumps({
                "technical_score": 50, "fundamental_score": 50, "sentiment_score": 50,
                "chart_pattern_score": 50, "overall_score": 50,
                "key_factors": ["API_ERROR"], "risks": ["API 호출 실패"], "opportunities": []
            })
    
    def _parse_gemini_response(self, response: str) -> AnalysisResult:
        """Gemini 응답 파싱"""
        try:
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                raise ValueError("JSON 형식을 찾을 수 없음")
            
            # AnalysisResult 객체 생성
            return AnalysisResult(
                technical_score=float(data.get('technical_score', 50)),
                fundamental_score=float(data.get('fundamental_score', 50)),
                sentiment_score=float(data.get('sentiment_score', 50)),
                chart_pattern_score=float(data.get('chart_pattern_score', 50)),
                overall_score=float(data.get('overall_score', 50)),
                key_factors=data.get('key_factors', []),
                risks=data.get('risks', []),
                opportunities=data.get('opportunities', [])
            )
            
        except Exception as e:
            logger.error(f"❌ Gemini 응답 파싱 실패: {e}")
            # 기본 분석 결과
            return AnalysisResult(
                technical_score=50.0,
                fundamental_score=50.0,
                sentiment_score=50.0,
                chart_pattern_score=50.0,
                overall_score=50.0,
                key_factors=["응답 파싱 오류"],
                risks=["분석 결과 불확실"],
                opportunities=[]
            )
    
    def _determine_action(self, analysis: AnalysisResult) -> str:
        """매매 액션 결정"""
        if analysis.overall_score >= 70:
            return "BUY"
        elif analysis.overall_score <= 30:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence(self, analysis: AnalysisResult) -> float:
        """신뢰도 계산"""
        # 각 점수의 편차를 기반으로 신뢰도 계산
        scores = [
            analysis.technical_score,
            analysis.fundamental_score,
            analysis.sentiment_score,
            analysis.chart_pattern_score
        ]
        
        # 점수들의 일관성 확인
        std_dev = np.std(scores)
        consistency = max(0, 1 - (std_dev / 50))  # 표준편차가 낮을수록 높은 신뢰도
        
        # 전체 점수의 극값 정도
        extremeness = abs(analysis.overall_score - 50) / 50
        
        # 최종 신뢰도 계산
        confidence = (consistency * 0.6 + extremeness * 0.4)
        return max(0.1, min(0.95, confidence))
    
    def _assess_risk_level(self, analysis: AnalysisResult, confidence: float) -> str:
        """리스크 레벨 평가"""
        risk_score = len(analysis.risks) * 20  # 리스크 요인당 20점
        
        if confidence < 0.5 or risk_score > 60:
            return "HIGH"
        elif confidence > 0.8 and risk_score < 20:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _determine_time_horizon(self, analysis: AnalysisResult) -> str:
        """투자 기간 결정"""
        # 기술적 점수가 높으면 단기, 펀더멘털이 높으면 장기
        if analysis.technical_score > analysis.fundamental_score + 20:
            return "SCALPING"
        elif analysis.fundamental_score > analysis.technical_score + 20:
            return "POSITION"
        else:
            return "SWING"
    
    def _generate_reasoning(self, analysis: AnalysisResult, action: str, 
                          confidence: float) -> str:
        """매매 근거 생성"""
        reasoning_parts = [
            f"종합 점수: {analysis.overall_score:.1f}점",
            f"신뢰도: {confidence:.1%}",
            f"주요 요인: {', '.join(analysis.key_factors[:3])}"
        ]
        
        if analysis.opportunities:
            reasoning_parts.append(f"기회 요인: {', '.join(analysis.opportunities[:2])}")
        
        if analysis.risks:
            reasoning_parts.append(f"리스크: {', '.join(analysis.risks[:2])}")
        
        return " | ".join(reasoning_parts)
    
    def _get_volatility_adjustment(self, stock_code: str) -> float:
        """변동성 기반 포지션 사이즈 조정"""
        try:
            # data_provider를 통해 변동성 데이터 가져오기
            volatility_data = self.data_provider._get_instant_volatility(stock_code)
            
            if volatility_data['level'] == 'HIGH':
                return 0.7  # 변동성 높으면 포지션 축소
            elif volatility_data['level'] == 'LOW':
                return 1.2  # 변동성 낮으면 포지션 확대
            else:
                return 1.0
        except Exception as e:
            logger.warning(f"⚠️ 변동성 데이터 조회 실패: {e}")
            return 1.0 # 기본값
    
    def get_trading_summary(self, signal: TradingSignal, market_data: MarketData) -> Dict[str, Any]:
        """📋 매매 신호 요약"""
        return {
            "timestamp": datetime.now().isoformat(),
            "stock_code": market_data.stock_code,
            "action": signal.action,
            "confidence": f"{signal.confidence:.1%}",
            "position_size": f"{signal.position_size:.1%}",
            "entry_price": f"{signal.entry_price:,}원",
            "stop_loss": f"{signal.stop_loss:,}원",
            "target_price": f"{signal.target_price:,}원",
            "risk_level": signal.risk_level,
            "time_horizon": signal.time_horizon,
            "reasoning": signal.reasoning,
            "expected_return": f"{((signal.target_price / signal.entry_price) - 1) * 100:.1f}%",
            "max_loss": f"{((signal.stop_loss / signal.entry_price) - 1) * 100:.1f}%"
        }
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 AIAnalyzer 리소스 정리")
        if self.chart_manager:
            self.chart_manager.cleanup()
    
    def get_trailing_stop_price(self, entry_price: float, current_price: float, 
                               high_price: float) -> Tuple[bool, float, str]:
        """🔄 트레일링 스탑 가격 계산"""
        try:
            # 트레일링 활성화 조건 확인
            trailing_activation_price = entry_price * (1 + self.optimized_params['trailing_activation'] / 100)
            
            # 초기 손절가
            initial_stop_loss = entry_price * (1 - self.optimized_params['initial_stop_loss'] / 100)
            
            # 트레일링이 활성화되었는지 확인
            is_trailing_activated = high_price >= trailing_activation_price
            
            if is_trailing_activated:
                # 트레일링 스탑 가격 계산 (최고가 기준)
                trailing_stop_price = high_price * (1 - self.optimized_params['trailing_stop'] / 100)
                
                # 트레일링 스탑이 초기 손절가보다 높아야 함
                trailing_stop_price = max(trailing_stop_price, initial_stop_loss)
                
                # 현재가가 트레일링 스탑에 걸렸는지 확인
                should_sell = current_price <= trailing_stop_price
                
                status = "트레일링 스탑 활성화" if not should_sell else f"트레일링 스탑 매도 ({self.optimized_params['trailing_stop']:.1f}%)"
                
                return should_sell, trailing_stop_price, status
            else:
                # 트레일링 미활성화 - 초기 손절만 확인
                should_sell = current_price <= initial_stop_loss
                status = "초기 손절 대기" if not should_sell else f"초기 손절 매도 ({self.optimized_params['initial_stop_loss']:.1f}%)"
                
                return should_sell, initial_stop_loss, status
                
        except Exception as e:
            logger.error(f"❌ 트레일링 스탑 계산 실패: {e}")
            # 안전한 기본값
            return False, entry_price * 0.96, "계산 오류"
    
    def update_trailing_stop(self, position_info: Dict[str, Any], 
                           current_price: float) -> Dict[str, Any]:
        """📈 포지션의 트레일링 스탑 업데이트"""
        try:
            entry_price = position_info['entry_price']
            high_price = max(position_info.get('high_price', entry_price), current_price)
            
            # 최고가 업데이트
            position_info['high_price'] = high_price
            
            # 트레일링 스탑 상태 계산
            should_sell, stop_price, status = self.get_trailing_stop_price(
                entry_price, current_price, high_price
            )
            
            # 포지션 정보 업데이트
            position_info.update({
                'current_price': current_price,
                'stop_loss': stop_price,
                'should_sell': should_sell,
                'status': status,
                'profit_rate': ((current_price - entry_price) / entry_price) * 100,
                'trailing_activated': high_price >= (entry_price * (1 + self.optimized_params['trailing_activation'] / 100))
            })
            
            if should_sell:
                logger.warning(f"🚨 매도 신호 발생: {status}")
                logger.info(f"📊 진입가: {entry_price:,.0f}원 → 현재가: {current_price:,.0f}원")
                logger.info(f"📊 수익률: {position_info['profit_rate']:+.2f}%")
            elif position_info['trailing_activated']:
                logger.info(f"🔄 트레일링 스탑 업데이트: {stop_price:,.0f}원")
            
            return position_info
            
        except Exception as e:
            logger.error(f"❌ 트레일링 스탑 업데이트 실패: {e}")
            return position_info

    # ===================================================================
    # Advanced AI Trader를 위한 새로운 메서드들 (v3.0 - 동적 분석/전략)
    # ===================================================================

    def _format_data_for_prompt(self, data: Optional[Any], title: str, empty_message: str = "N/A") -> str:
        """프롬프트에 사용될 데이터 포맷을 생성하는 헬퍼 함수"""
        if not data:
            return f"**{title}:**\n- {empty_message}\n"
        
        formatted_string = f"**{title}:**\n"
        
        if isinstance(data, list) and data:
            # 리스트 형태의 데이터 (뉴스, 공시 등)
            for item in data:
                if isinstance(item, dict):
                    details = []
                    # 뉴스 포맷
                    if "title" in item and "source" in item:
                        details.append(f"[{item['source']}] {item['title']}")
                        if "content" in item and item['content']:
                             # 본문은 최대 200자까지만 요약해서 보여줌
                            content_preview = item['content'][:200] + '...' if len(item['content']) > 200 else item['content']
                            details.append(f"  - 본문: {content_preview}")
                    # 공시 포맷
                    elif "report_nm" in item and "rcept_dt" in item:
                        details.append(f"[{item['rcept_dt']}] {item['report_nm']} ({item['flr_nm']})")

                    if details:
                        formatted_string += "- " + "\n".join(details) + "\n"
                else:
                    formatted_string += f"- {str(item)}\n"

        elif isinstance(data, dict):
             # 딕셔너리 형태의 데이터 (수급, 재무 등)
            for key, value in data.items():
                # 숫자인 경우 포맷팅
                if isinstance(value, (int, float)):
                    formatted_string += f"- {key}: {value:,.0f}\n"
                else:
                    formatted_string += f"- {key}: {value}\n"
        else:
            return f"**{title}:**\n- {str(data)}\n"
            
        return formatted_string
        
    async def run_advanced_stock_discovery(self, stock_code: str, stock_name: str, theme: str) -> Optional[Dict]:
        """
        [업그레이드 v3] DART공시, 세부수급, 뉴스본문을 포함한 종합 데이터 기반 심층 종목 분석
        """
        logger.info(f"🔬 고급 AI 분석 시작: [{stock_name}({stock_code})] (테마: {theme})")
        
        try:
            # 1. 모든 종합 데이터를 한 번에 가져오기
            comprehensive_data = await self.data_provider.get_comprehensive_stock_data(stock_code)
            if not comprehensive_data:
                logger.error(f"❌ [{stock_code}] 분석에 필요한 종합 데이터를 가져올 수 없습니다.")
                return None

            # 2. 모든 데이터를 종합하여 AI에게 최종 분석 요청
            final_decision = await self._get_final_decision_from_ai(stock_code, stock_name, theme, comprehensive_data)

            # 3. AI의 최종 결정을 터미널에 상세히 출력
            if final_decision:
                self.console.print(Panel(
                    Syntax(json.dumps(final_decision, indent=4, ensure_ascii=False), "json", theme="monokai", line_numbers=True),
                    title=f"[bold green]🤖 AI 최종 분석 리포트: {stock_name}({stock_code})[/bold green]",
                    subtitle=f"[bold yellow]테마: {theme}[/bold yellow]",
                    border_style="blue"
                ))
            
            return final_decision

        except Exception as e:
            logger.error(f"❌ [{stock_code}] 고급 AI 분석 중 심각한 오류 발생: {e}", exc_info=True)
            return None

    async def _get_final_decision_from_ai(self, stock_code: str, stock_name: str, theme: str, data: Dict) -> Optional[Dict]:
        """AI를 통해 최종 투자 결정을 얻어옵니다."""
        prompt = self._create_advanced_discovery_prompt(stock_code, stock_name, theme, data)
        try:
            # 모델을 flash_model로 명시
            response = await self.flash_model.generate_content_async(prompt)
            
            # 응답에서 JSON 부분만 추출
            json_text = self._extract_json_from_response(response.text)
            if not json_text:
                logger.warning(f"[{stock_code}] AI 응답에서 JSON을 찾을 수 없습니다. 원본 응답: {response.text}")
                return None

            decision = json.loads(json_text)
            return decision
        except Exception as e:
            logger.error(f"❌ [{stock_code}] 최종 AI 결정 생성 중 오류: {e}\n프롬프트: {prompt[:500]}...", exc_info=True)
            return None

    def _create_advanced_discovery_prompt(self, stock_code: str, stock_name: str, theme: str, data: Dict) -> str:
        """
        AI 최종 분석을 위한 프롬프트를 생성합니다.
        모든 데이터를 구조화하여 제공합니다.
        """
        # 시스템 메시지 (역할 및 지침)
        system_message = """
        당신은 대한민국 최고의 애널리스트이자 펀드매니저입니다. 당신의 임무는 주어진 모든 데이터를 종합하여 특정 종목에 대한 깊이 있는 투자 보고서를 작성하는 것입니다. 반드시 최종 결론을 JSON 형식으로 제공해야 합니다.
        """

        # 분석 대상 정보
        company_overview_str = f"""
        - **종목명 (코드)**: {stock_name} ({stock_code})
        - **소속 테마**: {theme}
        """

        # 유저 메시지 (분석 요청)
        user_message = f"""
        위 정보를 바탕으로, 다음 항목들을 분석하고, 최종 결론을 **반드시 JSON 형식으로만** 작성해주세요.

        1.  **Sentiment Analysis (감성 분석)**:
            - 뉴스, 공시, 커뮤니티 반응(가정)을 종합하여 시장의 투자 심리를 '매우 긍정적', '긍정적', '중립', '부정적', '매우 부정적' 중 하나로 평가하고, 그 핵심 근거를 제시해주세요.

        2.  **Key Factors Analysis (핵심 동인 분석)**:
            - **상승 요인 (Bull Case)**: 현재 주가에 긍정적인 핵심 요인 2~3가지를 구체적인 데이터에 기반하여 서술해주세요. (예: 연기금의 연속 순매수, 2분기 실적 컨센서스 상회 등)
            - **하락 요인 (Bear Case)**: 주가에 부정적인 리스크 요인 2~3가지를 구체적인 데이터에 기반하여 서술해주세요. (예: 단기 이평선 이탈, 주력 제품 수요 둔화 뉴스 등)

        3.  **Executive Summary (투자 결정 요약)**:
            - 위 모든 분석을 종합하여, 이 종목에 대한 당신의 최종 투자 의견을 한 문장으로 명확하게 요약해주세요. (예: "2차전지 테마 강세와 연기금 수급을 바탕으로 단기 상승 모멘텀이 유효하다고 판단됨.")

        4.  **Actionable Advice (실행 가능한 조언)**:
            - **Investment Score (투자 매력도 점수)**: 0점에서 100점 사이의 종합 점수를 부여해주세요. (높을수록 매력적)
            - **Optimal Entry Timing (최적 진입 시점)**: '즉시 매수', '눌림목 매수', '돌파 매수', '관망' 중 가장 적절한 전략을 선택해주세요.
            - **Recommended Allocation (추천 투자 비중)**: 전체 포트폴리오 대비 이 종목에 할당할 비중을 퍼센트(%)로 제시해주세요. (예: 5.5)
            - **Primary Stop-Loss Price (1차 손절 가격)**: 현재가 기준 합리적인 손절 가격을 구체적인 원화(KRW) 금액으로 제시해주세요.
        """

        # 최종 프롬프트 조합
        return f"{system_message}\n\n## 분석 대상 정보\n{company_overview_str}\n\n{user_message}"

    async def get_adaptive_strategy_adjustment(self) -> Optional[Dict]:
        """
        현재 시장 상황을 기반으로 AI에게 동적 전략 조정을 요청합니다.
        (예: 현금 비중, 선호 업종, 리스크 관리 수준 등)
        """
        try:
            market_condition = await self.data_provider.get_market_regime()
            
            prompt = f"""
            당신은 매크로 전략 분석가입니다. 현재 시장 상황을 보고 단기 트레이딩 전략을 어떻게 조정해야 할지 조언해주세요.

            **현재 시장 상황:**
            - 시장 구분: {market_condition.get('market', 'N/A')}
            - 상태: {market_condition.get('status', 'N/A')}
            - 설명: {market_condition.get('description', 'N/A')}
            - 주요 지수 변동률: {market_condition.get('change_rate', 0.0):.2f}%

            **[MISSION]**
            위 상황을 고려하여, 아래 항목들에 대한 구체적인 조언을 JSON 형식으로만 답변해주세요.
            - `cash_ratio_adjustment`: 현금 비중 조정 (+10%, -5%, 0% 등)
            - `preferred_sectors`: 현재 가장 유망해 보이는 섹터 (리스트)
            - `risk_management_level`: 리스크 관리 수준 ('강화', '유지', '완화')

            **응답 형식 (JSON):**
            {{
              "cash_ratio_adjustment": "<증감 퍼센트>",
              "preferred_sectors": ["<섹터1>", "<섹터2>"],
              "risk_management_level": "<'강화'|'유지'|'완화'>"
            }}
            """
            
            # Flash 모델 사용
            response = await self.flash_model.generate_content_async(prompt)
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            adjustment = json.loads(cleaned_response)
            
            logger.info(f"💡 AI 동적 전략 조정: {adjustment}")
            return adjustment

        except Exception as e:
            logger.error(f"❌ AI 동적 전략 조정 실패: {e}", exc_info=True)
            return None 