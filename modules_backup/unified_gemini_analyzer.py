"""
🤖 통합 Gemini AI 분석기 (Unified Gemini Analyzer)
====================================================

실제 Gemini AI와 완전 연동하여 전 세계 최고 애널리스트 수준의 
주식 투자 분석을 제공하는 고품질 시스템입니다.

주요 기능:
1. 실제 Gemini AI API 연동 (gemini-1.5-pro 모델 사용)
2. 투자 대가 17개 전략 종합 분석 (워런 버핏, 벤저민 그레이엄 등)
3. 전 세계 최고 애널리스트 수준의 분석 (Goldman Sachs, JP Morgan 수준)
4. 고도화된 프롬프트 엔지니어링 및 structured 응답
5. 강력한 오류 처리 및 재시도 로직
6. 캐싱 시스템으로 성능 최적화
7. 중복 코드 제거 및 통합 최적화

투자 대가별 가중치:
- 워런 버핏: 15% (가치투자의 대가)
- 벤저민 그레이엄: 12% (가치투자 창시자)
- 피터 린치: 10% (성장주 투자)
- 기타 14개 전략: 각 4-8%
"""

import asyncio
import os
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import google.generativeai as genai
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경 변수에서 API 키 로드
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


@dataclass
class InvestmentAnalysis:
    """투자 분석 결과 구조체"""
    symbol: str
    company_name: str
    overall_score: float  # 종합 점수 (0-100)
    investment_action: str  # 'BUY', 'HOLD', 'SELL'
    target_price: Optional[float]  # 목표가
    expected_return: float  # 기대수익률 (%)
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    confidence_level: float  # 신뢰도 (0-100)
    
    # 상세 분석
    competitive_advantage: str  # 경쟁우위 분석
    financial_health: str  # 재무건전성 평가
    growth_potential: str  # 성장잠재력
    valuation_analysis: str  # 밸류에이션 분석
    market_position: str  # 시장지위
    management_quality: str  # 경영진 품질
    esg_factors: str  # ESG 요소
    sector_comparison: str  # 섹터 비교분석
    
    # 투자 대가별 전략 점수
    strategy_scores: Dict[str, float]  # 각 전략별 점수
    
    # 기술적 분석
    technical_indicators: Dict[str, Any]
    chart_pattern: str
    momentum_score: float
    
    # AI 분석 근거
    analysis_reasoning: str
    key_catalysts: List[str]
    major_risks: List[str]
    
    # 메타 정보
    analysis_timestamp: str
    gemini_model_version: str


@dataclass
class MarketInsight:
    """시장 통찰력 구조체"""
    market_sentiment: str  # 시장 센티먼트
    key_trends: List[str]  # 핵심 트렌드
    risk_factors: List[str]  # 리스크 요인
    investment_opportunities: List[str]  # 투자 기회
    market_outlook: str  # 시장 전망
    recommended_sectors: List[str]  # 추천 섹터
    macro_environment: str  # 거시환경
    sector_rotation: str  # 섹터 로테이션
    
    # 메타 정보
    insight_timestamp: str
    confidence_level: float


class IAIAnalyzer(ABC):
    """AI 분석기 인터페이스"""
    
    @abstractmethod
    async def analyze_stock(self, stock_data, market_context: Dict[str, Any]) -> Optional[InvestmentAnalysis]:
        """개별 종목 분석"""
        pass
        
    @abstractmethod
    async def generate_market_insight(self, market_data: Dict) -> MarketInsight:
        """시장 통찰력 생성"""
        pass
        
    @abstractmethod
    async def select_top_stocks(self, analyses: List[InvestmentAnalysis], count: int = 5) -> List[InvestmentAnalysis]:
        """상위 종목 선정"""
        pass


class UnifiedGeminiAnalyzer(IAIAnalyzer):
    """
    통합 Gemini AI 분석기
    
    전 세계 최고 애널리스트 수준의 분석을 제공하는 완전한 시스템
    중복 코드를 제거하고 최적화된 단일 클래스로 통합
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro"):
        """초기화"""
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.model = None
        
        # 캐싱
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 1800  # 30분 캐시
        
        # 통계
        self.analysis_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # 투자 대가별 전략 가중치
        self.strategy_weights = {
            'warren_buffett': 0.15,  # 15% - 가치투자의 대가
            'benjamin_graham': 0.12,  # 12% - 가치투자 창시자
            'peter_lynch': 0.10,  # 10% - 성장주 투자
            'philip_fisher': 0.08,  # 8% - 성장주 분석
            'john_templeton': 0.07,  # 7% - 글로벌 가치투자
            'george_soros': 0.06,  # 6% - 반사성 이론
            'jesse_livermore': 0.05,  # 5% - 추세 매매
            'bill_ackman': 0.05,  # 5% - 액티비스트
            'carl_icahn': 0.05,  # 5% - 액티비스트
            'ray_dalio': 0.05,  # 5% - 전천후 포트폴리오
            'stanley_druckenmiller': 0.04,  # 4% - 거시경제 분석
            'david_tepper': 0.04,  # 4% - 디스트레스드 투자
            'seth_klarman': 0.04,  # 4% - 절대수익 추구
            'howard_marks': 0.03,  # 3% - 리스크 관리
            'joel_greenblatt': 0.03,  # 3% - 마법공식
            'thomas_rowe_price': 0.02,  # 2% - 성장주 투자
            'john_bogle': 0.02   # 2% - 인덱스 투자
        }
        
        self._initialize_gemini()
        logger.info("🤖 통합 Gemini AI 분석기 초기화 완료")
    
    def _initialize_gemini(self):
        """Gemini AI 초기화"""
        try:
            if not self.api_key:
                logger.warning("⚠️ Gemini API 키가 설정되지 않았습니다. 환경변수 GEMINI_API_KEY를 설정하세요.")
                return
            
            # API 키 설정
            genai.configure(api_key=self.api_key)
            
            # 모델 초기화 - 투자 분석에 최적화
            generation_config = {
                "temperature": 0.2,  # 일관성 있는 분석을 위해 낮은 temperature
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"✅ Gemini AI ({self.model_name}) 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ Gemini AI 초기화 실패: {e}")
            self.model = None
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            del self.cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        """캐시에 데이터 저장"""
        self.cache[key] = (data, time.time())
    
    async def analyze_stock(self, stock_data, market_context: Dict[str, Any]) -> Optional[InvestmentAnalysis]:
        """개별 종목 분석"""
        if not self.model:
            logger.error("Gemini AI 모델이 초기화되지 않았습니다.")
            return None
        
        self.analysis_count += 1
        symbol = getattr(stock_data, 'symbol', 'Unknown')
        
        # 캐시 확인
        cache_key = f"stock_analysis_{symbol}_{hash(str(stock_data))}"
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info(f"📋 {symbol} 캐시된 분석 결과 반환")
            return cached_result
        
        try:
            logger.info(f"🔍 {symbol} 종목 분석 시작...")
            
            # 분석 프롬프트 생성
            prompt = self._create_analysis_prompt(stock_data, market_context)
            
            # Gemini AI 호출
            response = await self._call_gemini_async(prompt)
            if not response:
                logger.error(f"❌ {symbol} Gemini AI 응답 없음")
                return None
            
            # 응답 파싱
            analysis = self._parse_analysis_response(response, stock_data)
            if analysis:
                self._set_cache(cache_key, analysis)
                self.success_count += 1
                logger.info(f"✅ {symbol} 분석 완료 (점수: {analysis.overall_score:.1f})")
                return analysis
            else:
                logger.error(f"❌ {symbol} 응답 파싱 실패")
                return None
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ {symbol} 분석 실패: {e}")
            return None
    
    def _create_analysis_prompt(self, stock_data, market_context: Dict[str, Any]) -> str:
        """분석 프롬프트 생성"""
        symbol = getattr(stock_data, 'symbol', 'Unknown')
        company_name = getattr(stock_data, 'name', 'Unknown Company')
        current_price = getattr(stock_data, 'current_price', 0)
        change_percent = getattr(stock_data, 'change_percent', 0)
        
        prompt = f"""
당신은 Goldman Sachs, JP Morgan 수준의 전 세계 최고 주식 애널리스트입니다.
다음 종목에 대해 투자 대가 17명의 전략을 종합하여 심층 분석해주세요.

## 분석 대상 종목
- 종목코드: {symbol}
- 회사명: {company_name}
- 현재가: {current_price:,.2f}
- 등락률: {change_percent:+.2f}%

## 투자 대가별 전략 분석 (각각 0-100점으로 점수화)
1. 워런 버핏 (15%): 장기 가치투자 관점
2. 벤저민 그레이엄 (12%): 안전마진과 내재가치
3. 피터 린치 (10%): 성장주 투자와 PEG 비율
4. 필립 피셔 (8%): 성장주 질적 분석
5. 존 템플턴 (7%): 글로벌 가치투자
6. 조지 소로스 (6%): 반사성 이론과 거시경제
7. 제시 리버모어 (5%): 추세 매매와 기술적 분석
8. 빌 애크먼 (5%): 액티비스트 투자
9. 칼 아이칸 (5%): 기업 구조조정 가치
10. 레이 달리오 (5%): 전천후 포트폴리오 적합성
11. 스탠리 드러켄밀러 (4%): 거시경제 투자
12. 데이비드 테퍼 (4%): 디스트레스드 투자
13. 세스 클라만 (4%): 절대수익 추구
14. 하워드 막스 (3%): 리스크 조정 수익
15. 조엘 그린블랫 (3%): 마법공식
16. 토마스 로우 프라이스 (2%): 성장주 투자
17. 존 보글 (2%): 인덱스 투자 철학

## 요구사항
다음 JSON 형식으로 정확히 응답해주세요:

{{
    "symbol": "{symbol}",
    "company_name": "{company_name}",
    "overall_score": 85.5,
    "investment_action": "BUY/HOLD/SELL",
    "target_price": 목표가,
    "expected_return": 기대수익률(%),
    "risk_level": "LOW/MEDIUM/HIGH",
    "confidence_level": 95.0,
    "competitive_advantage": "경쟁우위 상세 분석",
    "financial_health": "재무건전성 평가",
    "growth_potential": "성장잠재력 분석",
    "valuation_analysis": "밸류에이션 분석",
    "market_position": "시장지위 분석",
    "management_quality": "경영진 품질 평가",
    "esg_factors": "ESG 요소 분석",
    "sector_comparison": "섹터 비교분석",
    "strategy_scores": {{
        "warren_buffett": 88.0,
        "benjamin_graham": 75.0,
        "peter_lynch": 92.0,
        ...각 전략별 점수
    }},
    "technical_indicators": {{
        "rsi": 45.2,
        "moving_averages": "상승 추세",
        "volume_analysis": "거래량 분석"
    }},
    "chart_pattern": "차트 패턴 분석",
    "momentum_score": 78.5,
    "analysis_reasoning": "종합 분석 근거",
    "key_catalysts": ["주요 상승 요인1", "상승 요인2"],
    "major_risks": ["주요 리스크1", "리스크2"]
}}

## 주의사항
- 모든 점수는 0-100 범위
- 투자 대가별 전략 특성을 정확히 반영
- 실제 시장 데이터와 기업 현황을 고려
- 객관적이고 전문적인 분석 제공
- JSON 형식을 정확히 준수
"""
        
        return prompt
    
    async def _call_gemini_async(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Gemini AI 비동기 호출"""
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
                
                if response and response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"⚠️ Gemini 응답 없음 (시도 {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                logger.error(f"❌ Gemini 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프
        
        return None
    
    def _parse_analysis_response(self, response_text: str, stock_data) -> Optional[InvestmentAnalysis]:
        """분석 응답 파싱"""
        try:
            # JSON 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error("JSON 형식을 찾을 수 없습니다.")
                return None
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            # InvestmentAnalysis 객체 생성
            analysis = InvestmentAnalysis(
                symbol=data.get('symbol', getattr(stock_data, 'symbol', 'Unknown')),
                company_name=data.get('company_name', getattr(stock_data, 'name', 'Unknown')),
                overall_score=float(data.get('overall_score', 50.0)),
                investment_action=data.get('investment_action', 'HOLD'),
                target_price=data.get('target_price'),
                expected_return=float(data.get('expected_return', 0.0)),
                risk_level=data.get('risk_level', 'MEDIUM'),
                confidence_level=float(data.get('confidence_level', 70.0)),
                competitive_advantage=data.get('competitive_advantage', ''),
                financial_health=data.get('financial_health', ''),
                growth_potential=data.get('growth_potential', ''),
                valuation_analysis=data.get('valuation_analysis', ''),
                market_position=data.get('market_position', ''),
                management_quality=data.get('management_quality', ''),
                esg_factors=data.get('esg_factors', ''),
                sector_comparison=data.get('sector_comparison', ''),
                strategy_scores=data.get('strategy_scores', {}),
                technical_indicators=data.get('technical_indicators', {}),
                chart_pattern=data.get('chart_pattern', ''),
                momentum_score=float(data.get('momentum_score', 50.0)),
                analysis_reasoning=data.get('analysis_reasoning', ''),
                key_catalysts=data.get('key_catalysts', []),
                major_risks=data.get('major_risks', []),
                analysis_timestamp=datetime.now().isoformat(),
                gemini_model_version=self.model_name
            )
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return None
    
    async def generate_market_insight(self, market_data: Dict) -> MarketInsight:
        """시장 통찰력 생성"""
        if not self.model:
            logger.error("Gemini AI 모델이 초기화되지 않았습니다.")
            return self._create_fallback_insight()
        
        try:
            logger.info("🌍 시장 통찰력 생성 시작...")
            
            # 시장 분석 프롬프트 생성
            prompt = self._create_market_insight_prompt(market_data)
            
            # Gemini AI 호출
            response = await self._call_gemini_async(prompt)
            if not response:
                logger.error("시장 통찰력 생성 실패")
                return self._create_fallback_insight()
            
            # 응답 파싱
            insight = self._parse_insight_response(response)
            if insight:
                logger.info("✅ 시장 통찰력 생성 완료")
                return insight
            else:
                logger.error("시장 통찰력 파싱 실패")
                return self._create_fallback_insight()
                
        except Exception as e:
            logger.error(f"시장 통찰력 생성 실패: {e}")
            return self._create_fallback_insight()
    
    def _create_market_insight_prompt(self, market_data: Dict) -> str:
        """시장 통찰력 프롬프트 생성"""
        market_summary = self._create_market_summary(market_data)
        
        prompt = f"""
당신은 전 세계 최고 수준의 거시경제 및 시장 전략 애널리스트입니다.
다음 시장 데이터를 분석하여 투자 통찰력을 제공해주세요.

## 시장 현황
{market_summary}

## 요구사항
다음 JSON 형식으로 정확히 응답해주세요:

{{
    "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
    "key_trends": ["트렌드1", "트렌드2", "트렌드3"],
    "risk_factors": ["리스크1", "리스크2", "리스크3"],
    "investment_opportunities": ["기회1", "기회2", "기회3"],
    "market_outlook": "시장 전망 상세 분석",
    "recommended_sectors": ["섹터1", "섹터2", "섹터3"],
    "macro_environment": "거시경제 환경 분석",
    "sector_rotation": "섹터 로테이션 분석",
    "confidence_level": 85.0
}}

## 분석 관점
- 거시경제 지표와 시장 동향
- 지정학적 리스크와 기회
- 통화정책과 금리 환경
- 섹터별 투자 매력도
- 기술적 분석과 시장 심리
"""
        
        return prompt
    
    def _create_market_summary(self, market_data: Dict) -> str:
        """시장 데이터 요약"""
        summary_parts = []
        
        for market_name, stocks in market_data.items():
            if stocks:
                avg_change = sum(getattr(stock, 'change_percent', 0) for stock in stocks) / len(stocks)
                summary_parts.append(f"- {market_name}: {len(stocks)}개 종목, 평균 등락률 {avg_change:+.2f}%")
        
        return "\n".join(summary_parts) if summary_parts else "시장 데이터 없음"
    
    def _parse_insight_response(self, response_text: str) -> Optional[MarketInsight]:
        """시장 통찰력 응답 파싱"""
        try:
            # JSON 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            return MarketInsight(
                market_sentiment=data.get('market_sentiment', 'NEUTRAL'),
                key_trends=data.get('key_trends', []),
                risk_factors=data.get('risk_factors', []),
                investment_opportunities=data.get('investment_opportunities', []),
                market_outlook=data.get('market_outlook', ''),
                recommended_sectors=data.get('recommended_sectors', []),
                macro_environment=data.get('macro_environment', ''),
                sector_rotation=data.get('sector_rotation', ''),
                insight_timestamp=datetime.now().isoformat(),
                confidence_level=float(data.get('confidence_level', 70.0))
            )
            
        except Exception as e:
            logger.error(f"시장 통찰력 파싱 실패: {e}")
            return None
    
    def _create_fallback_insight(self) -> MarketInsight:
        """기본 시장 통찰력 생성"""
        return MarketInsight(
            market_sentiment="NEUTRAL",
            key_trends=["시장 데이터 부족으로 분석 제한"],
            risk_factors=["데이터 부족"],
            investment_opportunities=["추가 분석 필요"],
            market_outlook="시장 데이터가 부족하여 상세 분석이 제한됩니다.",
            recommended_sectors=[],
            macro_environment="분석 불가",
            sector_rotation="분석 불가",
            insight_timestamp=datetime.now().isoformat(),
            confidence_level=30.0
        )
    
    async def select_top_stocks(self, analyses: List[InvestmentAnalysis], count: int = 5) -> List[InvestmentAnalysis]:
        """상위 종목 선정"""
        if not analyses:
            return []
        
        # 점수순으로 정렬
        sorted_analyses = sorted(analyses, key=lambda x: x.overall_score, reverse=True)
        
        # 상위 종목 선정
        top_candidates = sorted_analyses[:min(count * 2, len(sorted_analyses))]
        
        # Gemini AI로 최종 검증
        final_selection = await self._final_validation(top_candidates, count)
        
        return final_selection or top_candidates[:count]
    
    async def _final_validation(self, candidates: List[InvestmentAnalysis], count: int) -> Optional[List[InvestmentAnalysis]]:
        """최종 검증 및 선정"""
        if not self.model or not candidates:
            return None
        
        try:
            # 후보 종목 정보 생성
            candidates_info = []
            for analysis in candidates:
                candidates_info.append({
                    'symbol': analysis.symbol,
                    'company_name': analysis.company_name,
                    'score': analysis.overall_score,
                    'action': analysis.investment_action,
                    'reasoning': analysis.analysis_reasoning[:200]  # 요약
                })
            
            prompt = f"""
다음 {len(candidates)}개 후보 종목 중에서 최종 Top {count}개를 선정해주세요.

## 후보 종목들
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}

## 선정 기준
1. 종합 점수 (가장 중요)
2. 투자 액션 (BUY > HOLD > SELL)
3. 리스크 대비 수익률
4. 포트폴리오 다각화
5. 시장 환경 적합성

다음 형식으로 응답해주세요:
{{
    "selected_symbols": ["AAPL", "MSFT", "GOOGL", ...],
    "selection_reasoning": "선정 근거"
}}
"""
            
            response = await self._call_gemini_async(prompt)
            if response:
                data = json.loads(response)
                selected_symbols = data.get('selected_symbols', [])
                
                # 선정된 종목들 반환
                selected_analyses = []
                for symbol in selected_symbols[:count]:
                    for analysis in candidates:
                        if analysis.symbol == symbol:
                            selected_analyses.append(analysis)
                            break
                
                return selected_analyses
                
        except Exception as e:
            logger.error(f"최종 검증 실패: {e}")
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """분석 통계 반환"""
        success_rate = (self.success_count / self.analysis_count * 100) if self.analysis_count > 0 else 0
        
        return {
            'total_analyses': self.analysis_count,
            'successful_analyses': self.success_count,
            'failed_analyses': self.error_count,
            'success_rate': f"{success_rate:.1f}%",
            'cache_size': len(self.cache),
            'model_name': self.model_name
        }


def get_unified_gemini_analyzer(api_key: Optional[str] = None) -> UnifiedGeminiAnalyzer:
    """통합 Gemini 분석기 인스턴스 반환"""
    return UnifiedGeminiAnalyzer(api_key)


if __name__ == "__main__":
    # 테스트 코드
    async def main():
        analyzer = get_unified_gemini_analyzer()
        
        class DummyStock:
            def __init__(self, symbol, name, price, change, volume):
                self.symbol = symbol
                self.name = name
                self.current_price = price
                self.change_percent = change
                self.volume = volume
        
        # 테스트 종목
        test_stock = DummyStock("AAPL", "Apple Inc", 150.0, 1.2, 1000000)
        
        print("🤖 통합 Gemini 분석기 테스트")
        print("=" * 50)
        
        # 종목 분석 테스트
        analysis = await analyzer.analyze_stock(test_stock, {})
        if analysis:
            print(f"✅ 분석 완료: {analysis.symbol}")
            print(f"   종합 점수: {analysis.overall_score:.1f}")
            print(f"   투자 액션: {analysis.investment_action}")
        else:
            print("❌ 분석 실패")
        
        # 통계 출력
        stats = analyzer.get_statistics()
        print(f"\n📊 분석 통계: {stats}")
    
    asyncio.run(main()) 