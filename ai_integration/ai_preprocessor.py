#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 AI 데이터 전처리기
투자 분석을 위한 데이터 전처리 및 프롬프트 생성 모듈
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """주식 데이터 구조"""
    symbol: str
    name: str
    price: float
    market_cap: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    roe: Optional[float]
    roa: Optional[float]
    debt_ratio: Optional[float]
    current_ratio: Optional[float]
    dividend_yield: Optional[float]
    revenue_growth: Optional[float]
    profit_growth: Optional[float]
    sector: Optional[str]
    industry: Optional[str]

class AIDataPreprocessor:
    """AI 분석을 위한 데이터 전처리기 - 100% 성능 최적화"""
    
    def __init__(self):
        self.financial_metrics = [
            'pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_ratio', 
            'current_ratio', 'dividend_yield', 'revenue_growth', 'profit_growth'
        ]
        
        # 100% 최적화 설정
        self.ultra_prompt_cache = {}
        self.advanced_templates = self._init_advanced_templates()
        
    def _init_advanced_templates(self) -> Dict[str, str]:
        """고급 프롬프트 템플릿 초기화"""
        return {
            "world_class_analyst": """
당신은 워렌 버핏, 피터 린치, 벤저민 그레이엄, 레이 달리오의 투자 철학을 완벽히 체득한 세계 최고 수준의 AI 투자 분석가입니다.
다음 주식에 대해 가치투자, 성장투자, 퀀트 분석을 종합하여 최고 수준의 분석을 제공하세요.

🎯 분석 미션: 세계 최고 수준의 투자 분석 수행
🧠 분석 방법론:
1. 워렌 버핏의 내재가치 평가 (Intrinsic Value Analysis)
2. 피터 린치의 성장주 발굴 기법 (Growth Stock Identification)
3. 벤저민 그레이엄의 안전마진 원칙 (Margin of Safety)
4. 레이 달리오의 리스크 패리티 (Risk Parity)
5. AI 빅데이터 패턴 분석 (Big Data Pattern Analysis)

🔍 핵심 평가 요소:
- 경제적 해자 (Economic Moat) 분석
- 경영진 품질 및 지배구조
- 산업 생명주기 및 경쟁 환경
- 거시경제 변수 영향도
- ESG 요소 통합 평가
""",
            "ultra_comprehensive": """
🚀 ULTRA AI 종합 분석 프레임워크

📊 정량 분석 (Quantitative Analysis):
- 밸류에이션: PER, PBR, EV/EBITDA, PEG
- 수익성: ROE, ROA, ROIC, 영업이익률
- 안전성: 부채비율, 유동비율, 이자보상배수
- 성장성: 매출/이익 성장률, 시장점유율 변화
- 효율성: 자산회전율, 재고회전율

🔍 정성 분석 (Qualitative Analysis):
- 비즈니스 모델의 지속가능성
- 경쟁우위 및 진입장벽
- 경영진의 비전과 실행력
- 기술혁신 및 R&D 역량
- 브랜드 가치 및 고객 충성도

🌍 거시 환경 분석:
- 글로벌 경제 트렌드
- 산업별 성장 전망
- 정책 변화 영향
- 환율 및 금리 민감도
""",
            "risk_assessment": """
⚠️ 리스크 평가 전문 프레임워크

🎯 리스크 카테고리 분석:
1. 시장 리스크 (Market Risk)
   - 주가 변동성, 베타 계수
   - 시장 상관관계 분석

2. 신용 리스크 (Credit Risk)
   - 재무건전성, 부도 확률
   - 신용등급 변화 가능성

3. 유동성 리스크 (Liquidity Risk)
   - 거래량, 매매 스프레드
   - 시장 충격 시 매도 가능성

4. 운영 리스크 (Operational Risk)
   - 경영진 변화, 지배구조
   - 규제 변화, 법적 이슈

5. 전략적 리스크 (Strategic Risk)
   - 사업 모델 변화 필요성
   - 경쟁 환경 악화 가능성
"""
        }
    
    def prepare_ultra_stock_analysis_prompt(self, stock_data: StockData, 
                                          strategy: str = "world_class") -> str:
        """울트라 주식 분석용 프롬프트 생성 - 100% 최적화"""
        
        # 캐시 확인
        cache_key = f"{stock_data.symbol}_{strategy}_{datetime.now().date()}"
        if cache_key in self.ultra_prompt_cache:
            return self.ultra_prompt_cache[cache_key]
        
        # 기본 정보 정리 - 고도화
        basic_info = f"""
🏢 기업 정보:
- 종목명: {stock_data.name} ({stock_data.symbol})
- 현재가: {stock_data.price:,.0f}원
- 시가총액: {stock_data.market_cap:,.0f}원 ({self._format_market_cap(stock_data.market_cap)})
- 섹터: {stock_data.sector or '미분류'}
- 업종: {stock_data.industry or '미분류'}
- 분석일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 재무 지표 정리 - 고도화
        financial_info = self._format_advanced_financial_metrics(stock_data)
        
        # 전략별 프롬프트 - 고도화
        strategy_template = self.advanced_templates.get(strategy, self.advanced_templates["world_class_analyst"])
        
        # 투자 등급 시스템 정의
        grading_system = """
🏆 투자 등급 시스템:
- SSS: 초월적 투자 기회 (95-100점) - 워렌 버핏도 인정할 종목
- SS+: 최고급 투자 대상 (90-94점) - 포트폴리오 핵심 종목
- SS: 우수한 투자 기회 (85-89점) - 적극 매수 권장
- S+: 양호한 투자 대상 (80-84점) - 매수 고려
- S: 보통 수준 (70-79점) - 신중한 접근
- A+: 평균 이하 (60-69점) - 보유 또는 관망
- A: 부진한 상태 (50-59점) - 매도 검토
- B: 위험 수준 (40-49점) - 매도 권장
- C: 고위험 (30-39점) - 즉시 매도
- D: 투자 부적격 (0-29점) - 투자 금지
"""
        
        # JSON 응답 형식 정의
        json_format = """
{
    "ultra_grade": "SSS/SS+/SS/S+/S/A+/A/B/C/D",
    "investment_score": 0-100,
    "target_price": 숫자,
    "upside_potential": "퍼센트",
    "investment_opinion": "초강력매수/강력매수/매수/보유/매도/강력매도",
    "time_horizon": "단기(3개월)/중기(1년)/장기(3년+)",
    "position_size": "권장 포트폴리오 비중(%)",
    "strengths": ["핵심강점1", "핵심강점2", "핵심강점3"],
    "weaknesses": ["주요약점1", "주요약점2"],
    "risk_factors": ["리스크1", "리스크2", "리스크3"],
    "catalysts": ["상승촉매1", "상승촉매2"],
    "investment_strategy": "맞춤형 투자전략",
    "entry_timing": "진입 시점 분석",
    "exit_strategy": "청산 전략",
    "risk_management": "리스크 관리 방안",
    "buffett_perspective": "워렌 버핏 관점 분석",
    "lynch_perspective": "피터 린치 관점 분석",
    "graham_perspective": "벤저민 그레이엄 관점 분석",
    "ai_unique_insight": "AI만의 독특한 통찰",
    "confidence_level": 1-10,
    "market_conditions_impact": "시장 상황 영향도",
    "sector_outlook": "섹터 전망",
    "final_recommendation": "최종 투자 권고사항"
}
"""
        
        # 최종 프롬프트 조합
        ultra_prompt = f"""
{strategy_template}

{basic_info}

{financial_info}

{grading_system}

🎯 분석 요청:
위 정보를 바탕으로 해당 종목에 대한 세계 최고 수준의 투자 분석을 수행해주세요.
반드시 다음 JSON 형식으로 응답하세요:

{json_format}

🧠 분석 시 고려사항:
- 정량적 지표와 정성적 요소를 균형있게 평가
- 현재 시장 상황과 업계 트렌드 반영
- 투자자 유형별 적합성 고려
- 리스크 대비 수익률 분석
- 글로벌 경제 환경 영향 고려
- ESG 요소 통합 평가
- 기술적 분석과 펀더멘털 분석 결합
"""
        
        # 캐시 저장
        self.ultra_prompt_cache[cache_key] = ultra_prompt
        
        return ultra_prompt.strip()
    
    def _format_advanced_financial_metrics(self, stock_data: StockData) -> str:
        """고급 재무 지표 포맷팅"""
        
        metrics_info = """📊 핵심 재무 지표:

💰 밸류에이션 지표:"""
        
        # PER 분석
        if stock_data.pe_ratio:
            per_analysis = self._analyze_per(stock_data.pe_ratio)
            metrics_info += f"\n- PER: {stock_data.pe_ratio:.2f}배 ({per_analysis})"
        else:
            metrics_info += "\n- PER: N/A"
            
        # PBR 분석
        if stock_data.pb_ratio:
            pbr_analysis = self._analyze_pbr(stock_data.pb_ratio)
            metrics_info += f"\n- PBR: {stock_data.pb_ratio:.2f}배 ({pbr_analysis})"
        else:
            metrics_info += "\n- PBR: N/A"
        
        metrics_info += "\n\n🎯 수익성 지표:"
        
        # ROE 분석
        if stock_data.roe:
            roe_analysis = self._analyze_roe(stock_data.roe)
            metrics_info += f"\n- ROE: {stock_data.roe:.1f}% ({roe_analysis})"
        else:
            metrics_info += "\n- ROE: N/A"
            
        # ROA 분석
        if stock_data.roa:
            roa_analysis = self._analyze_roa(stock_data.roa)
            metrics_info += f"\n- ROA: {stock_data.roa:.1f}% ({roa_analysis})"
        else:
            metrics_info += "\n- ROA: N/A"
        
        metrics_info += "\n\n🛡️ 안전성 지표:"
        
        # 부채비율 분석
        if stock_data.debt_ratio:
            debt_analysis = self._analyze_debt_ratio(stock_data.debt_ratio)
            metrics_info += f"\n- 부채비율: {stock_data.debt_ratio:.1f}% ({debt_analysis})"
        else:
            metrics_info += "\n- 부채비율: N/A"
            
        # 유동비율 분석
        if stock_data.current_ratio:
            current_analysis = self._analyze_current_ratio(stock_data.current_ratio)
            metrics_info += f"\n- 유동비율: {stock_data.current_ratio:.1f}% ({current_analysis})"
        else:
            metrics_info += "\n- 유동비율: N/A"
        
        metrics_info += "\n\n📈 성장성 지표:"
        
        # 매출 성장률
        if stock_data.revenue_growth:
            revenue_analysis = self._analyze_growth_rate(stock_data.revenue_growth, "매출")
            metrics_info += f"\n- 매출 성장률: {stock_data.revenue_growth:.1f}% ({revenue_analysis})"
        else:
            metrics_info += "\n- 매출 성장률: N/A"
            
        # 순이익 성장률
        if stock_data.profit_growth:
            profit_analysis = self._analyze_growth_rate(stock_data.profit_growth, "순이익")
            metrics_info += f"\n- 순이익 성장률: {stock_data.profit_growth:.1f}% ({profit_analysis})"
        else:
            metrics_info += "\n- 순이익 성장률: N/A"
            
        # 배당 정보
        if stock_data.dividend_yield:
            dividend_analysis = self._analyze_dividend_yield(stock_data.dividend_yield)
            metrics_info += f"\n\n💎 배당 정보:\n- 배당수익률: {stock_data.dividend_yield:.2f}% ({dividend_analysis})"
        
        return metrics_info
    
    def _analyze_per(self, per: float) -> str:
        """PER 분석"""
        if per < 10:
            return "저평가"
        elif per < 15:
            return "적정"
        elif per < 25:
            return "약간 고평가"
        else:
            return "고평가"
    
    def _analyze_pbr(self, pbr: float) -> str:
        """PBR 분석"""
        if pbr < 1:
            return "저평가"
        elif pbr < 2:
            return "적정"
        elif pbr < 3:
            return "약간 고평가"
        else:
            return "고평가"
    
    def _analyze_roe(self, roe: float) -> str:
        """ROE 분석"""
        if roe > 20:
            return "매우 우수"
        elif roe > 15:
            return "우수"
        elif roe > 10:
            return "양호"
        elif roe > 5:
            return "보통"
        else:
            return "부진"
    
    def _analyze_roa(self, roa: float) -> str:
        """ROA 분석"""
        if roa > 10:
            return "매우 우수"
        elif roa > 7:
            return "우수"
        elif roa > 5:
            return "양호"
        elif roa > 3:
            return "보통"
        else:
            return "부진"
    
    def _analyze_debt_ratio(self, debt_ratio: float) -> str:
        """부채비율 분석"""
        if debt_ratio < 30:
            return "매우 안전"
        elif debt_ratio < 50:
            return "안전"
        elif debt_ratio < 100:
            return "보통"
        elif debt_ratio < 200:
            return "주의"
        else:
            return "위험"
    
    def _analyze_current_ratio(self, current_ratio: float) -> str:
        """유동비율 분석"""
        if current_ratio > 200:
            return "매우 안전"
        elif current_ratio > 150:
            return "안전"
        elif current_ratio > 100:
            return "보통"
        else:
            return "위험"
    
    def _analyze_growth_rate(self, growth_rate: float, metric_type: str) -> str:
        """성장률 분석"""
        if growth_rate > 20:
            return "고성장"
        elif growth_rate > 10:
            return "양호한 성장"
        elif growth_rate > 5:
            return "보통 성장"
        elif growth_rate > 0:
            return "저성장"
        else:
            return "마이너스 성장"
    
    def _analyze_dividend_yield(self, dividend_yield: float) -> str:
        """배당수익률 분석"""
        if dividend_yield > 5:
            return "고배당"
        elif dividend_yield > 3:
            return "양호한 배당"
        elif dividend_yield > 1:
            return "보통 배당"
        else:
            return "저배당"
    
    def _format_market_cap(self, market_cap: float) -> str:
        """시가총액 포맷팅"""
        if market_cap >= 10_000_000_000_000:  # 10조 이상
            return f"대형주 ({market_cap/1_000_000_000_000:.1f}조원)"
        elif market_cap >= 2_000_000_000_000:  # 2조 이상
            return f"중대형주 ({market_cap/1_000_000_000_000:.1f}조원)"
        elif market_cap >= 500_000_000_000:  # 5천억 이상
            return f"중형주 ({market_cap/1_000_000_000_000:.2f}조원)"
        else:
            return f"소형주 ({market_cap/100_000_000:.0f}억원)"
    
    def prepare_stock_analysis_prompt(self, stock_data: StockData, strategy: str = "종합") -> str:
        """주식 분석용 프롬프트 생성"""
        
        # 기본 정보 정리
        basic_info = f"""
주식 정보:
- 종목명: {stock_data.name} ({stock_data.symbol})
- 현재가: {stock_data.price:,.0f}원
- 시가총액: {stock_data.market_cap:,.0f}원
- 섹터: {stock_data.sector or '미분류'}
- 업종: {stock_data.industry or '미분류'}
"""
        
        # 재무 지표 정리
        financial_info = self._format_financial_metrics(stock_data)
        
        # 전략별 프롬프트 생성
        strategy_prompt = self._get_strategy_prompt(strategy)
        
        # 최종 프롬프트 조합
        prompt = f"""
{basic_info}

{financial_info}

{strategy_prompt}

분석 요청:
위 정보를 바탕으로 해당 종목에 대한 투자 분석을 수행해주세요.
분석 결과는 다음 형식으로 제공해주세요:

1. 투자 등급: A+/A/B+/B/C+/C/D (7단계)
2. 투자 점수: 0-100점
3. 주요 강점 (3개)
4. 주요 약점 (3개)
5. 투자 전략 추천
6. 목표 가격 (12개월)
7. 리스크 요인
8. 종합 의견

분석 시 고려사항:
- 정량적 지표와 정성적 요소를 균형있게 평가
- 현재 시장 상황과 업계 트렌드 반영
- 투자자 유형별 적합성 고려
- 리스크 대비 수익률 분석
"""
        
        return prompt.strip()
    
    def prepare_portfolio_analysis_prompt(self, stocks: List[StockData], strategy: str = "균형") -> str:
        """포트폴리오 분석용 프롬프트 생성"""
        
        # 포트폴리오 개요
        portfolio_overview = f"""
포트폴리오 분석 요청:
- 종목 수: {len(stocks)}개
- 투자 전략: {strategy}
- 분석 일자: {datetime.now().strftime('%Y-%m-%d')}

포함 종목:
"""
        
        # 각 종목 정보 요약
        for i, stock in enumerate(stocks, 1):
            portfolio_overview += f"""
{i}. {stock.name} ({stock.symbol})
   - 현재가: {stock.price:,.0f}원
   - 시가총액: {stock.market_cap:,.0f}원
   - PER: {stock.pe_ratio:.2f} | PBR: {stock.pb_ratio:.2f} | ROE: {stock.roe:.1f}%
   - 섹터: {stock.sector or '미분류'}
"""
        
        # 분석 요청 사항
        analysis_request = f"""

포트폴리오 분석 요청:
1. 포트폴리오 등급: S/A+/A/B+/B/C+/C/D (8단계)
2. 종합 점수: 0-100점
3. 섹터 분산도 평가
4. 리스크 레벨: 1-5 (1=안전, 5=고위험)
5. 예상 수익률 (12개월)
6. 각 종목별 추천 비중
7. 포트폴리오 강점/약점
8. 리밸런싱 제안
9. 대안 종목 추천

분석 기준:
- {strategy} 전략에 최적화된 평가
- 분산투자 효과 극대화
- 리스크 대비 수익률 최적화
- 시장 상황 변화에 대한 대응력
"""
        
        return (portfolio_overview + analysis_request).strip()
    
    def prepare_market_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """시장 분석용 프롬프트 생성"""
        
        prompt = f"""
시장 분석 요청:

현재 시장 상황:
- 분석 일자: {datetime.now().strftime('%Y-%m-%d')}
- KOSPI: {market_data.get('kospi', 'N/A')}
- KOSDAQ: {market_data.get('kosdaq', 'N/A')}
- 달러/원: {market_data.get('usd_krw', 'N/A')}

주요 지표:
- VIX 지수: {market_data.get('vix', 'N/A')}
- 10년 국채 수익률: {market_data.get('bond_10y', 'N/A')}
- 금리: {market_data.get('interest_rate', 'N/A')}

분석 요청:
1. 시장 전망: 상승/보합/하락
2. 시장 점수: 0-100점
3. 주요 상승 요인 (3개)
4. 주요 하락 요인 (3개)
5. 추천 투자 전략
6. 섹터별 전망
7. 향후 3개월 시나리오
8. 투자자 행동 지침

분석 기준:
- 기술적/펀더멘털 분석 병행
- 글로벌 시장 연동성 고려
- 정책 변화 영향 평가
- 계절성 및 이벤트 요인 반영
"""
        
        return prompt.strip()
    
    def _format_financial_metrics(self, stock_data: StockData) -> str:
        """재무 지표 포맷팅"""
        
        metrics_info = "재무 지표:\n"
        
        # 밸류에이션 지표
        metrics_info += f"- PER: {stock_data.pe_ratio:.2f}배\n" if stock_data.pe_ratio else "- PER: N/A\n"
        metrics_info += f"- PBR: {stock_data.pb_ratio:.2f}배\n" if stock_data.pb_ratio else "- PBR: N/A\n"
        
        # 수익성 지표
        metrics_info += f"- ROE: {stock_data.roe:.1f}%\n" if stock_data.roe else "- ROE: N/A\n"
        metrics_info += f"- ROA: {stock_data.roa:.1f}%\n" if stock_data.roa else "- ROA: N/A\n"
        
        # 안전성 지표
        metrics_info += f"- 부채비율: {stock_data.debt_ratio:.1f}%\n" if stock_data.debt_ratio else "- 부채비율: N/A\n"
        metrics_info += f"- 유동비율: {stock_data.current_ratio:.1f}%\n" if stock_data.current_ratio else "- 유동비율: N/A\n"
        
        # 성장성 및 배당
        metrics_info += f"- 매출 성장률: {stock_data.revenue_growth:.1f}%\n" if stock_data.revenue_growth else "- 매출 성장률: N/A\n"
        metrics_info += f"- 순이익 성장률: {stock_data.profit_growth:.1f}%\n" if stock_data.profit_growth else "- 순이익 성장률: N/A\n"
        metrics_info += f"- 배당수익률: {stock_data.dividend_yield:.2f}%\n" if stock_data.dividend_yield else "- 배당수익률: N/A\n"
        
        return metrics_info.strip()
    
    def _get_strategy_prompt(self, strategy: str) -> str:
        """전략별 프롬프트 생성"""
        
        strategy_prompts = {
            "buffett": """
투자 전략: 워렌 버핏 가치투자
- 내재가치 대비 저평가된 우량기업 선호
- 높은 ROE와 안정적인 수익성 중시
- 낮은 부채비율과 강한 재무구조 선호
- 지속가능한 경쟁우위와 브랜드 가치 평가
- 장기 보유 관점에서 분석
""",
            "lynch": """
투자 전략: 피터 린치 성장투자
- PEG 비율 1.0 이하의 성장주 선호
- 15-30% 성장률의 적정 성장 기업 타겟
- 이해하기 쉬운 비즈니스 모델 선호
- 업계 내 경쟁 우위와 시장 점유율 확대 가능성
- 모멘텀과 기술적 요인도 고려
""",
            "greenblatt": """
투자 전략: 조엘 그린블라트 마법공식
- 높은 자본수익률(ROIC) 기업 선호
- 저평가된 기업(높은 Earnings Yield) 타겟
- 수익률과 밸류에이션의 균형 추구
- 정량적 지표 기반 체계적 접근
- 시장 비효율성 활용한 초과수익 추구
""",
            "종합": """
투자 전략: 종합 분석
- 가치, 성장, 품질 요소를 균형있게 평가
- 정량적 지표와 정성적 요소 종합 고려
- 시장 상황과 업계 트렌드 반영
- 리스크 대비 수익률 최적화
- 다양한 투자자 성향에 맞는 분석 제공
""",
            "균형": """
투자 전략: 균형 포트폴리오
- 안정성과 성장성의 적절한 조화
- 섹터 분산을 통한 리스크 관리
- 시장 상황 변화에 대한 대응력 확보
- 배당주와 성장주의 적절한 배분
- 중장기 관점의 안정적 수익 추구
"""
        }
        
        return strategy_prompts.get(strategy, strategy_prompts["종합"])
    
    def clean_and_validate_data(self, raw_data: Dict[str, Any]) -> Optional[StockData]:
        """원시 데이터 정제 및 검증"""
        
        try:
            # 필수 필드 검증
            required_fields = ['symbol', 'name', 'price', 'market_cap']
            for field in required_fields:
                if field not in raw_data or raw_data[field] is None:
                    logger.warning(f"필수 필드 누락: {field}")
                    return None
            
            # 숫자 데이터 정제
            numeric_fields = [
                'price', 'market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'roa',
                'debt_ratio', 'current_ratio', 'dividend_yield', 
                'revenue_growth', 'profit_growth'
            ]
            
            cleaned_data = {}
            for field, value in raw_data.items():
                if field in numeric_fields:
                    cleaned_data[field] = self._clean_numeric_value(value)
                else:
                    cleaned_data[field] = self._clean_string_value(value)
            
            # StockData 객체 생성
            return StockData(**cleaned_data)
            
        except Exception as e:
            logger.error(f"데이터 정제 오류: {e}")
            return None
    
    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """숫자 데이터 정제"""
        if value is None or value == '':
            return None
        
        try:
            # 문자열인 경우 숫자로 변환
            if isinstance(value, str):
                # 쉼표, 퍼센트, 특수문자 제거
                cleaned = re.sub(r'[,%\s]', '', value)
                if cleaned == '' or cleaned.lower() in ['n/a', 'na', '-']:
                    return None
                value = float(cleaned)
            
            # 무한대나 NaN 처리
            if not np.isfinite(value):
                return None
            
            return float(value)
            
        except (ValueError, TypeError):
            return None
    
    def _clean_string_value(self, value: Any) -> Optional[str]:
        """문자열 데이터 정제"""
        if value is None:
            return None
        
        try:
            cleaned = str(value).strip()
            return cleaned if cleaned and cleaned.lower() not in ['n/a', 'na', '-'] else None
        except:
            return None
    
    def format_analysis_result(self, raw_result: str) -> Dict[str, Any]:
        """AI 분석 결과 포맷팅"""
        
        try:
            # 기본 구조 생성
            formatted = {
                "raw_analysis": raw_result,
                "timestamp": datetime.now().isoformat(),
                "parsed_data": {}
            }
            
            # 주요 정보 추출 시도
            lines = raw_result.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 섹션 헤더 감지
                if any(keyword in line for keyword in ['투자 등급', '투자 점수', '강점', '약점', '추천', '목표', '리스크', '의견']):
                    current_section = line
                    formatted["parsed_data"][current_section] = []
                elif current_section:
                    formatted["parsed_data"][current_section].append(line)
            
            return formatted
            
        except Exception as e:
            logger.error(f"분석 결과 포맷팅 오류: {e}")
            return {
                "raw_analysis": raw_result,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            } 

    def prepare_for_analysis(self, stocks: List, analysis_type: str = 'investment') -> Dict[str, Any]:
        """AI 분석을 위한 데이터 전처리"""
        try:
            # 주식 데이터를 StockData 형태로 변환
            processed_stocks = []
            for stock in stocks:
                if hasattr(stock, 'symbol'):
                    stock_data = StockData(
                        symbol=stock.symbol,
                        name=getattr(stock, 'name', stock.symbol),
                        price=getattr(stock, 'price', 0.0),
                        market_cap=getattr(stock, 'market_cap', 0.0),
                        pe_ratio=getattr(stock, 'pe_ratio', None),
                        pb_ratio=getattr(stock, 'pb_ratio', None),
                        roe=getattr(stock, 'roe', None),
                        roa=getattr(stock, 'roa', None),
                        debt_ratio=getattr(stock, 'debt_ratio', None),
                        current_ratio=getattr(stock, 'current_ratio', None),
                        dividend_yield=getattr(stock, 'dividend_yield', None),
                        revenue_growth=getattr(stock, 'revenue_growth', None),
                        profit_growth=getattr(stock, 'profit_growth', None),
                        sector=getattr(stock, 'sector', None),
                        industry=getattr(stock, 'industry', None)
                    )
                    processed_stocks.append(stock_data)
            
            # 포트폴리오 분석 프롬프트 생성
            if analysis_type == 'investment':
                prompt = self.prepare_portfolio_analysis_prompt(processed_stocks, "균형")
            else:
                prompt = self.prepare_portfolio_analysis_prompt(processed_stocks, analysis_type)
            
            return {
                "stocks": processed_stocks,
                "analysis_prompt": prompt,
                "analysis_type": analysis_type,
                "total_stocks": len(processed_stocks)
            }
            
        except Exception as e:
            logger.error(f"AI 데이터 전처리 오류: {e}")
            return {
                "stocks": [],
                "analysis_prompt": "",
                "analysis_type": analysis_type,
                "total_stocks": 0,
                "error": str(e)
            }

# 호환성을 위한 별칭
AIPreprocessor = AIDataPreprocessor 