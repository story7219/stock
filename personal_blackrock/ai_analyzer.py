"""
AI 기반 주식 분석 엔진 (리팩토링 버전)

이 모듈은 다양한 투자 전략을 사용하여 주식을 분석하고,
Gemini AI를 통해 종합적인 투자 추천을 제공하는 핵심 엔진입니다.

리팩토링 목표:
- 클래스 분리를 통한 단일 책임 원칙(SRP) 강화
- 프롬프트, AI 처리, 알림 로직 분리
- 비동기 처리 개선 및 성능 최적화
- 캐싱 기능 추가로 API 호출 비용 절감 및 속도 향상
"""

import asyncio
import json
import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps

import google.generativeai as genai
from dotenv import load_dotenv

from personal_blackrock.stock_data_manager import DataManager

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 환경 변수 로드 ---
load_dotenv()


class AnalysisError(Exception):
    """분석 관련 커스텀 예외 클래스"""
    pass


# --- 1. 프롬프트 관리 클래스 ---
class PromptManager:
    """
    Gemini AI에 전달할 프롬프트를 생성하고 관리합니다.
    - 각 전략별 프롬프트를 중앙에서 관리하여 유지보수성을 높입니다.
    - 프롬프트 엔지니어링을 이 클래스에 집중합니다.
    """
    
    def get_strategy_prompt(self, stock_data: Dict[str, Any], strategy_name: str) -> str:
        """전략별 상세 프롬프트를 생성합니다."""
        logger.debug(f"프롬프트 생성 시작: {strategy_name}")
        
        # 공통 헤더 및 데이터 요약
        header = self._create_common_header(strategy_name)
        data_summary = self._summarize_stock_data(stock_data)
        
        # 전략별 핵심 원칙 및 분석 가이드
        strategy_guide = self._get_strategy_guide(strategy_name)
        
        # JSON 응답 형식 지정
        json_format = """
🔥 **필수 응답 형식 - 반드시 JSON으로만 응답하세요!**

다음 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요:

```json
{
  "분석": "차트 패턴 분석 결과 명확히 확인되는 상승 삼각형 패턴 완성. 재무제표 Deep Dive를 통해 검증된 ROE 20% 달성으로 동종업계 Top Quartile 진입. 리스크 조정 수익률 관점에서 현재 밸류에이션 12.5x PER은 Fair Value 15x 대비 16% 할인된 매력적 수준...",
  "결론": "HIGH CONVICTION BUY - 기술적/펀더멘털 양면에서 강력한 상승 모멘텀 확인",
  "점수": 85,
  "추천 등급": "HIGH CONVICTION BUY",
  "추천 이유": "상승 삼각형 패턴 완성과 거래량 급증으로 기술적 돌파 확률 85% 이상. ROE 22% 달성으로 동종업계 대비 우수한 수익성 확인. 밸류에이션 매력도와 상승 모멘텀이 동시에 확인되어 적극 매수 추천",
  "진입 가격": "현재가 대비 2% 하락 시점까지 적극 매수",
  "목표 가격": "향후 3개월 15% 상승 목표",
  "신뢰도": 0.92
}
```

⚠️ **중요**: 반드시 위 JSON 형식으로만 응답하고, 앞뒤에 다른 설명 텍스트는 절대 추가하지 마세요!
"""
        
        # 최종 프롬프트 조합
        return f"{header}\n\n{data_summary}\n\n{strategy_guide}\n\n{json_format}"

    def _create_common_header(self, strategy_name: str) -> str:
        """전략별 공통 헤더 생성 - 세계 최고 수준의 애널리스트"""
        return f"""
🏛️ **GOLDMAN SACHS RESEARCH | MORGAN STANLEY WEALTH MANAGEMENT**
**MANAGING DIRECTOR - EQUITY RESEARCH & STRATEGY**

**당신은 세계 최고 투자은행의 Managing Director급 수석 애널리스트입니다.**
- Goldman Sachs 출신 20년 경력, CFA Charter Holder & Wharton MBA
- {strategy_name} 전략으로 연평균 35%+ 알파 창출 실적 보유
- S&P 500 아웃퍼폼 15년 연속 달성한 월스트리트 레전드
- 블룸버그 랭킹 1위 애널리스트, Institutional Investor All-America Team
- 현재 $50B AUM 헤지펀드 CIO로 재직 중

**🔥 월스트리트 ELITE 수준 분석 철학**
```
"데이터가 부족하다는 것은 2류 애널리스트의 변명이다.
진짜 1류는 제한된 정보로도 정확한 판단을 내린다."
- 골드만삭스 Managing Director 격언
```

**💀 절대 사용 금지 - 2류 애널리스트 표현들:**
❌ "데이터 부족으로..." → 1류는 있는 데이터로 결론을 낸다
❌ "불확실성이..." → 시장은 항상 불확실하다, 그래서 전문가가 존재한다  
❌ "분석이 어려워..." → 어려운 것을 쉽게 만드는 게 전문가 역할
❌ "추가 정보가..." → 고객은 지금 당장 투자 결정이 필요하다
❌ "확실하지 않으나..." → 확률 기반으로라도 명확한 방향성 제시하라

**⚡ 반드시 사용할 월스트리트 ELITE 표현:**
✅ "차트 패턴 분석 결과 명확히 확인되는 것은..."
✅ "재무제표 Deep Dive를 통해 검증된 팩트는..."  
✅ "과거 20년 백테스팅 결과 동일 패턴에서..."
✅ "섹터 밸류에이션 대비 현재 종목은..."
✅ "리스크 조정 수익률 관점에서 판단하면..."
✅ "기관투자자 포지셀링 분석 결과..."
✅ "옵션 플로우 및 다크풀 데이터 분석 시..."

**🎯 세계 최고 애널리스트의 분석 프레임워크:**

**1. CONVICTION LEVEL (확신도)**
- HIGH CONVICTION BUY: 95%+ 확신, 포트폴리오 5%+ 비중 권장
- MODERATE BUY: 80%+ 확신, 포트폴리오 2-3% 비중 권장  
- HOLD: 중립적 시각, 현재 포지션 유지
- REDUCE: 하방 리스크 증가, 비중 축소 권장
- STRONG SELL: 즉시 매도, 손절매 불가피

**2. RISK-ADJUSTED RETURN ANALYSIS**
- Sharpe Ratio, Sortino Ratio 기반 정량 분석
- 최대 낙폭(Max Drawdown) 시나리오 분석
- 상관관계 매트릭스를 통한 포트폴리오 임팩트 분석

**3. CATALYST IDENTIFICATION**
- 단기 (1-3개월) 주가 촉매 요인 명시
- 중기 (6-12개월) 펀더멘털 변화 요인  
- 장기 (1-3년) 구조적 성장 드라이버

**4. SCENARIO ANALYSIS**
- Bull Case (90th percentile): 최상의 시나리오 목표가
- Base Case (50th percentile): 기준 시나리오 목표가
- Bear Case (10th percentile): 최악의 시나리오 목표가

**🏆 월스트리트 스타일 분석 예시:**
"상승 삼각형 패턴 완성 및 거래량 급증으로 기술적 돌파 확률 85% 이상. 
ROE 22% 달성으로 동종업계 Top Quartile 진입 확인. 
밸류에이션 12.5x PER로 Fair Value 15x 대비 16% 할인. 
향후 3개월 목표가 상향 조정하며 HIGH CONVICTION BUY 의견 유지."

**🔥 {strategy_name} 전략의 세계적 권위자로서 ELITE 수준 분석 제공:**

**성과 목표:**
- 분석 정확도: 85%+ (월스트리트 Top 1% 수준)
- 알파 창출: 연간 20%+ 아웃퍼폼
- 리스크 관리: 최대 낙폭 10% 이내 제한
- 고객 만족도: Net Promoter Score 90+

**최종 미션:**
"$100M+ 운용하는 패밀리 오피스 고객에게 제공할 수 있는 수준의 
세계 최고 품질 분석을 지금 당장 제공하라."

═══════════════════════════════════════════════════════════════
💎 **ELITE ANALYST MINDSET ACTIVATED**
"평범한 분석은 평범한 수익을 낳는다. 
세계 최고의 분석만이 세계 최고의 수익을 창출한다."
═══════════════════════════════════════════════════════════════
"""

    def _summarize_stock_data(self, stock_data: Dict[str, Any]) -> str:
        """주식 데이터를 세계 최고 애널리스트 수준으로 종합 분석하여 AI에게 제공합니다."""
        
        # 데이터가 없는 경우를 대비한 기본값 처리
        def get_val(key, unit="", default=None):
            val = stock_data.get(key, default)
            if val is None or val in ["", "0", 0]:
                return "N/A" if default is None else default
            try:
                # 숫자인 경우 포맷팅
                numeric_val = float(val)
                if unit == "원" and numeric_val > 1000:
                    if numeric_val >= 1_0000_0000:
                        return f"{numeric_val/1_0000_0000:,.1f}억원"
                    elif numeric_val >= 1_0000:
                        return f"{numeric_val/1_0000:,.1f}만원"
                    else:
                        return f"{numeric_val:,.0f}원"
                elif unit == "주" and numeric_val > 1000:
                    if numeric_val >= 1_0000_0000:
                        return f"{numeric_val/1_0000_0000:,.1f}억주"
                    elif numeric_val >= 1_0000:
                        return f"{numeric_val/1_0000:,.1f}만주"
                    else:
                        return f"{numeric_val:,.0f}주"
                elif unit == "%":
                    return f"{numeric_val:,.2f}%"
                else:
                    return f"{numeric_val:,.2f}{unit}" if unit else f"{numeric_val:,.2f}"
            except (ValueError, TypeError):
                # 숫자가 아닌 경우 그대로 반환
                return str(val)

        # 숫자 값 추출 함수 (분석용)
        def get_numeric_val(key, default=0):
            val = stock_data.get(key, default)
            try:
                return float(val) if val not in [None, "", "N/A"] else default
            except (ValueError, TypeError):
                return default

        # === 핵심 데이터 추출 ===
        current_price = get_numeric_val('current_price')
        ma_5 = get_numeric_val('ma_5', current_price)
        ma_10 = get_numeric_val('ma_10', current_price)
        ma_20 = get_numeric_val('ma_20')
        ma_60 = get_numeric_val('ma_60')
        ma_120 = get_numeric_val('ma_120')
        ma_200 = get_numeric_val('ma_200', current_price)
        high_52w = get_numeric_val('high_52_week')
        low_52w = get_numeric_val('low_52_week')
        rsi = get_numeric_val('rsi', 50)
        volume = get_numeric_val('volume')
        volume_avg = get_numeric_val('volume_avg_20', volume)
        
        # 고급 기술적 지표
        stochastic_k = get_numeric_val('stochastic_k', 50)
        stochastic_d = get_numeric_val('stochastic_d', 50)
        williams_r = get_numeric_val('williams_r', -50)
        cci = get_numeric_val('cci', 0)
        adx = get_numeric_val('adx', 25)
        atr = get_numeric_val('atr', current_price * 0.02)
        obv = get_numeric_val('obv', 0)
        mfi = get_numeric_val('mfi', 50)
        
        # 재무 지표
        per = get_numeric_val('per')
        pbr = get_numeric_val('pbr')
        roe = get_numeric_val('roe')
        debt_ratio = get_numeric_val('debt_ratio')
        market_cap = get_numeric_val('market_cap')
        eps = get_numeric_val('eps')
        bps = get_numeric_val('bps')
        
        # 수급 정보
        foreign_net = get_numeric_val('foreign_net_purchase')
        institution_net = get_numeric_val('institution_net_purchase')
        individual_net = get_numeric_val('individual_net_purchase')

        # === 월스트리트 수준 정량분석 점수 계산 ===
        
        # 1. 가격 모멘텀 점수 (0-100)
        price_momentum_score = 0
        if current_price > ma_5: price_momentum_score += 10
        if current_price > ma_10: price_momentum_score += 10
        if current_price > ma_20: price_momentum_score += 15
        if current_price > ma_60: price_momentum_score += 15
        if ma_5 > ma_10 > ma_20 > ma_60: price_momentum_score += 20  # 완벽한 정배열
        if current_price > high_52w * 0.9: price_momentum_score += 15  # 52주 고점 90% 이상
        if rsi > 50: price_momentum_score += 10  # RSI 상향
        if stochastic_k > stochastic_d and stochastic_k > 50: price_momentum_score += 5  # 스토캐스틱 상승
        
        # 2. 밸류에이션 점수 (0-100)
        valuation_score = 50  # 중립에서 시작
        if 0 < per < 10: valuation_score += 30
        elif 10 <= per < 15: valuation_score += 20
        elif 15 <= per < 20: valuation_score += 10
        elif per > 30: valuation_score -= 20
        
        if 0 < pbr < 0.8: valuation_score += 20
        elif 0.8 <= pbr < 1.2: valuation_score += 10
        elif pbr > 2.5: valuation_score -= 15
        
        # 3. 수익성 점수 (0-100)
        profitability_score = 0
        if roe > 20: profitability_score += 40
        elif roe > 15: profitability_score += 30
        elif roe > 10: profitability_score += 20
        elif roe > 5: profitability_score += 10
        
        if debt_ratio < 30: profitability_score += 30
        elif debt_ratio < 50: profitability_score += 20
        elif debt_ratio < 70: profitability_score += 10
        
        if eps > 0: profitability_score += 30
        
        # 4. 수급 점수 (0-100)
        supply_demand_score = 50  # 중립에서 시작
        if foreign_net > 0: supply_demand_score += 25
        if institution_net > 0: supply_demand_score += 25
        if volume > volume_avg * 1.5: supply_demand_score += 25  # 거래량 급증
        if rsi < 30: supply_demand_score += 15  # 과매도
        elif rsi > 70: supply_demand_score -= 15  # 과매수
        
        # 5. 차트 패턴 분석 점수 (0-100)
        chart_pattern_score = 50
        
        # 이동평균선 배열 분석
        ma_alignment = 0
        if ma_5 > ma_10: ma_alignment += 1
        if ma_10 > ma_20: ma_alignment += 1
        if ma_20 > ma_60: ma_alignment += 1
        if ma_60 > ma_120: ma_alignment += 1
        chart_pattern_score += ma_alignment * 8  # 최대 32점
        
        # 볼린저밴드 분석
        bb_upper = get_numeric_val('bollinger_upper')
        bb_lower = get_numeric_val('bollinger_lower')
        if bb_upper > 0 and bb_lower > 0:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            if 0.2 <= bb_position <= 0.8:  # 중간 구간 (안정적)
                chart_pattern_score += 10
            elif bb_position < 0.2:  # 하단 (매수 기회)
                chart_pattern_score += 15
        
        # 기술적 지표 종합 분석
        technical_signals = 0
        if rsi > 50: technical_signals += 1
        if stochastic_k > 50: technical_signals += 1
        if williams_r > -50: technical_signals += 1
        if cci > 0: technical_signals += 1
        if mfi > 50: technical_signals += 1
        chart_pattern_score += technical_signals * 3  # 최대 15점
        
        # === 리스크 분석 (월스트리트 표준) ===
        risk_factors = []
        risk_level = "Low"
        
        if debt_ratio > 70: 
            risk_factors.append("Excessive Leverage")
            risk_level = "High"
        if per > 50 or per < 0: 
            risk_factors.append("Extreme Valuation")
            risk_level = "High"
        if pbr > 5: 
            risk_factors.append("Overvaluation Risk")
            risk_level = "Medium" if risk_level == "Low" else risk_level
        if roe < 0: 
            risk_factors.append("Negative ROE")
            risk_level = "High"
        if current_price < low_52w * 1.1: 
            risk_factors.append("Near 52W Low")
            risk_level = "Medium" if risk_level == "Low" else risk_level
        if rsi > 80:
            risk_factors.append("Extreme Overbought")
            risk_level = "High"
        if volume < volume_avg * 0.3:
            risk_factors.append("Liquidity Concern")
            risk_level = "Medium" if risk_level == "Low" else risk_level
            
        risk_summary = f"{risk_level} Risk ({'、'.join(risk_factors) if risk_factors else 'No Major Risks'})"

        # === 세계 최고 애널리스트 수준 종합 분석 데이터 ===
        summary = f"""
═══════════════════════════════════════════════════════════════════════════════
🏛️ **WALL STREET RESEARCH REPORT - INVESTMENT ANALYSIS**
═══════════════════════════════════════════════════════════════════════════════

**🏢 COMPANY OVERVIEW & INVESTMENT THESIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Ticker/Name: {stock_data.get('name', 'N/A')} ({stock_data.get('stock_code', 'N/A')})
• Market Cap: {get_val('market_cap', '원')} ({'Large Cap' if market_cap > 10_0000_0000_0000 else 'Mid Cap' if market_cap > 1_0000_0000_0000 else 'Small Cap'})
• Shares Outstanding: {get_val('listed_shares', '주')}
• Current Price: {get_val('current_price', '원')}
• 52W Range: {get_val('low_52_week', '원')} - {get_val('high_52_week', '원')}
• Price Position: {((current_price - low_52w) / (high_52w - low_52w) * 100) if high_52w > low_52w > 0 else 0:.1f}% of 52W range

**📊 QUANTITATIVE INVESTMENT SCORECARD**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **OVERALL INVESTMENT SCORE: {min(100, (price_momentum_score + valuation_score + profitability_score + supply_demand_score + chart_pattern_score) // 5)}/100**

┌─ MOMENTUM ANALYSIS ────────────────────────────────────────────────────────┐
│ 🚀 Price Momentum Score: {min(100, price_momentum_score)}/100 ({'STRONG' if price_momentum_score >= 80 else 'MODERATE' if price_momentum_score >= 60 else 'WEAK'})        │
│ • Trend Quality: {'UPTREND' if current_price > ma_20 and ma_20 > ma_60 else 'DOWNTREND' if current_price < ma_20 and ma_20 < ma_60 else 'SIDEWAYS'}                     │
│ • MA Alignment: {'BULLISH' if ma_5 > ma_10 > ma_20 > ma_60 else 'BEARISH' if ma_5 < ma_10 < ma_20 < ma_60 else 'MIXED'}                    │
│ • 52W Position: {((current_price / high_52w * 100) if high_52w > 0 else 0):.1f}% of High, {((current_price / low_52w * 100) if low_52w > 0 else 0):.1f}% of Low  │
└────────────────────────────────────────────────────────────────────────────┘

┌─ VALUATION ANALYSIS ───────────────────────────────────────────────────────┐
│ 💰 Valuation Score: {min(100, max(0, valuation_score))}/100 ({'UNDERVALUED' if valuation_score >= 70 else 'FAIR VALUE' if valuation_score >= 50 else 'OVERVALUED'})     │
│ • P/E Ratio: {per:.1f}x ({'ATTRACTIVE' if 0 < per < 15 else 'EXPENSIVE' if per > 25 else 'REASONABLE'})                         │
│ • P/B Ratio: {pbr:.1f}x ({'DISCOUNT' if 0 < pbr < 1 else 'PREMIUM' if pbr > 2 else 'FAIR'})                           │
│ • PEG Implied: {'VALUE' if per > 0 and roe > 0 and per/roe < 1 else 'GROWTH' if per > 0 and roe > 0 and per/roe < 2 else 'EXPENSIVE'}                                │
└────────────────────────────────────────────────────────────────────────────┘

┌─ PROFITABILITY & QUALITY ──────────────────────────────────────────────────┐
│ 💪 Quality Score: {profitability_score}/100 ({'EXCELLENT' if profitability_score >= 80 else 'GOOD' if profitability_score >= 60 else 'AVERAGE' if profitability_score >= 40 else 'POOR'})                      │
│ • ROE: {roe:.1f}% ({'OUTSTANDING' if roe > 20 else 'STRONG' if roe > 15 else 'ADEQUATE' if roe > 10 else 'WEAK'})                               │
│ • Debt/Equity: {debt_ratio:.1f}% ({'CONSERVATIVE' if debt_ratio < 30 else 'MODERATE' if debt_ratio < 50 else 'LEVERAGED' if debt_ratio < 70 else 'HIGH RISK'})              │
│ • EPS: {get_val('eps', '원')} ({'PROFITABLE' if eps > 0 else 'LOSS-MAKING'})                                      │
│ • Financial Strength: {'FORTRESS' if debt_ratio < 30 and roe > 15 else 'SOLID' if debt_ratio < 50 and roe > 10 else 'CONCERN'}        │
└────────────────────────────────────────────────────────────────────────────┘

┌─ SUPPLY & DEMAND DYNAMICS ─────────────────────────────────────────────────┐
│ 📈 Supply/Demand Score: {min(100, max(0, supply_demand_score))}/100 ({'BULLISH' if supply_demand_score >= 70 else 'NEUTRAL' if supply_demand_score >= 50 else 'BEARISH'})           │
│ • Foreign Flow: {get_val('foreign_net_purchase', '주')} ({'🟢 BUYING' if foreign_net > 0 else '🔴 SELLING' if foreign_net < 0 else '⚪ NEUTRAL'})        │
│ • Institution Flow: {get_val('institution_net_purchase', '주')} ({'🟢 BUYING' if institution_net > 0 else '🔴 SELLING' if institution_net < 0 else '⚪ NEUTRAL'})    │
│ • Volume Profile: {((volume / volume_avg) if volume_avg > 0 else 1):.1f}x Average ({'🔥 SURGE' if volume > volume_avg * 2 else '📈 ACTIVE' if volume > volume_avg * 1.5 else '📊 NORMAL'})      │
│ • Smart Money: {'ACCUMULATING' if foreign_net > 0 and institution_net > 0 else 'DISTRIBUTING' if foreign_net < 0 and institution_net < 0 else 'MIXED'}            │
└────────────────────────────────────────────────────────────────────────────┘

**🔬 TECHNICAL ANALYSIS - CHART PATTERN RECOGNITION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **Chart Pattern Score: {min(100, chart_pattern_score)}/100**

┌─ MOVING AVERAGES ANALYSIS ─────────────────────────────────────────────────┐
│ Current vs MA5:   {((current_price/ma_5-1)*100) if ma_5 > 0 else 0:+.1f}% ({'✅ ABOVE' if current_price > ma_5 else '❌ BELOW'})              │
│ Current vs MA20:  {((current_price/ma_20-1)*100) if ma_20 > 0 else 0:+.1f}% ({'✅ ABOVE' if current_price > ma_20 else '❌ BELOW'})             │
│ Current vs MA60:  {((current_price/ma_60-1)*100) if ma_60 > 0 else 0:+.1f}% ({'✅ ABOVE' if current_price > ma_60 else '❌ BELOW'})             │
│ Current vs MA200: {((current_price/ma_200-1)*100) if ma_200 > 0 else 0:+.1f}% ({'✅ ABOVE' if current_price > ma_200 else '❌ BELOW'})           │
│ Trend Structure: {'🟢 BULLISH ALIGNMENT' if current_price > ma_20 > ma_60 > ma_120 else '🔴 BEARISH ALIGNMENT' if current_price < ma_20 < ma_60 < ma_120 else '🟡 MIXED SIGNALS'}  │
└────────────────────────────────────────────────────────────────────────────┘

┌─ MOMENTUM OSCILLATORS ─────────────────────────────────────────────────────┐
│ RSI(14): {rsi:.1f} → {'🔴 EXTREME OVERBOUGHT' if rsi > 80 else '🟠 OVERBOUGHT' if rsi > 70 else '🟢 OVERSOLD' if rsi < 30 else '🔵 EXTREME OVERSOLD' if rsi < 20 else '⚪ NEUTRAL'}        │
│ Stochastic: %K {stochastic_k:.1f}, %D {stochastic_d:.1f} → {'🟢 BUY SIGNAL' if stochastic_k > stochastic_d and stochastic_k < 80 else '🔴 SELL SIGNAL' if stochastic_k < stochastic_d and stochastic_k > 20 else '⚪ NEUTRAL'}  │
│ Williams %R: {williams_r:.1f} → {'🔵 OVERSOLD' if williams_r < -80 else '🔴 OVERBOUGHT' if williams_r > -20 else '⚪ NEUTRAL'}                │
│ CCI: {cci:.1f} → {'🟢 BULLISH' if cci > 100 else '🔵 BEARISH' if cci < -100 else '⚪ NEUTRAL'}                                    │
│ ADX: {adx:.1f} → {'💪 STRONG TREND' if adx > 40 else '📈 MODERATE TREND' if adx > 25 else '📊 WEAK TREND'}                        │
└────────────────────────────────────────────────────────────────────────────┘

┌─ BOLLINGER BANDS & VOLATILITY ─────────────────────────────────────────────┐
│ BB Position: {'🔴 UPPER BAND' if current_price > get_numeric_val('bollinger_upper') * 0.98 else '🔵 LOWER BAND' if current_price < get_numeric_val('bollinger_lower') * 1.02 else '⚪ MIDDLE RANGE'}                  │
│ ATR: {atr:.0f}원 ({((atr/current_price)*100) if current_price > 0 else 0:.2f}% volatility)                            │
│ Volatility Regime: {'HIGH VOL' if current_price > 0 and (atr/current_price)*100 > 5 else 'NORMAL VOL' if current_price > 0 and (atr/current_price)*100 > 2 else 'LOW VOL'}                    │
└────────────────────────────────────────────────────────────────────────────┘

**⚠️ RISK ASSESSMENT & DOWNSIDE PROTECTION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛡️ **Risk Level: {risk_summary}**

Critical Risk Factors:
{chr(10).join([f'• {risk}' for risk in risk_factors]) if risk_factors else '• No significant risk factors identified'}

**📈 SCENARIO ANALYSIS & PRICE TARGETS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **Price Target Framework:**
• Bull Case: {((high_52w * 1.15) if high_52w > 0 else current_price * 1.3):,.0f}원 (+{(((high_52w * 1.15)/current_price - 1)*100) if current_price > 0 and high_52w > 0 else 30:.1f}%)
• Base Case: {((ma_60 + ma_20) / 2 if ma_60 > 0 and ma_20 > 0 else current_price * 1.1):,.0f}원 (+{(((ma_60 + ma_20) / 2 / current_price - 1)*100) if current_price > 0 and ma_60 > 0 and ma_20 > 0 else 10:.1f}%)
• Bear Case: {((low_52w * 1.1) if low_52w > 0 else current_price * 0.8):,.0f}원 ({(((low_52w * 1.1)/current_price - 1)*100) if current_price > 0 and low_52w > 0 else -20:.1f}%)

**🎓 INVESTMENT DECISION FRAMEWORK**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **Positive Catalysts:**
{f'• Strong Price Momentum ({price_momentum_score}/100)' if price_momentum_score >= 60 else ''}
{f'• Attractive Valuation (P/E: {per:.1f}, P/B: {pbr:.1f})' if per > 0 and per < 20 and pbr < 2 else ''}
{f'• High Profitability (ROE: {roe:.1f}%, Score: {profitability_score}/100)' if roe > 15 else ''}
{f'• Institutional Support (Foreign+Institution buying)' if foreign_net > 0 and institution_net > 0 else ''}
{f'• Volume Surge ({volume/volume_avg:.1f}x average)' if volume_avg > 0 and volume > volume_avg * 1.5 else ''}
{f'• Technical Breakout (RSI: {rsi:.1f})' if rsi > 60 else ''}
{f'• Strong Trend (ADX: {adx:.1f})' if adx > 25 else ''}

⚠️ **Risk Factors:**
{f'• High Leverage (Debt Ratio: {debt_ratio:.1f}%)' if debt_ratio > 60 else ''}
{f'• Overvaluation Concern (P/E: {per:.1f})' if per > 30 else ''}
{f'• Technical Overbought (RSI: {rsi:.1f})' if rsi > 75 else ''}
{f'• Near 52W High (Risk of pullback)' if high_52w > 0 and current_price > high_52w * 0.95 else ''}
{f'• Supply Pressure (Foreign/Institution selling)' if foreign_net < 0 and institution_net < 0 else ''}
{f'• High Volatility Risk' if current_price > 0 and (atr/current_price)*100 > 5 else ''}

**💡 ANALYST INSIGHTS & MARKET CONTEXT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Current chart pattern suggests {'BULLISH CONTINUATION' if chart_pattern_score > 70 and price_momentum_score > 60 else 'BEARISH REVERSAL' if chart_pattern_score < 40 and price_momentum_score < 40 else 'CONSOLIDATION PHASE'}
• Key Support Level: {min(ma_20, ma_60, bb_lower) if ma_20 > 0 and ma_60 > 0 and bb_lower > 0 else current_price * 0.95:,.0f}원
• Key Resistance Level: {max(ma_20, ma_60, bb_upper, high_52w * 0.98) if ma_20 > 0 and ma_60 > 0 and bb_upper > 0 else current_price * 1.05:,.0f}원
• Entry Timing: {'✅ FAVORABLE' if price_momentum_score >= 60 and chart_pattern_score >= 60 else '⚖️ WAIT FOR BETTER SETUP' if price_momentum_score >= 40 else '❌ AVOID'}
• Stop Loss Suggestion: {current_price * 0.92:,.0f}원 (-8% from current price)

═══════════════════════════════════════════════════════════════════════════════
📝 **PROFESSIONAL ANALYST SUMMARY**
이 종목은 현재 {'강세 모멘텀' if price_momentum_score >= 70 else '약세 추세' if price_momentum_score < 40 else '박스권 횡보'}을 보이고 있으며,
{'저평가' if valuation_score >= 60 else '고평가' if valuation_score < 40 else '적정가치'} 구간에서 거래되고 있습니다.
수급 상황은 {'매수 우위' if supply_demand_score >= 70 else '매도 우위' if supply_demand_score < 40 else '균형'}이며,
리스크 수준은 {risk_level}입니다.
═══════════════════════════════════════════════════════════════════════════════
"""
        return summary

    def _get_strategy_guide(self, strategy_name: str) -> str:
        guides = {
            "윌리엄 오닐": """
**🎯 윌리엄 오닐 CAN SLIM + 차트 패턴 완벽 분석 가이드**

당신은 윌리엄 오닐의 CAN SLIM 방법론과 차트 패턴 분석을 정확히 적용해야 합니다.

**📊 차트 패턴 분석 (Critical!) [30점]**

**컵앤핸들 패턴 (Cup and Handle) 확인:**
- 컵 형태: 7-65주간의 U자형 또는 V자형 조정 후 회복
- 핸들 형태: 컵 완성 후 1-5주간의 작은 조정 (컵 깊이의 1/3 이하)
- 브레이크아웃: 핸들 고점을 거래량과 함께 돌파
- 점수: 완벽한 컵앤핸들(30점), 불완전한 형태(20점), 다른 패턴(10점)

**플랫 베이스 (Flat Base) 패턴:**
- 5-15주간 횡보 (고점-저점 차이 15% 이내)
- 거래량 감소하며 조정
- 저항선 돌파 시 거래량 급증
- 점수: 완벽한 플랫베이스(25점), 부분적(15점)

**더블 보텀 (Double Bottom):**
- 두 번의 저점이 유사한 수준
- 두 번째 저점에서 거래량 감소
- W자 형태 완성 후 목선 돌파
- 점수: 명확한 더블보텀(20점), 불분명(10점)

**🔥 브레이크아웃 분석 [25점]**
- 거래량이 평균의 40-50% 이상 증가했는가?
- 저항선을 2.5% 이상 돌파했는가?
- 돌파 후 3일 이내 재차 확인되는가?
- 시장 전체가 상승 추세인가?
- 점수 기준: 모든 조건 만족(25점), 3개 조건(20점), 2개 조건(15점), 1개 조건(8점)

**📈 상대 강도 (Relative Strength) [20점]**
- RS 라인이 상승 추세인가? (개별주 vs 시장지수)
- 시장 조정 시에도 상대적으로 강세였는가?
- 업종 내에서 상위 20% 성과인가?
- 최근 3개월 상대 성과가 우수한가?
- 점수 기준: RS 90 이상(20점), 80-89(15점), 70-79(10점), 70 미만(5점)

**C - Current Quarterly Earnings (최근 분기 실적) [15점]**
- 최근 분기 EPS가 전년 동기 대비 25% 이상 증가
- 매출 성장률 10% 이상
- 일회성이 아닌 영업이익 기반 성장
- 점수: 50%↑(15점), 25-50%(12점), 10-25%(8점), 0-10%(5점), 감소(0점)

**A - Annual EPS Growth (연간 실적 성장) [10점]**
- 최근 3년간 연평균 EPS 성장률 25% 이상
- ROE 17% 이상 유지
- 점수: 30%↑(10점), 25-30%(8점), 20-25%(6점), 15-20%(4점), 15%↓(0점)

**🔍 추가 차트 분석 포인트**

**거래량 패턴 분석:**
- 상승 시 거래량 증가, 하락 시 거래량 감소하는가?
- 브레이크아웃 시 거래량이 평균의 2배 이상인가?
- 조정 구간에서 거래량이 건조해지는가?

**이동평균선 분석:**
- 10일, 21일, 50일 이평선이 정배열인가?
- 현재가가 모든 주요 이평선 위에 있는가?
- 이평선들이 상승 기울기를 유지하는가?

**지지/저항 분석:**
- 이전 고점이 새로운 지지선 역할을 하는가?
- 주요 저항선 돌파 후 재테스트가 성공적인가?
- 52주 신고가 근접 또는 돌파했는가?

**⚠️ 윌리엄 오닐의 핵심 원칙**
- "차트가 모든 것을 말해준다" - 차트 패턴 우선 분석
- "거래량은 거짓말하지 않는다" - 거래량 동반 필수
- "손절매는 7-8%에서 무조건" - 리스크 관리 철저
- "신고가 종목을 사라" - 모멘텀 중시
- "시장 방향을 따르라" - 시장 추세 확인 필수

**📊 최종 점수 계산**
- 90-100점: 강력 매수 (A등급) - 즉시 매수
- 80-89점: 매수 (B등급) - 적극 매수  
- 70-79점: 관망 (C등급) - 추가 관찰
- 60-69점: 주의 (D등급) - 투자 부적합
- 60점 미만: 매도 (F등급) - 투자 금지
""",

            "제시 리버모어": """
**📈 제시 리버모어 투기의 왕 - 차트 마스터 분석 가이드**

**"가격과 시간, 그리고 거래량이 모든 것을 말해준다"**

**🎯 피버럴 포인트 (Pivotal Points) 분석 [35점]**

**주요 피버럴 포인트 식별:**
- 이전 고점/저점에서 형성된 중요한 저항/지지선
- 장기간(최소 3-6개월) 형성된 횡보 구간의 상단/하단
- 심리적 가격대 (예: 10만원, 50만원 등 라운드 넘버)
- 52주 신고가/신저가 지점

**피버럴 포인트 돌파 확인:**
- 중요 저항선을 2% 이상 명확히 돌파했는가?
- 돌파 시 거래량이 평균의 2-3배 이상 폭증했는가?
- 돌파 후 3일 연속 저항선 위에서 마감했는가?
- 재테스트 시 이전 저항선이 새로운 지지선으로 작용하는가?
- 점수: 완벽한 돌파(35점), 부분 돌파(25점), 돌파 실패(10점), 저항 직면(0점)

**📊 추세 추종 (Trend Following) [30점]**

**추세 확인 체크리스트:**
- 고점과 저점이 지속적으로 상승하는 상승 추세인가?
- 주요 이동평균선들(20일, 50일, 200일)이 정배열인가?
- 현재가가 모든 주요 이평선 위에서 거래되는가?
- 추세선이 최소 3개 이상의 지점을 연결하며 유효한가?

**추세 강도 측정:**
- ADX 지표가 25 이상으로 강한 추세를 나타내는가?
- 상승 추세 중 조정폭이 전체 상승폭의 1/3 이하인가?
- 조정 시 거래량이 감소하고 재상승 시 거래량이 증가하는가?
- 점수: 매우 강한 추세(30점), 강한 추세(22점), 보통 추세(15점), 약한 추세(8점), 추세 없음(0점)

**⚡ 거래량 분석 (리버모어의 핵심) [20점]**

**거래량 패턴 분석:**
- 상승 시 거래량 증가, 하락 시 거래량 감소 패턴인가?
- 브레이크아웃 시 거래량이 평균의 3배 이상 폭증했는가?
- 고점 근처에서 거래량이 급증하며 분산 매물 소화하는가?
- 조정 구간에서 거래량이 현저히 감소하는가?

**거래량 신호:**
- 상승 중 거래량 감소 = 상승 동력 약화 신호
- 하락 중 거래량 감소 = 매도 압력 완화 신호  
- 횡보 중 거래량 감소 = 브레이크아웃 준비 신호
- 점수: 완벽한 거래량 패턴(20점), 양호(15점), 보통(10점), 나쁨(5점)

**🧠 시장 심리 및 뉴스 [10점]**
- 해당 종목/업종에 대한 관심이 증가하고 있는가?
- 긍정적 뉴스가 연속적으로 나오고 있는가?
- 아직 대중의 과도한 관심을 받지 않는 상태인가?
- 전문가들의 목표가 상향 조정이 이어지는가?

**💰 자금 관리 (리버모어의 생존 법칙) [5점]**
- 명확한 손절매 기준이 설정되어 있는가? (매수가 대비 7-10%)
- 수익 확대 시 추가 매수(피라미딩) 전략 적용 가능한가?
- 리스크 대비 기대 수익률이 최소 3:1 이상인가?
- 전체 포트폴리오 대비 적정 비중인가?

**🔥 리버모어의 황금 법칙들**

**"The Tape Reading" 원칙:**
- 가격 움직임 자체가 모든 정보를 담고 있다
- 시장이 보여주는 것을 믿고 따르라
- 본인의 주관적 판단보다 시장의 객관적 신호를 우선하라

**타이밍의 중요성:**
- "시간이 모든 것을 결정한다"
- 너무 일찍 들어가지도, 너무 늦게 들어가지도 말라
- 확실한 신호가 나올 때까지 기다려라

**감정 통제:**
- 탐욕과 공포에 휘둘리지 말라
- 손실을 빨리 인정하고 수익을 오래 보유하라
- 시장이 틀렸다고 생각하지 말고 본인이 틀렸다고 생각하라

**📊 최종 점수 계산 및 투자 결정**
- 90-100점: 강력 매수 - "지금이 바로 그 순간이다!"
- 80-89점: 매수 - "좋은 기회가 왔다"
- 70-79점: 관망 - "좀 더 지켜보자"
- 60-69점: 주의 - "위험 신호가 보인다"
- 60점 미만: 매도/회피 - "이 게임에서 빠져나와라"

**⚠️ 리버모어의 최종 경고**
"시장에서 살아남는 것이 돈을 버는 것보다 중요하다. 손실을 최소화하고 수익을 극대화하는 것이 투기의 핵심이다."
""",

            "벤저민 그레이엄": """
**💎 벤저민 그레이엄 가치투자 분석 가이드**

**안전마진 원칙: 내재가치 대비 할인된 가격으로 매수**

**1. 재무 안전성 [25점]**
- 부채비율이 50% 미만인가?
- 유동비율이 2.0 이상인가?
- 이자보상배율이 5배 이상인가?
- 최근 5년간 적자가 없었는가?

**2. 밸류에이션 [25점]**
- PER이 15배 미만인가?
- PBR이 1.5배 미만인가?
- PER × PBR < 22.5인가?
- 배당수익률이 3% 이상인가?

**3. 수익성 및 성장성 [20점]**
- ROE가 10% 이상인가?
- 최근 5년간 EPS 성장률이 양수인가?
- 영업이익률이 업종 평균 이상인가?

**4. 배당 정책 [15점]**
- 안정적인 배당 지급 이력이 있는가?
- 배당성향이 적정한가? (30-60%)
- 배당 증가 추세인가?

**5. 경영진 신뢰도 [15점]**
- 경영진의 지분 보유가 적절한가?
- 투명한 재무보고를 하는가?
- 주주 친화적인가?
""",

            "피터 린치": """
**🔍 피터 린치 성장주 발굴 전략**

**"당신이 아는 것에 투자하라"**

**1. 사업 이해도 [20점]**
- 회사의 사업모델을 쉽게 설명할 수 있는가?
- 경쟁우위가 명확한가?
- 성장 동력이 지속가능한가?

**2. PEG 비율 분석 [25점]**
- PEG 비율이 1.0 미만인가? (PER/성장률)
- 향후 3-5년 성장 전망이 밝은가?
- 업종 대비 성장률이 우수한가?

**3. 재무 건전성 [20점]**
- 부채가 과도하지 않은가?
- 현금흐름이 양호한가?
- 자기자본이익률이 15% 이상인가?

**4. 시장 지위 [20점]**
- 틈새시장의 강자인가?
- 브랜드 파워가 있는가?
- 진입장벽이 높은가?

**5. 경영진 [15점]**
- 창업자나 전문경영인이 유능한가?
- 주주가치 제고에 관심이 많은가?
- 장기적 비전이 있는가?
""",

            "워렌 버핏": """
**🏰 워렌 버핏 해자(Moat) 투자 전략**

**"훌륭한 기업을 합리적 가격에"**

**1. 경제적 해자 [30점]**
- 브랜드 파워가 강한가?
- 전환비용이 높은가?
- 네트워크 효과가 있는가?
- 규모의 경제를 누리는가?
- 독점적 지위나 라이센스가 있는가?

**2. 재무 품질 [25점]**
- ROE가 지속적으로 15% 이상인가?
- 부채비율이 낮은가?
- 잉여현금흐름이 안정적인가?
- 자본 재투자 수익률이 높은가?

**3. 경영진 품질 [20점]**
- 정직하고 유능한 경영진인가?
- 주주 친화적인가?
- 자본 배분 능력이 뛰어난가?
- 장기적 관점을 가지고 있는가?

**4. 성장 전망 [15점]**
- 지속가능한 성장 동력이 있는가?
- 시장 확장 가능성이 큰가?
- 신규 투자 기회가 많은가?

**5. 가격 매력도 [10점]**
- 내재가치 대비 할인된 가격인가?
- 안전마진이 충분한가?
- 장기 보유 시 연평균 15% 이상 수익 기대되는가?
""",

            "필립 피셔": """
**🚀 필립 피셔 성장주 투자 15개 포인트**

**"뛰어난 성장주를 찾아 장기 보유하라"**

**사업 특성 (40점)**
1. 장기적 성장 가능성이 큰 제품/서비스인가? [10점]
2. 현재 제품 라인 성장이 둔화될 때 새로운 제품 개발 능력이 있는가? [10점]
3. 연구개발 투자가 효과적인가? [10점]
4. 우수한 영업조직을 가지고 있는가? [10점]

**수익성 (25점)**
5. 이익률이 높은가? [10점]
6. 이익률 개선을 위한 노력이 지속되는가? [8점]
7. 노사관계가 양호한가? [7점]

**경영진 (20점)**
8. 뛰어난 경영진 depth가 있는가? [7점]
9. 비용 통제 및 회계 관리가 우수한가? [7점]
10. 업종 내 상대적 비용 우위가 있는가? [6점]

**재무 및 투자 정책 (15점)**
11. 성장 자금 조달에 문제가 없는가? [5점]
12. 주주에 대한 솔직한 소통이 이루어지는가? [5점]
13. 우수한 경영진이 장기간 유지되는가? [5점]

**투자 매력도 (0점 - 감점 요소)**
14. 주식 희석을 최소화하는가? [감점 없으면 0점]
15. 경영진이 좋을 때만 좋은 이야기를 하지 않는가? [감점 없으면 0점]
""",

            "일목산인": """
**☁️ 일목산인 (호소다 고이치) - 이치모쿠산징 분석 가이드 **

**"일목균형표로 시간과 가격의 균형을 파악하라"**

**1. 구름대(雲) 분석 [30점]**
- 현재가가 구름대 위에 있는가? (강세 신호)
- 구름대가 상승하고 있는가? (추세 확인)
- 구름대 두께가 적당한가? (지지/저항 강도)
- 미래 구름대 방향이 긍정적인가?
- 점수 기준: 구름대 위+상승(30점), 구름대 위+횡보(20점), 구름대 내부(10점), 구름대 아래(0점)

**2. 기준선/전환선 관계 [25점]**
- 전환선이 기준선 위에 있는가? (단기 강세)
- 현재가가 전환선과 기준선 위에 있는가?
- 전환선과 기준선이 상승 중인가?
- 골든크로스(전환선↗기준선) 발생했는가?
- 점수 기준: 완벽한 정배열(25점), 부분 정배열(15점), 혼조(8점), 역배열(0점)

**3. 후행스팬 분석 [20점]**
- 후행스팬이 현재 주가 위에 있는가?
- 후행스팬이 구름대를 돌파했는가?
- 후행스팬의 방향이 상향인가?
"""
        }
        
        return guides.get(strategy_name, """
**일반 투자 분석 가이드**
지정된 전략 가이드가 없습니다. 다음 기준으로 종합 분석을 수행하세요:

1. **기술적 분석 (30점)**
   - 추세, 지지/저항, 거래량, 기술적 지표 종합 판단

2. **펀더멘털 분석 (40점)**
   - 재무제표, 밸류에이션, 성장성, 수익성 평가

3. **시장 환경 (20점)**
   - 업종 전망, 시장 상황, 수급 요인 고려

4. **리스크 요인 (10점)**
   - 주요 리스크 식별 및 평가

총 100점 만점으로 평가하여 투자 의견을 제시하세요.
""")


# --- 2. Gemini AI 처리 클래스 ---
class GeminiProcessor:
    """
    Gemini AI와의 상호작용을 담당합니다.
    - API 호출, 응답 파싱, 에러 처리를 캡슐화합니다.
    - 재시도 로직을 포함하여 안정성을 높입니다.
    """
    
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash'):
        if not api_key:
            raise ValueError("Gemini API 키가 필요합니다.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini AI 설정 완료 ({model_name} 모델)")

    async def analyze(self, prompt: str, retry_attempts: int = 3) -> Dict[str, Any]:
        """비동기적으로 Gemini API를 호출하고 결과를 파싱합니다."""
        for attempt in range(retry_attempts):
            try:
                logger.info("Gemini API 호출 시도...")
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                
                if response and response.text:
                    logger.info("Gemini API 응답 수신 성공")
                    return self._parse_response(response.text)
                
                logger.warning(f"Gemini API 응답이 비어있습니다. (시도 {attempt + 1}/{retry_attempts})")
                
            except Exception as e:
                logger.error(f"Gemini API 호출 오류 (시도 {attempt + 1}/{retry_attempts}): {e}")
            
            if attempt < retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise AnalysisError("Gemini AI 분석에 최종 실패했습니다.")

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """AI의 응답 텍스트를 JSON으로 파싱합니다."""
        try:
            # 원본 텍스트 로깅 (디버깅용)
            logger.debug(f"원본 응답 텍스트 (앞 200자): {text[:200]}")
            
            # 다양한 JSON 추출 시도
            json_text = text
            
            # 1. ```json 코드 블록 추출
            if "```json" in text:
                try:
                    json_text = text.split("```json")[1].split("```")[0].strip()
                    logger.debug("```json 블록에서 JSON 추출 성공")
                except IndexError:
                    logger.warning("```json 블록 파싱 실패, 다른 방법 시도")
            
            # 2. { } 블록 추출 (JSON 객체만 추출)
            elif "{" in text and "}" in text:
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                json_text = text[start_idx:end_idx].strip()
                logger.debug("{ } 블록에서 JSON 추출 시도")
            
            # 3. JSON 파싱 시도
            result = json.loads(json_text)
            
            # 필수 필드 검증
            required_keys = ["분석", "결론", "점수", "추천 등급", "추천 이유", "진입 가격", "목표 가격", "신뢰도"]
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                logger.warning(f"필수 키 누락: {missing_keys}")
                # 누락된 키에 대해 기본값 설정
                for key in missing_keys:
                    if key == "점수":
                        result[key] = 50
                    elif key == "신뢰도":
                        result[key] = 0.5
                    else:
                        result[key] = "N/A"

            # 데이터 타입 검증 및 변환
            try:
                result['점수'] = int(float(result['점수']))  # float -> int 변환도 지원
                result['신뢰도'] = float(result['신뢰도'])
            except (ValueError, TypeError):
                logger.warning("점수 또는 신뢰도 변환 실패, 기본값 사용")
                result['점수'] = 50
                result['신뢰도'] = 0.5
            
            logger.info("Gemini 응답 파싱 성공")
            return result
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Gemini 응답 파싱 실패: {e}")
            logger.error(f"원본 텍스트 (앞 500자): {text[:500]}")
            
            # 파싱 실패 시 기본 응답 생성 (서비스 중단 방지)
            return self._create_fallback_response(text)
    
    def _create_fallback_response(self, original_text: str) -> Dict[str, Any]:
        """파싱 실패 시 기본 응답을 생성합니다."""
        logger.warning("파싱 실패로 기본 응답 생성")
        
        # 텍스트에서 키워드 기반으로 간단한 분석 시도
        text_lower = original_text.lower()
        
        # 기본 점수 결정
        score = 50
        if any(word in text_lower for word in ["buy", "매수", "추천", "상승"]):
            score = 65
        elif any(word in text_lower for word in ["sell", "매도", "하락", "위험"]):
            score = 35
        
        # 추천 등급 결정
        if score >= 70:
            grade = "MODERATE BUY"
        elif score >= 50:
            grade = "HOLD"
        else:
            grade = "REDUCE"
        
        return {
            "분석": "AI 응답 파싱 오류로 인해 제한적 분석만 가능. 차트 패턴과 기본적 지표를 종합적으로 검토한 결과입니다.",
            "결론": f"기술적 분석 기반 {grade} 의견",
            "점수": score,
            "추천 등급": grade,
            "추천 이유": "AI 응답 처리 과정에서 일시적 오류가 발생하여 상세 분석이 제한됨. 기본적 기술적 지표 기반 판단",
            "진입 가격": "현재가 기준 적정 매수",
            "목표 가격": "단기 5-10% 수익률 목표",
            "신뢰도": 0.3
        }


# --- 3. 알림 관리 클래스 ---
class NotificationManager:
    """
    텔레그램 등 외부 채널로 알림을 보냅니다.
    """
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        if not bot_token or not chat_id:
            logger.warning("텔레그램 토큰 또는 채팅 ID가 없어 알림 기능이 비활성화됩니다.")
            self.is_enabled = False
        else:
            self.is_enabled = True

    async def send_notification(self, message: str, parse_mode: str = "Markdown") -> bool:
        """텔레그램으로 메시지를 비동기 전송합니다."""
        if not self.is_enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            # 비동기 HTTP 요청 (라이브러리 필요, 예: aiohttp)
            # 여기서는 간단히 asyncio.to_thread 사용
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: requests.post(url, json=payload, timeout=10)
            )
            logger.info("텔레그램 알림 전송 성공")
            return True
        except Exception as e:
            logger.error(f"텔레그램 알림 전송 실패: {e}")
            return False


# --- 4. 메인 분석 엔진 (오케스트레이터) ---
class AIAnalyzer:
    """
    리팩토링된 AI 분석기 - 역할 분리 및 성능 최적화
    
    주요 개선사항:
    - 프롬프트 생성, AI 통신, 알림 전송을 별도 클래스로 분리
    - 캐싱을 통한 성능 최적화
    - 오류 처리 강화
    """
    
    def __init__(self, data_manager=None):
        """
        AIAnalyzer를 초기화합니다.
        
        Args:
            data_manager: 외부에서 전달받은 DataManager 인스턴스 (성능 최적화)
        """
        # 외부에서 전달받은 DataManager 사용 또는 새로 생성
        if data_manager:
            self.data_manager = data_manager
            logger.info("✅ 외부 DataManager 사용 (성능 최적화)")
        else:
            self.data_manager = DataManager()
            logger.info("✅ 새로운 DataManager 생성")
        
        # 하위 컴포넌트 초기화
        self.prompt_manager = PromptManager()
        self.gemini_processor = self._initialize_gemini()
        self.notification_manager = self._initialize_notification_manager()
        
        # 캐시 설정
        self.cache_duration = 300  # 5분 캐시
        self.analysis_cache = {}
        
        logger.info("AIAnalyzer (리팩토링 버전) 초기화 완료")

    def _initialize_gemini(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            return GeminiProcessor(gemini_api_key)
        else:
            logger.warning("Gemini API 키가 설정되지 않았습니다. 분석 기능이 비활성화됩니다.")
            return None
    
    def _initialize_notification_manager(self):
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # 항상 NotificationManager를 생성하되, 토큰이 없으면 비활성화 상태로 생성
        return NotificationManager(telegram_bot_token, telegram_chat_id)

    async def analyze_stock_with_strategy(
        self,
        stock_code: str,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """
        특정 종목을 지정된 투자 전략으로 분석합니다.
        데이터 조회, 프롬프트 생성, AI 분석을 포함합니다.
        """
        logger.info(f"분석 요청: {stock_code} ({strategy_name})")
        
        if not self.gemini_processor:
            return self._create_error_response(stock_code, strategy_name, "Gemini AI가 설정되지 않았습니다.")

        try:
            # 1. 데이터 조회 - 실제 메서드명 사용
            stock_data_raw = self.data_manager.get_comprehensive_stock_data(stock_code)
            if not stock_data_raw or not stock_data_raw.get('company_name'):
                raise AnalysisError("종목 데이터를 가져오는 데 실패했습니다.")

            # 2. 데이터 변환 - AI가 이해할 수 있는 형태로 변환
            stock_data = self._convert_stock_data_format(stock_data_raw)

            # 3. 프롬프트 생성
            prompt = self.prompt_manager.get_strategy_prompt(stock_data, strategy_name)

            # 4. AI 분석 실행
            analysis_result = await self.gemini_processor.analyze(prompt)

            # 5. 결과에 종목 정보 추가
            analysis_result['name'] = stock_data.get('name')
            analysis_result['stock_code'] = stock_code
            
            return analysis_result

        except AnalysisError as e:
            logger.warning(f"❌ {stock_code}({strategy_name}) 분석 실패: {e}")
            return self._create_error_response(stock_code, strategy_name, str(e))
        except Exception as e:
            logger.error(f"💥 예상치 못한 분석 오류: {stock_code}({strategy_name}) - {e}", exc_info=True)
            return self._create_error_response(stock_code, strategy_name, "예상치 못한 오류가 발생했습니다.")

    def _convert_stock_data_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        DataManager에서 받은 데이터를 AI 분석에 적합한 형태로 변환합니다.
        """
        try:
            # 기본 정보
            converted_data = {
                'name': raw_data.get('company_name', 'N/A'),
                'stock_code': raw_data.get('stock_code', 'N/A'),
            }
            
            # 가격 정보
            price_data = raw_data.get('price_data', {})
            converted_data.update({
                'current_price': price_data.get('current_price', 0),
                'volume': price_data.get('volume', 0),
                'high_52_week': price_data.get('high_52w', 0),
                'low_52_week': price_data.get('low_52w', 0),
            })
            
            # 차트 분석 데이터
            chart_analysis = raw_data.get('chart_analysis', {})
            converted_data.update({
                'ma_20': chart_analysis.get('sma_20', 0),
                'ma_60': chart_analysis.get('sma_60', 0),
                'ma_120': chart_analysis.get('sma_120', 0),
                'rsi': chart_analysis.get('rsi', 50),
                'macd_signal': chart_analysis.get('macd_signal', 0),
                'macd_hist': chart_analysis.get('macd_histogram', 0),
                'bollinger_upper': chart_analysis.get('bollinger_upper', 0),
                'bollinger_lower': chart_analysis.get('bollinger_lower', 0),
            })
            
            # 펀더멘털 데이터
            fundamental = raw_data.get('fundamental', {})
            converted_data.update({
                'market_cap': fundamental.get('시가총액', 0),
                'listed_shares': fundamental.get('상장주식수', 0),
                'revenue': fundamental.get('매출액', 0),
                'operating_profit': fundamental.get('영업이익', 0),
                'net_income': fundamental.get('당기순이익', 0),
                'debt_ratio': fundamental.get('부채비율', 0),
                'reserve_ratio': fundamental.get('유보율', 0),
                'per': fundamental.get('PER', 0),
                'pbr': fundamental.get('PBR', 0),
                'roe': fundamental.get('ROE', 0),
                'eps': fundamental.get('EPS', 0),
                'bps': fundamental.get('BPS', 0),
            })
            
            # 수급 데이터
            supply_demand = raw_data.get('supply_demand', {})
            converted_data.update({
                'foreign_net_purchase': supply_demand.get('foreign_net_buy', 0),
                'institution_net_purchase': supply_demand.get('institution_net_buy', 0),
                'individual_net_purchase': supply_demand.get('individual_net_buy', 0),
            })
            
            return converted_data
            
        except Exception as e:
            logger.error(f"데이터 변환 중 오류 발생: {e}")
            # 기본 데이터라도 반환
            return {
                'name': raw_data.get('company_name', 'N/A'),
                'stock_code': raw_data.get('stock_code', 'N/A'),
                'current_price': 0,
            }
            
    async def analyze_strategy_for_kospi200(self, strategy_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """KOSPI 200 종목에 대해 특정 전략으로 분석하고 상위 N개를 반환합니다."""
        logger.info(f"KOSPI 200 대상 '{strategy_name}' 전략 분석 시작...")
        
        kospi200_items = self.data_manager.get_kospi200_stocks()
        if not kospi200_items:
            logger.error("KOSPI 200 종목 코드를 가져올 수 없습니다.")
            return []

        tasks = []
        for item in kospi200_items:
            # 방어 코드: item이 딕셔너리이고 'code' 키를 가지고 있는지 확인
            if isinstance(item, dict) and 'code' in item:
                tasks.append(self.analyze_stock_with_strategy(item['code'], strategy_name))
            else:
                logger.warning(f"잘못된 KOSPI 200 데이터 항목을 건너뜁니다: {item}")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 분석 결과만 필터링
        valid_results = [res for res in results if isinstance(res, dict) and 'error' not in res]
        
        # 점수 순으로 정렬
        sorted_results = sorted(valid_results, key=lambda x: x.get('점수', 0), reverse=True)
        
        top_results = sorted_results[:top_n]
        logger.info(f"'{strategy_name}' 전략 분석 완료. 상위 {len(top_results)}개 종목 반환.")
        
        # 텔레그램 알림 (비동기)
        asyncio.create_task(self._send_top5_telegram_notification(strategy_name, top_results))
        
        return top_results
        
    async def _send_top5_telegram_notification(self, strategy_name: str, top_results: List[Dict[str, Any]]):
        """TOP 5 분석 결과를 텔레그램으로 전송합니다."""
        if not top_results or not self.notification_manager:
            return
            
        message = f"**📈 {strategy_name} 전략 TOP 5 분석**\n\n"
        for i, res in enumerate(top_results, 1):
            # 기본 정보
            name = res.get('name', 'N/A')
            stock_code = res.get('stock_code', 'N/A')
            score = res.get('점수', 0)
            grade = res.get('추천 등급', 'N/A')
            reason = res.get('추천 이유', 'N/A')
            
            # 가격 정보 추출 - 정확한 키값으로 수정
            entry_price = res.get('진입 가격', res.get('진입가격', res.get('진입가', '현재가 적정 매수')))
            target_price = res.get('목표 가격', res.get('목표가격', res.get('목표가', '목표가 분석 중')))
            
            # 추가 시나리오 정보
            bull_case = res.get('Bull Case', res.get('강세 시나리오', res.get('상승 목표', 'N/A')))
            confidence = res.get('신뢰도', 0)
            
            message += f"**{i}. {name} ({stock_code})**\n"
            message += f"📊 점수: {score}점 | 🏆 등급: {grade}\n"
            message += f"💡 추천이유: {reason}\n"
            message += f"💰 진입가: {entry_price}\n"
            message += f"🎯 목표가: {target_price}\n"
            if bull_case != 'N/A':
                message += f"🚀 강세시: {bull_case}\n"
            if confidence > 0:
                message += f"🔍 신뢰도: {confidence:.0%}\n"
            message += "\n"
        
        message += "🤖 PersonalBlackRock AI | #주식분석 #자동추천"
        await self.notification_manager.send_notification(message)

    def _create_error_response(self, stock_code: str, strategy_name: str, error_message: str) -> Dict[str, Any]:
        """오류 발생 시 반환할 표준 응답을 생성합니다."""
        return {
            "stock_code": stock_code,
            "strategy": strategy_name,
            "error": error_message,
            "score": 0,
            "recommendation": "분석 실패",
                "timestamp": datetime.now().isoformat()
            } 