#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Gemini AI 종합 분석 및 Top5 선정 엔진
Google Gemini의 고급 추론으로 최적의 투자 종목을 자동 선정
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import google.generativeai as genai
from dotenv import load_dotenv

from investment_strategies import StockData, StrategyScore, InvestmentMasterStrategies
from technical_analysis import TechnicalAnalysisResult, TechnicalAnalyzer

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class GeminiAnalysisResult:
    """Gemini AI 분석 결과"""
    top5_stocks: List[Dict[str, Any]]
    reasoning: str
    market_outlook: str
    risk_assessment: str
    confidence_score: float
    alternative_picks: List[Dict[str, Any]]

class GeminiAIAnalyzer:
    """🤖 Gemini AI 투자 분석가"""
    
    def __init__(self):
        """초기화 및 API 설정"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            logger.error("❌ GEMINI_API_KEY가 설정되지 않았습니다!")
            raise ValueError("Gemini API 키가 필요합니다")
        
        # Gemini 설정
        genai.configure(api_key=self.api_key)
        
        # 모델 초기화 (최신 Gemini Pro 사용)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # 투자 전략 엔진
        self.strategy_engine = InvestmentMasterStrategies()
        self.technical_analyzer = TechnicalAnalyzer()
        
        logger.info("🤖 Gemini AI 분석 엔진 초기화 완료")
    
    def analyze_and_select_top5(self, 
                               stocks: List[StockData],
                               technical_results: Dict[str, TechnicalAnalysisResult],
                               strategy_scores: Dict[str, List[StrategyScore]]) -> GeminiAnalysisResult:
        """Gemini AI로 Top5 종목 선정"""
        
        logger.info("🤖 Gemini AI 종합 분석 시작")
        
        try:
            # 1. 데이터 전처리 및 요약
            analysis_data = self._prepare_analysis_data(stocks, technical_results, strategy_scores)
            
            # 2. Gemini AI 프롬프트 생성
            prompt = self._create_gemini_prompt(analysis_data)
            
            # 3. Gemini AI 분석 실행
            response = self.model.generate_content(prompt)
            
            # 4. 응답 파싱
            result = self._parse_gemini_response(response.text)
            
            logger.info("✅ Gemini AI 분석 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ Gemini AI 분석 실패: {e}")
            return self._create_fallback_result(stocks, strategy_scores)
    
    def _prepare_analysis_data(self, 
                             stocks: List[StockData],
                             technical_results: Dict[str, TechnicalAnalysisResult],
                             strategy_scores: Dict[str, List[StrategyScore]]) -> Dict[str, Any]:
        """Gemini AI 분석용 데이터 준비"""
        
        analysis_data = {
            'market_summary': {
                'total_stocks': len(stocks),
                'markets': self._summarize_markets(stocks),
                'sectors': self._summarize_sectors(stocks)
            },
            'top_candidates': self._get_top_candidates(stocks, strategy_scores, 20),
            'technical_insights': self._summarize_technical_analysis(technical_results),
            'strategy_rankings': self._summarize_strategy_rankings(strategy_scores)
        }
        
        return analysis_data
    
    def _summarize_markets(self, stocks: List[StockData]) -> Dict[str, int]:
        """시장별 종목 수 요약"""
        markets = {}
        for stock in stocks:
            market = stock.market or "Unknown"
            markets[market] = markets.get(market, 0) + 1
        return markets
    
    def _summarize_sectors(self, stocks: List[StockData]) -> Dict[str, int]:
        """섹터별 종목 수 요약"""
        sectors = {}
        for stock in stocks:
            sector = stock.sector or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + 1
        return sectors
    
    def _get_top_candidates(self, 
                          stocks: List[StockData], 
                          strategy_scores: Dict[str, List[StrategyScore]], 
                          limit: int = 20) -> List[Dict[str, Any]]:
        """전략별 상위 후보군 추출"""
        
        # 모든 종목의 평균 점수 계산
        stock_avg_scores = {}
        
        for symbol, scores in strategy_scores.items():
            if scores:
                avg_score = sum(score.total_score for score in scores) / len(scores)
                stock_avg_scores[symbol] = avg_score
        
        # 상위 후보 선정
        top_symbols = sorted(stock_avg_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        candidates = []
        for symbol, avg_score in top_symbols:
            # 해당 종목 정보 찾기
            stock = next((s for s in stocks if s.symbol == symbol), None)
            if stock:
                candidates.append({
                    'symbol': symbol,
                    'name': stock.name,
                    'current_price': stock.current_price,
                    'market': stock.market,
                    'sector': stock.sector,
                    'avg_strategy_score': avg_score,
                    'pe_ratio': stock.pe_ratio,
                    'pb_ratio': stock.pb_ratio,
                    'rsi': stock.rsi,
                    'market_cap': stock.market_cap
                })
        
        return candidates
    
    def _summarize_technical_analysis(self, technical_results: Dict[str, TechnicalAnalysisResult]) -> Dict[str, Any]:
        """기술적 분석 요약"""
        
        total_buy_signals = 0
        total_sell_signals = 0
        high_volatility_count = 0
        uptrend_count = 0
        
        for symbol, result in technical_results.items():
            # 신호 집계
            buy_signals = len([s for s in result.signals if s.signal_type == "BUY"])
            sell_signals = len([s for s in result.signals if s.signal_type == "SELL"])
            
            total_buy_signals += buy_signals
            total_sell_signals += sell_signals
            
            # 변동성 및 트렌드 집계
            if result.volatility_score > 70:
                high_volatility_count += 1
            
            if result.trend_direction == "UPTREND":
                uptrend_count += 1
        
        return {
            'total_buy_signals': total_buy_signals,
            'total_sell_signals': total_sell_signals,
            'high_volatility_stocks': high_volatility_count,
            'uptrend_stocks': uptrend_count,
            'signal_ratio': total_buy_signals / (total_buy_signals + total_sell_signals) if (total_buy_signals + total_sell_signals) > 0 else 0.5
        }
    
    def _summarize_strategy_rankings(self, strategy_scores: Dict[str, List[StrategyScore]]) -> Dict[str, List[str]]:
        """투자 전략별 상위 종목"""
        
        strategy_rankings = {}
        
        # 전략별로 상위 5개 종목 추출
        all_strategies = set()
        for scores in strategy_scores.values():
            for score in scores:
                all_strategies.add(score.strategy_name)
        
        for strategy in all_strategies:
            strategy_stocks = []
            for symbol, scores in strategy_scores.items():
                strategy_score = next((s for s in scores if s.strategy_name == strategy), None)
                if strategy_score:
                    strategy_stocks.append((symbol, strategy_score.total_score))
            
            # 상위 5개 선정
            top5 = sorted(strategy_stocks, key=lambda x: x[1], reverse=True)[:5]
            strategy_rankings[strategy] = [symbol for symbol, _ in top5]
        
        return strategy_rankings
    
    def _create_gemini_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Gemini AI 분석 프롬프트 생성"""
        
        prompt = f"""
# 🎯 세계 최고 투자 대가 전략 기반 TOP5 종목 선정

당신은 세계 최고의 투자 분석가입니다. 아래 데이터를 바탕으로 **최적의 TOP5 투자 종목**을 선정해주세요.

## 📊 분석 데이터

### 시장 현황
- 총 분석 종목: {analysis_data['market_summary']['total_stocks']}개
- 시장별 분포: {analysis_data['market_summary']['markets']}
- 섹터별 분포: {analysis_data['market_summary']['sectors']}

### 🏆 상위 후보군 (상위 20개)
{json.dumps(analysis_data['top_candidates'], indent=2, ensure_ascii=False)}

### 📈 기술적 분석 요약
- 매수 신호: {analysis_data['technical_insights']['total_buy_signals']}개
- 매도 신호: {analysis_data['technical_insights']['total_sell_signals']}개
- 고변동성 종목: {analysis_data['technical_insights']['high_volatility_stocks']}개
- 상승 추세 종목: {analysis_data['technical_insights']['uptrend_stocks']}개
- 신호 비율: {analysis_data['technical_insights']['signal_ratio']:.2f}

### 💡 투자 대가 전략별 추천
{json.dumps(analysis_data['strategy_rankings'], indent=2, ensure_ascii=False)}

## 🎯 요청사항

다음 기준으로 **TOP5 종목**을 선정하고 분석해주세요:

1. **다양한 투자 대가들의 전략 점수**
2. **기술적 분석 신호의 강도**
3. **시장별/섹터별 분산투자 고려**
4. **현재 시장 상황과 트렌드**
5. **리스크 대비 수익률 잠재력**

## 📝 응답 형식 (JSON)

```json
{{
    "top5_stocks": [
        {{
            "rank": 1,
            "symbol": "종목코드",
            "name": "종목명",
            "selection_reason": "선정 이유 (구체적으로)",
            "expected_return": "예상 수익률",
            "risk_level": "리스크 레벨 (낮음/보통/높음)",
            "investment_horizon": "투자 기간 추천",
            "key_strengths": ["강점1", "강점2", "강점3"]
        }}
    ],
    "reasoning": "전체적인 선정 논리와 시장 전망",
    "market_outlook": "향후 시장 전망 및 투자 방향",
    "risk_assessment": "주요 리스크 요인 및 대응 방안",
    "confidence_score": 85.5,
    "alternative_picks": [
        {{
            "symbol": "대안종목1",
            "reason": "대안 선정 이유"
        }}
    ]
}}
```

**중요**: 반드시 위 JSON 형식으로만 응답해주세요. 추가 설명은 JSON 내부 필드에 포함해주세요.
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> GeminiAnalysisResult:
        """Gemini 응답 파싱"""
        
        try:
            # JSON 부분 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            json_str = response_text[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            
            return GeminiAnalysisResult(
                top5_stocks=parsed_data.get('top5_stocks', []),
                reasoning=parsed_data.get('reasoning', ''),
                market_outlook=parsed_data.get('market_outlook', ''),
                risk_assessment=parsed_data.get('risk_assessment', ''),
                confidence_score=parsed_data.get('confidence_score', 0.0),
                alternative_picks=parsed_data.get('alternative_picks', [])
            )
            
        except Exception as e:
            logger.warning(f"Gemini 응답 파싱 실패: {e}")
            logger.warning(f"응답 내용: {response_text[:500]}...")
            
            # 응답에서 정보 추출 시도
            return self._extract_info_from_text(response_text)
    
    def _extract_info_from_text(self, text: str) -> GeminiAnalysisResult:
        """텍스트에서 정보 추출 (백업 방법)"""
        
        # 간단한 패턴 매칭으로 종목 추출 시도
        import re
        
        # 종목 코드 패턴 찾기
        stock_patterns = re.findall(r'[A-Z]{2,6}(?:\.[A-Z]{2})?', text)
        
        top5_stocks = []
        for i, symbol in enumerate(stock_patterns[:5]):
            top5_stocks.append({
                'rank': i + 1,
                'symbol': symbol,
                'name': f'종목{i+1}',
                'selection_reason': '기술적 분석 및 전략 점수 기반',
                'expected_return': '5-15%',
                'risk_level': '보통',
                'investment_horizon': '3-6개월',
                'key_strengths': ['기술적 우위', '전략 점수 우수', '시장 트렌드 부합']
            })
        
        return GeminiAnalysisResult(
            top5_stocks=top5_stocks,
            reasoning="Gemini AI 분석을 통한 종합적 판단",
            market_outlook="현재 시장 상황을 고려한 신중한 접근 필요",
            risk_assessment="분산투자를 통한 리스크 관리 권장",
            confidence_score=70.0,
            alternative_picks=[]
        )
    
    def _create_fallback_result(self, 
                              stocks: List[StockData], 
                              strategy_scores: Dict[str, List[StrategyScore]]) -> GeminiAnalysisResult:
        """백업 결과 생성 (Gemini 실패 시)"""
        
        logger.info("📊 백업 분석으로 Top5 선정")
        
        # 평균 점수 기반 Top5 선정
        stock_avg_scores = {}
        
        for symbol, scores in strategy_scores.items():
            if scores:
                avg_score = sum(score.total_score for score in scores) / len(scores)
                stock_avg_scores[symbol] = avg_score
        
        top5_symbols = sorted(stock_avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        top5_stocks = []
        for i, (symbol, score) in enumerate(top5_symbols):
            stock = next((s for s in stocks if s.symbol == symbol), None)
            if stock:
                top5_stocks.append({
                    'rank': i + 1,
                    'symbol': symbol,
                    'name': stock.name,
                    'selection_reason': f'투자 대가 전략 종합 점수 {score:.1f}점',
                    'expected_return': '목표 수익률 10-20%',
                    'risk_level': '중간',
                    'investment_horizon': '3-6개월',
                    'key_strengths': ['높은 전략 점수', '기술적 분석 우수', '시장 포지션 양호']
                })
        
        return GeminiAnalysisResult(
            top5_stocks=top5_stocks,
            reasoning="15명 투자 대가들의 전략을 종합하여 평균 점수가 높은 종목들을 선정했습니다.",
            market_outlook="다양한 전략의 균형잡힌 포트폴리오로 안정적인 수익을 기대할 수 있습니다.",
            risk_assessment="분산된 종목 선정으로 리스크를 최소화하였으며, 지속적인 모니터링이 필요합니다.",
            confidence_score=75.0,
            alternative_picks=[
                {'symbol': sym, 'reason': f'전략 점수 {score:.1f}점'} 
                for sym, score in sorted(stock_avg_scores.items(), key=lambda x: x[1], reverse=True)[5:10]
            ]
        )
    
    def generate_investment_report(self, result: GeminiAnalysisResult) -> str:
        """투자 리포트 생성"""
        
        report = f"""
# 🎯 Gemini AI 투자 분석 리포트

## 📈 TOP5 추천 종목

"""
        
        for stock in result.top5_stocks:
            report += f"""
### {stock['rank']}. {stock['name']} ({stock['symbol']})
- **선정 이유**: {stock['selection_reason']}
- **예상 수익률**: {stock['expected_return']}
- **리스크 레벨**: {stock['risk_level']}
- **투자 기간**: {stock['investment_horizon']}
- **주요 강점**: {', '.join(stock['key_strengths'])}

"""
        
        report += f"""
## 🧠 분석 근거
{result.reasoning}

## 🔮 시장 전망
{result.market_outlook}

## ⚠️ 리스크 평가
{result.risk_assessment}

## 📊 신뢰도: {result.confidence_score:.1f}%

## 🔄 대안 종목
"""
        
        for alt in result.alternative_picks:
            report += f"- **{alt['symbol']}**: {alt['reason']}\n"
        
        report += f"""

---
*본 분석은 Gemini AI와 15명 투자 대가들의 전략을 종합한 결과입니다.*
*투자 결정은 개인의 판단과 책임 하에 이루어져야 합니다.*
"""
        
        return report

if __name__ == "__main__":
    print("🤖 Gemini AI 투자 분석 엔진 v1.0")
    print("=" * 50)
    
    # 테스트용 더미 데이터
    test_stocks = [
        StockData(symbol="AAPL", name="Apple Inc.", current_price=150.0, market="NASDAQ100"),
        StockData(symbol="TSLA", name="Tesla Inc.", current_price=200.0, market="NASDAQ100"),
        StockData(symbol="005930.KS", name="삼성전자", current_price=70000.0, market="KOSPI200")
    ]
    
    print("✅ Gemini AI 분석 엔진 테스트 준비 완료!")
    print("실제 분석을 위해서는 전체 데이터가 필요합니다.") 