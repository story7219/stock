"""
Gemini AI 분석 모듈
투자 대가 전략별 후보군을 종합 분석하여 Top5 종목 자동 선정
"""

import asyncio
import logging
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
from .data_collector import StockData
from .strategies import StrategyScore

logger = logging.getLogger(__name__)

@dataclass
class Top5Selection:
    """Top5 선정 결과 클래스"""
    symbol: str
    name: str
    rank: int
    final_score: float
    selection_reason: str
    strategy_scores: Dict[str, float]
    technical_analysis: str
    risk_assessment: str
    gemini_reasoning: str

@dataclass
class GeminiAnalysisResult:
    """Gemini AI 분석 결과"""
    top5_selections: List[Top5Selection]
    analysis_summary: str
    market_outlook: str
    risk_warnings: List[str]
    alternative_candidates: List[str]
    confidence_score: float
    analysis_timestamp: datetime

class GeminiAnalyzer:
    """Gemini AI 기반 종목 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Gemini AI 분석기 초기화
        
        Args:
            api_key: Google Gemini API 키 (환경변수에서 자동 로드)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        # Gemini AI 설정
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 분석 파라미터
        self.analysis_config = {
            'temperature': 0.7,  # 창의성 수준
            'top_p': 0.9,       # 다양성 수준
            'max_output_tokens': 4000,
            'candidate_count': 1
        }
        
    async def analyze_candidates(self, 
                               strategy_results: Dict[str, List[StrategyScore]], 
                               market_data: Dict[str, List[StockData]]) -> GeminiAnalysisResult:
        """
        전략별 후보군을 종합 분석하여 Top5 종목 선정
        
        Args:
            strategy_results: 각 전략별 점수 결과
            market_data: 시장별 원본 데이터
            
        Returns:
            GeminiAnalysisResult: Gemini AI 분석 결과
        """
        logger.info("Gemini AI 종합 분석 시작")
        
        try:
            # 1. 분석용 데이터 준비
            analysis_data = self._prepare_analysis_data(strategy_results, market_data)
            
            # 2. Gemini AI 프롬프트 생성
            prompt = self._create_analysis_prompt(analysis_data)
            
            # 3. Gemini AI 분석 실행
            response = await self._call_gemini_api(prompt)
            
            # 4. 응답 파싱 및 결과 생성
            analysis_result = self._parse_gemini_response(response, analysis_data)
            
            logger.info("Gemini AI 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Gemini AI 분석 실패: {e}")
            # 백업 분석 수행
            return self._fallback_analysis(strategy_results, market_data)
    
    def _prepare_analysis_data(self, 
                             strategy_results: Dict[str, List[StrategyScore]], 
                             market_data: Dict[str, List[StockData]]) -> Dict[str, Any]:
        """분석 데이터 준비 - 고품질 데이터셋 활용"""
        try:
            logger.info("🧠 Gemini AI 분석용 고품질 데이터 준비 중...")
            
            # DataCollector의 고품질 데이터셋 활용
            from .data_collector import DataCollector
            data_collector = DataCollector()
            
            # 고품질 데이터셋 생성
            gemini_dataset = data_collector.prepare_gemini_dataset(market_data)
            
            # 전략 결과 통합
            strategy_candidates = {}
            all_symbols_with_scores = {}
            
            for strategy_name, scores in strategy_results.items():
                strategy_candidates[strategy_name] = []
                for score in scores[:20]:  # 각 전략별 상위 20개
                    candidate_info = {
                        'symbol': score.symbol,
                        'name': score.name,
                        'total_score': score.total_score,
                        'individual_scores': score.individual_scores,
                        'analysis_reason': score.analysis_reason
                    }
                    strategy_candidates[strategy_name].append(candidate_info)
                    
                    # 심볼별 전략 점수 수집
                    if score.symbol not in all_symbols_with_scores:
                        all_symbols_with_scores[score.symbol] = {}
                    all_symbols_with_scores[score.symbol][strategy_name] = score.total_score
            
            # 고급 분석 데이터 구성
            analysis_data = {
                # 기존 전략 결과
                'strategy_candidates': strategy_results,
                'strategy_summary': strategy_candidates,
                
                # 고품질 데이터셋 (Gemini AI 최적화)
                'gemini_dataset': gemini_dataset,
                
                # 종합 분석용 데이터
                'comprehensive_analysis': {
                    'total_stocks_analyzed': gemini_dataset.get('total_stocks', 0),
                    'markets_covered': gemini_dataset.get('markets', []),
                    'data_quality_score': gemini_dataset.get('data_quality_summary', {}).get('avg_quality_score', 0),
                    'market_statistics': gemini_dataset.get('market_statistics', {}),
                    'technical_patterns': gemini_dataset.get('technical_patterns', {}),
                    'top_performers': gemini_dataset.get('top_performers', {}),
                    'sector_analysis': gemini_dataset.get('sector_analysis', {})
                },
                
                # 투자 전략 매핑
                'strategy_mapping': {
                    'buffett_candidates': [s for s in strategy_candidates.get('buffett', [])],
                    'lynch_candidates': [s for s in strategy_candidates.get('lynch', [])],
                    'graham_candidates': [s for s in strategy_candidates.get('graham', [])]
                },
                
                # 종목별 종합 점수
                'symbol_comprehensive_scores': all_symbols_with_scores,
                
                # 분석 메타데이터
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'data_freshness': 'real-time',
                    'quality_threshold': 70.0,
                    'analysis_scope': 'kospi200_nasdaq100_sp500',
                    'optimization_target': 'gemini_ai_analysis'
                }
            }
            
            logger.info(f"✅ 고품질 분석 데이터 준비 완료: {len(all_symbols_with_scores)}개 종목, 평균 품질 점수 {gemini_dataset.get('data_quality_summary', {}).get('avg_quality_score', 0):.1f}")
            return analysis_data
            
        except Exception as e:
            logger.error(f"분석 데이터 준비 실패: {e}")
            # 백업 데이터 준비
            return {
                'strategy_candidates': strategy_results,
                'market_data_summary': self._create_market_summary(market_data),
                'error': str(e)
            }
    
    def _create_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Gemini AI 분석 프롬프트 생성 - 고품질 데이터 활용"""
        
        # 고품질 데이터셋 추출
        gemini_dataset = analysis_data.get('gemini_dataset', {})
        comprehensive_analysis = analysis_data.get('comprehensive_analysis', {})
        strategy_mapping = analysis_data.get('strategy_mapping', {})
        
        prompt = f"""
🚀 **Ultra HTS v5.0 - Gemini AI 고급 종목 분석 및 Top5 선정**

당신은 세계 최고 수준의 AI 투자 분석가입니다. 아래 고품질 데이터를 바탕으로 최적의 Top5 종목을 선정해주세요.

## 📊 **고품질 데이터셋 정보**
- **분석 대상**: {comprehensive_analysis.get('total_stocks_analyzed', 0)}개 종목
- **커버 시장**: {', '.join(comprehensive_analysis.get('markets_covered', []))}
- **데이터 품질 점수**: {comprehensive_analysis.get('data_quality_score', 0):.1f}/100
- **분석 시점**: {analysis_data.get('analysis_metadata', {}).get('timestamp', 'N/A')}

## 🎯 **투자 대가 전략 결과**

### 워런 버핏 전략 (가치투자)
{self._format_strategy_candidates(strategy_mapping.get('buffett_candidates', []))}

### 피터 린치 전략 (성장투자)  
{self._format_strategy_candidates(strategy_mapping.get('lynch_candidates', []))}

### 벤저민 그레이엄 전략 (가치투자)
{self._format_strategy_candidates(strategy_mapping.get('graham_candidates', []))}

## 📈 **시장 통계 및 기술적 패턴**

### 시장별 현황
{self._format_market_statistics(comprehensive_analysis.get('market_statistics', {}))}

### 기술적 패턴 분석
- **강세 신호**: {comprehensive_analysis.get('technical_patterns', {}).get('bullish_signals', 0)}개 종목
- **약세 신호**: {comprehensive_analysis.get('technical_patterns', {}).get('bearish_signals', 0)}개 종목
- **강한 모멘텀**: {', '.join(comprehensive_analysis.get('technical_patterns', {}).get('strong_momentum', [])[:5])}
- **과매도 기회**: {', '.join(comprehensive_analysis.get('technical_patterns', {}).get('oversold_opportunities', [])[:5])}

### 상위 성과 종목
{self._format_top_performers(comprehensive_analysis.get('top_performers', {}))}

## 🎯 **분석 지침**

### 선정 기준 (우선순위)
1. **기술적 분석 우선**: RSI, MACD, 볼린저밴드, 스토캐스틱 등 기술적 지표 종합 평가
2. **모멘텀 분석**: 단기/중기 가격 모멘텀 및 거래량 패턴
3. **리스크 관리**: 변동성 대비 수익률, 베타 계수 고려
4. **전략 다각화**: 워런 버핏, 피터 린치, 벤저민 그레이엄 전략 균형 반영
5. **시장 상황 고려**: 현재 시장 환경에 최적화된 종목 선정

### 필수 고려사항
- 재무정보 제외, 순수 기술적 분석 기반 선정
- 각 선정 종목의 구체적 기술적 근거 제시
- 리스크 요인 및 대안 후보 제시
- 포트폴리오 다각화 고려 (시장/섹터 분산)

## 📋 **요구 응답 형식**

반드시 아래 JSON 형식으로만 응답하세요:

```json
{{
  "top5_selections": [
    {{
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "rank": 1,
      "final_score": 92.5,
      "selection_reason": "강력한 기술적 지표와 모멘텀 우수",
      "technical_analysis": "RSI 65.2 (적정), MACD 상승 크로스오버, 볼린저밴드 상단 돌파",
      "risk_assessment": "베타 1.2, 변동성 중간 수준, 단기 조정 가능성",
      "gemini_reasoning": "현재 시장 환경에서 기술적 우위와 모멘텀을 동시에 보유한 최적 종목"
    }},
    // ... 나머지 4개 종목
  ],
  "analysis_summary": "현재 시장은 기술적 분석 관점에서 선별적 강세를 보이고 있으며, 모멘텀과 기술적 지표가 우수한 종목들이 부각되고 있습니다...",
  "market_outlook": "향후 3-6개월 시장 전망: 기술적 분석 기반으로 볼 때...",
  "risk_warnings": ["금리 변동성", "지정학적 리스크", "기술적 조정 가능성"],
  "alternative_candidates": ["MSFT", "GOOGL", "005930.KS", "NVDA", "TSLA"],
  "confidence_score": 87.5
}}
```

## 🔥 **핵심 미션**
고품질 데이터와 투자 대가들의 전략을 바탕으로, 현재 시장에서 최고의 성과를 낼 수 있는 Top5 종목을 선정하고, 그 이유를 명확히 제시해주세요. 당신의 분석이 투자자들의 성공을 좌우합니다!
"""
        
        return prompt
    
    def _format_strategy_candidates(self, candidates: List[Dict]) -> str:
        """전략별 후보 종목 포맷팅"""
        if not candidates:
            return "- 해당 전략 후보 없음"
        
        formatted = []
        for i, candidate in enumerate(candidates[:5], 1):  # 상위 5개만
            formatted.append(f"- {i}. {candidate['symbol']} ({candidate['name']}) - 점수: {candidate['total_score']:.1f}")
        
        return "\n".join(formatted)
    
    def _format_market_statistics(self, market_stats: Dict) -> str:
        """시장 통계 포맷팅"""
        if not market_stats:
            return "- 시장 통계 정보 없음"
        
        formatted = []
        for market, stats in market_stats.items():
            formatted.append(f"- **{market.upper()}**: {stats.get('total_stocks', 0)}개 종목, 평균 RSI: {stats.get('avg_rsi', 0):.1f}")
        
        return "\n".join(formatted)
    
    def _format_top_performers(self, top_performers: Dict) -> str:
        """상위 성과 종목 포맷팅"""
        if not top_performers:
            return "- 상위 성과 종목 정보 없음"
        
        formatted = []
        
        # 수익률 상위 종목
        top_returns = top_performers.get('top_20_returns', [])
        if top_returns:
            symbols = [stock['symbol'] for stock in top_returns]
            formatted.append(f"- **수익률 상위**: {', '.join(symbols)}")
        
        # RSI 적정 종목
        good_rsi = top_performers.get('good_rsi_stocks', [])
        if good_rsi:
            symbols = [stock['symbol'] for stock in good_rsi[:5]]
            formatted.append(f"- **RSI 적정 구간**: {', '.join(symbols)}")
        
        return "\n".join(formatted) if formatted else "- 상위 성과 종목 정보 없음"
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출"""
        try:
            # 비동기 처리를 위한 래퍼
            def _sync_generate():
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.analysis_config['temperature'],
                        top_p=self.analysis_config['top_p'],
                        max_output_tokens=self.analysis_config['max_output_tokens'],
                        candidate_count=self.analysis_config['candidate_count']
                    )
                )
                return response.text
            
            # 별도 스레드에서 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response_text = await asyncio.get_event_loop().run_in_executor(
                    executor, _sync_generate
                )
            
            logger.info("Gemini API 호출 성공")
            return response_text
            
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            raise
    
    def _parse_gemini_response(self, response: str, analysis_data: Dict[str, Any]) -> GeminiAnalysisResult:
        """Gemini 응답 파싱"""
        try:
            # JSON 응답 파싱
            response_data = json.loads(response)
            
            # Top5 선정 결과 생성
            top5_selections = []
            for selection_data in response_data.get('top5_selections', []):
                
                # 전략별 점수 수집
                symbol = selection_data['symbol']
                strategy_scores = {}
                for strategy_name, candidates in analysis_data['strategy_candidates'].items():
                    for candidate in candidates:
                        if candidate.symbol == symbol:
                            strategy_scores[strategy_name] = candidate.total_score
                            break
                
                top5_selection = Top5Selection(
                    symbol=selection_data['symbol'],
                    name=selection_data['name'],
                    rank=selection_data['rank'],
                    final_score=selection_data['final_score'],
                    selection_reason=selection_data['selection_reason'],
                    strategy_scores=strategy_scores,
                    technical_analysis=selection_data['technical_analysis'],
                    risk_assessment=selection_data['risk_assessment'],
                    gemini_reasoning=selection_data['gemini_reasoning']
                )
                top5_selections.append(top5_selection)
            
            # 분석 결과 생성
            analysis_result = GeminiAnalysisResult(
                top5_selections=top5_selections,
                analysis_summary=response_data.get('analysis_summary', ''),
                market_outlook=response_data.get('market_outlook', ''),
                risk_warnings=response_data.get('risk_warnings', []),
                alternative_candidates=response_data.get('alternative_candidates', []),
                confidence_score=response_data.get('confidence_score', 0.0),
                analysis_timestamp=datetime.now()
            )
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Gemini 응답 JSON 파싱 실패: {e}")
            # 백업 분석 수행
            return self._fallback_analysis(analysis_data['strategy_candidates'], {})
        except Exception as e:
            logger.error(f"Gemini 응답 처리 실패: {e}")
            return self._fallback_analysis(analysis_data['strategy_candidates'], {})
    
    def _fallback_analysis(self, 
                          strategy_results: Dict[str, List[StrategyScore]], 
                          market_data: Dict[str, List[StockData]]) -> GeminiAnalysisResult:
        """백업 분석 (Gemini API 실패 시)"""
        logger.warning("백업 분석 모드로 전환")
        
        # 간단한 점수 기반 Top5 선정
        all_candidates = []
        for strategy_name, scores in strategy_results.items():
            for score in scores[:10]:  # 각 전략별 상위 10개
                all_candidates.append((score, strategy_name))
        
        # 점수 순으로 정렬
        all_candidates.sort(key=lambda x: x[0].total_score, reverse=True)
        
        # Top5 선정
        top5_selections = []
        seen_symbols = set()
        rank = 1
        
        for candidate, strategy_name in all_candidates:
            if candidate.symbol not in seen_symbols and rank <= 5:
                top5_selection = Top5Selection(
                    symbol=candidate.symbol,
                    name=candidate.name,
                    rank=rank,
                    final_score=candidate.total_score,
                    selection_reason=f"{strategy_name} 전략에서 높은 점수 획득",
                    strategy_scores={strategy_name: candidate.total_score},
                    technical_analysis="기술적 분석 데이터 기반 선정",
                    risk_assessment="일반적인 주식 투자 리스크 적용",
                    gemini_reasoning="백업 분석 모드 - 점수 기반 자동 선정"
                )
                top5_selections.append(top5_selection)
                seen_symbols.add(candidate.symbol)
                rank += 1
        
        return GeminiAnalysisResult(
            top5_selections=top5_selections,
            analysis_summary="백업 분석 모드로 수행된 결과입니다.",
            market_outlook="상세한 시장 전망은 Gemini AI 분석이 필요합니다.",
            risk_warnings=["일반적인 주식 투자 리스크"],
            alternative_candidates=[],
            confidence_score=60.0,  # 낮은 신뢰도
            analysis_timestamp=datetime.now()
        )

class Top5Selector:
    """Top5 종목 선정 관리자"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_analyzer = GeminiAnalyzer(gemini_api_key)
        
    async def select_top5_stocks(self, 
                               strategy_results: Dict[str, List[StrategyScore]], 
                               market_data: Dict[str, List[StockData]]) -> GeminiAnalysisResult:
        """
        투자 전략 결과를 바탕으로 Top5 종목 선정
        
        Args:
            strategy_results: 전략별 점수 결과
            market_data: 시장 데이터
            
        Returns:
            GeminiAnalysisResult: Top5 선정 결과
        """
        logger.info("Top5 종목 선정 프로세스 시작")
        
        try:
            # Gemini AI 분석 수행
            result = await self.gemini_analyzer.analyze_candidates(strategy_results, market_data)
            
            # 결과 검증
            if len(result.top5_selections) < 5:
                logger.warning(f"Top5 미만 선정됨: {len(result.top5_selections)}개")
            
            # 로그 출력
            logger.info("=== Top5 종목 선정 결과 ===")
            for selection in result.top5_selections:
                logger.info(f"{selection.rank}. {selection.symbol} ({selection.name}) - 점수: {selection.final_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"Top5 선정 실패: {e}")
            raise
    
    def export_results(self, result: GeminiAnalysisResult, output_format: str = 'json') -> str:
        """결과 내보내기"""
        if output_format == 'json':
            return self._export_json(result)
        elif output_format == 'markdown':
            return self._export_markdown(result)
        else:
            raise ValueError(f"지원하지 않는 형식: {output_format}")
    
    def _export_json(self, result: GeminiAnalysisResult) -> str:
        """JSON 형식으로 내보내기"""
        export_data = {
            'analysis_timestamp': result.analysis_timestamp.isoformat(),
            'confidence_score': result.confidence_score,
            'top5_selections': [
                {
                    'rank': sel.rank,
                    'symbol': sel.symbol,
                    'name': sel.name,
                    'final_score': sel.final_score,
                    'selection_reason': sel.selection_reason,
                    'strategy_scores': sel.strategy_scores,
                    'technical_analysis': sel.technical_analysis,
                    'risk_assessment': sel.risk_assessment,
                    'gemini_reasoning': sel.gemini_reasoning
                }
                for sel in result.top5_selections
            ],
            'analysis_summary': result.analysis_summary,
            'market_outlook': result.market_outlook,
            'risk_warnings': result.risk_warnings,
            'alternative_candidates': result.alternative_candidates
        }
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def _export_markdown(self, result: GeminiAnalysisResult) -> str:
        """Markdown 형식으로 내보내기"""
        md = f"""# Gemini AI Top5 종목 선정 결과

## 분석 개요
- **분석 시간**: {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **신뢰도**: {result.confidence_score:.1f}%

## Top5 선정 종목

"""
        
        for selection in result.top5_selections:
            md += f"""### {selection.rank}. {selection.symbol} - {selection.name}
- **최종 점수**: {selection.final_score:.1f}점
- **선정 이유**: {selection.selection_reason}
- **기술적 분석**: {selection.technical_analysis}
- **리스크 평가**: {selection.risk_assessment}
- **Gemini 분석**: {selection.gemini_reasoning}

"""
        
        md += f"""## 시장 분석 요약
{result.analysis_summary}

## 시장 전망
{result.market_outlook}

## 위험 요소
"""
        for warning in result.risk_warnings:
            md += f"- {warning}\n"
        
        md += f"""
## 대안 후보
{', '.join(result.alternative_candidates)}
"""
        
        return md 