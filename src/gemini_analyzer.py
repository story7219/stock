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
        """분석용 데이터 준비"""
        
        # 전략별 상위 후보 추출 (각 전략별 Top 10)
        top_candidates = {}
        all_symbols = set()
        
        for strategy_name, scores in strategy_results.items():
            top_10 = scores[:10]
            top_candidates[strategy_name] = top_10
            all_symbols.update([score.symbol for score in top_10])
        
        # 종목별 상세 데이터 수집
        stock_details = {}
        for market, stocks in market_data.items():
            for stock in stocks:
                if stock.symbol in all_symbols:
                    stock_details[stock.symbol] = {
                        'market': market,
                        'basic_data': stock,
                        'technical_indicators': {
                            'rsi': stock.rsi,
                            'macd': stock.macd,
                            'macd_signal': stock.macd_signal,
                            'bollinger_upper': stock.bollinger_upper,
                            'bollinger_lower': stock.bollinger_lower,
                            'moving_avg_20': stock.moving_avg_20,
                            'moving_avg_60': stock.moving_avg_60
                        }
                    }
        
        return {
            'strategy_candidates': top_candidates,
            'stock_details': stock_details,
            'analysis_timestamp': datetime.now(),
            'total_candidates': len(all_symbols)
        }
    
    def _create_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Gemini AI 분석 프롬프트 생성"""
        
        prompt = f"""
        당신은 세계적인 투자 전문가입니다. 다음 데이터를 바탕으로 최고의 Top5 종목을 선정해주세요.

        ## 분석 대상
        - 코스피200, 나스닥100, S&P500 전체 종목
        - 총 {analysis_data['total_candidates']}개 후보 종목
        - 분석 시점: {analysis_data['analysis_timestamp'].strftime('%Y-%m-%d %H:%M')}

        ## 투자 전략별 후보군
        """
        
        # 각 전략별 후보 정보 추가
        for strategy_name, candidates in analysis_data['strategy_candidates'].items():
            prompt += f"\n### {strategy_name} 전략 Top 10:\n"
            for i, candidate in enumerate(candidates[:5], 1):  # 상위 5개만 표시
                stock_detail = analysis_data['stock_details'].get(candidate.symbol, {})
                basic_data = stock_detail.get('basic_data')
                
                if basic_data:
                    prompt += f"{i}. {candidate.symbol} ({candidate.name})\n"
                    prompt += f"   - 점수: {candidate.total_score:.1f}점\n"
                    prompt += f"   - 현재가: ${basic_data.price:.2f}\n"
                    prompt += f"   - 시가총액: ${basic_data.market_cap/1e9:.1f}B (if available)\n"
                    prompt += f"   - PER: {basic_data.pe_ratio:.1f} (if available)\n"
                    prompt += f"   - RSI: {basic_data.rsi:.1f} (if available)\n"
                    prompt += f"   - 선정 이유: {candidate.reasoning[:200]}...\n\n"
        
        prompt += """
        ## 요청사항
        위 후보군을 종합적으로 분석하여 다음 기준으로 Top5 종목을 선정해주세요:

        1. **다각도 분석**: 각 투자 전략의 장단점을 고려
        2. **기술적 분석**: RSI, MACD, 볼린저밴드 등 기술적 지표 해석
        3. **리스크 평가**: 각 종목의 위험 요소 분석
        4. **포트폴리오 균형**: 시장별, 섹터별 분산 고려
        5. **현재 시장 상황**: 최근 시장 트렌드와 경제 상황 반영

        ## 응답 형식 (JSON)
        다음 JSON 형식으로 정확히 응답해주세요:

        {
          "top5_selections": [
            {
              "rank": 1,
              "symbol": "AAPL",
              "name": "Apple Inc.",
              "final_score": 95.5,
              "selection_reason": "강력한 기술적 지표와 워런 버핏 전략에서 높은 점수...",
              "technical_analysis": "RSI 65로 적정 수준, MACD 골든크로스 형성...",
              "risk_assessment": "높은 유동성으로 리스크 낮음, 단 기술주 변동성 주의...",
              "gemini_reasoning": "종합적으로 판단할 때 가장 안정적이면서도 성장 가능성이 높은 종목..."
            }
            // ... 5개 종목
          ],
          "analysis_summary": "현재 시장은 기술주 중심의 회복세를 보이고 있으며...",
          "market_outlook": "향후 3-6개월 시장 전망...",
          "risk_warnings": ["금리 인상 리스크", "지정학적 위험"],
          "alternative_candidates": ["MSFT", "GOOGL", "005930.KS"],
          "confidence_score": 85.5
        }

        중요: 반드시 유효한 JSON 형식으로만 응답하고, 추가 설명은 JSON 내부에 포함해주세요.
        """
        
        return prompt
    
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