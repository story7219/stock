#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 파이프라인 매니저
데이터 수집 → 정제 → AI 분석 → 전략 적용 → 추천 생성 전체 흐름 관리
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass, asdict

# 로컬 모듈 임포트
from data.data_loader import DataLoader, CleanedStockData
from ai_integration.gemini_client import GeminiClient
from ai_integration.ai_preprocessor import AIPreprocessor
from strategies.value.buffett import BuffettStrategy
from strategies.growth.lynch import LynchStrategy
from strategies.quantitative.greenblatt import GreenblattStrategy
from recommenders.recommender import InvestmentRecommender
from recommenders.scorer import StockScorer

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    market: str
    total_stocks: int
    processed_stocks: int
    ai_analysis_completed: bool
    strategies_applied: List[str]
    top_recommendations: List[Dict[str, Any]]
    execution_time: float
    quality_score: float
    errors: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

class PipelineManager:
    """투자 추천 파이프라인 매니저"""
    
    def __init__(self):
        # 핵심 컴포넌트 초기화
        self.data_loader = DataLoader()
        self.gemini_client = GeminiClient()
        self.ai_preprocessor = AIPreprocessor()
        self.stock_scorer = StockScorer()
        self.recommender = InvestmentRecommender()
        
        # 투자 전략들
        self.strategies = {
            'buffett': BuffettStrategy(),
            'lynch': LynchStrategy(),
            'greenblatt': GreenblattStrategy()
        }
        
        # 파이프라인 설정
        self.pipeline_config = {
            'enable_ai_analysis': True,
            'enable_caching': True,
            'min_data_quality': 60.0,
            'max_recommendations': 5,
            'strategy_weights': {
                'buffett': 0.4,
                'lynch': 0.35,
                'greenblatt': 0.25
            }
        }
        
        # 결과 저장 경로
        self.results_dir = Path("data/processed/pipeline_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_full_pipeline(self, market: str = 'KR', 
                               symbols: Optional[List[str]] = None,
                               custom_config: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        errors = []
        
        try:
            logger.info(f"🚀 {market} 시장 투자 추천 파이프라인 시작")
            
            # 설정 업데이트
            if custom_config:
                self.pipeline_config.update(custom_config)
            
            # 1단계: 데이터 수집 및 정제
            logger.info("📊 1단계: 데이터 수집 및 정제")
            stocks_data = await self._collect_and_clean_data(market, symbols)
            
            if not stocks_data:
                raise Exception("데이터 수집 실패")
            
            logger.info(f"✅ 데이터 수집 완료: {len(stocks_data)} 종목")
            
            # 2단계: AI 분석
            ai_analysis_completed = False
            if self.pipeline_config['enable_ai_analysis']:
                logger.info("🤖 2단계: AI 분석")
                ai_analysis_completed = await self._perform_ai_analysis(stocks_data)
            
            # 3단계: 투자 전략 적용
            logger.info("📈 3단계: 투자 전략 적용")
            strategy_results = await self._apply_strategies(stocks_data)
            
            # 4단계: 종합 점수 계산
            logger.info("📊 4단계: 종합 점수 계산")
            scored_stocks = await self._calculate_comprehensive_scores(stocks_data, strategy_results)
            
            # 5단계: 최종 추천 생성
            logger.info("🎯 5단계: 최종 추천 생성")
            recommendations = await self._generate_final_recommendations(scored_stocks)
            
            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 품질 점수 계산
            quality_score = self._calculate_pipeline_quality(stocks_data, recommendations)
            
            # 결과 생성
            result = PipelineResult(
                success=True,
                market=market,
                total_stocks=len(stocks_data),
                processed_stocks=len(scored_stocks),
                ai_analysis_completed=ai_analysis_completed,
                strategies_applied=list(self.strategies.keys()),
                top_recommendations=recommendations,
                execution_time=execution_time,
                quality_score=quality_score,
                errors=errors,
                timestamp=datetime.now().isoformat()
            )
            
            # 결과 저장
            await self._save_pipeline_result(result)
            
            logger.info(f"🎉 파이프라인 완료: {len(recommendations)}개 추천, 실행시간: {execution_time:.2f}초")
            return result
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 오류: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                success=False,
                market=market,
                total_stocks=0,
                processed_stocks=0,
                ai_analysis_completed=False,
                strategies_applied=[],
                top_recommendations=[],
                execution_time=execution_time,
                quality_score=0.0,
                errors=[str(e)],
                timestamp=datetime.now().isoformat()
            )
    
    async def _collect_and_clean_data(self, market: str, symbols: Optional[List[str]]) -> List[CleanedStockData]:
        """데이터 수집 및 정제"""
        try:
            # 데이터 로더를 통해 데이터 수집
            stocks_data = await self.data_loader.load_market_data(
                market=market,
                symbols=symbols,
                use_cache=self.pipeline_config['enable_caching']
            )
            
            # 품질 필터링
            min_quality = self.pipeline_config['min_data_quality']
            filtered_stocks = [
                stock for stock in stocks_data 
                if stock.data_quality >= min_quality
            ]
            
            logger.info(f"품질 필터링: {len(filtered_stocks)}/{len(stocks_data)} 종목 (최소 품질: {min_quality})")
            
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            return []
    
    async def _perform_ai_analysis(self, stocks_data: List[CleanedStockData]) -> bool:
        """AI 분석 수행"""
        try:
            # AI 분석용 데이터 준비
            ai_data = self.ai_preprocessor.prepare_for_analysis(stocks_data, 'investment')
            
            # Gemini AI 분석 요청
            async with self.gemini_client as client:
                # 배치 분석 요청
                analysis_tasks = []
                for stock in stocks_data:
                    prompt = self._create_ai_analysis_prompt(stock)
                    task = client.analyze_stock_async(prompt)
                    analysis_tasks.append(task)
                
                # 모든 분석 완료 대기
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                # 결과 처리
                successful_analyses = 0
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"AI 분석 실패 ({stocks_data[i].symbol}): {result}")
                    else:
                        # AI 분석 결과를 주식 데이터에 추가
                        stocks_data[i].ai_analysis = result
                        successful_analyses += 1
                
                logger.info(f"AI 분석 완료: {successful_analyses}/{len(stocks_data)} 종목")
                return successful_analyses > 0
                
        except Exception as e:
            logger.error(f"AI 분석 오류: {e}")
            return False
    
    async def _apply_strategies(self, stocks_data: List[CleanedStockData]) -> Dict[str, List[Dict[str, Any]]]:
        """투자 전략 적용"""
        strategy_results = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                logger.info(f"📊 {strategy_name} 전략 적용 중...")
                
                # 전략별 분석 실행
                results = []
                for stock in stocks_data:
                    try:
                        score = strategy.analyze_stock(stock)
                        results.append({
                            'symbol': stock.symbol,
                            'name': stock.name,
                            'score': score,
                            'strategy': strategy_name
                        })
                    except Exception as e:
                        logger.debug(f"{strategy_name} 전략 분석 오류 ({stock.symbol}): {e}")
                        continue
                
                # 전략별 결과 정렬
                results.sort(key=lambda x: x['score'], reverse=True)
                strategy_results[strategy_name] = results
                
                logger.info(f"✅ {strategy_name} 전략 완료: {len(results)} 종목 분석")
                
            except Exception as e:
                logger.error(f"{strategy_name} 전략 적용 오류: {e}")
                strategy_results[strategy_name] = []
        
        return strategy_results
    
    async def _calculate_comprehensive_scores(self, stocks_data: List[CleanedStockData], 
                                            strategy_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """종합 점수 계산"""
        try:
            scored_stocks = []
            
            for stock in stocks_data:
                # 기본 점수 계산
                base_scores = self.stock_scorer.calculate_comprehensive_score(stock)
                
                # 전략별 점수 가중합
                strategy_score = 0.0
                strategy_weights = self.pipeline_config['strategy_weights']
                
                for strategy_name, results in strategy_results.items():
                    # 해당 전략에서 이 종목의 점수 찾기
                    stock_result = next((r for r in results if r['symbol'] == stock.symbol), None)
                    if stock_result:
                        weight = strategy_weights.get(strategy_name, 0.0)
                        strategy_score += stock_result['score'] * weight
                
                # 최종 종합 점수
                final_score = (base_scores['comprehensive_score'] * 0.6) + (strategy_score * 0.4)
                
                scored_stock = {
                    'symbol': stock.symbol,
                    'name': stock.name,
                    'price': stock.price,
                    'market_cap': stock.market_cap,
                    'sector': stock.sector,
                    'base_scores': base_scores,
                    'strategy_score': strategy_score,
                    'final_score': final_score,
                    'data_quality': stock.data_quality,
                    'ai_analysis': getattr(stock, 'ai_analysis', None)
                }
                
                scored_stocks.append(scored_stock)
            
            # 최종 점수 기준 정렬
            scored_stocks.sort(key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"종합 점수 계산 완료: {len(scored_stocks)} 종목")
            return scored_stocks
            
        except Exception as e:
            logger.error(f"종합 점수 계산 오류: {e}")
            return []
    
    async def _generate_final_recommendations(self, scored_stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """최종 추천 생성"""
        try:
            max_recommendations = self.pipeline_config['max_recommendations']
            
            # 상위 종목 선택
            top_stocks = scored_stocks[:max_recommendations * 2]  # 여유분 확보
            
            # 추천 시스템을 통한 최종 선별
            recommendations = self.recommender.generate_recommendations(
                top_stocks, 
                max_recommendations
            )
            
            # 추천 이유 생성
            for rec in recommendations:
                rec['recommendation_reason'] = self._generate_recommendation_reason(rec)
            
            logger.info(f"최종 추천 생성 완료: {len(recommendations)} 종목")
            return recommendations
            
        except Exception as e:
            logger.error(f"최종 추천 생성 오류: {e}")
            return []
    
    def _create_ai_analysis_prompt(self, stock: CleanedStockData) -> str:
        """AI 분석용 프롬프트 생성"""
        prompt = f"""
        다음 주식에 대한 투자 분석을 수행해주세요:
        
        종목명: {stock.name} ({stock.symbol})
        현재가: {stock.price:,}원
        시가총액: {stock.market_cap:,}원
        섹터: {stock.sector or '미분류'}
        
        재무 지표:
        - PER: {stock.pe_ratio}
        - PBR: {stock.pb_ratio}
        - ROE: {stock.roe}%
        - 부채비율: {stock.debt_ratio}%
        - 배당수익률: {stock.dividend_yield}%
        
        성장성:
        - 매출성장률: {stock.revenue_growth}%
        - 순이익성장률: {stock.profit_growth}%
        
        다음 관점에서 분석해주세요:
        1. 재무 건전성 평가
        2. 성장 잠재력 분석
        3. 밸류에이션 적정성
        4. 투자 위험 요소
        5. 종합 투자 의견 (매수/보유/매도)
        
        간결하고 명확한 분석을 제공해주세요.
        """
        return prompt
    
    def _generate_recommendation_reason(self, recommendation: Dict[str, Any]) -> str:
        """추천 이유 생성"""
        reasons = []
        
        # 점수 기반 이유
        if recommendation['final_score'] > 80:
            reasons.append("매우 높은 종합 점수")
        elif recommendation['final_score'] > 70:
            reasons.append("높은 종합 점수")
        
        # 전략별 이유
        base_scores = recommendation.get('base_scores', {})
        if base_scores.get('value_score', 0) > 70:
            reasons.append("우수한 가치 투자 지표")
        if base_scores.get('growth_score', 0) > 70:
            reasons.append("강한 성장 잠재력")
        if base_scores.get('quality_score', 0) > 70:
            reasons.append("뛰어난 재무 건전성")
        
        # AI 분석 결과
        if recommendation.get('ai_analysis'):
            reasons.append("AI 분석 긍정적 평가")
        
        return " | ".join(reasons) if reasons else "종합적 분석 결과"
    
    def _calculate_pipeline_quality(self, stocks_data: List[CleanedStockData], 
                                   recommendations: List[Dict[str, Any]]) -> float:
        """파이프라인 품질 점수 계산"""
        try:
            # 데이터 품질 점수
            data_qualities = [stock.data_quality for stock in stocks_data]
            avg_data_quality = sum(data_qualities) / len(data_qualities) if data_qualities else 0
            
            # 추천 품질 점수
            if recommendations:
                recommendation_scores = [rec['final_score'] for rec in recommendations]
                avg_recommendation_score = sum(recommendation_scores) / len(recommendation_scores)
            else:
                avg_recommendation_score = 0
            
            # 전체 품질 점수 (데이터 품질 40% + 추천 품질 60%)
            pipeline_quality = (avg_data_quality * 0.4) + (avg_recommendation_score * 0.6)
            
            return round(pipeline_quality, 2)
            
        except Exception as e:
            logger.error(f"품질 점수 계산 오류: {e}")
            return 0.0
    
    async def _save_pipeline_result(self, result: PipelineResult):
        """파이프라인 결과 저장"""
        try:
            filename = f"pipeline_result_{result.market}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"파이프라인 결과 저장: {filename}")
            
        except Exception as e:
            logger.error(f"결과 저장 오류: {e}")
    
    async def run_quick_analysis(self, symbol: str, market: str = 'KR') -> Dict[str, Any]:
        """빠른 개별 종목 분석"""
        try:
            logger.info(f"🔍 빠른 분석 시작: {symbol}")
            
            # 개별 종목 데이터 로드
            stock_data = await self.data_loader.load_stock_data(symbol, market)
            
            if not stock_data:
                return {"error": f"종목 데이터를 불러올 수 없습니다: {symbol}"}
            
            # 기본 점수 계산
            base_scores = self.stock_scorer.calculate_comprehensive_score(stock_data)
            
            # 전략별 분석
            strategy_scores = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    score = strategy.analyze_stock(stock_data)
                    strategy_scores[strategy_name] = score
                except Exception as e:
                    logger.debug(f"{strategy_name} 전략 분석 오류: {e}")
                    strategy_scores[strategy_name] = 0
            
            # AI 분석 (선택적)
            ai_analysis = None
            if self.pipeline_config['enable_ai_analysis']:
                try:
                    prompt = self._create_ai_analysis_prompt(stock_data)
                    async with self.gemini_client as client:
                        ai_analysis = await client.analyze_stock_async(prompt)
                except Exception as e:
                    logger.debug(f"AI 분석 오류: {e}")
            
            # 결과 구성
            result = {
                "symbol": stock_data.symbol,
                "name": stock_data.name,
                "price": stock_data.price,
                "market_cap": stock_data.market_cap,
                "sector": stock_data.sector,
                "data_quality": stock_data.data_quality,
                "base_scores": base_scores,
                "strategy_scores": strategy_scores,
                "ai_analysis": ai_analysis,
                "analysis_time": datetime.now().isoformat()
            }
            
            logger.info(f"✅ 빠른 분석 완료: {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"빠른 분석 오류 ({symbol}): {e}")
            return {"error": str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 정보"""
        return {
            "components": {
                "data_loader": "활성화",
                "ai_client": "활성화" if self.pipeline_config['enable_ai_analysis'] else "비활성화",
                "strategies": list(self.strategies.keys()),
                "recommender": "활성화"
            },
            "config": self.pipeline_config,
            "last_results": self._get_recent_results()
        }
    
    def _get_recent_results(self) -> List[Dict[str, Any]]:
        """최근 파이프라인 결과 조회"""
        try:
            result_files = list(self.results_dir.glob("pipeline_result_*.json"))
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            recent_results = []
            for result_file in result_files[:5]:  # 최근 5개
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    # 요약 정보만 추출
                    summary = {
                        "timestamp": result_data.get("timestamp"),
                        "market": result_data.get("market"),
                        "success": result_data.get("success"),
                        "total_stocks": result_data.get("total_stocks"),
                        "recommendations_count": len(result_data.get("top_recommendations", [])),
                        "quality_score": result_data.get("quality_score"),
                        "execution_time": result_data.get("execution_time")
                    }
                    recent_results.append(summary)
                    
                except Exception as e:
                    logger.debug(f"결과 파일 읽기 오류: {e}")
                    continue
            
            return recent_results
            
        except Exception as e:
            logger.error(f"최근 결과 조회 오류: {e}")
            return []

# 편의 함수들
async def run_korean_market_analysis(symbols: List[str] = None) -> PipelineResult:
    """한국 시장 분석 실행"""
    manager = PipelineManager()
    return await manager.run_full_pipeline('KR', symbols)

async def run_us_market_analysis(symbols: List[str] = None) -> PipelineResult:
    """미국 시장 분석 실행"""
    manager = PipelineManager()
    return await manager.run_full_pipeline('US', symbols)

async def analyze_stock_quick(symbol: str, market: str = 'KR') -> Dict[str, Any]:
    """빠른 종목 분석"""
    manager = PipelineManager()
    return await manager.run_quick_analysis(symbol, market) 