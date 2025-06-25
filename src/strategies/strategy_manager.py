"""
전략 관리자 - 다양한 투자 전략을 통합 관리
"""

import asyncio
from typing import Dict, List, Any, Optional, Type
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, StrategyResult
from .druckenmiller_strategy import DruckenmillerStrategy
from ..data.models import StockData, MarketType
from ..core.cache_manager import CacheManager
from ..core.async_executor import AsyncExecutor

logger = logging.getLogger(__name__)


@dataclass
class StrategyAnalysisResult:
    """전략 분석 종합 결과"""
    market_type: MarketType
    analysis_time: datetime
    strategy_results: Dict[str, List[StrategyResult]] = field(default_factory=dict)
    top_picks: List[StrategyResult] = field(default_factory=list)
    market_summary: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class StrategyManager:
    """투자 전략 통합 관리자"""
    
    def __init__(self, cache_manager: CacheManager, async_executor: AsyncExecutor):
        self.cache_manager = cache_manager
        self.async_executor = async_executor
        self.strategies: Dict[str, BaseStrategy] = {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 전략 등록
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """기본 전략들 등록"""
        try:
            # 드러켄밀러 전략 등록
            druckenmiller = DruckenmillerStrategy()
            self.register_strategy(druckenmiller)
            
            self.logger.info(f"등록된 전략: {list(self.strategies.keys())}")
            
        except Exception as e:
            self.logger.error(f"기본 전략 등록 실패: {e}")
    
    def register_strategy(self, strategy: BaseStrategy):
        """새로운 전략 등록"""
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("전략은 BaseStrategy를 상속해야 합니다")
        
        self.strategies[strategy.name] = strategy
        self.logger.info(f"전략 등록 완료: {strategy.name}")
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """전략 인스턴스 반환"""
        return self.strategies.get(strategy_name)
    
    def list_strategies(self) -> List[str]:
        """등록된 전략 목록 반환"""
        return list(self.strategies.keys())
    
    async def analyze_stocks_with_strategy(
        self,
        stocks_data: List[StockData],
        strategy_name: str,
        market_type: MarketType,
        top_n: int = 5
    ) -> StrategyAnalysisResult:
        """특정 전략으로 종목들 분석"""
        
        if strategy_name not in self.strategies:
            raise ValueError(f"등록되지 않은 전략: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        # 캐시 키 생성
        cache_key = f"strategy_analysis:{strategy_name}:{market_type.value}:{len(stocks_data)}:{top_n}"
        
        # 캐시에서 확인
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 전략 분석 결과 반환: {strategy_name}")
            return cached_result
        
        try:
            self.logger.info(f"{strategy_name} 전략으로 {len(stocks_data)}개 종목 분석 시작")
            
            # 비동기 병렬 분석
            analysis_tasks = [
                self._analyze_single_stock(strategy, stock_data)
                for stock_data in stocks_data
            ]
            
            # 배치 실행
            results = await self.async_executor.execute_batch(
                analysis_tasks,
                batch_size=20,
                max_concurrent=10
            )
            
            # 결과 정리
            valid_results = [result for result in results if result and result.score > 0]
            
            # 점수순 정렬
            valid_results.sort(key=lambda x: x.score, reverse=True)
            
            # Top N 선정
            top_picks = valid_results[:top_n]
            
            # 시장 요약 생성
            market_summary = self._generate_market_summary(valid_results, market_type)
            
            # 성능 지표 계산
            performance_metrics = self._calculate_performance_metrics(valid_results)
            
            # 결과 객체 생성
            analysis_result = StrategyAnalysisResult(
                market_type=market_type,
                analysis_time=datetime.now(),
                strategy_results={strategy_name: valid_results},
                top_picks=top_picks,
                market_summary=market_summary,
                performance_metrics=performance_metrics
            )
            
            # 캐시에 저장 (30분)
            await self.cache_manager.set(cache_key, analysis_result, ttl=1800)
            
            self.logger.info(
                f"{strategy_name} 전략 분석 완료: "
                f"총 {len(valid_results)}개 유효 결과, Top {len(top_picks)}개 선정"
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"전략 분석 실패 ({strategy_name}): {e}")
            raise
    
    async def _analyze_single_stock(self, strategy: BaseStrategy, stock_data: StockData) -> Optional[StrategyResult]:
        """단일 종목 분석"""
        try:
            result = await strategy.analyze_stock(stock_data)
            return result
        except Exception as e:
            self.logger.warning(f"종목 분석 실패 ({stock_data.symbol}): {e}")
            return None
    
    async def analyze_multiple_strategies(
        self,
        stocks_data: List[StockData],
        strategy_names: List[str],
        market_type: MarketType,
        top_n: int = 5
    ) -> Dict[str, StrategyAnalysisResult]:
        """여러 전략으로 동시 분석"""
        
        # 유효한 전략만 필터링
        valid_strategies = [name for name in strategy_names if name in self.strategies]
        
        if not valid_strategies:
            raise ValueError("유효한 전략이 없습니다")
        
        self.logger.info(f"{len(valid_strategies)}개 전략으로 다중 분석 시작")
        
        # 각 전략별 분석 태스크 생성
        strategy_tasks = [
            self.analyze_stocks_with_strategy(stocks_data, strategy_name, market_type, top_n)
            for strategy_name in valid_strategies
        ]
        
        # 병렬 실행
        results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
        
        # 결과 정리
        strategy_results = {}
        for i, result in enumerate(results):
            strategy_name = valid_strategies[i]
            if isinstance(result, Exception):
                self.logger.error(f"전략 {strategy_name} 분석 실패: {result}")
            else:
                strategy_results[strategy_name] = result
        
        return strategy_results
    
    def _generate_market_summary(self, results: List[StrategyResult], market_type: MarketType) -> Dict[str, Any]:
        """시장 요약 생성"""
        if not results:
            return {"message": "분석 결과 없음"}
        
        # 기본 통계
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        
        # 리스크 레벨 분포
        risk_distribution = {}
        for result in results:
            risk_level = result.risk_level
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        # 주요 투자 사유 분석
        all_reasons = []
        for result in results:
            all_reasons.extend(result.reasons)
        
        reason_counts = {}
        for reason in all_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "market_type": market_type.value,
            "total_analyzed": len(results),
            "score_statistics": {
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "std": pd.Series(scores).std()
            },
            "confidence_statistics": {
                "mean": sum(confidences) / len(confidences),
                "max": max(confidences),
                "min": min(confidences)
            },
            "risk_distribution": risk_distribution,
            "top_investment_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons],
            "high_score_count": len([s for s in scores if s >= 70]),
            "medium_score_count": len([s for s in scores if 50 <= s < 70]),
            "low_score_count": len([s for s in scores if s < 50])
        }
    
    def _calculate_performance_metrics(self, results: List[StrategyResult]) -> Dict[str, float]:
        """성능 지표 계산"""
        if not results:
            return {}
        
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        
        # 고득점 종목 비율
        high_score_ratio = len([s for s in scores if s >= 70]) / len(scores)
        
        # 평균 신뢰도
        avg_confidence = sum(confidences) / len(confidences)
        
        # 점수 분산 (일관성 지표)
        score_variance = pd.Series(scores).var()
        
        # 품질 지표 (고득점 + 고신뢰도)
        quality_stocks = [
            r for r in results 
            if r.score >= 70 and r.confidence >= 70
        ]
        quality_ratio = len(quality_stocks) / len(results)
        
        return {
            "high_score_ratio": high_score_ratio,
            "average_confidence": avg_confidence,
            "score_variance": score_variance,
            "quality_ratio": quality_ratio,
            "total_analyzed": len(results),
            "analysis_efficiency": len([r for r in results if r.score > 0]) / len(results)
        }
    
    async def get_strategy_performance_report(self, strategy_name: str) -> Dict[str, Any]:
        """전략 성능 보고서 생성"""
        if strategy_name not in self.strategies:
            raise ValueError(f"등록되지 않은 전략: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        # 전략 기본 정보
        strategy_info = {
            "name": strategy.name,
            "description": strategy.description,
            "parameters": strategy.parameters
        }
        
        # 전략별 성능 정보 (실제 구현시 데이터베이스에서 조회)
        performance_info = {
            "total_analyses": 0,
            "success_rate": 0.0,
            "average_score": 0.0,
            "average_confidence": 0.0,
            "last_used": None
        }
        
        # 전략 파라미터 정보
        if hasattr(strategy, 'get_strategy_parameters'):
            additional_info = strategy.get_strategy_parameters()
            strategy_info.update(additional_info)
        
        return {
            "strategy_info": strategy_info,
            "performance_metrics": performance_info,
            "status": "active" if strategy_name in self.strategies else "inactive"
        }
    
    async def optimize_strategy_parameters(
        self,
        strategy_name: str,
        sample_data: List[StockData],
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """전략 파라미터 최적화 (향후 구현)"""
        # 현재는 기본 구현만 제공
        self.logger.info(f"전략 파라미터 최적화 요청: {strategy_name}")
        
        return {
            "message": "파라미터 최적화는 향후 구현 예정",
            "strategy": strategy_name,
            "current_parameters": self.strategies[strategy_name].parameters if strategy_name in self.strategies else {}
        }
    
    def get_strategy_comparison_report(self, results: Dict[str, StrategyAnalysisResult]) -> Dict[str, Any]:
        """전략 비교 보고서 생성"""
        if not results:
            return {"message": "비교할 전략 결과가 없습니다"}
        
        comparison = {}
        
        for strategy_name, result in results.items():
            if result.top_picks:
                comparison[strategy_name] = {
                    "top_score": result.top_picks[0].score,
                    "average_top5_score": sum(r.score for r in result.top_picks[:5]) / min(5, len(result.top_picks)),
                    "total_valid_picks": len([r for r in result.strategy_results.get(strategy_name, []) if r.score > 0]),
                    "high_confidence_picks": len([r for r in result.top_picks if r.confidence > 70]),
                    "market_summary": result.market_summary
                }
        
        # 최고 성능 전략 식별
        if comparison:
            best_strategy = max(comparison.keys(), key=lambda k: comparison[k]["top_score"])
            comparison["best_performing_strategy"] = best_strategy
        
        return comparison 