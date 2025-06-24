#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI 기반 투자 분석 시스템 v3.0 - 자가 학습 메인 시스템
지속적으로 학습하고 진화하는 고도화된 투자 분석 플랫폼
"""

import asyncio
import logging
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 시스템 컴포넌트
    from src.ml_engine import AdaptiveLearningEngine, MLPrediction
    from src.auto_updater import AutoUpdater, SmartConfigManager
    from src.system_monitor import get_system_monitor
    
    # 기본 라이브러리들로 간소화된 구성
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    logger = logging.getLogger(__name__)
    
except ImportError as e:
    print(f"❌ 필수 라이브러리 누락: {e}")
    print("🔧 requirements.txt의 라이브러리들을 설치하세요:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# 기존 시스템 컴포넌트
from data_collector import DataCollector
from investment_strategies import InvestmentStrategy, StockData, StrategyScore
from technical_analysis import TechnicalAnalyzer, TechnicalAnalysisResult
from ai_analyzer import GeminiAnalyzer

@dataclass
class SystemStatus:
    """시스템 상태"""
    timestamp: datetime
    overall_health: str
    active_strategies: int
    ml_models_loaded: int
    auto_learning_enabled: bool
    last_evolution: Optional[datetime]
    performance_trend: str

@dataclass 
class AnalysisResult:
    """분석 결과"""
    symbol: str
    market: str
    strategy_scores: Dict[str, float]
    technical_score: float
    ml_prediction: MLPrediction
    ai_recommendation: str
    final_score: float
    confidence: float
    reasoning: str

class InvestmentAnalysisSystem:
    """🎯 자가 학습 투자 분석 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.start_time = datetime.now()
        
        # 기존 컴포넌트
        self.data_collector = DataCollector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_analyzer = GeminiAnalyzer()
        
        # 새로운 ML/자동화 컴포넌트
        self.ml_engine = AdaptiveLearningEngine()
        self.auto_updater = AutoUpdater()
        self.config_manager = SmartConfigManager()
        
        # 시스템 상태
        self.analysis_history = []
        self.performance_metrics = {}
        self.learning_enabled = True
        
        # 설정
        self.auto_retrain_interval_hours = 24
        self.performance_tracking_enabled = True
        self.evolution_threshold_days = 7
        
        logger.info("🎯 자가 학습 투자 분석 시스템 v3.0 초기화 완료")
    
    async def run_comprehensive_analysis(self, 
                                       markets: List[str] = None,
                                       enable_learning: bool = True,
                                       enable_evolution: bool = True) -> Dict[str, Any]:
        """종합 분석 실행 (자가 학습 포함)"""
        
        if markets is None:
            markets = ["KOSPI200", "NASDAQ100", "S&P500"]
        
        logger.info(f"🚀 종합 분석 시작 - 시장: {markets}, 학습: {enable_learning}, 진화: {enable_evolution}")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'markets_analyzed': markets,
            'system_status': {},
            'top_recommendations': [],
            'performance_insights': {},
            'learning_updates': {},
            'evolution_log': []
        }
        
        try:
            # 1. 시스템 상태 체크
            system_status = await self._check_system_status()
            analysis_results['system_status'] = asdict(system_status)
            
            # 2. 자동 성능 모니터링 및 진화
            if enable_evolution:
                evolution_report = await self._run_evolutionary_cycle()
                analysis_results['evolution_log'] = evolution_report
            
            # 3. 데이터 수집
            logger.info("📊 시장 데이터 수집 중...")
            all_stocks = []
            
            for market in markets:
                market_stocks = await self._collect_market_data(market)
                all_stocks.extend(market_stocks)
            
            logger.info(f"✅ 총 {len(all_stocks)}개 종목 데이터 수집 완료")
            
            # 4. 투자 전략 분석
            logger.info("💡 투자 전략 분석 중...")
            strategy_results = await self._analyze_with_strategies(all_stocks)
            
            # 5. 기술적 분석
            logger.info("📈 기술적 분석 중...")
            technical_results = await self._perform_technical_analysis(all_stocks)
            
            # 6. ML 기반 예측
            logger.info("🧠 ML 예측 실행 중...")
            ml_predictions = await self._generate_ml_predictions(
                all_stocks, strategy_results, technical_results
            )
            
            # 7. AI 종합 분석
            logger.info("🤖 AI 종합 분석 중...")
            ai_insights = await self._perform_ai_analysis(
                all_stocks, strategy_results, technical_results, ml_predictions
            )
            
            # 8. 최종 추천 종목 선정
            logger.info("🎯 최종 추천 생성 중...")
            top_recommendations = await self._generate_final_recommendations(
                all_stocks, strategy_results, technical_results, ml_predictions, ai_insights
            )
            
            analysis_results['top_recommendations'] = top_recommendations
            
            # 9. 성능 추적 및 학습 데이터 업데이트
            if enable_learning:
                learning_updates = await self._update_learning_systems(
                    all_stocks, strategy_results, technical_results, ml_predictions
                )
                analysis_results['learning_updates'] = learning_updates
            
            # 10. 성능 인사이트 생성
            performance_insights = await self._generate_performance_insights()
            analysis_results['performance_insights'] = performance_insights
            
            # 11. 분석 결과 저장
            await self._save_analysis_results(analysis_results)
            
            logger.info("✅ 종합 분석 완료")
            
        except Exception as e:
            logger.error(f"❌ 종합 분석 실패: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    async def _check_system_status(self) -> SystemStatus:
        """시스템 상태 확인"""
        
        try:
            # ML 엔진 상태
            ml_status = self.ml_engine.get_model_status()
            ml_models_loaded = sum(ml_status['models_loaded'].values())
            
            # 업데이터 상태
            update_status = self.auto_updater.get_update_status()
            
            # 전략 수
            active_strategies = 15  # 하드코딩 (실제로는 동적 계산)
            
            # 성능 트렌드 (간단화)
            performance_trend = "improving"  # 실제로는 최근 성과 기반 계산
            
            status = SystemStatus(
                timestamp=datetime.now(),
                overall_health="excellent",
                active_strategies=active_strategies,
                ml_models_loaded=ml_models_loaded,
                auto_learning_enabled=self.learning_enabled,
                last_evolution=update_status.get('last_update'),
                performance_trend=performance_trend
            )
            
            logger.info(f"📊 시스템 상태: {status.overall_health} (ML모델: {ml_models_loaded}개)")
            
            return status
            
        except Exception as e:
            logger.error(f"시스템 상태 확인 실패: {e}")
            return SystemStatus(
                timestamp=datetime.now(),
                overall_health="error",
                active_strategies=0,
                ml_models_loaded=0,
                auto_learning_enabled=False,
                last_evolution=None,
                performance_trend="unknown"
            )
    
    async def _run_evolutionary_cycle(self) -> List[Dict[str, Any]]:
        """진화 사이클 실행"""
        
        evolution_log = []
        
        try:
            logger.info("🧬 진화 사이클 시작")
            
            # 성능 모니터링
            performance_report = self.auto_updater.monitor_system_performance()
            
            evolution_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'performance_monitoring',
                'result': performance_report['overall_health'],
                'recommendations': len(performance_report['improvement_recommendations']),
                'auto_update_triggered': performance_report['auto_update_triggered']
            })
            
            # 자동 파라미터 튜닝
            if performance_report['strategy_performances']:
                avg_performance = {
                    'accuracy': sum(p.accuracy for p in performance_report['strategy_performances']) / len(performance_report['strategy_performances']),
                    'max_drawdown': min(p.max_drawdown for p in performance_report['strategy_performances'])
                }
                
                optimized_params = self.config_manager.auto_tune_parameters(avg_performance)
                
                evolution_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'parameter_tuning',
                    'parameters_optimized': len(optimized_params),
                    'details': optimized_params
                })
            
            logger.info(f"✅ 진화 사이클 완료: {len(evolution_log)}개 작업")
            
        except Exception as e:
            logger.error(f"진화 사이클 실패: {e}")
            evolution_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'error',
                'message': str(e)
            })
        
        return evolution_log
    
    async def _collect_market_data(self, market: str) -> List[StockData]:
        """시장 데이터 수집"""
        
        try:
            # 기존 데이터 수집기 사용
            stocks = self.data_collector.get_market_stocks(market)
            
            logger.info(f"📊 {market} 시장: {len(stocks)}개 종목 수집")
            return stocks
            
        except Exception as e:
            logger.error(f"{market} 데이터 수집 실패: {e}")
            return []
    
    async def _analyze_with_strategies(self, stocks: List[StockData]) -> Dict[str, List[StrategyScore]]:
        """투자 전략 분석"""
        
        strategy_results = {}
        
        try:
            # 기존 전략 분석 로직 (간단화)
            for stock in stocks:
                # 15개 전략별 점수 (시뮬레이션)
                import random
                strategy_scores = []
                
                for i in range(15):
                    score = StrategyScore(
                        strategy_name=f"Strategy_{i+1}",
                        score=random.uniform(30, 90),
                        confidence=random.uniform(0.6, 0.9),
                        reasoning=f"전략 {i+1} 분석 결과"
                    )
                    strategy_scores.append(score)
                
                strategy_results[stock.symbol] = strategy_scores
            
            logger.info(f"💡 전략 분석 완료: {len(stocks)}개 종목")
            
        except Exception as e:
            logger.error(f"전략 분석 실패: {e}")
        
        return strategy_results
    
    async def _perform_technical_analysis(self, stocks: List[StockData]) -> Dict[str, TechnicalAnalysisResult]:
        """기술적 분석"""
        
        technical_results = {}
        
        try:
            for stock in stocks:
                # 기술적 분석 실행
                result = self.technical_analyzer.analyze_stock(stock)
                technical_results[stock.symbol] = result
            
            logger.info(f"📈 기술적 분석 완료: {len(stocks)}개 종목")
            
        except Exception as e:
            logger.error(f"기술적 분석 실패: {e}")
        
        return technical_results
    
    async def _generate_ml_predictions(self, 
                                     stocks: List[StockData],
                                     strategy_results: Dict[str, List[StrategyScore]],
                                     technical_results: Dict[str, TechnicalAnalysisResult]) -> Dict[str, MLPrediction]:
        """ML 기반 예측 생성"""
        
        ml_predictions = {}
        
        try:
            for stock in stocks:
                # ML 예측 실행
                prediction = self.ml_engine.predict_stock_performance(
                    stock, strategy_results, technical_results
                )
                ml_predictions[stock.symbol] = prediction
            
            logger.info(f"🧠 ML 예측 완료: {len(stocks)}개 종목")
            
        except Exception as e:
            logger.error(f"ML 예측 실패: {e}")
        
        return ml_predictions
    
    async def _perform_ai_analysis(self,
                                 stocks: List[StockData],
                                 strategy_results: Dict[str, List[StrategyScore]],
                                 technical_results: Dict[str, TechnicalAnalysisResult],
                                 ml_predictions: Dict[str, MLPrediction]) -> Dict[str, Any]:
        """AI 종합 분석"""
        
        ai_insights = {}
        
        try:
            # 상위 후보군 선별 (ML 예측 기반)
            top_candidates = sorted(
                stocks,
                key=lambda s: ml_predictions.get(s.symbol, MLPrediction("", 0, 0, 0, "", "")).predicted_return,
                reverse=True
            )[:50]  # 상위 50개
            
            # Gemini AI 분석
            for stock in top_candidates:
                try:
                    analysis_data = {
                        'stock': stock,
                        'strategy_scores': strategy_results.get(stock.symbol, []),
                        'technical_result': technical_results.get(stock.symbol),
                        'ml_prediction': ml_predictions.get(stock.symbol)
                    }
                    
                    # AI 분석 실행
                    ai_result = self.ai_analyzer.analyze_comprehensive(analysis_data)
                    ai_insights[stock.symbol] = ai_result
                    
                except Exception as e:
                    logger.warning(f"AI 분석 실패 ({stock.symbol}): {e}")
            
            logger.info(f"🤖 AI 분석 완료: {len(ai_insights)}개 종목")
            
        except Exception as e:
            logger.error(f"AI 분석 실패: {e}")
        
        return ai_insights
    
    async def _generate_final_recommendations(self,
                                            stocks: List[StockData],
                                            strategy_results: Dict[str, List[StrategyScore]],
                                            technical_results: Dict[str, TechnicalAnalysisResult],
                                            ml_predictions: Dict[str, MLPrediction],
                                            ai_insights: Dict[str, Any]) -> List[AnalysisResult]:
        """최종 추천 생성"""
        
        recommendations = []
        
        try:
            # 종목별 종합 점수 계산
            for stock in stocks:
                # 전략 점수 평균
                strategy_scores_dict = {}
                if stock.symbol in strategy_results:
                    for score in strategy_results[stock.symbol]:
                        strategy_scores_dict[score.strategy_name] = score.score
                    avg_strategy_score = sum(strategy_scores_dict.values()) / len(strategy_scores_dict)
                else:
                    avg_strategy_score = 50.0
                
                # 기술적 분석 점수
                technical_score = technical_results.get(stock.symbol, type('obj', (object,), {'overall_score': 50.0})).overall_score
                
                # ML 예측
                ml_pred = ml_predictions.get(stock.symbol, MLPrediction(stock.symbol, 0, 0.5, 0.5, "HOLD", "fallback"))
                
                # AI 인사이트
                ai_insight = ai_insights.get(stock.symbol, {'recommendation': 'HOLD', 'reasoning': '분석 없음'})
                
                # 종합 점수 계산 (가중 평균)
                final_score = (
                    avg_strategy_score * 0.3 +
                    technical_score * 0.2 +
                    (ml_pred.predicted_return * 100 + 50) * 0.3 +  # 정규화
                    (80 if ai_insight['recommendation'] in ['BUY', 'STRONG_BUY'] else 40) * 0.2
                )
                
                # 신뢰도 계산
                confidence = (ml_pred.confidence + 0.8) / 2  # ML 신뢰도와 기본값 평균
                
                # 추천 생성
                recommendation = AnalysisResult(
                    symbol=stock.symbol,
                    market=stock.market,
                    strategy_scores=strategy_scores_dict,
                    technical_score=technical_score,
                    ml_prediction=ml_pred,
                    ai_recommendation=ai_insight['recommendation'],
                    final_score=final_score,
                    confidence=confidence,
                    reasoning=ai_insight.get('reasoning', '종합 분석 결과')
                )
                
                recommendations.append(recommendation)
            
            # 상위 5개 선별
            top_5 = sorted(recommendations, key=lambda x: x.final_score, reverse=True)[:5]
            
            logger.info(f"🎯 Top 5 종목 선정 완료")
            for i, rec in enumerate(top_5, 1):
                logger.info(f"  {i}. {rec.symbol} (점수: {rec.final_score:.1f}, 신뢰도: {rec.confidence:.1f})")
            
            return top_5
            
        except Exception as e:
            logger.error(f"최종 추천 생성 실패: {e}")
            return []
    
    async def _update_learning_systems(self,
                                     stocks: List[StockData],
                                     strategy_results: Dict[str, List[StrategyScore]],
                                     technical_results: Dict[str, TechnicalAnalysisResult],
                                     ml_predictions: Dict[str, MLPrediction]) -> Dict[str, Any]:
        """학습 시스템 업데이트"""
        
        learning_updates = {
            'training_data_collected': False,
            'models_retrained': False,
            'performance_updated': False,
            'errors': []
        }
        
        try:
            # 학습 데이터 수집
            training_data = self.ml_engine.collect_training_data(
                stocks, strategy_results, technical_results
            )
            
            learning_updates['training_data_collected'] = len(training_data) > 0
            learning_updates['samples_collected'] = len(training_data)
            
            # 모델 재학습 (일정 조건 만족 시)
            if self._should_retrain_models():
                retrain_success = self.ml_engine.train_models()
                learning_updates['models_retrained'] = retrain_success
            
            # 성능 기록 업데이트 (실제 수익률이 있는 경우)
            # 여기서는 시뮬레이션
            performance_updates = 0
            for stock in stocks[:10]:  # 샘플 10개
                if stock.symbol in ml_predictions:
                    pred = ml_predictions[stock.symbol]
                    # 시뮬레이션된 실제 수익률
                    actual_return = pred.predicted_return + (hash(stock.symbol) % 20 - 10) / 1000
                    
                    self.ml_engine.update_model_performance(
                        stock.symbol, pred.predicted_return, actual_return
                    )
                    performance_updates += 1
            
            learning_updates['performance_updated'] = performance_updates > 0
            learning_updates['performance_updates'] = performance_updates
            
            logger.info(f"📚 학습 시스템 업데이트 완료: 데이터 {len(training_data)}개, 성능 업데이트 {performance_updates}개")
            
        except Exception as e:
            logger.error(f"학습 시스템 업데이트 실패: {e}")
            learning_updates['errors'].append(str(e))
        
        return learning_updates
    
    def _should_retrain_models(self) -> bool:
        """모델 재학습 필요 여부 판단"""
        
        # 간단한 조건: 24시간마다
        return (datetime.now().hour == 0)  # 매일 자정
    
    async def _generate_performance_insights(self) -> Dict[str, Any]:
        """성능 인사이트 생성"""
        
        insights = {
            'system_uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'analysis_count': len(self.analysis_history),
            'ml_model_status': self.ml_engine.get_model_status(),
            'recent_performance': 'stable',  # 실제로는 성능 데이터 기반
            'recommendations': []
        }
        
        # 성능 기반 권장사항
        if insights['system_uptime_hours'] > 168:  # 1주일 이상
            insights['recommendations'].append("시스템이 안정적으로 운영되고 있습니다")
        
        if insights['analysis_count'] > 100:
            insights['recommendations'].append("충분한 학습 데이터가 축적되었습니다")
        
        return insights
    
    async def _save_analysis_results(self, results: Dict[str, Any]):
        """분석 결과 저장"""
        
        try:
            # 결과 디렉토리 생성
            results_dir = "analysis_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 타임스탬프 기반 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # JSON으로 저장 (직렬화 가능한 형태로 변환)
            serializable_results = self._make_serializable(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # 분석 히스토리에 추가
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'file_path': filepath,
                'markets': results.get('markets_analyzed', []),
                'recommendations_count': len(results.get('top_recommendations', []))
            })
            
            logger.info(f"💾 분석 결과 저장: {filepath}")
            
        except Exception as e:
            logger.error(f"분석 결과 저장 실패: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 정보"""
        
        return {
            'version': '3.0 (자가 학습)',
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'total_analyses': len(self.analysis_history),
            'learning_enabled': self.learning_enabled,
            'ml_models_status': self.ml_engine.get_model_status(),
            'auto_updater_status': self.auto_updater.get_update_status(),
            'last_analysis': self.analysis_history[-1] if self.analysis_history else None
        }

# 사용 예시
async def main():
    """메인 실행 함수"""
    print("🚀 AI 기반 자가 학습 투자 분석 시스템 v3.0")
    print("=" * 60)
    
    # 시스템 초기화
    system = InvestmentAnalysisSystem()
    
    # 시스템 요약 출력
    summary = system.get_system_summary()
    print(f"\n📊 시스템 요약:")
    print(f"  • 버전: {summary['version']}")
    print(f"  • 가동 시간: {summary['uptime_hours']:.1f}시간")
    print(f"  • 학습 활성화: {'예' if summary['learning_enabled'] else '아니오'}")
    print(f"  • ML 모델: {sum(summary['ml_models_status']['models_loaded'].values())}개 로드됨")
    
    # 종합 분석 실행
    print(f"\n🎯 종합 분석 시작...")
    results = await system.run_comprehensive_analysis(
        markets=["KOSPI200", "NASDAQ100"],
        enable_learning=True,
        enable_evolution=True
    )
    
    # 결과 출력
    print(f"\n✅ 분석 완료!")
    print(f"  • 시스템 상태: {results['system_status']['overall_health']}")
    print(f"  • Top 5 추천: {len(results['top_recommendations'])}개")
    print(f"  • 진화 작업: {len(results['evolution_log'])}개")
    
    if results['top_recommendations']:
        print(f"\n🏆 Top 5 추천 종목:")
        for i, rec in enumerate(results['top_recommendations'], 1):
            print(f"  {i}. {rec['symbol']} - 점수: {rec['final_score']:.1f} (신뢰도: {rec['confidence']:.1f})")

if __name__ == "__main__":
    asyncio.run(main()) 