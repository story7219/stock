#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 향상된 주식 투자 분석 시스템 실행 스크립트
================================================

완전히 리팩토링된 고품질 주식 투자 분석 시스템을 실행합니다.

주요 기능:
- 코스피200·나스닥100·S&P500 전체 종목 실시간 수집
- 투자 대가 17개 전략 종합 분석
- Gemini AI를 통한 전문가 수준 분석
- 비동기 고속 병렬 처리
- 다양한 출력 형식 지원

사용법:
    python run_enhanced_system.py [옵션]

옵션:
    --output-format CONSOLE|JSON|HTML    출력 형식 선택 (기본: CONSOLE)
    --quick-mode                        빠른 테스트 모드 (상위 10개 종목만)
    --enable-gemini                     Gemini AI 분석 활성화
    --enable-telegram                   텔레그램 알림 활성화
    --save-results                      결과 파일 저장
    --verbose                          상세 로그 출력

예시:
    python run_enhanced_system.py --output-format JSON --enable-gemini --save-results
    python run_enhanced_system.py --quick-mode --verbose
"""

import asyncio
import argparse
import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback
import pandas as pd
import colorlog

# 로깅 설정
from loguru import logger

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from enhanced_data_collector import EnhancedDataCollector, StockData
    from enhanced_gemini_analyzer_fixed import EnhancedGeminiAnalyzer, InvestmentAnalysis, MarketInsight
    from optimized_investment_strategies import OptimizedInvestmentStrategies, StrategyScore
    logger.info("✅ 모든 핵심 모듈 임포트 성공")
except ImportError as e:
    logger.error(f"❌ 모듈 임포트 실패: {e}")
    logger.error("필수 파일들이 현재 디렉토리에 있는지 확인하세요:")
    logger.error("- enhanced_data_collector.py")
    logger.error("- enhanced_gemini_analyzer_fixed.py") 
    logger.error("- optimized_investment_strategies.py")
    sys.exit(1)


class EnhancedSystemRunner:
    """향상된 종합 투자 분석 시스템 실행기"""
    
    def __init__(self, output_dir: str = "results"):
        """
        초기화
        
        Args:
            output_dir: 결과 파일 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self._setup_logging()
        
        # 컴포넌트 초기화
        self.data_collector = None
        self.gemini_analyzer = None
        self.strategies = OptimizedInvestmentStrategies()
        
        # 결과 저장용
        self.results = {
            'timestamp': None,
            'markets': {},
            'top_stocks_by_market': {},
            'overall_top_stocks': [],
            'analysis_summary': {},
            'execution_time': 0
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        # 컬러 로그 포맷터
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        
        # 파일 핸들러
        log_file = self.output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 루트 로거 설정
        logging.basicConfig(
            level=logging.INFO,
            handlers=[console_handler, file_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("로깅 시스템 초기화 완료")
    
    async def initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.logger.info("=== 시스템 컴포넌트 초기화 시작 ===")
            
            # 데이터 수집기 초기화
            self.data_collector = EnhancedDataCollector()
            await self.data_collector.__aenter__()
            self.logger.info("✓ 데이터 수집기 초기화 완료")
            
            # Gemini AI 분석기 초기화
            self.gemini_analyzer = EnhancedGeminiAnalyzer()
            self.logger.info("✓ Gemini AI 분석기 초기화 완료")
            
            # 투자 전략 초기화 (이미 완료)
            self.logger.info("✓ 투자 전략 엔진 초기화 완료")
            
            self.logger.info("=== 모든 컴포넌트 초기화 완료 ===\n")
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    async def step1_collect_market_data(self) -> Dict[str, pd.DataFrame]:
        """1단계: 전체 시장 데이터 수집"""
        try:
            self.logger.info("🔍 1단계: 전체 시장 데이터 수집 시작")
            start_time = time.time()
            
            # 전체 시장 데이터 수집
            market_data = await self.data_collector.collect_all_market_data()
            
            if not market_data:
                raise Exception("시장 데이터 수집 실패")
            
            # 수집 결과 로깅
            total_stocks = sum(len(df) for df in market_data.values())
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"✓ 시장 데이터 수집 완료:")
            for market, df in market_data.items():
                self.logger.info(f"  - {market}: {len(df)}개 종목")
            self.logger.info(f"  - 총 {total_stocks}개 종목, {elapsed_time:.1f}초 소요\n")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패: {e}")
            return {}
    
    def step2_analyze_investment_strategies(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """2단계: 투자 대가 17개 전략 분석"""
        try:
            self.logger.info("📊 2단계: 투자 대가 17개 전략 분석 시작")
            start_time = time.time()
            
            strategy_results = {}
            
            for market, df in market_data.items():
                self.logger.info(f"  분석 중: {market} ({len(df)}개 종목)")
                
                # DataFrame을 딕셔너리 리스트로 변환
                stocks_data = df.to_dict('records')
                
                # 투자 전략 분석
                strategy_scores = self.strategies.analyze_multiple_stocks(stocks_data)
                
                # 원본 데이터에 전략 점수 추가
                for i, scores in enumerate(strategy_scores):
                    stocks_data[i]['strategy_scores'] = scores
                
                strategy_results[market] = stocks_data
            
            elapsed_time = time.time() - start_time
            total_analyzed = sum(len(stocks) for stocks in strategy_results.values())
            
            self.logger.info(f"✓ 투자 전략 분석 완료:")
            self.logger.info(f"  - 총 {total_analyzed}개 종목, {elapsed_time:.1f}초 소요")
            self.logger.info(f"  - 적용 전략: 17개 (워런 버핏, 벤저민 그레이엄, 피터 린치 등)\n")
            
            return strategy_results
            
        except Exception as e:
            self.logger.error(f"투자 전략 분석 실패: {e}")
            return {}
    
    def step3_calculate_technical_analysis(self, strategy_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """3단계: 기술적 분석 강화"""
        try:
            self.logger.info("📈 3단계: 기술적 분석 강화 수행")
            start_time = time.time()
            
            enhanced_results = {}
            
            for market, stocks in strategy_results.items():
                enhanced_stocks = []
                
                for stock in stocks:
                    # 기존 기술적 지표에 추가 분석 수행
                    tech_indicators = stock.get('technical_indicators', {})
                    
                    # 종합 기술적 점수 계산
                    tech_score = self._calculate_technical_score(tech_indicators, stock)
                    stock['technical_score'] = tech_score
                    
                    # 전략 점수와 기술적 점수 결합
                    strategy_scores = stock.get('strategy_scores', {})
                    combined_score = self._calculate_combined_score(strategy_scores, tech_score)
                    stock['combined_score'] = combined_score
                    
                    enhanced_stocks.append(stock)
                
                enhanced_results[market] = enhanced_stocks
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✓ 기술적 분석 강화 완료: {elapsed_time:.1f}초 소요\n")
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"기술적 분석 실패: {e}")
            return strategy_results
    
    async def step4_gemini_ai_analysis(self, enhanced_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """4단계: Gemini AI 종합 분석"""
        try:
            self.logger.info("🤖 4단계: Gemini AI 종합 분석 시작")
            start_time = time.time()
            
            gemini_results = {}
            
            for market, stocks in enhanced_results.items():
                self.logger.info(f"  Gemini AI 분석 중: {market} ({len(stocks)}개 종목)")
                
                # 상위 후보군 선정 (종합 점수 기준 상위 20%)
                sorted_stocks = sorted(stocks, key=lambda x: x.get('combined_score', 0), reverse=True)
                top_candidates = sorted_stocks[:max(10, len(sorted_stocks) // 5)]  # 최소 10개
                
                # Gemini AI 분석
                strategy_scores_list = [stock.get('strategy_scores', {}) for stock in top_candidates]
                
                if self.gemini_analyzer:
                    ai_analyses = await self.gemini_analyzer.analyze_stocks(top_candidates, strategy_scores_list)
                    
                    # 분석 결과를 원본 데이터에 매핑
                    ai_analysis_dict = {analysis.get('symbol'): analysis for analysis in ai_analyses}
                    
                    for stock in top_candidates:
                        symbol = stock.get('symbol')
                        if symbol in ai_analysis_dict:
                            stock['gemini_analysis'] = ai_analysis_dict[symbol]
                else:
                    self.logger.warning("Gemini AI 분석기 없음, 기본 분석 수행")
                    for stock in top_candidates:
                        stock['gemini_analysis'] = self._generate_basic_analysis(stock)
                
                gemini_results[market] = sorted_stocks  # 전체 리스트 유지
            
            elapsed_time = time.time() - start_time
            total_ai_analyzed = sum(
                len([s for s in stocks if 'gemini_analysis' in s]) 
                for stocks in gemini_results.values()
            )
            
            self.logger.info(f"✓ Gemini AI 분석 완료:")
            self.logger.info(f"  - AI 분석 종목: {total_ai_analyzed}개")
            self.logger.info(f"  - 소요 시간: {elapsed_time:.1f}초\n")
            
            return gemini_results
            
        except Exception as e:
            self.logger.error(f"Gemini AI 분석 실패: {e}")
            return enhanced_results
    
    def step5_select_top_stocks(self, gemini_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """5단계: 시장별 Top5 종목 선정"""
        try:
            self.logger.info("🏆 5단계: 시장별 Top5 종목 선정")
            
            market_top_stocks = {}
            overall_candidates = []
            
            for market, stocks in gemini_results.items():
                # Gemini 분석이 있는 종목들 우선 정렬
                gemini_analyzed = [s for s in stocks if 'gemini_analysis' in s]
                others = [s for s in stocks if 'gemini_analysis' not in s]
                
                # Gemini 점수로 정렬 (없으면 combined_score 사용)
                gemini_analyzed.sort(
                    key=lambda x: x.get('gemini_analysis', {}).get('종합_점수', 0), 
                    reverse=True
                )
                others.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
                
                # Top 5 선정
                top_5 = (gemini_analyzed + others)[:5]
                market_top_stocks[market] = top_5
                
                # 전체 후보에 추가 (시장 정보 포함)
                for stock in top_5:
                    stock_copy = stock.copy()
                    stock_copy['source_market'] = market
                    overall_candidates.append(stock_copy)
                
                # 로깅
                self.logger.info(f"  {market} Top 5:")
                for i, stock in enumerate(top_5, 1):
                    symbol = stock.get('symbol', 'Unknown')
                    score = stock.get('gemini_analysis', {}).get('종합_점수', 0)
                    if score == 0:
                        score = stock.get('combined_score', 0)
                    self.logger.info(f"    {i}. {symbol}: {score:.1f}점")
            
            # 전체 Top 10 선정
            overall_candidates.sort(
                key=lambda x: x.get('gemini_analysis', {}).get('종합_점수', x.get('combined_score', 0)),
                reverse=True
            )
            overall_top = overall_candidates[:10]
            
            self.logger.info(f"\n  🌟 전체 Top 10:")
            for i, stock in enumerate(overall_top, 1):
                symbol = stock.get('symbol', 'Unknown')
                market = stock.get('source_market', 'Unknown')
                score = stock.get('gemini_analysis', {}).get('종합_점수', 0)
                if score == 0:
                    score = stock.get('combined_score', 0)
                self.logger.info(f"    {i}. {symbol} ({market}): {score:.1f}점")
            
            self.results['top_stocks_by_market'] = market_top_stocks
            self.results['overall_top_stocks'] = overall_top
            
            self.logger.info("✓ Top 종목 선정 완료\n")
            
            return market_top_stocks
            
        except Exception as e:
            self.logger.error(f"Top 종목 선정 실패: {e}")
            return {}
    
    def step6_generate_reports(self, final_results: Dict) -> Dict:
        """6단계: 상세 분석 리포트 생성"""
        try:
            self.logger.info("📋 6단계: 상세 분석 리포트 생성")
            
            # 분석 요약 생성
            analysis_summary = {
                'timestamp': datetime.now().isoformat(),
                'total_markets': len(final_results),
                'total_analyzed_stocks': sum(len(stocks) for stocks in final_results.values()),
                'analysis_method': '17개 투자 대가 전략 + Gemini AI 종합 분석',
                'top_selection_criteria': 'Gemini AI 종합점수 우선, 기술적 분석 보조',
                'markets_covered': list(final_results.keys())
            }
            
            self.results['analysis_summary'] = analysis_summary
            
            # JSON 리포트 저장
            json_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
            
            # CSV 리포트 저장
            self._save_csv_reports(final_results)
            
            # HTML 리포트 생성
            self._generate_html_report()
            
            self.logger.info(f"✓ 리포트 생성 완료:")
            self.logger.info(f"  - JSON: {json_file}")
            self.logger.info(f"  - CSV: {self.output_dir}/top_stocks_*.csv")
            self.logger.info(f"  - HTML: {self.output_dir}/analysis_report.html\n")
            
            return analysis_summary
            
        except Exception as e:
            self.logger.error(f"리포트 생성 실패: {e}")
            return {}
    
    def step7_send_notifications(self, analysis_summary: Dict) -> bool:
        """7단계: 알림 발송 (선택적)"""
        try:
            self.logger.info("📱 7단계: 알림 발송 (선택적)")
            
            # 텔레그램 봇 설정이 있는 경우에만 발송
            telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
            telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if telegram_token and telegram_chat_id:
                # 텔레그램 알림 발송 (구현 생략)
                self.logger.info("  (텔레그램 알림 기능은 선택적으로 구현 가능)")
            else:
                self.logger.info("  텔레그램 설정 없음, 알림 생략")
            
            self.logger.info("✓ 알림 단계 완료\n")
            return True
            
        except Exception as e:
            self.logger.error(f"알림 발송 실패: {e}")
            return False
    
    def _calculate_technical_score(self, tech_indicators: Dict, stock_data: Dict) -> float:
        """기술적 점수 계산"""
        try:
            score = 50.0
            
            # RSI 점수 (25점)
            rsi = tech_indicators.get('RSI')
            if rsi:
                if 40 <= rsi <= 60:
                    score += 20
                elif 30 <= rsi <= 70:
                    score += 15
                else:
                    score += 5
            
            # 추세 점수 (25점)
            current_price = stock_data.get('current_price', 0)
            sma_20 = tech_indicators.get('SMA_20')
            if current_price and sma_20:
                if current_price > sma_20:
                    score += 20
                else:
                    score += 5
            
            # MACD 점수 (25점)
            macd = tech_indicators.get('MACD')
            if macd:
                if macd > 0:
                    score += 20
                else:
                    score += 10
            
            # 변동성 점수 (25점)
            volatility = tech_indicators.get('Volatility', 30)
            if volatility < 25:
                score += 20
            elif volatility < 40:
                score += 15
            else:
                score += 10
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"기술적 점수 계산 실패: {e}")
            return 50.0
    
    def _calculate_combined_score(self, strategy_scores: Dict, tech_score: float) -> float:
        """전략 점수와 기술적 점수 결합"""
        try:
            if not strategy_scores:
                return tech_score
            
            # 전략 점수 가중 평균 (70%)
            strategy_weights = {
                'warren_buffett': 0.15,
                'benjamin_graham': 0.12,
                'peter_lynch': 0.10,
                'john_templeton': 0.08,
                'philip_fisher': 0.08,
                'john_bogle': 0.07
            }
            
            weighted_strategy_score = 0.0
            total_weight = 0.0
            
            for strategy, score in strategy_scores.items():
                weight = strategy_weights.get(strategy, 0.03)  # 기본 3%
                weighted_strategy_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_strategy_score /= total_weight
            
            # 전략 70% + 기술적 30%
            combined = weighted_strategy_score * 0.7 + tech_score * 0.3
            
            return round(combined, 2)
            
        except Exception as e:
            self.logger.error(f"종합 점수 계산 실패: {e}")
            return tech_score
    
    def _generate_basic_analysis(self, stock_data: Dict) -> Dict:
        """기본 분석 생성 (Gemini AI 없을 때)"""
        return {
            '종합_점수': stock_data.get('combined_score', 50),
            '투자_액션': '보유',
            '목표가': stock_data.get('current_price', 0) * 1.1,
            '기대수익률': 10.0,
            '투자_등급': '보유',
            '리스크_수준': '보통',
            '투자_기간': '중기',
            '핵심_강점': ['기술적 분석 기반'],
            '주요_리스크': ['시장 변동성'],
            '분석_요약': '기본 분석 결과',
            'api_mode': 'basic'
        }
    
    def _save_csv_reports(self, final_results: Dict):
        """CSV 리포트 저장"""
        try:
            # 시장별 CSV 저장
            for market, stocks in final_results.items():
                if not stocks:
                    continue
                
                # 기본 정보 추출
                csv_data = []
                for stock in stocks:
                    gemini_analysis = stock.get('gemini_analysis', {})
                    row = {
                        'Symbol': stock.get('symbol', ''),
                        'Market': market,
                        'Current_Price': stock.get('current_price', 0),
                        'Market_Cap': stock.get('market_cap', 0),
                        'Combined_Score': stock.get('combined_score', 0),
                        'Gemini_Score': gemini_analysis.get('종합_점수', 0),
                        'Investment_Action': gemini_analysis.get('투자_액션', ''),
                        'Target_Price': gemini_analysis.get('목표가', 0),
                        'Expected_Return': gemini_analysis.get('기대수익률', 0),
                        'Risk_Level': gemini_analysis.get('리스크_수준', ''),
                        'Investment_Grade': gemini_analysis.get('투자_등급', ''),
                        'Sector': stock.get('sector', ''),
                        'PE_Ratio': stock.get('pe_ratio', ''),
                        'RSI': stock.get('technical_indicators', {}).get('RSI', ''),
                        'Analysis_Summary': gemini_analysis.get('분석_요약', '')
                    }
                    csv_data.append(row)
                
                # CSV 저장
                df = pd.DataFrame(csv_data)
                csv_file = self.output_dir / f"top_stocks_{market.lower()}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            self.logger.error(f"CSV 리포트 저장 실패: {e}")
    
    def _generate_html_report(self):
        """HTML 리포트 생성"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>투자 분석 리포트</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                    .summary {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
                    .market-section {{ margin: 20px 0; }}
                    .stock-item {{ padding: 10px; margin: 5px 0; border: 1px solid #ddd; }}
                    .score {{ font-weight: bold; color: #007bff; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>AI 기반 투자 분석 리포트</h1>
                    <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <h2>분석 요약</h2>
                    <p>• 분석 대상: 코스피200, 나스닥100, S&P500 전체 종목</p>
                    <p>• 분석 방법: 17개 투자 대가 전략 + Gemini AI 종합 분석</p>
                    <p>• 선정 기준: 기술적 분석 중심, 재무정보 제외</p>
                </div>
                
                <div class="market-section">
                    <h2>시장별 Top 5 종목</h2>
                    <p>상세 결과는 생성된 CSV 파일을 참조하세요.</p>
                </div>
            </body>
            </html>
            """
            
            html_file = self.output_dir / "analysis_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"HTML 리포트 생성 실패: {e}")
    
    async def run_complete_analysis(self) -> Dict:
        """완전한 7단계 분석 프로세스 실행"""
        try:
            start_time = time.time()
            self.results['timestamp'] = datetime.now().isoformat()
            
            self.logger.info("🚀 향상된 종합 투자 분석 시스템 시작")
            self.logger.info("=" * 60)
            
            # 컴포넌트 초기화
            await self.initialize_components()
            
            # 7단계 분석 프로세스
            market_data = await self.step1_collect_market_data()
            if not market_data:
                raise Exception("시장 데이터 수집 실패로 분석 중단")
            
            strategy_results = self.step2_analyze_investment_strategies(market_data)
            if not strategy_results:
                raise Exception("투자 전략 분석 실패로 분석 중단")
            
            enhanced_results = self.step3_calculate_technical_analysis(strategy_results)
            
            gemini_results = await self.step4_gemini_ai_analysis(enhanced_results)
            
            top_stocks = self.step5_select_top_stocks(gemini_results)
            
            analysis_summary = self.step6_generate_reports(top_stocks)
            
            self.step7_send_notifications(analysis_summary)
            
            # 실행 시간 기록
            total_time = time.time() - start_time
            self.results['execution_time'] = round(total_time, 2)
            
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 전체 분석 완료! 총 소요 시간: {total_time:.1f}초")
            self.logger.info(f"📁 결과 파일 위치: {self.output_dir.absolute()}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"분석 시스템 실행 실패: {e}")
            raise
        finally:
            # 리소스 정리
            if self.data_collector:
                await self.data_collector.__aexit__(None, None, None)
    
    async def run_quick_test(self) -> Dict:
        """빠른 테스트 실행 (소수 종목만)"""
        try:
            self.logger.info("🔬 빠른 테스트 모드 실행")
            
            await self.initialize_components()
            
            # 테스트용 소수 종목만 수집
            test_stocks = {
                'NASDAQ100': ['AAPL', 'MSFT', 'GOOGL'],
                'SP500': ['JPM', 'XOM', 'JNJ']
            }
            
            results = {}
            for market, symbols in test_stocks.items():
                stocks_data = []
                for symbol in symbols:
                    stock_data = await self.data_collector.get_stock_data(symbol, market)
                    if stock_data:
                        stocks_data.append(stock_data)
                
                if stocks_data:
                    # 투자 전략 분석
                    strategy_scores = self.strategies.analyze_multiple_stocks(stocks_data)
                    for i, scores in enumerate(strategy_scores):
                        stocks_data[i]['strategy_scores'] = scores
                    
                    results[market] = stocks_data
            
            self.logger.info(f"✓ 테스트 완료: {sum(len(stocks) for stocks in results.values())}개 종목")
            return results
            
        except Exception as e:
            self.logger.error(f"테스트 실행 실패: {e}")
            return {}
        finally:
            if self.data_collector:
                await self.data_collector.__aexit__(None, None, None)


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="향상된 주식 투자 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s --output-format JSON --enable-gemini --save-results
  %(prog)s --quick-mode --verbose
  %(prog)s --output-format HTML --enable-gemini
        """
    )
    
    parser.add_argument(
        '--output-format',
        choices=['CONSOLE', 'JSON', 'HTML'],
        default='CONSOLE',
        help='출력 형식 (기본: CONSOLE)'
    )
    
    parser.add_argument(
        '--quick-mode',
        action='store_true',
        help='빠른 테스트 모드 (각 시장당 10개 종목만)'
    )
    
    parser.add_argument(
        '--enable-gemini',
        action='store_true',
        help='Gemini AI 분석 활성화'
    )
    
    parser.add_argument(
        '--enable-telegram',
        action='store_true',
        help='텔레그램 알림 활성화'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='결과를 JSON 파일로 저장'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세한 로그 출력'
    )
    
    return parser.parse_args()


async def main():
    """메인 실행 함수"""
    try:
        # 실행 모드 선택
        mode = input("실행 모드를 선택하세요 (1: 전체 분석, 2: 빠른 테스트): ").strip()
        
        system = EnhancedSystemRunner()
        
        if mode == "2":
            results = await system.run_quick_test()
            print("\n=== 테스트 결과 ===")
            for market, stocks in results.items():
                print(f"{market}: {len(stocks)}개 종목")
                for stock in stocks:
                    symbol = stock.get('symbol', 'Unknown')
                    scores = stock.get('strategy_scores', {})
                    avg_score = sum(scores.values()) / len(scores) if scores else 0
                    print(f"  {symbol}: 평균 점수 {avg_score:.1f}")
        else:
            results = await system.run_complete_analysis()
            print("\n=== 전체 분석 완료 ===")
            print(f"총 실행 시간: {results.get('execution_time', 0)}초")
            print(f"결과 파일: {system.output_dir.absolute()}")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"실행 실패: {e}")


if __name__ == "__main__":
    # 환경 변수 확인 및 안내
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️ 참고: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   Gemini AI 분석을 사용하려면 다음과 같이 설정하세요:")
        print("   Windows: set GEMINI_API_KEY=your_api_key")
        print("   Linux/Mac: export GEMINI_API_KEY=your_api_key")
        print("   또는 .env 파일에 GEMINI_API_KEY=your_api_key 추가")
        print()
    
    asyncio.run(main()) 