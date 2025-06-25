#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제시 리버모어 전략 자동 추천 테스트
한국주식(코스피200) 및 미국주식(나스닥100, S&P500) Top5 종목 추천 테스트
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from modules.independent_strategy_runner import (
    IndependentStrategyRunner, 
    run_livermore_strategy,
    StrategyResult,
    MarketRecommendation
)

class LivermoreRecommendationTester:
    """제시 리버모어 전략 추천 테스터"""
    
    def __init__(self):
        self.runner = IndependentStrategyRunner()
        self.test_results = {}
        
    async def test_korean_stocks_recommendation(self) -> Dict[str, StrategyResult]:
        """한국주식 Top5 추천 테스트"""
        print("🇰🇷 제시 리버모어 전략 - 한국주식 Top5 추천 테스트 시작...")
        print("=" * 80)
        
        try:
            # 코스피200 종목 대상으로 제시 리버모어 전략 실행
            results = await self.runner.run_single_strategy(
                strategy_name='jesse_livermore',
                market='kospi200',
                top_n=5
            )
            
            if 'kospi200' in results:
                result = results['kospi200']
                print(f"✅ 분석 완료!")
                print(f"📊 분석 종목 수: {result.total_analyzed}개")
                print(f"⏱️ 실행 시간: {result.execution_time:.2f}초")
                print(f"🕐 실행 시각: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"\n🏆 한국주식 Top5 추천 종목:")
                print("-" * 60)
                
                for rec in result.recommendations:
                    print(f"\n{rec.rank}. 🏢 {rec.name} ({rec.symbol})")
                    print(f"   💰 현재가: {rec.current_price:,.0f}원")
                    print(f"   📈 진입가: {rec.entry_price:,.0f}원")
                    print(f"   🎯 목표가: {rec.target_price:,.0f}원")
                    print(f"   🛑 손절가: {rec.stop_loss:,.0f}원")
                    print(f"   📊 기대수익률: {rec.expected_return:.1f}%")
                    print(f"   ⭐ 점수: {rec.score:.1f}/100")
                    print(f"   🎲 신뢰도: {rec.confidence:.1%}")
                    print(f"   ⚠️ 리스크: {rec.risk_level}")
                    print(f"   📅 투자기간: {rec.investment_period}")
                    print(f"   💡 추천이유: {rec.reasoning}")
                
                print(f"\n🤖 Gemini AI 종합 분석:")
                print("-" * 60)
                print(result.gemini_analysis)
                
                return results
            else:
                print("❌ 한국주식 분석 결과를 찾을 수 없습니다.")
                return {}
                
        except Exception as e:
            print(f"❌ 한국주식 분석 중 오류 발생: {e}")
            return {}
    
    async def test_us_stocks_recommendation(self) -> Dict[str, StrategyResult]:
        """미국주식 Top5 추천 테스트"""
        print("\n" + "=" * 80)
        print("🇺🇸 제시 리버모어 전략 - 미국주식 Top5 추천 테스트 시작...")
        print("=" * 80)
        
        try:
            # 나스닥100 + S&P500 종목 대상으로 제시 리버모어 전략 실행
            nasdaq_results = await self.runner.run_single_strategy(
                strategy_name='jesse_livermore',
                market='nasdaq100',
                top_n=3  # 나스닥에서 3개
            )
            
            sp500_results = await self.runner.run_single_strategy(
                strategy_name='jesse_livermore',
                market='sp500',
                top_n=2  # S&P500에서 2개
            )
            
            # 결과 통합
            all_us_results = {}
            all_us_results.update(nasdaq_results)
            all_us_results.update(sp500_results)
            
            print("✅ 미국주식 분석 완료!")
            
            # 나스닥100 결과 출력
            if 'nasdaq100' in nasdaq_results:
                nasdaq_result = nasdaq_results['nasdaq100']
                print(f"\n📈 나스닥100 Top3 추천:")
                print("-" * 60)
                print(f"📊 분석 종목 수: {nasdaq_result.total_analyzed}개")
                print(f"⏱️ 실행 시간: {nasdaq_result.execution_time:.2f}초")
                
                for rec in nasdaq_result.recommendations:
                    print(f"\n{rec.rank}. 🏢 {rec.name} ({rec.symbol})")
                    print(f"   💰 현재가: ${rec.current_price:,.2f}")
                    print(f"   📈 진입가: ${rec.entry_price:,.2f}")
                    print(f"   🎯 목표가: ${rec.target_price:,.2f}")
                    print(f"   🛑 손절가: ${rec.stop_loss:,.2f}")
                    print(f"   📊 기대수익률: {rec.expected_return:.1f}%")
                    print(f"   ⭐ 점수: {rec.score:.1f}/100")
                    print(f"   🎲 신뢰도: {rec.confidence:.1%}")
                    print(f"   ⚠️ 리스크: {rec.risk_level}")
                    print(f"   📅 투자기간: {rec.investment_period}")
                    print(f"   💡 추천이유: {rec.reasoning}")
                
                print(f"\n🤖 나스닥100 Gemini AI 분석:")
                print("-" * 40)
                print(nasdaq_result.gemini_analysis)
            
            # S&P500 결과 출력
            if 'sp500' in sp500_results:
                sp500_result = sp500_results['sp500']
                print(f"\n📈 S&P500 Top2 추천:")
                print("-" * 60)
                print(f"📊 분석 종목 수: {sp500_result.total_analyzed}개")
                print(f"⏱️ 실행 시간: {sp500_result.execution_time:.2f}초")
                
                for rec in sp500_result.recommendations:
                    print(f"\n{rec.rank}. 🏢 {rec.name} ({rec.symbol})")
                    print(f"   💰 현재가: ${rec.current_price:,.2f}")
                    print(f"   📈 진입가: ${rec.entry_price:,.2f}")
                    print(f"   🎯 목표가: ${rec.target_price:,.2f}")
                    print(f"   🛑 손절가: ${rec.stop_loss:,.2f}")
                    print(f"   📊 기대수익률: {rec.expected_return:.1f}%")
                    print(f"   ⭐ 점수: {rec.score:.1f}/100")
                    print(f"   🎲 신뢰도: {rec.confidence:.1%}")
                    print(f"   ⚠️ 리스크: {rec.risk_level}")
                    print(f"   📅 투자기간: {rec.investment_period}")
                    print(f"   💡 추천이유: {rec.reasoning}")
                
                print(f"\n🤖 S&P500 Gemini AI 분석:")
                print("-" * 40)
                print(sp500_result.gemini_analysis)
            
            return all_us_results
            
        except Exception as e:
            print(f"❌ 미국주식 분석 중 오류 발생: {e}")
            return {}
    
    async def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🎯 제시 리버모어 전략 자동 추천 시스템 테스트")
        print("=" * 80)
        print("📅 테스트 시작 시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("👨‍💼 투자 대가: 제시 리버모어 (Jesse Livermore)")
        print("📈 전략 특징: 시장 심리와 모멘텀 기반 단기 트레이딩")
        print("🎯 목표: 한국주식 Top5 + 미국주식 Top5 자동 추천")
        
        start_time = datetime.now()
        
        # 1. 한국주식 테스트
        korean_results = await self.test_korean_stocks_recommendation()
        
        # 2. 미국주식 테스트  
        us_results = await self.test_us_stocks_recommendation()
        
        # 3. 결과 통합 및 저장
        all_results = {}
        all_results.update(korean_results)
        all_results.update(us_results)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 4. 최종 결과 요약
        print("\n" + "=" * 80)
        print("📊 제시 리버모어 전략 자동 추천 테스트 완료!")
        print("=" * 80)
        print(f"⏱️ 총 실행 시간: {total_time:.2f}초")
        print(f"🕐 완료 시각: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_recommendations = sum(len(result.recommendations) for result in all_results.values())
        print(f"🏆 총 추천 종목 수: {total_recommendations}개")
        
        # 시장별 요약
        if korean_results:
            kr_count = sum(len(result.recommendations) for result in korean_results.values())
            print(f"🇰🇷 한국주식 추천: {kr_count}개")
        
        if us_results:
            us_count = sum(len(result.recommendations) for result in us_results.values())
            print(f"🇺🇸 미국주식 추천: {us_count}개")
        
        # 5. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_test_results(all_results, f"livermore_recommendation_test_{timestamp}")
        
        print(f"\n✅ 테스트 완료! 결과가 저장되었습니다.")
        
        return all_results
    
    def save_test_results(self, results: Dict[str, StrategyResult], filename: str):
        """테스트 결과 저장"""
        try:
            # 결과 디렉토리 생성
            results_dir = Path("reports/livermore_tests")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # JSON 직렬화 가능하도록 변환
            serializable_results = {}
            for market, result in results.items():
                serializable_results[market] = {
                    'strategy_name': result.strategy_name,
                    'market': result.market,
                    'total_analyzed': result.total_analyzed,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat(),
                    'gemini_analysis': result.gemini_analysis,
                    'recommendations': []
                }
                
                for rec in result.recommendations:
                    serializable_results[market]['recommendations'].append({
                        'symbol': rec.symbol,
                        'name': rec.name,
                        'market': rec.market,
                        'strategy_name': rec.strategy_name,
                        'current_price': rec.current_price,
                        'entry_price': rec.entry_price,
                        'target_price': rec.target_price,
                        'stop_loss': rec.stop_loss,
                        'expected_return': rec.expected_return,
                        'score': rec.score,
                        'rank': rec.rank,
                        'reasoning': rec.reasoning,
                        'confidence': rec.confidence,
                        'risk_level': rec.risk_level,
                        'investment_period': rec.investment_period
                    })
            
            # JSON 파일 저장
            json_filepath = results_dir / f"{filename}.json"
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            # 리포트 파일 저장
            report_filepath = results_dir / f"{filename}_report.txt"
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(self.generate_detailed_report(results))
            
            print(f"📁 JSON 결과: {json_filepath}")
            print(f"📄 리포트: {report_filepath}")
            
        except Exception as e:
            print(f"❌ 결과 저장 중 오류: {e}")
    
    def generate_detailed_report(self, results: Dict[str, StrategyResult]) -> str:
        """상세 리포트 생성"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("🎯 제시 리버모어 전략 자동 추천 시스템 테스트 리포트")
        report_lines.append("=" * 100)
        report_lines.append(f"📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"👨‍💼 투자 대가: 제시 리버모어 (Jesse Livermore)")
        report_lines.append(f"📈 전략 특징: 시장 심리와 모멘텀 기반 단기 트레이딩")
        
        total_recommendations = sum(len(result.recommendations) for result in results.values())
        report_lines.append(f"🏆 총 추천 종목: {total_recommendations}개")
        
        for market, result in results.items():
            market_names = {
                'kospi200': '🇰🇷 한국 코스피200',
                'nasdaq100': '🇺🇸 미국 나스닥100', 
                'sp500': '🇺🇸 미국 S&P500'
            }
            
            market_name = market_names.get(market, market.upper())
            
            report_lines.append(f"\n" + "=" * 80)
            report_lines.append(f"📈 {market_name}")
            report_lines.append("=" * 80)
            report_lines.append(f"📊 분석 종목 수: {result.total_analyzed}개")
            report_lines.append(f"⏱️ 실행 시간: {result.execution_time:.2f}초")
            report_lines.append(f"🕐 실행 시각: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            report_lines.append(f"\n🏆 Top {len(result.recommendations)} 추천 종목:")
            report_lines.append("-" * 80)
            
            for rec in result.recommendations:
                currency = "원" if market == 'kospi200' else "$"
                report_lines.append(f"\n{rec.rank}. 🏢 {rec.name} ({rec.symbol})")
                report_lines.append(f"   💰 현재가: {currency}{rec.current_price:,.2f}")
                report_lines.append(f"   📈 진입가: {currency}{rec.entry_price:,.2f}")
                report_lines.append(f"   🎯 목표가: {currency}{rec.target_price:,.2f}")
                report_lines.append(f"   🛑 손절가: {currency}{rec.stop_loss:,.2f}")
                report_lines.append(f"   📊 기대수익률: {rec.expected_return:.1f}%")
                report_lines.append(f"   ⭐ 점수: {rec.score:.1f}/100")
                report_lines.append(f"   🎲 신뢰도: {rec.confidence:.1%}")
                report_lines.append(f"   ⚠️ 리스크: {rec.risk_level}")
                report_lines.append(f"   📅 투자기간: {rec.investment_period}")
                report_lines.append(f"   💡 추천이유: {rec.reasoning}")
            
            report_lines.append(f"\n🤖 Gemini AI 종합 분석:")
            report_lines.append("-" * 80)
            report_lines.append(result.gemini_analysis)
        
        report_lines.append(f"\n" + "=" * 100)
        report_lines.append("✅ 테스트 완료")
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)

async def main():
    """메인 테스트 실행 함수"""
    tester = LivermoreRecommendationTester()
    
    try:
        results = await tester.run_comprehensive_test()
        
        if results:
            print("\n🎉 제시 리버모어 전략 자동 추천 테스트가 성공적으로 완료되었습니다!")
            print("📊 추천 결과를 확인하여 투자 결정에 참고하시기 바랍니다.")
        else:
            print("\n❌ 테스트 실행 중 문제가 발생했습니다.")
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 제시 리버모어 전략 자동 추천 테스트 시작...")
    asyncio.run(main()) 