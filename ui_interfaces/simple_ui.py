"""
투자 추천 시스템 간단 테스트

20명의 유명 투자자 전략을 종합하여 최적의 투자 종목을 추천합니다.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict
import json
from datetime import datetime

from recommenders.recommender import InvestmentRecommender
from data.sample_data import get_sample_stocks

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_recommendations(recommendations: List[Dict], report: Dict):
    """추천 결과 출력"""
    print("\n" + "="*80)
    print("🏆 투자 추천 시스템 - TOP 5 종목 추천")
    print("="*80)
    
    # 요약 정보
    summary = report.get('summary', {})
    print(f"\n📊 분석 요약:")
    print(f"   • 총 추천 종목: {summary.get('total_recommendations', 0)}개")
    print(f"   • 평균 종합 점수: {summary.get('average_composite_score', 0):.1f}점")
    print(f"   • 평균 신뢰도: {summary.get('average_confidence', 0):.1f}%")
    print(f"   • 평균 유효 전략 수: {summary.get('average_valid_strategies', 0):.1f}개")
    
    # 투자 스타일 분포
    style_dist = summary.get('investment_style_distribution', {})
    if style_dist:
        print(f"\n🎯 투자 스타일 분포:")
        for style, count in style_dist.items():
            print(f"   • {style}: {count}개")
    
    # TOP 5 종목 상세
    print(f"\n🌟 TOP 5 추천 종목:")
    print("-" * 80)
    
    for i, stock in enumerate(recommendations[:5], 1):
        print(f"\n{i}. {stock.get('name', 'Unknown')} ({stock.get('symbol', 'N/A')})")
        print(f"   종합점수: {stock.get('composite_score', 0):.1f}점")
        print(f"   추천강도: {stock.get('recommendation_strength', 'N/A')}")
        print(f"   투자스타일: {stock.get('investment_style', 'N/A')}")
        print(f"   지배전략: {stock.get('dominant_strategy', 'N/A')}")
        print(f"   신뢰도: {stock.get('confidence', 0):.1f}%")
        print(f"   유효전략수: {stock.get('valid_strategies_count', 0)}개")
        
        # 상위 3개 전략 점수
        strategy_scores = []
        for strategy_name in ['buffett', 'lynch', 'graham', 'fisher', 'soros', 'dalio']:
            score = stock.get(f'{strategy_name}_score', 0)
            if score > 0:
                strategy_scores.append((strategy_name, score))
        
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        if strategy_scores:
            print(f"   상위전략: ", end="")
            for j, (strategy, score) in enumerate(strategy_scores[:3]):
                print(f"{strategy}({score:.0f})", end="")
                if j < min(2, len(strategy_scores) - 1):
                    print(", ", end="")
            print()
    
    # 최고 성과 전략
    top_strategies = report.get('top_performing_strategies', {})
    if top_strategies:
        print(f"\n🏅 최고 성과 전략 TOP 5:")
        for i, (strategy, score) in enumerate(list(top_strategies.items())[:5], 1):
            print(f"   {i}. {strategy}: {score:.1f}점")

def save_results_to_file(recommendations: List[Dict], report: Dict):
    """결과를 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"투자추천결과_{timestamp}.json"
    
    results = {
        'timestamp': timestamp,
        'recommendations': recommendations,
        'report': report
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 '{filename}' 파일로 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 투자 추천 시스템 시작")
    print("20명의 유명 투자자 전략을 종합 분석합니다...")
    
    try:
        # 1. 추천 시스템 초기화
        print("\n📈 추천 시스템 초기화 중...")
        recommender = InvestmentRecommender()
        
        # 2. 샘플 데이터 로드
        print("📊 주식 데이터 로드 중...")
        stocks_data = get_sample_stocks()
        print(f"총 {len(stocks_data)}개 종목 로드 완료")
        
        # 3. TOP 5 추천 생성
        print("\n🔍 종합 분석 및 추천 생성 중...")
        recommendations = recommender.get_top_recommendations(stocks_data, top_n=5)
        
        # 4. 추천 보고서 생성
        print("📋 추천 보고서 생성 중...")
        report = recommender.generate_recommendation_report(recommendations)
        
        # 5. 결과 출력
        display_recommendations(recommendations, report)
        
        # 6. 결과 파일 저장
        save_results_to_file(recommendations, report)
        
        # 7. 다양화 포트폴리오 추천 (옵션)
        print(f"\n🎯 다양화 포트폴리오 생성 중...")
        portfolio = recommender.get_diversified_portfolio(stocks_data, portfolio_size=10)
        
        print(f"\n📊 다양화 포트폴리오 (10개 종목):")
        for i, stock in enumerate(portfolio, 1):
            print(f"{i:2d}. {stock.get('name', 'Unknown'):20s} "
                  f"({stock.get('investment_style', 'N/A'):10s}) "
                  f"{stock.get('composite_score', 0):5.1f}점")
        
        print(f"\n✅ 투자 추천 시스템 완료!")
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {e}")
        print(f"\n❌ 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 