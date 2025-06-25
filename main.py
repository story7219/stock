#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 코스피200·나스닥100·S&P500 투자 대가 전략 기반 Top5 종목 자동 선정 시스템
=======================================================================
한국투자증권 API + Gemini AI + 투자 대가 전략 (워런 버핏, 피터 린치, 벤저민 그레이엄)
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List

# 프로젝트 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from kis_stock_analyzer import KISStockAnalyzer
from gemini_ai_analyzer import GeminiAIAnalyzer

class MasterStockAnalyzer:
    """마스터 주식 분석 시스템"""
    
    def __init__(self):
        print("🚀 코스피200·나스닥100·S&P500 투자 대가 전략 기반 분석 시스템")
        print("=" * 70)
        
        # 서브 시스템 초기화
        self.kis_analyzer = KISStockAnalyzer()
        self.gemini_analyzer = GeminiAIAnalyzer()
        
        print("🔴 한국투자증권 API 시스템 로드 완료")
        print("🤖 Gemini AI 분석 시스템 로드 완료")
        print("=" * 70)
    
    def run_complete_analysis(self) -> Dict:
        """전체 분석 프로세스 실행"""
        print("\n🎯 1단계: 한국투자증권 API 데이터 수집")
        print("-" * 50)
        
        # 1. 한투 API로 데이터 수집
        stock_data = self.kis_analyzer.run_data_collection()
        
        if not stock_data:
            print("❌ 데이터 수집 실패. 분석을 중단합니다.")
            return {}
        
        print(f"✅ 데이터 수집 완료: 총 {self._count_collected_stocks(stock_data)}개 종목")
        
        print("\n🎯 2단계: Gemini AI 투자 대가 전략 분석")
        print("-" * 50)
        
        # 2. Gemini AI 분석
        analysis_result = self.gemini_analyzer.run_complete_analysis(stock_data)
        
        print("\n🎯 3단계: 최종 결과 정리")
        print("-" * 50)
        
        # 3. 최종 결과 통합
        final_result = {
            'system_info': {
                'name': '코스피200·나스닥100·S&P500 투자 대가 전략 기반 Top5 종목 자동 선정 시스템',
                'version': '1.0.0',
                'execution_time': datetime.now(),
                'data_sources': ['한국투자증권 API', 'Gemini AI'],
                'strategies': ['워런 버핏', '피터 린치', '벤저민 그레이엄']
            },
            'collected_data': stock_data,
            'analysis_result': analysis_result,
            'summary': self._generate_summary(stock_data, analysis_result)
        }
        
        return final_result
    
    def _count_collected_stocks(self, stock_data: Dict) -> int:
        """수집된 종목 수 계산"""
        total = 0
        for category, stocks in stock_data.items():
            if isinstance(stocks, list):
                total += len(stocks)
        return total
    
    def _generate_summary(self, stock_data: Dict, analysis_result: Dict) -> Dict:
        """결과 요약 생성"""
        final_top5 = analysis_result.get('final_selection', {}).get('final_top5', [])
        
        return {
            'data_collection': {
                'sp500_futures': len(stock_data.get('sp500_futures', [])),
                'nasdaq100': len(stock_data.get('nasdaq100', [])),
                'kospi200_related': len(stock_data.get('kospi200_related', [])),
                'total_collected': self._count_collected_stocks(stock_data)
            },
            'strategy_analysis': {
                'warren_buffett_candidates': len(analysis_result.get('strategies', {}).get('warren_buffett', {}).get('top_candidates', [])),
                'peter_lynch_candidates': len(analysis_result.get('strategies', {}).get('peter_lynch', {}).get('top_candidates', [])),
                'benjamin_graham_candidates': len(analysis_result.get('strategies', {}).get('benjamin_graham', {}).get('top_candidates', []))
            },
            'final_selection': {
                'top5_count': len(final_top5),
                'selection_method': analysis_result.get('final_selection', {}).get('reasoning', 'Unknown'),
                'top5_symbols': [stock['symbol'] for stock in final_top5]
            }
        }
    
    def save_results(self, results: Dict) -> str:
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"master_analysis_result_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        return filename
    
    def print_final_report(self, results: Dict):
        """최종 리포트 출력"""
        print("\n" + "=" * 70)
        print("🏆 최종 분석 결과 리포트")
        print("=" * 70)
        
        # 시스템 정보
        system_info = results.get('system_info', {})
        print(f"📊 시스템: {system_info.get('name', 'Unknown')}")
        print(f"🕐 실행 시간: {system_info.get('execution_time', 'Unknown')}")
        print(f"📈 데이터 소스: {', '.join(system_info.get('data_sources', []))}")
        print(f"🎯 투자 전략: {', '.join(system_info.get('strategies', []))}")
        
        # 데이터 수집 결과
        summary = results.get('summary', {})
        data_collection = summary.get('data_collection', {})
        print(f"\n📊 데이터 수집 결과:")
        print(f"  📈 S&P500 선물: {data_collection.get('sp500_futures', 0)}개")
        print(f"  📊 나스닥100: {data_collection.get('nasdaq100', 0)}개")
        print(f"  🇰🇷 코스피200 관련: {data_collection.get('kospi200_related', 0)}개")
        print(f"  🎯 총 수집 종목: {data_collection.get('total_collected', 0)}개")
        
        # 전략별 분석 결과
        strategy_analysis = summary.get('strategy_analysis', {})
        print(f"\n🎯 투자 대가 전략별 분석:")
        print(f"  💼 워런 버핏 후보: {strategy_analysis.get('warren_buffett_candidates', 0)}개")
        print(f"  📈 피터 린치 후보: {strategy_analysis.get('peter_lynch_candidates', 0)}개")
        print(f"  💎 벤저민 그레이엄 후보: {strategy_analysis.get('benjamin_graham_candidates', 0)}개")
        
        # 최종 Top5 결과
        final_selection = summary.get('final_selection', {})
        final_top5 = results.get('analysis_result', {}).get('final_selection', {}).get('final_top5', [])
        
        print(f"\n🏆 Gemini AI 선정 Top5 종목:")
        for stock in final_top5:
            print(f"  {stock['rank']}. {stock['symbol']} (점수: {stock['final_score']:.1f}, 비중: {stock['recommended_weight']}%)")
            print(f"     └─ {stock['ai_reasoning']}")
        
        print(f"\n📝 선정 방식: {final_selection.get('selection_method', 'Unknown')}")
        
        print("\n" + "=" * 70)
        print("✅ 분석 완료!")
        print("=" * 70)

def main():
    """메인 실행 함수"""
    try:
        # 마스터 분석기 생성
        master_analyzer = MasterStockAnalyzer()
        
        # 전체 분석 실행
        results = master_analyzer.run_complete_analysis()
        
        if not results:
            print("❌ 분석 실행 실패")
            return
        
        # 결과 저장
        filename = master_analyzer.save_results(results)
        print(f"\n💾 결과 저장 완료: {filename}")
        
        # 최종 리포트 출력
        master_analyzer.print_final_report(results)
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 