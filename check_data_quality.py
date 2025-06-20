#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 품질 확인 스크립트
실제로 데이터가 부족한지 확인해봅니다.
"""

import asyncio
import sys
import json
from datetime import datetime
from pprint import pprint

# 프로젝트 루트 경로 추가
sys.path.append('.')

from personal_blackrock.stock_data_manager import DataManager


async def check_data_quality():
    """데이터 품질 상세 확인"""
    print("🔍 데이터 품질 상세 확인 시작")
    print("=" * 80)
    
    try:
        # DataManager 생성
        print("📊 DataManager 초기화 중...")
        data_manager = DataManager(preload_data=False)
        print("✅ DataManager 초기화 완료")
        
        # 테스트 종목들
        test_stocks = [
            '005930',  # 삼성전자
            '000660',  # SK하이닉스
            '035420',  # NAVER
            '051910',  # LG화학
        ]
        
        for stock_code in test_stocks:
            print(f"\n📈 {stock_code} 데이터 품질 검사")
            print("-" * 60)
            
            # 1. 종합 데이터 조회
            comprehensive_data = data_manager.get_comprehensive_stock_data(stock_code)
            
            if comprehensive_data:
                print(f"✅ 종합 데이터 조회 성공")
                print(f"   - 종목명: {comprehensive_data.get('company_name', 'N/A')}")
                print(f"   - 현재가: {comprehensive_data.get('current_price', 'N/A'):,}원")
                print(f"   - 시가총액: {comprehensive_data.get('market_cap', 0):,}원")
                print(f"   - 데이터 소스: {comprehensive_data.get('data_source', 'N/A')}")
                
                # 2. 차트 데이터 확인
                chart_data = comprehensive_data.get('chart_data')
                if chart_data is not None and not chart_data.empty:
                    print(f"   - 차트 데이터: {len(chart_data)}일치 (최신: {chart_data.index[-1]})")
                else:
                    print("   - ❌ 차트 데이터 없음")
                
                # 3. 펀더멘털 데이터 확인
                fundamentals = comprehensive_data.get('fundamentals', {})
                if fundamentals:
                    print(f"   - 펀더멘털 데이터:")
                    print(f"     • PER: {fundamentals.get('per', 'N/A')}")
                    print(f"     • PBR: {fundamentals.get('pbr', 'N/A')}")
                    print(f"     • ROE: {fundamentals.get('roe', 'N/A')}%")
                    print(f"     • EPS: {fundamentals.get('eps', 'N/A'):,}원")
                    print(f"     • 부채비율: {fundamentals.get('debt_ratio', 'N/A')}%")
                else:
                    print("   - ❌ 펀더멘털 데이터 없음")
                
                # 4. 수급 데이터 확인
                supply_demand = comprehensive_data.get('supply_demand', {})
                if supply_demand:
                    print(f"   - 수급 데이터:")
                    print(f"     • 외국인 순매수: {supply_demand.get('foreign_net_buy', 'N/A'):,}주")
                    print(f"     • 기관 순매수: {supply_demand.get('institution_net_buy', 'N/A'):,}주")
                    print(f"     • 개인 순매수: {supply_demand.get('individual_net_buy', 'N/A'):,}주")
                else:
                    print("   - ❌ 수급 데이터 없음")
                
                # 5. 기술적 지표 확인
                technical = comprehensive_data.get('technical_indicators', {})
                if technical:
                    print(f"   - 기술적 지표:")
                    print(f"     • RSI: {technical.get('rsi', 'N/A')}")
                    print(f"     • MACD: {technical.get('macd', 'N/A')}")
                    print(f"     • 볼린저 밴드: {technical.get('bollinger_position', 'N/A')}")
                else:
                    print("   - ❌ 기술적 지표 없음")
                
                # 6. 데이터 완성도 점수 계산
                completeness_score = 0
                max_score = 5
                
                if comprehensive_data.get('current_price', 0) > 0:
                    completeness_score += 1
                if chart_data is not None and not chart_data.empty:
                    completeness_score += 1
                if fundamentals:
                    completeness_score += 1
                if supply_demand:
                    completeness_score += 1
                if technical:
                    completeness_score += 1
                
                completeness_percentage = (completeness_score / max_score) * 100
                print(f"   - 📊 데이터 완성도: {completeness_score}/{max_score} ({completeness_percentage:.1f}%)")
                
                if completeness_percentage < 60:
                    print(f"   - ⚠️ 데이터 부족 상태 (60% 미만)")
                elif completeness_percentage < 80:
                    print(f"   - 🔶 데이터 보통 상태 (60-80%)")
                else:
                    print(f"   - ✅ 데이터 양호 상태 (80% 이상)")
                    
            else:
                print("❌ 종합 데이터 조회 실패")
        
        print("\n" + "=" * 80)
        print("🎯 데이터 품질 검사 완료")
        
        # 결론
        print("\n📋 결론:")
        print("1. 실제 API에서 데이터를 가져오지 못하는 경우 샘플 데이터를 사용합니다.")
        print("2. 샘플 데이터는 완전하지 않아 AI 분석 시 '데이터 부족' 메시지가 나올 수 있습니다.")
        print("3. 실제 운영 환경에서는 한국투자증권 API 연동이 필요합니다.")
        
    except Exception as e:
        print(f"❌ 데이터 품질 검사 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_data_quality()) 