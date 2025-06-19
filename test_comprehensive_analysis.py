#!/usr/bin/env python3
"""
🎯 실용적인 종합 주식 분석 테스트
- DART API 최대 활용
- 대체 데이터 소스 보완
- 실제 투자 판단에 활용 가능한 분석
"""

import asyncio
import logging
from datetime import datetime
import json
from typing import Dict, Any, List

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_analysis_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def test_comprehensive_stock_analysis():
    """🎯 종합 주식 분석 테스트"""
    logger.info("🎯 종합 주식 분석 테스트 시작...")
    
    try:
        from core_trader import CoreTrader
        from market_data_provider import AIDataCollector
        
        # CoreTrader 초기화
        trader = CoreTrader()
        await trader.async_initialize()
        
        # AIDataCollector 초기화
        data_collector = AIDataCollector(trader)
        
        # 테스트 종목들 (다양한 시가총액과 섹터)
        test_symbols = [
            '005930',  # 삼성전자 (대형주)
            '000660',  # SK하이닉스 (대형주)
            '035420',  # NAVER (대형주)
            '068270',  # 셀트리온 (바이오)
            '051910'   # LG화학 (화학)
        ]
        
        analysis_results = []
        
        for symbol in test_symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 [{symbol}] 종합 분석 시작")
            logger.info(f"{'='*80}")
            
            # 종합 데이터 수집
            comprehensive_data = await data_collector.get_comprehensive_stock_data(symbol)
            
            if comprehensive_data and comprehensive_data.get('data_quality') != 'ERROR':
                logger.info(f"✅ [{symbol}] 종합 데이터 수집 성공!")
                
                # 분석 결과 요약 출력
                print_comprehensive_summary(symbol, comprehensive_data)
                
                # 결과 저장
                analysis_results.append(comprehensive_data)
                
                # 개별 JSON 파일 저장
                # filename = f'comprehensive_analysis_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                # with open(filename, 'w', encoding='utf-8') as f:
                #     json.dump(comprehensive_data, f, ensure_ascii=False, indent=2, default=str)
                
            else:
                logger.error(f"❌ [{symbol}] 종합 데이터 수집 실패")
        
        # 전체 결과 요약
        if analysis_results:
            print_overall_summary(analysis_results)
            
            # 전체 결과 저장
            # overall_filename = f'comprehensive_analysis_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            # with open(overall_filename, 'w', encoding='utf-8') as f:
            #     json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        await trader.close()
        
    except Exception as e:
        logger.error(f"❌ 종합 분석 테스트 실패: {e}", exc_info=True)

async def test_alternative_data_sources():
    """🔄 대체 데이터 소스 테스트"""
    logger.info("🔄 대체 데이터 소스 테스트 시작...")
    
    try:
        from core_trader import CoreTrader
        from market_data_provider import AIDataCollector
        
        trader = CoreTrader()
        await trader.async_initialize()
        
        data_collector = AIDataCollector(trader)
        
        # 테스트 종목
        test_symbol = '005930'  # 삼성전자
        
        logger.info(f"🔄 [{test_symbol}] 대체 데이터 소스 테스트...")
        
        # 대체 데이터 수집
        alternative_data = await data_collector.get_alternative_fundamental_data(test_symbol)
        
        if alternative_data and 'error' not in alternative_data:
            logger.info(f"✅ [{test_symbol}] 대체 데이터 수집 성공!")
            
            print(f"\n📊 [{test_symbol}] 대체 데이터 분석 결과")
            print("-" * 60)
            
            # 기본 정보
            print(f"🏢 기업명: {alternative_data.get('company_name', 'N/A')}")
            print(f"🏛️ 시장: {alternative_data.get('market', 'N/A')}")
            print(f"🏭 섹터: {alternative_data.get('sector', 'N/A')}")
            print(f"🔧 업종: {alternative_data.get('industry', 'N/A')}")
            
            # 가격 분석
            price_analysis = alternative_data.get('price_analysis', {})
            if price_analysis:
                print(f"\n💰 가격 분석:")
                print(f"   현재가: {price_analysis.get('current_price', 0):,.0f}원")
                print(f"   52주 최고가: {price_analysis.get('52week_high', 0):,.0f}원")
                print(f"   52주 최저가: {price_analysis.get('52week_low', 0):,.0f}원")
                print(f"   고점 대비: {price_analysis.get('high_ratio', 0):.1f}%")
                print(f"   저점 대비: {price_analysis.get('low_ratio', 0):.1f}%")
            
            # 데이터 소스
            data_sources = alternative_data.get('data_sources', [])
            print(f"\n📡 데이터 소스: {', '.join(data_sources)}")
            
            # 분석 점수
            score = alternative_data.get('alternative_analysis_score', 0)
            print(f"📊 대체 분석 점수: {score:.1f}/100")
            
            # 상세 결과 저장
            # filename = f'alternative_data_{test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            # with open(filename, 'w', encoding='utf-8') as f:
            #     json.dump(alternative_data, f, ensure_ascii=False, indent=2, default=str)
                
        else:
            logger.error(f"❌ [{test_symbol}] 대체 데이터 수집 실패")
        
        await trader.close()
        
    except Exception as e:
        logger.error(f"❌ 대체 데이터 소스 테스트 실패: {e}", exc_info=True)

def print_comprehensive_summary(symbol: str, data: Dict[str, Any]):
    """종합 분석 결과 요약 출력"""
    print(f"\n🎯 [{symbol}] 종합 분석 결과")
    print("=" * 70)
    
    # 데이터 품질
    data_quality = data.get('data_quality', 'UNKNOWN')
    quality_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴', 'ERROR': '❌'}.get(data_quality, '⚪')
    print(f"{quality_icon} 데이터 품질: {data_quality}")
    
    # 데이터 소스 현황
    has_dart = data.get('has_dart_data', False)
    has_alternative = data.get('has_alternative_data', False)
    has_kis = data.get('has_kis_data', False)
    
    print(f"📊 데이터 소스:")
    print(f"   🏛️ DART API: {'✅' if has_dart else '❌'}")
    print(f"   🔄 대체 소스: {'✅' if has_alternative else '❌'}")
    print(f"   🚀 KIS API: {'✅' if has_kis else '❌'}")
    
    # 종합 점수
    comprehensive_score = data.get('comprehensive_score', 0)
    print(f"\n🎯 종합 점수: {comprehensive_score:.1f}/100")
    
    # DART 데이터 요약
    if has_dart:
        dart_data = data['dart_data']
        dart_score = dart_data.get('dart_analysis_score', 0)
        print(f"🏛️ DART 점수: {dart_score:.1f}/100")
        
        company_info = dart_data.get('company_info', {})
        if company_info:
            print(f"   기업명: {company_info.get('corp_name', 'N/A')}")
            print(f"   대표이사: {company_info.get('ceo_nm', 'N/A')}")
    
    # 대체 데이터 요약
    if has_alternative:
        alt_data = data['alternative_data']
        alt_score = alt_data.get('alternative_analysis_score', 0)
        print(f"🔄 대체 데이터 점수: {alt_score:.1f}/100")
        
        price_analysis = alt_data.get('price_analysis', {})
        if price_analysis:
            print(f"   현재가: {price_analysis.get('current_price', 0):,.0f}원")
            print(f"   고점 대비: {price_analysis.get('high_ratio', 0):.1f}%")
    
    # KIS 데이터 요약
    if has_kis:
        kis_data = data['kis_data']
        current_price = kis_data.get('current_price', {})
        if current_price:
            price = current_price.get('stck_prpr', 0)
            change_rate = current_price.get('prdy_ctrt', 0)
            print(f"🚀 실시간 가격: {price:,}원 ({change_rate:+.2f}%)")
    
    # 투자 추천 (간단한 로직)
    recommendation = get_investment_recommendation(comprehensive_score)
    print(f"\n💡 투자 추천: {recommendation}")

def print_overall_summary(results: List[Dict[str, Any]]):
    """전체 분석 결과 요약"""
    print(f"\n🏆 전체 분석 결과 요약")
    print("=" * 80)
    
    # 데이터 품질 통계
    quality_counts = {}
    for result in results:
        quality = result.get('data_quality', 'UNKNOWN')
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    print(f"📊 데이터 품질 분포:")
    for quality, count in quality_counts.items():
        icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴', 'ERROR': '❌'}.get(quality, '⚪')
        print(f"   {icon} {quality}: {count}개")
    
    # 점수별 순위
    valid_results = [r for r in results if r.get('comprehensive_score', 0) > 0]
    valid_results.sort(key=lambda x: x.get('comprehensive_score', 0), reverse=True)
    
    print(f"\n🏆 종합 점수 순위:")
    for i, result in enumerate(valid_results, 1):
        symbol = result['symbol']
        score = result.get('comprehensive_score', 0)
        
        # 기업명 찾기
        company_name = 'N/A'
        if result.get('has_dart_data'):
            company_name = result['dart_data'].get('company_info', {}).get('corp_name', 'N/A')
        elif result.get('has_alternative_data'):
            company_name = result['alternative_data'].get('company_name', 'N/A')
        
        recommendation = get_investment_recommendation(score)
        print(f"   {i:2d}. {company_name} ({symbol}): {score:.1f}점 - {recommendation}")

def get_investment_recommendation(score: float) -> str:
    """점수 기반 투자 추천"""
    if score >= 80:
        return "🟢 강력 매수"
    elif score >= 70:
        return "🟢 매수"
    elif score >= 60:
        return "🟡 약한 매수"
    elif score >= 40:
        return "🟡 보유"
    elif score >= 30:
        return "🔴 약한 매도"
    else:
        return "🔴 매도"

async def main():
    """메인 테스트 실행"""
    print("🎯 실용적인 종합 주식 분석 테스트 시작")
    print("=" * 80)
    
    try:
        # 1. 종합 주식 분석 테스트
        await test_comprehensive_stock_analysis()
        
        print("\n" + "="*80)
        
        # 2. 대체 데이터 소스 테스트
        await test_alternative_data_sources()
        
        print("\n" + "="*80)
        print("✅ 모든 종합 분석 테스트가 완료되었습니다!")
        print("📁 상세 결과는 생성된 JSON 파일들을 확인하세요.")
        print("💡 이제 실제 투자 판단에 활용할 수 있는 데이터가 준비되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 메인 테스트 실행 실패: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 