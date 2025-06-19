#!/usr/bin/env python3
"""
🏛️ DART API 최대 활용 통합 테스트
- DART API의 모든 기능을 테스트
- KIS API와의 완전 통합 분석 테스트
- 실제 투자 판단에 활용할 수 있는 종합 데이터 수집 검증
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
        logging.FileHandler('dart_integration_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def test_dart_comprehensive_data():
    """🏛️ DART 종합 데이터 수집 테스트"""
    logger.info("🏛️ DART 종합 데이터 수집 테스트 시작...")
    
    try:
        from core_trader import CoreTrader
        from market_data_provider import AIDataCollector
        
        # CoreTrader 초기화
        trader = CoreTrader()
        await trader.async_initialize()
        
        # AIDataCollector 초기화
        data_collector = AIDataCollector(trader)
        
        # 테스트 종목들 (주요 대형주)
        test_symbols = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
        
        for symbol in test_symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 [{symbol}] DART 종합 데이터 수집 테스트")
            logger.info(f"{'='*60}")
            
            # DART 종합 데이터 수집
            dart_data = await data_collector.get_dart_comprehensive_data(symbol)
            
            if dart_data and 'error' not in dart_data:
                logger.info(f"✅ [{symbol}] DART 데이터 수집 성공!")
                
                # 수집된 데이터 요약 출력
                print_dart_summary(symbol, dart_data)
                
                # JSON 파일로 저장 (상세 분석용)
                # with open(f'dart_data_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w', encoding='utf-8') as f:
                #     json.dump(dart_data, f, ensure_ascii=False, indent=2, default=str)
                
            else:
                logger.error(f"❌ [{symbol}] DART 데이터 수집 실패: {dart_data.get('error', '알 수 없는 오류')}")
        
        await trader.close()
        
    except Exception as e:
        logger.error(f"❌ DART 종합 데이터 테스트 실패: {e}", exc_info=True)

async def test_ultimate_stock_analysis():
    """🎯 완전 통합 분석 테스트 (DART + KIS)"""
    logger.info("🎯 완전 통합 분석 테스트 시작...")
    
    try:
        from core_trader import CoreTrader
        from market_data_provider import AIDataCollector
        
        trader = CoreTrader()
        await trader.async_initialize()
        
        data_collector = AIDataCollector(trader)
        
        # 테스트 종목
        test_symbol = '005930'  # 삼성전자
        
        logger.info(f"🎯 [{test_symbol}] 완전 통합 분석 실행...")
        
        # 완전 통합 분석 실행
        ultimate_analysis = await data_collector.get_ultimate_stock_analysis(test_symbol)
        
        if ultimate_analysis and 'error' not in ultimate_analysis:
            logger.info(f"✅ [{test_symbol}] 완전 통합 분석 성공!")
            
            # 분석 결과 출력
            print_ultimate_analysis_summary(test_symbol, ultimate_analysis)
            
            # 상세 결과 저장
            # with open(f'ultimate_analysis_{test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w', encoding='utf-8') as f:
            #     json.dump(ultimate_analysis, f, ensure_ascii=False, indent=2, default=str)
                
        else:
            logger.error(f"❌ [{test_symbol}] 완전 통합 분석 실패: {ultimate_analysis.get('error', '알 수 없는 오류')}")
        
        await trader.close()
        
    except Exception as e:
        logger.error(f"❌ 완전 통합 분석 테스트 실패: {e}", exc_info=True)

async def test_dart_market_leaders():
    """🏛️ DART 기반 시장 리더 발굴 테스트"""
    logger.info("🏛️ DART 기반 시장 리더 발굴 테스트 시작...")
    
    try:
        from core_trader import CoreTrader
        from market_data_provider import AIDataCollector
        
        trader = CoreTrader()
        await trader.async_initialize()
        
        data_collector = AIDataCollector(trader)
        
        # 시장 리더 발굴
        market_leaders = await data_collector.get_dart_market_leaders(limit=10)
        
        if market_leaders:
            logger.info(f"✅ DART 기반 시장 리더 {len(market_leaders)}개 종목 발굴 성공!")
            
            print("\n" + "="*80)
            print("🏆 DART 기반 시장 리더 TOP 10")
            print("="*80)
            
            for i, leader in enumerate(market_leaders, 1):
                print(f"\n{i:2d}. {leader['company_name']} ({leader['symbol']})")
                print(f"    💰 현재가: {leader['current_price']:,}원")
                print(f"    📊 DART 점수: {leader['dart_score']:.1f}/100")
                print(f"    🏥 재무건전성: {leader['financial_health']}")
                print(f"    📢 공시품질: {leader['disclosure_quality']}")
                print(f"    💎 배당매력도: {leader['dividend_attractiveness']}")
                if leader['key_highlights']:
                    print(f"    ✨ 주요특징: {', '.join(leader['key_highlights'])}")
            
            # 결과 저장
            # with open(f'dart_market_leaders_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w', encoding='utf-8') as f:
            #     json.dump(market_leaders, f, ensure_ascii=False, indent=2, default=str)
                
        else:
            logger.warning("⚠️ DART 기반 시장 리더 발굴 결과가 없습니다.")
        
        await trader.close()
        
    except Exception as e:
        logger.error(f"❌ DART 시장 리더 발굴 테스트 실패: {e}", exc_info=True)

async def test_dart_risk_alerts():
    """⚠️ DART 기반 리스크 알림 테스트"""
    logger.info("⚠️ DART 기반 리스크 알림 테스트 시작...")
    
    try:
        from core_trader import CoreTrader
        from market_data_provider import AIDataCollector
        
        trader = CoreTrader()
        await trader.async_initialize()
        
        data_collector = AIDataCollector(trader)
        
        # 테스트 종목들 (다양한 리스크 수준 포함)
        test_symbols = ['005930', '000660', '035420', '068270', '051910']
        
        # 리스크 알림 분석
        risk_alerts = await data_collector.get_dart_risk_alerts(test_symbols)
        
        if risk_alerts:
            logger.info(f"⚠️ {len(risk_alerts)}개 종목에서 리스크 발견!")
            
            print("\n" + "="*80)
            print("⚠️ DART 기반 리스크 알림")
            print("="*80)
            
            for alert in risk_alerts:
                print(f"\n🚨 {alert['company_name']} ({alert['symbol']})")
                print(f"   📊 리스크 수준: {alert['overall_risk_level']}")
                print(f"   📝 리스크 건수: {alert['risk_count']}건")
                
                for risk in alert['risks']:
                    severity_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(risk['severity'], '⚪')
                    print(f"   {severity_icon} {risk['type']}: {risk['description']}")
            
            # 결과 저장
            # with open(f'dart_risk_alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w', encoding='utf-8') as f:
            #     json.dump(risk_alerts, f, ensure_ascii=False, indent=2, default=str)
                
        else:
            logger.info("✅ 테스트 종목들에서 특별한 리스크가 발견되지 않았습니다.")
        
        await trader.close()
        
    except Exception as e:
        logger.error(f"❌ DART 리스크 알림 테스트 실패: {e}", exc_info=True)

def print_dart_summary(symbol: str, dart_data: Dict[str, Any]):
    """DART 데이터 요약 출력"""
    print(f"\n📊 [{symbol}] DART 종합 데이터 요약")
    print("-" * 50)
    
    # 기업 정보
    company_info = dart_data.get('company_info', {})
    if company_info:
        print(f"🏢 기업명: {company_info.get('corp_name', 'N/A')}")
        print(f"👨‍💼 대표이사: {company_info.get('ceo_nm', 'N/A')}")
        print(f"📅 설립일: {company_info.get('est_dt', 'N/A')}")
        print(f"👥 직원수: {company_info.get('employee_count', 'N/A'):,}명" if company_info.get('employee_count') else "👥 직원수: N/A")
    
    # 재무 정보
    financial = dart_data.get('financial_statements', {})
    if financial:
        ratios = financial.get('financial_ratios', {})
        print(f"💰 ROE: {ratios.get('roe', 0):.1f}%")
        print(f"📊 부채비율: {ratios.get('debt_ratio', 0):.1f}%")
        print(f"📈 영업이익률: {ratios.get('operating_margin', 0):.1f}%")
        
        trend = financial.get('trend_analysis', {})
        if trend.get('revenue_growth'):
            print(f"📊 매출성장률: {trend['revenue_growth']:.1f}%")
    
    # 공시 정보
    disclosures = dart_data.get('recent_disclosures', [])
    if disclosures:
        print(f"📢 최근 공시: {len(disclosures)}건")
        important_disclosures = [d for d in disclosures if d.get('importance_score', 0) > 7]
        if important_disclosures:
            print(f"⚠️ 중요 공시: {len(important_disclosures)}건")
    
    # 배당 정보
    dividend_info = dart_data.get('dividend_info', {})
    if dividend_info:
        recent_years = sorted(dividend_info.keys(), reverse=True)[:3]
        if recent_years:
            avg_yield = sum(dividend_info[year].get('dividend_yield', 0) for year in recent_years) / len(recent_years)
            print(f"💎 평균 배당수익률: {avg_yield:.2f}%")
    
    # DART 분석 점수
    dart_score = dart_data.get('dart_analysis_score', 0)
    print(f"🏛️ DART 종합 점수: {dart_score:.1f}/100")

def print_ultimate_analysis_summary(symbol: str, analysis: Dict[str, Any]):
    """완전 통합 분석 결과 요약 출력"""
    print(f"\n🎯 [{symbol}] 완전 통합 분석 결과")
    print("=" * 60)
    
    # 기본 정보
    processing_time = analysis.get('processing_time', 0)
    print(f"⏱️ 분석 소요시간: {processing_time:.2f}초")
    
    # 점수들
    fundamental_score = analysis.get('fundamental_score', 0)
    technical_score = analysis.get('technical_score', 0)
    timing_score = analysis.get('market_timing_score', 0)
    ultimate_score = analysis.get('ultimate_score', 0)
    
    print(f"\n📊 세부 점수:")
    print(f"   🏛️ 펀더멘털 점수: {fundamental_score:.1f}/100")
    print(f"   📈 기술적 점수: {technical_score:.1f}/100")
    print(f"   ⏰ 시장타이밍 점수: {timing_score:.1f}/100")
    print(f"   🎯 최종 통합 점수: {ultimate_score:.1f}/100")
    
    # 투자 추천
    recommendation = analysis.get('investment_recommendation', 'HOLD')
    risk_level = analysis.get('risk_level', 'MEDIUM')
    
    print(f"\n💡 투자 추천: {recommendation}")
    print(f"⚠️ 리스크 수준: {risk_level}")
    
    # 핵심 강점
    strengths = analysis.get('key_strengths', [])
    if strengths:
        print(f"\n✅ 핵심 강점:")
        for strength in strengths:
            print(f"   • {strength}")
    
    # 핵심 리스크
    risks = analysis.get('key_risks', [])
    if risks:
        print(f"\n⚠️ 핵심 리스크:")
        for risk in risks:
            print(f"   • {risk}")
    
    # 목표 주가
    target_price_info = analysis.get('target_price_range')
    if target_price_info:
        print(f"\n🎯 목표 주가: {target_price_info['target_price']:,}원")
        print(f"📈 상승 여력: {target_price_info['upside_potential']:.1f}%")
        print(f"📝 계산 방법: {target_price_info['method']}")

async def main():
    """메인 테스트 실행"""
    print("🏛️ DART API 최대 활용 통합 테스트 시작")
    print("=" * 80)
    
    try:
        # 1. DART 종합 데이터 수집 테스트
        await test_dart_comprehensive_data()
        
        print("\n" + "="*80)
        
        # 2. 완전 통합 분석 테스트
        await test_ultimate_stock_analysis()
        
        print("\n" + "="*80)
        
        # 3. DART 기반 시장 리더 발굴 테스트
        await test_dart_market_leaders()
        
        print("\n" + "="*80)
        
        # 4. DART 기반 리스크 알림 테스트
        await test_dart_risk_alerts()
        
        print("\n" + "="*80)
        print("✅ 모든 DART API 통합 테스트가 완료되었습니다!")
        print("📁 상세 결과는 생성된 JSON 파일들을 확인하세요.")
        
    except Exception as e:
        logger.error(f"❌ 메인 테스트 실행 실패: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 