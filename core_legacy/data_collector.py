"""
🚀 AI 투자 판단을 위한 데이터 수집 및 분석 파이프라인
- 각 데이터 제공자(DART, KIS, FDR)와 분석기(펀더멘털, 기술적)를 조립하여 최종 데이터를 생성합니다.
- 고수준의 데이터 수집 및 분석 흐름을 관리합니다.
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime

# 프로젝트 내 모듈 임포트
from core.models import StockInfo, FilterCriteria
from data_providers.dart_provider import DartProvider
from analysis import fundamental_analyzer, technical_analyzer
from core.core_trader import CoreTrader

logger = logging.getLogger(__name__)

class AIDataCollector:
    """데이터 수집 및 분석을 총괄하는 AI 데이터 수집기"""
    
    def __init__(self, trader: CoreTrader):
        """
        필요한 데이터 제공자 및 분석기를 초기화합니다.
        :param trader: KIS API와 통신하는 CoreTrader 인스턴스
        """
        self.trader = trader
        self.dart_provider = DartProvider()

    async def get_comprehensive_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        한 종목에 대한 모든 소스의 데이터를 종합하여 분석합니다.
        """
        logger.info(f"🎯 [{symbol}] 종합 주식 데이터 분석 시작...")
        start_time = time.time()
        
        dart_task = self.get_full_dart_analysis(symbol)
        kis_task = self.get_full_kis_analysis(symbol)
        
        dart_analysis, kis_analysis = await asyncio.gather(dart_task, kis_task)

        comprehensive_data = {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'dart_analysis': dart_analysis,
            'kis_analysis': kis_analysis,
            'data_quality': 'LOW'
        }

        if dart_analysis and not dart_analysis.get('error'):
            comprehensive_data['data_quality'] = 'HIGH'
        elif kis_analysis and not kis_analysis.get('error'):
            comprehensive_data['data_quality'] = 'MEDIUM'

        processing_time = time.time() - start_time
        logger.info(f"✅ [{symbol}] 종합 분석 완료 (품질: {comprehensive_data['data_quality']}, 소요시간: {processing_time:.2f}초)")
        return comprehensive_data

    async def get_full_dart_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """DART 데이터를 수집하고 펀더멘털 분석을 수행합니다."""
        if not self.dart_provider.dart_available:
            return {'error': 'DART API 사용 불가'}

        company_info = await self.dart_provider.get_company_info(symbol)
        if not company_info:
            return {'error': 'DART 기업 정보를 찾을 수 없습니다.'}

        financials = await self.dart_provider.get_financial_statements(symbol)
        disclosures = await self.dart_provider.get_recent_disclosures(symbol)
        shareholders = await self.dart_provider.get_major_shareholders(symbol)

        dart_data_bundle = {
            'company_info': company_info, 'financial_statements': financials,
            'recent_disclosures': disclosures, 'major_shareholders': shareholders
        }
        
        ratios = fundamental_analyzer.calculate_financial_ratios(financials)
        
        return {
            'corp_name': company_info.get('corp_name'),
            'fundamental_score': fundamental_analyzer.calculate_dart_analysis_score(dart_data_bundle),
            'financial_health': fundamental_analyzer.assess_financial_health(ratios),
            'financial_ratios': ratios,
            'financial_trends': fundamental_analyzer.analyze_financial_trends(financials),
            'recent_disclosures_count': len(disclosures) if disclosures else 0,
            'has_negative_disclosure': any(fundamental_analyzer.is_negative_disclosure(d.get('report_nm', '')) for d in disclosures) if disclosures else False
        }

    async def get_full_kis_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """KIS API로 시세 데이터를 수집하고 기술적 분석을 수행합니다."""
        try:
            tasks = {
                'current_price': self.trader.get_current_price(symbol),
                'daily_history': self.trader.fetch_daily_price_history(symbol, period=100),
                'investor_trends': self.trader.fetch_detailed_investor_trends(symbol)
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            kis_data = {key: res for key, res in zip(tasks.keys(), results) if not isinstance(res, Exception)}

            tech_indicators = {}
            if kis_data.get('daily_history'):
                tech_indicators = technical_analyzer.get_technical_indicators(kis_data['daily_history'])
            
            return {
                'current_price_info': kis_data.get('current_price'),
                'technical_indicators': tech_indicators,
                'investor_trends': kis_data.get('investor_trends')
            }
        except Exception as e:
            logger.error(f"❌ [{symbol}] KIS 종합 분석 실패: {e}")
            return {'error': str(e)} 