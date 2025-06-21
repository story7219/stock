"""
🎯 지능형 종목 필터링 전략
- 시가총액, 거래량 등 기본 조건과 AI 점수를 종합하여 유망 종목을 발굴합니다.
- 다양한 필터링 기준을 조합하여 사용할 수 있습니다.
"""
import logging
import asyncio
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 내 모듈 임포트
from core.models import StockInfo, FilterCriteria
from core.data_collector import AIDataCollector

logger = logging.getLogger(__name__)

class StockFilter:
    """시장 상황에 맞는 유망 종목을 발굴하는 지능형 필터"""

    def __init__(self, data_collector: AIDataCollector, criteria: Optional[FilterCriteria] = None):
        """
        :param data_collector: 데이터 수집을 위한 AIDataCollector 인스턴스
        :param criteria: 필터링 기준
        """
        self.data_collector = data_collector
        self.criteria = criteria or FilterCriteria()
        self.max_workers = 10

    def set_filter_criteria(self, criteria: FilterCriteria):
        """필터링 기준을 설정하거나 업데이트합니다."""
        self.criteria = criteria
        logger.info(f"📊 필터링 기준 업데이트: 시총 ≥ {self.criteria.min_market_cap}억, "
                    f"거래량 ≥ {self.criteria.min_volume:,}주")

    async def screen_stocks(self, force_update: bool = False) -> List[StockInfo]:
        """설정된 기준에 따라 종목을 필터링하고 AI 점수를 부여하여 상위 종목을 반환합니다."""
        logger.info("🔍 종목 스크리닝 시작...")
        start_time = time.time()

        # 1. KIS 순위 기반 후보 종목군 수집
        candidate_stocks = await self._fetch_candidate_stocks()
        if not candidate_stocks:
            logger.error("❌ 후보 종목 수집 실패. 스크리닝을 중단합니다.")
            return []

        # 2. 기본 조건 필터링
        primary_filtered = self._apply_primary_filters(candidate_stocks)
        logger.info(f"✅ 1차 필터링 완료: {len(candidate_stocks)} → {len(primary_filtered)}개 종목")

        # 3. AI 점수 계산 및 최종 필터링
        final_stocks = await self._score_and_finalize(primary_filtered)
        
        elapsed = time.time() - start_time
        logger.info(f"🎯 스크리닝 완료: 최종 {len(final_stocks)}개 종목 선별 (소요시간: {elapsed:.1f}초)")
        
        self._log_screening_summary(final_stocks)
        return final_stocks

    async def _fetch_candidate_stocks(self) -> List[StockInfo]:
        """KIS 순위 API를 통해 다양한 유형의 상위 종목들을 수집하여 후보군을 만듭니다."""
        logger.info("   - (1단계) KIS 순위 API로 후보 종목군 병렬 조회...")
        try:
            ranking_types = ["rise", "volume", "value", "institution_net_buy", "foreign_net_buy"]
            tasks = [self.data_collector.trader.fetch_ranking_data(rtype, limit=100) for rtype in ranking_types]
            results = await asyncio.gather(*tasks)
            
            unique_codes = set()
            for stock_list in results:
                if stock_list:
                    for item in stock_list:
                        unique_codes.add(item.get('mksc_shrn_iscd'))

            if not unique_codes:
                logger.warning("   - KIS 순위 조회 결과가 없습니다.")
                return []
            
            logger.info(f"   - (2단계) {len(unique_codes)}개 후보 종목 상세 정보 병렬 조회...")
            stock_details = await self.data_collector.trader.get_stock_details_parallel(list(unique_codes))
            return [stock for stock in stock_details if stock]

        except Exception as e:
            logger.error(f"❌ 후보 종목 수집 중 오류: {e}", exc_info=True)
            return []

    def _apply_primary_filters(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """시총, 거래량/대금, 시장, 업종 등 기본 필터링을 적용합니다."""
        return [
            stock for stock in stocks
            if (stock.market_cap >= self.criteria.min_market_cap and
                stock.volume >= self.criteria.min_volume and
                stock.volume_value >= self.criteria.min_volume_value and
                stock.market_type in self.criteria.market_types and
                stock.sector not in self.criteria.exclude_sectors)
        ]

    async def _score_and_finalize(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """각 종목에 대해 AI 분석 점수를 계산하고 최종 순위를 매깁니다."""
        # 이 부분은 추후 더 정교한 AI 점수 모델로 대체될 수 있습니다.
        # 현재는 간단히 펀더멘털 점수 + 기술적 점수를 사용한다고 가정합니다.
        # 실제 점수 계산은 AIDataCollector가 수행하고 여기서는 결과만 활용합니다.
        
        # 예시: 각 종목에 대해 종합 분석 데이터 요청
        tasks = [self.data_collector.get_comprehensive_stock_data(s.code) for s in stocks]
        analysis_results = await asyncio.gather(*tasks)

        for stock, analysis in zip(stocks, analysis_results):
            if analysis and analysis.get('dart_analysis'):
                # 간단한 점수 합산 예시
                dart_score = analysis['dart_analysis'].get('fundamental_score', 50)
                stock.score = dart_score # 임시로 DART 점수만 사용
        
        scored_stocks = sorted([s for s in stocks if s.score > 0], key=lambda x: x.score, reverse=True)
        return scored_stocks[:self.criteria.max_stocks]

    def _log_screening_summary(self, stocks: List[StockInfo]):
        """스크리닝 결과 상위 5개 종목을 로깅합니다."""
        logger.info("--- 筛选出的前5只股票 ---")
        for i, stock in enumerate(stocks[:5]):
            logger.info(f"{i+1}. {stock.name}({stock.code}) - 점수: {stock.score:.1f}, "
                        f"시총: {stock.market_cap}억, 거래대금: {stock.volume_value}백만")
        logger.info("-----------------------------") 