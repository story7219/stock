#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 스카우트 전략 매니저
- 유망 종목 자동 발굴 및 스크리닝
- 다양한 발굴 전략 통합 관리
- AI 기반 종목 평가 및 랭킹
- v2.5.0 (2024-12-24): NumPy 호환성 개선
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json

# NumPy 호환성 처리
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

logger = logging.getLogger(__name__)

@dataclass
class StockCandidate:
    """발굴된 종목 후보 데이터 클래스"""
    symbol: str
    name: str
    score: float
    reason: str
    price: float
    market_cap: int
    volume: int
    momentum_score: float = 0.0
    value_score: float = 0.0
    growth_score: float = 0.0
    technical_score: float = 0.0
    discovery_time: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)

class ScoutStrategyManager:
    """종목 발굴 전략 통합 관리자"""
    
    def __init__(self):
        self.discovered_stocks = []
        self.blacklist = set()  # 제외할 종목들
        
        # 발굴 전략 가중치
        self.strategy_weights = {
            'momentum': 0.3,
            'value': 0.25,
            'growth': 0.25,
            'technical': 0.2
        }
        
        # 최소 조건
        self.min_conditions = {
            'min_price': 1000,      # 최소 주가 1,000원
            'min_volume': 100000,   # 최소 거래량 10만주
            'min_market_cap': 100,  # 최소 시가총액 100억원
            'max_price': 500000     # 최대 주가 50만원
        }
        
        logger.info("🎯 스카우트 전략 매니저 초기화 완료")
    
    async def run_stock_discovery(self, max_candidates: int = 50) -> List[StockCandidate]:
        """종합 종목 발굴 실행"""
        logger.info("🔍 종목 발굴 시작...")
        
        try:
            # 1. 기본 종목 리스트 수집
            stock_universe = await self._get_stock_universe()
            logger.info(f"📋 분석 대상 종목: {len(stock_universe)}개")
            
            # 2. 병렬 분석 실행
            candidates = await self._analyze_candidates_parallel(stock_universe)
            
            # 3. 후보 종목 정렬 및 필터링
            top_candidates = self._rank_and_filter_candidates(candidates, max_candidates)
            
            # 4. 결과 저장
            await self._save_discovery_results(top_candidates)
            
            logger.info(f"✅ 종목 발굴 완료: {len(top_candidates)}개 후보 발견")
            return top_candidates
            
        except Exception as e:
            logger.error(f"❌ 종목 발굴 오류: {e}")
            return []
    
    async def _get_stock_universe(self) -> List[Dict[str, Any]]:
        """분석할 종목 리스트 수집"""
        try:
            # 여기서는 시뮬레이션을 위한 샘플 데이터
            # 실제로는 KIS API나 다른 데이터 소스에서 가져와야 함
            sample_stocks = [
                {'symbol': '005930', 'name': '삼성전자', 'market': 'KOSPI'},
                {'symbol': '000660', 'name': 'SK하이닉스', 'market': 'KOSPI'},
                {'symbol': '035420', 'name': 'NAVER', 'market': 'KOSPI'},
                {'symbol': '051910', 'name': 'LG화학', 'market': 'KOSPI'},
                {'symbol': '006400', 'name': '삼성SDI', 'market': 'KOSPI'},
                {'symbol': '035720', 'name': '카카오', 'market': 'KOSPI'},
                {'symbol': '028260', 'name': '삼성물산', 'market': 'KOSPI'},
                {'symbol': '066570', 'name': 'LG전자', 'market': 'KOSPI'},
                {'symbol': '096770', 'name': 'SK이노베이션', 'market': 'KOSPI'},
                {'symbol': '003550', 'name': 'LG', 'market': 'KOSPI'},
            ]
            
            # 블랙리스트 필터링
            filtered_stocks = [stock for stock in sample_stocks if stock['symbol'] not in self.blacklist]
            
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"❌ 종목 리스트 수집 오류: {e}")
            return []
    
    async def _analyze_candidates_parallel(self, stock_universe: List[Dict]) -> List[StockCandidate]:
        """병렬 처리로 후보 종목 분석"""
        candidates = []
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        with ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, self._analyze_single_stock, stock
                ) for stock in stock_universe
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, StockCandidate):
                    candidates.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"⚠️ 종목 분석 오류: {result}")
        
        return candidates
    
    def _analyze_single_stock(self, stock_info: Dict[str, Any]) -> Optional[StockCandidate]:
        """개별 종목 분석"""
        try:
            symbol = stock_info['symbol']
            name = stock_info['name']
            
            # 시뮬레이션을 위한 랜덤 데이터 생성
            # 실제로는 실시간 데이터를 가져와야 함
            import random
            
            price = random.randint(10000, 100000)
            volume = random.randint(100000, 1000000)
            market_cap = random.randint(1000, 50000)  # 억원 단위
            
            # 기본 조건 확인
            if not self._meets_basic_conditions(price, volume, market_cap):
                return None
            
            # 각 전략별 점수 계산
            momentum_score = self._calculate_momentum_score(symbol)
            value_score = self._calculate_value_score(symbol)
            growth_score = self._calculate_growth_score(symbol)
            technical_score = self._calculate_technical_score(symbol)
            
            # 종합 점수 계산
            total_score = (
                momentum_score * self.strategy_weights['momentum'] +
                value_score * self.strategy_weights['value'] +
                growth_score * self.strategy_weights['growth'] +
                technical_score * self.strategy_weights['technical']
            )
            
            # 발굴 이유 생성
            reason = self._generate_discovery_reason(momentum_score, value_score, growth_score, technical_score)
            
            return StockCandidate(
                symbol=symbol,
                name=name,
                score=round(total_score, 2),
                reason=reason,
                price=price,
                market_cap=market_cap,
                volume=volume,
                momentum_score=round(momentum_score, 2),
                value_score=round(value_score, 2),
                growth_score=round(growth_score, 2),
                technical_score=round(technical_score, 2)
            )
            
        except Exception as e:
            logger.warning(f"⚠️ {stock_info.get('symbol', '알수없음')} 분석 오류: {e}")
            return None
    
    def _meets_basic_conditions(self, price: float, volume: int, market_cap: int) -> bool:
        """기본 조건 충족 여부 확인"""
        return (
            self.min_conditions['min_price'] <= price <= self.min_conditions['max_price'] and
            volume >= self.min_conditions['min_volume'] and
            market_cap >= self.min_conditions['min_market_cap']
        )
    
    def _calculate_momentum_score(self, symbol: str) -> float:
        """모멘텀 점수 계산 (0-100)"""
        # 실제로는 가격 변화율, 거래량 증가율 등을 분석
        import random
        return random.uniform(0, 100)
    
    def _calculate_value_score(self, symbol: str) -> float:
        """가치 점수 계산 (0-100)"""
        # 실제로는 PER, PBR, ROE 등을 분석
        import random
        return random.uniform(0, 100)
    
    def _calculate_growth_score(self, symbol: str) -> float:
        """성장 점수 계산 (0-100)"""
        # 실제로는 매출/이익 성장률 등을 분석
        import random
        return random.uniform(0, 100)
    
    def _calculate_technical_score(self, symbol: str) -> float:
        """기술적 점수 계산 (0-100)"""
        # 실제로는 이동평균, RSI, MACD 등을 분석
        import random
        return random.uniform(0, 100)
    
    def _generate_discovery_reason(self, momentum: float, value: float, growth: float, technical: float) -> str:
        """발굴 이유 생성"""
        reasons = []
        
        if momentum > 70:
            reasons.append("강한 모멘텀")
        if value > 70:
            reasons.append("저평가 매력")
        if growth > 70:
            reasons.append("높은 성장성")
        if technical > 70:
            reasons.append("기술적 돌파")
        
        if not reasons:
            best_score = max(momentum, value, growth, technical)
            if best_score == momentum:
                reasons.append("모멘텀 주목")
            elif best_score == value:
                reasons.append("가치 발굴")
            elif best_score == growth:
                reasons.append("성장 기대")
            else:
                reasons.append("기술적 관심")
        
        return ", ".join(reasons)
    
    def _rank_and_filter_candidates(self, candidates: List[StockCandidate], max_count: int) -> List[StockCandidate]:
        """후보 종목 랭킹 및 필터링"""
        # 점수 기준 정렬
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # 상위 N개 선택
        top_candidates = sorted_candidates[:max_count]
        
        # 점수 임계값 적용 (50점 이상만)
        filtered_candidates = [c for c in top_candidates if c.score >= 50.0]
        
        return filtered_candidates
    
    async def _save_discovery_results(self, candidates: List[StockCandidate]) -> None:
        """발굴 결과 저장"""
        try:
            # JSON 형태로 저장
            results = {
                'discovery_time': datetime.now().isoformat(),
                'total_candidates': len(candidates),
                'candidates': [
                    {
                        'symbol': c.symbol,
                        'name': c.name,
                        'score': c.score,
                        'reason': c.reason,
                        'price': c.price,
                        'market_cap': c.market_cap,
                        'volume': c.volume,
                        'scores': {
                            'momentum': c.momentum_score,
                            'value': c.value_score,
                            'growth': c.growth_score,
                            'technical': c.technical_score
                        }
                    } for c in candidates
                ]
            }
            
            filename = f"scout_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 발굴 결과 저장: {filename}")
            
        except Exception as e:
            logger.error(f"❌ 결과 저장 오류: {e}")
    
    def add_to_blacklist(self, symbols: List[str]) -> None:
        """블랙리스트에 종목 추가"""
        self.blacklist.update(symbols)
        logger.info(f"🚫 블랙리스트 추가: {symbols}")
    
    def remove_from_blacklist(self, symbols: List[str]) -> None:
        """블랙리스트에서 종목 제거"""
        self.blacklist.difference_update(symbols)
        logger.info(f"✅ 블랙리스트 제거: {symbols}")
    
    def get_discovery_summary(self, candidates: List[StockCandidate]) -> Dict[str, Any]:
        """발굴 결과 요약"""
        if not candidates:
            return {"message": "발굴된 종목이 없습니다."}
        
        return {
            "총_후보수": len(candidates),
            "평균_점수": round(sum(c.score for c in candidates) / len(candidates), 2),
            "최고_점수": max(c.score for c in candidates),
            "상위_3종목": [
                {"종목명": c.name, "점수": c.score, "이유": c.reason}
                for c in candidates[:3]
            ],
            "전략별_평균점수": {
                "모멘텀": round(sum(c.momentum_score for c in candidates) / len(candidates), 2),
                "가치": round(sum(c.value_score for c in candidates) / len(candidates), 2),
                "성장": round(sum(c.growth_score for c in candidates) / len(candidates), 2),
                "기술적": round(sum(c.technical_score for c in candidates) / len(candidates), 2)
            }
        }

# 테스트 함수
async def test_scout_strategy():
    """스카우트 전략 테스트"""
    logger.info("🧪 스카우트 전략 테스트 시작")
    
    scout = ScoutStrategyManager()
    candidates = await scout.run_stock_discovery(max_candidates=10)
    
    if candidates:
        print(f"\n✅ 발굴된 종목: {len(candidates)}개")
        print("=" * 80)
        
        for i, candidate in enumerate(candidates[:5], 1):
            print(f"{i}. {candidate.name} ({candidate.symbol})")
            print(f"   점수: {candidate.score}점 | 이유: {candidate.reason}")
            print(f"   가격: {candidate.price:,}원 | 거래량: {candidate.volume:,}주")
            print(f"   세부점수 - 모멘텀:{candidate.momentum_score} 가치:{candidate.value_score} "
                  f"성장:{candidate.growth_score} 기술:{candidate.technical_score}")
            print()
        
        # 요약 정보
        summary = scout.get_discovery_summary(candidates)
        print("📊 발굴 요약:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("❌ 발굴된 종목이 없습니다.")

if __name__ == "__main__":
    asyncio.run(test_scout_strategy()) 