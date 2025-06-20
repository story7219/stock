"""
수급 데이터 최적화 분석기
- 실시간 수급 모니터링
- 수급 급변 알림 
- 세분화된 기관별 분석
- 수급 패턴 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

class SupplyDemandLevel(Enum):
    """수급 강도 레벨"""
    EXTREME_BUY = "극강매수"
    STRONG_BUY = "강매수"
    MODERATE_BUY = "보통매수"
    NEUTRAL = "중립"
    MODERATE_SELL = "보통매도"
    STRONG_SELL = "강매도"
    EXTREME_SELL = "극강매도"

@dataclass
class SupplyDemandData:
    """수급 데이터 구조"""
    stock_code: str
    stock_name: str
    date: str
    
    # 기본 수급 데이터
    foreign_net: int = 0          # 외국인 순매수
    institution_net: int = 0      # 기관 순매수
    individual_net: int = 0       # 개인 순매수
    
    # 세분화된 기관 데이터
    pension_fund: int = 0         # 연기금
    private_equity: int = 0       # 사모펀드
    insurance: int = 0            # 보험
    investment_trust: int = 0     # 투신
    bank: int = 0                 # 은행
    other_financial: int = 0      # 기타금융
    other_corp: int = 0           # 기타법인
    
    # 계산된 지표
    total_volume: int = 0         # 총 거래량
    supply_demand_score: float = 0.0  # 수급 점수
    level: SupplyDemandLevel = SupplyDemandLevel.NEUTRAL
    
    # 패턴 분석
    trend_days: int = 0           # 연속 매수/매도 일수
    momentum_score: float = 0.0   # 모멘텀 점수
    
    def to_dict(self) -> Dict[str, Any]:
        """데이터클래스 인스턴스를 딕셔너리로 변환합니다."""
        # asdict를 사용하여 기본 변환
        data_dict = asdict(self)
        
        # Enum 멤버를 값으로 변환
        if 'level' in data_dict and isinstance(data_dict['level'], Enum):
            data_dict['level'] = data_dict['level'].value
            
        return data_dict

@dataclass
class SupplyDemandAlert:
    """수급 급변 알림 데이터"""
    stock_code: str
    stock_name: str
    alert_type: str
    message: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: datetime
    current_data: SupplyDemandData
    previous_data: Optional[SupplyDemandData] = None

class SupplyDemandAnalyzer:
    """수급 데이터 최적화 분석기"""
    
    def __init__(self, data_manager=None, notifier=None):
        self.data_manager = data_manager
        self.notifier = notifier
        self.cache_file = Path("data/supply_demand_cache.json")
        self.alert_history = []
        self.monitoring_stocks = set()
        
        # 수급 분석 설정
        self.config = {
            # 급변 감지 임계값
            'extreme_change_threshold': 1000000,  # 100만주 이상
            'strong_change_threshold': 500000,    # 50만주 이상
            'moderate_change_threshold': 100000,  # 10만주 이상
            
            # 연속 패턴 감지
            'trend_days_threshold': 3,            # 3일 연속
            'momentum_threshold': 0.7,            # 모멘텀 임계값
            
            # 기관별 가중치
            'institution_weights': {
                'pension_fund': 0.25,      # 연기금 가중치
                'private_equity': 0.20,    # 사모펀드 가중치
                'insurance': 0.15,         # 보험 가중치
                'investment_trust': 0.15,  # 투신 가중치
                'bank': 0.10,              # 은행 가중치
                'other_financial': 0.10,   # 기타금융 가중치
                'other_corp': 0.05         # 기타법인 가중치
            }
        }
        
        # 캐시 디렉토리 생성
        self.cache_file.parent.mkdir(exist_ok=True)
        
        logger.info("✅ 수급 데이터 분석기 초기화 완료")

    async def analyze_supply_demand(self, stock_code: str, days: int = 30) -> SupplyDemandData:
        """종목의 수급 데이터 종합 분석"""
        try:
            logger.info(f"📊 [{stock_code}] 수급 데이터 분석 시작 (기간: {days}일)")
            
            # 기본 데이터 수집
            basic_data = await self._collect_basic_supply_demand(stock_code, days)
            detailed_data = await self._collect_detailed_supply_demand(stock_code)
            
            if not basic_data and not detailed_data:
                logger.warning(f"⚠️ [{stock_code}] 수급 데이터 수집 실패")
                return self._create_empty_data(stock_code)
            
            # 수급 데이터 통합
            supply_demand_data = self._integrate_supply_demand_data(
                stock_code, basic_data, detailed_data
            )
            
            # 수급 점수 계산
            supply_demand_data.supply_demand_score = self._calculate_supply_demand_score(supply_demand_data)
            supply_demand_data.level = self._determine_supply_demand_level(supply_demand_data.supply_demand_score)
            
            # 패턴 분석
            pattern_data = await self._analyze_supply_demand_pattern(stock_code, days)
            supply_demand_data.trend_days = pattern_data.get('trend_days', 0)
            supply_demand_data.momentum_score = pattern_data.get('momentum_score', 0.0)
            
            logger.info(f"✅ [{stock_code}] 수급 분석 완료 - 점수: {supply_demand_data.supply_demand_score:.2f}")
            return supply_demand_data
            
        except Exception as e:
            logger.error(f"❌ [{stock_code}] 수급 분석 중 오류: {e}")
            return self._create_empty_data(stock_code)

    async def _collect_basic_supply_demand(self, stock_code: str, days: int) -> Optional[Dict]:
        """기본 수급 데이터 수집 (PyKRX)"""
        try:
            if not self.data_manager:
                return None
                
            # 데이터 매니저를 통해 실제 수급 데이터 수집
            supply_demand_data = await self.data_manager.get_supply_demand_data(stock_code, days)
            
            if supply_demand_data:
                logger.info(f"✅ [{stock_code}] 기본 수급 데이터 수집 완료")
                return supply_demand_data
            else:
                logger.warning(f"⚠️ [{stock_code}] 기본 수급 데이터 없음")
                return None
                
        except Exception as e:
            logger.error(f"❌ [{stock_code}] 기본 수급 데이터 수집 실패: {e}")
            return None

    async def _collect_detailed_supply_demand(self, stock_code: str) -> Optional[Dict]:
        """세분화된 수급 데이터 수집 (한투 API)"""
        try:
            if not self.data_manager:
                return None
                
            # 데이터 매니저를 통해 상세 투자자별 데이터 수집
            detailed_data = await self.data_manager.get_detailed_investor_data(stock_code)
            
            if detailed_data:
                logger.info(f"✅ [{stock_code}] 상세 수급 데이터 수집 완료")
                return detailed_data
            else:
                logger.warning(f"⚠️ [{stock_code}] 상세 수급 데이터 없음")
                return None
                
        except Exception as e:
            logger.error(f"❌ [{stock_code}] 상세 수급 데이터 수집 실패: {e}")
            return None

    def _integrate_supply_demand_data(self, stock_code: str, basic_data: Dict, detailed_data: Dict) -> SupplyDemandData:
        """수급 데이터 통합"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # 종목명 가져오기 (임시)
        stock_name = f"종목_{stock_code}"
        
        supply_demand_data = SupplyDemandData(
            stock_code=stock_code,
            stock_name=stock_name,
            date=current_date
        )
        
        # 기본 데이터 통합
        if basic_data:
            supply_demand_data.foreign_net = basic_data.get('foreign_net', 0)
            supply_demand_data.institution_net = basic_data.get('institution_net', 0)
            supply_demand_data.individual_net = basic_data.get('individual_net', 0)
            supply_demand_data.total_volume = basic_data.get('total_volume', 0)
        
        # 상세 데이터 통합
        if detailed_data:
            supply_demand_data.pension_fund = detailed_data.get('pension_fund', 0)
            supply_demand_data.private_equity = detailed_data.get('private_equity', 0)
            supply_demand_data.insurance = detailed_data.get('insurance', 0)
            supply_demand_data.investment_trust = detailed_data.get('investment_trust', 0)
            supply_demand_data.bank = detailed_data.get('bank', 0)
            supply_demand_data.other_financial = detailed_data.get('other_financial', 0)
            supply_demand_data.other_corp = detailed_data.get('other_corp', 0)
        
        return supply_demand_data

    def _calculate_supply_demand_score(self, data: SupplyDemandData) -> float:
        """수급 점수 계산 (0-100점)"""
        try:
            # 기본 점수 (외국인 + 기관)
            base_score = 50.0
            
            # 외국인 수급 점수 (±20점)
            if data.total_volume > 0:
                foreign_ratio = data.foreign_net / data.total_volume
                foreign_score = min(max(foreign_ratio * 100, -20), 20)
            else:
                foreign_score = 0
            
            # 기관 수급 점수 (가중치 적용, ±20점)
            institution_score = 0
            total_institution = (
                data.pension_fund + data.private_equity + data.insurance +
                data.investment_trust + data.bank + data.other_financial + data.other_corp
            )
            
            if data.total_volume > 0 and total_institution != 0:
                weights = self.config['institution_weights']
                weighted_score = (
                    data.pension_fund * weights['pension_fund'] +
                    data.private_equity * weights['private_equity'] +
                    data.insurance * weights['insurance'] +
                    data.investment_trust * weights['investment_trust'] +
                    data.bank * weights['bank'] +
                    data.other_financial * weights['other_financial'] +
                    data.other_corp * weights['other_corp']
                )
                institution_ratio = weighted_score / data.total_volume
                institution_score = min(max(institution_ratio * 100, -20), 20)
            
            # 개인 수급 점수 (±10점, 반대 방향)
            if data.total_volume > 0:
                individual_ratio = data.individual_net / data.total_volume
                individual_score = min(max(-individual_ratio * 50, -10), 10)  # 개인 매도 시 긍정적
            else:
                individual_score = 0
            
            # 최종 점수 계산
            final_score = base_score + foreign_score + institution_score + individual_score
            return min(max(final_score, 0), 100)
            
        except Exception as e:
            logger.error(f"❌ 수급 점수 계산 오류: {e}")
            return 50.0

    def _determine_supply_demand_level(self, score: float) -> SupplyDemandLevel:
        """수급 점수에 따른 레벨 결정"""
        if score >= 85:
            return SupplyDemandLevel.EXTREME_BUY
        elif score >= 70:
            return SupplyDemandLevel.STRONG_BUY
        elif score >= 60:
            return SupplyDemandLevel.MODERATE_BUY
        elif score >= 40:
            return SupplyDemandLevel.NEUTRAL
        elif score >= 30:
            return SupplyDemandLevel.MODERATE_SELL
        elif score >= 15:
            return SupplyDemandLevel.STRONG_SELL
        else:
            return SupplyDemandLevel.EXTREME_SELL

    async def _analyze_supply_demand_pattern(self, stock_code: str, days: int) -> Dict:
        """수급 패턴 분석"""
        try:
            # 과거 데이터 로드
            historical_data = await self._load_historical_supply_demand(stock_code, days)
            
            if not historical_data:
                return {'trend_days': 0, 'momentum_score': 0.0}
            
            # 연속 매수/매도 일수 계산
            trend_days = self._calculate_trend_days(historical_data)
            
            # 모멘텀 점수 계산
            momentum_score = self._calculate_momentum_score(historical_data)
            
            return {
                'trend_days': trend_days,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"❌ [{stock_code}] 패턴 분석 오류: {e}")
            return {'trend_days': 0, 'momentum_score': 0.0}

    async def _load_historical_supply_demand(self, stock_code: str, days: int) -> List[Dict]:
        """과거 수급 데이터 로드"""
        # 실제 구현에서는 데이터베이스나 캐시에서 로드
        return []

    def _calculate_trend_days(self, historical_data: List[Dict]) -> int:
        """연속 매수/매도 일수 계산"""
        if not historical_data:
            return 0
        
        trend_days = 0
        last_direction = None
        
        for data in reversed(historical_data):
            net_buy = data.get('foreign_net', 0) + data.get('institution_net', 0)
            current_direction = 'buy' if net_buy > 0 else 'sell' if net_buy < 0 else 'neutral'
            
            if last_direction is None:
                last_direction = current_direction
                trend_days = 1
            elif current_direction == last_direction and current_direction != 'neutral':
                trend_days += 1
            else:
                break
        
        return trend_days

    def _calculate_momentum_score(self, historical_data: List[Dict]) -> float:
        """모멘텀 점수 계산"""
        if len(historical_data) < 5:
            return 0.0
        
        # 최근 5일간의 수급 변화 추세 분석
        recent_data = historical_data[-5:]
        net_buys = []
        
        for data in recent_data:
            net_buy = data.get('foreign_net', 0) + data.get('institution_net', 0)
            net_buys.append(net_buy)
        
        # 선형 회귀를 통한 추세 강도 계산
        if len(net_buys) >= 2:
            x = np.arange(len(net_buys))
            slope = np.polyfit(x, net_buys, 1)[0]
            momentum_score = min(max(slope / 1000000, -1), 1)  # -1 ~ 1 정규화
            return momentum_score
        
        return 0.0

    async def monitor_supply_demand_changes(self, stock_codes: List[str]) -> List[SupplyDemandAlert]:
        """실시간 수급 변화 모니터링"""
        alerts = []
        
        for stock_code in stock_codes:
            try:
                # 현재 수급 데이터 분석
                current_data = await self.analyze_supply_demand(stock_code)
                
                # 이전 데이터와 비교
                previous_data = await self._get_previous_supply_demand_data(stock_code)
                
                if previous_data:
                    # 급변 감지
                    alert = self._detect_supply_demand_changes(current_data, previous_data)
                    if alert:
                        alerts.append(alert)
                        
                        # 알림 발송
                        if self.notifier:
                            await self._send_supply_demand_alert(alert)
                
                # 현재 데이터 캐시 저장
                await self._cache_supply_demand_data(current_data)
                
            except Exception as e:
                logger.error(f"❌ [{stock_code}] 수급 모니터링 오류: {e}")
        
        return alerts

    def _detect_supply_demand_changes(self, current: SupplyDemandData, previous: SupplyDemandData) -> Optional[SupplyDemandAlert]:
        """수급 급변 감지"""
        try:
            # 외국인 수급 변화
            foreign_change = abs(current.foreign_net - previous.foreign_net)
            institution_change = abs(current.institution_net - previous.institution_net)
            
            # 급변 임계값 확인
            if foreign_change >= self.config['extreme_change_threshold']:
                severity = "CRITICAL"
                alert_type = "외국인 극급변"
            elif institution_change >= self.config['extreme_change_threshold']:
                severity = "CRITICAL"
                alert_type = "기관 극급변"
            elif foreign_change >= self.config['strong_change_threshold']:
                severity = "HIGH"
                alert_type = "외국인 급변"
            elif institution_change >= self.config['strong_change_threshold']:
                severity = "HIGH"
                alert_type = "기관 급변"
            else:
                return None
            
            # 알림 메시지 생성
            direction = "매수" if (current.foreign_net + current.institution_net) > (previous.foreign_net + previous.institution_net) else "매도"
            message = f"""
🚨 {alert_type} 감지!
종목: {current.stock_name} ({current.stock_code})
변화: {direction} 전환
외국인: {current.foreign_net - previous.foreign_net:+,}주
기관: {current.institution_net - previous.institution_net:+,}주
수급점수: {previous.supply_demand_score:.1f} → {current.supply_demand_score:.1f}
"""
            
            return SupplyDemandAlert(
                stock_code=current.stock_code,
                stock_name=current.stock_name,
                alert_type=alert_type,
                message=message.strip(),
                severity=severity,
                timestamp=datetime.now(),
                current_data=current,
                previous_data=previous
            )
            
        except Exception as e:
            logger.error(f"❌ 수급 급변 감지 오류: {e}")
            return None

    async def _send_supply_demand_alert(self, alert: SupplyDemandAlert):
        """수급 알림 발송"""
        try:
            if self.notifier:
                await self.notifier.send_message(alert.message)
                logger.info(f"📢 수급 알림 발송: {alert.stock_code} - {alert.alert_type}")
        except Exception as e:
            logger.error(f"❌ 수급 알림 발송 실패: {e}")

    async def _get_previous_supply_demand_data(self, stock_code: str) -> Optional[SupplyDemandData]:
        """이전 수급 데이터 조회"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if stock_code in cache_data:
                        data_dict = cache_data[stock_code]
                        return SupplyDemandData(**data_dict)
        except Exception as e:
            logger.error(f"❌ 이전 수급 데이터 조회 실패: {e}")
        return None

    async def _cache_supply_demand_data(self, data: SupplyDemandData):
        """수급 데이터 캐시 저장"""
        try:
            cache_data = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # 데이터 직렬화를 위해 딕셔너리로 변환
            data_dict = data.to_dict()
            
            cache_data[data.stock_code] = data_dict
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"❌ 수급 데이터 캐시 저장 실패: {e}")

    def _create_empty_data(self, stock_code: str) -> SupplyDemandData:
        """빈 수급 데이터 생성"""
        return SupplyDemandData(
            stock_code=stock_code,
            stock_name=f"종목_{stock_code}",
            date=datetime.now().strftime('%Y-%m-%d')
        )

    async def get_supply_demand_ranking(self, stock_codes: List[str], limit: int = 20) -> List[SupplyDemandData]:
        """수급 순위 조회"""
        try:
            logger.info(f"📊 수급 순위 분석 시작 (대상: {len(stock_codes)}개 종목)")
            
            # 병렬로 수급 데이터 분석
            tasks = [self.analyze_supply_demand(code) for code in stock_codes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 유효한 결과만 필터링
            valid_results = [r for r in results if isinstance(r, SupplyDemandData)]
            
            # 수급 점수 기준 정렬
            ranked_results = sorted(valid_results, key=lambda x: x.supply_demand_score, reverse=True)
            
            logger.info(f"✅ 수급 순위 분석 완료 (상위 {min(limit, len(ranked_results))}개 종목)")
            return ranked_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ 수급 순위 분석 오류: {e}")
            return []

    def generate_supply_demand_report(self, data_list: List[SupplyDemandData]) -> str:
        """수급 분석 리포트 생성"""
        if not data_list:
            return "📊 수급 분석 데이터가 없습니다."
        
        report = "📊 **수급 분석 리포트**\n"
        report += "=" * 50 + "\n\n"
        
        # 상위 10개 종목 리포트
        top_stocks = data_list[:10]
        
        report += "🔥 **수급 상위 10개 종목**\n"
        report += "-" * 30 + "\n"
        
        for i, data in enumerate(top_stocks, 1):
            report += f"{i:2d}. {data.stock_name} ({data.stock_code})\n"
            report += f"    수급점수: {data.supply_demand_score:.1f}점 ({data.level.value})\n"
            report += f"    외국인: {data.foreign_net:+,}주 | 기관: {data.institution_net:+,}주\n"
            
            if data.trend_days > 0:
                report += f"    연속패턴: {data.trend_days}일 연속\n"
            
            report += "\n"
        
        # 통계 정보
        report += "📈 **수급 통계**\n"
        report += "-" * 20 + "\n"
        
        avg_score = sum(d.supply_demand_score for d in data_list) / len(data_list)
        report += f"평균 수급점수: {avg_score:.1f}점\n"
        
        # 레벨별 분포
        level_counts = {}
        for data in data_list:
            level = data.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        report += "\n레벨별 분포:\n"
        for level, count in sorted(level_counts.items()):
            report += f"  {level}: {count}개 종목\n"
        
        return report

    async def cleanup(self):
        """리소스 정리"""
        try:
            logger.info("🧹 수급 분석기 리소스 정리 중...")
            self.monitoring_stocks.clear()
            self.alert_history.clear()
            logger.info("✅ 수급 분석기 정리 완료")
        except Exception as e:
            logger.error(f"❌ 수급 분석기 정리 중 오류: {e}") 