"""
수급 분석기 모듈 (Supply & Demand Analyzer)
GitHub 참고: TradingView 수급 분석 지표 기반
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class SupplyDemandZone:
    """수급 존(Supply/Demand Zone) 데이터 클래스"""
    zone_type: str  # 'supply' 또는 'demand'
    high_price: float
    low_price: float
    volume: float
    strength: float  # 존의 강도 (0-100)
    created_at: datetime
    touches: int = 0  # 테스트 횟수
    is_active: bool = True

@dataclass
class SupplyDemandData:
    """수급 분석 결과 데이터 클래스"""
    stock_code: str
    current_price: float
    supply_zones: List[SupplyDemandZone]
    demand_zones: List[SupplyDemandZone]
    net_supply_demand: float  # 순 수급 (-100 ~ 100)
    volume_analysis: Dict[str, Any]
    price_action_strength: float
    recommendation: str
    confidence: float
    analysis_time: datetime

class SupplyDemandAnalyzer:
    """수급 분석기 클래스"""
    
    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 분석 파라미터
        self.zone_difference_scale = 2.0  # 존 생성을 위한 캔들 크기 배수
        self.min_volume_multiplier = 1.5  # 최소 볼륨 배수
        self.zone_strength_threshold = 30.0  # 존 강도 임계값
        self.max_zone_age_days = 30  # 존의 최대 유효 기간
        
    async def analyze_supply_demand(self, stock_code: str, days: int = 30) -> SupplyDemandData:
        """
        주식의 수급 분석 수행
        
        Args:
            stock_code: 주식 코드
            days: 분석 기간 (일)
            
        Returns:
            SupplyDemandData: 수급 분석 결과
        """
        try:
            # 주가 데이터 가져오기
            df = await self._get_stock_data(stock_code, days)
            if df is None or len(df) < 20:
                return self._create_empty_data(stock_code)
            
            # 수급 존 식별
            supply_zones = await self._identify_supply_zones(df)
            demand_zones = await self._identify_demand_zones(df)
            
            # 볼륨 분석
            volume_analysis = await self._analyze_volume_patterns(df)
            
            # 현재 가격과 수급 존 관계 분석
            current_price = float(df['Close'].iloc[-1])
            net_supply_demand = self._calculate_net_supply_demand(
                current_price, supply_zones, demand_zones
            )
            
            # 가격 행동 강도 분석
            price_action_strength = self._analyze_price_action_strength(df)
            
            # 추천 및 신뢰도 계산
            recommendation, confidence = self._generate_recommendation(
                net_supply_demand, volume_analysis, price_action_strength
            )
            
            return SupplyDemandData(
                stock_code=stock_code,
                current_price=current_price,
                supply_zones=supply_zones,
                demand_zones=demand_zones,
                net_supply_demand=net_supply_demand,
                volume_analysis=volume_analysis,
                price_action_strength=price_action_strength,
                recommendation=recommendation,
                confidence=confidence,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"수급 분석 오류 ({stock_code}): {e}")
            return self._create_empty_data(stock_code)
    
    async def _get_stock_data(self, stock_code: str, days: int) -> Optional[pd.DataFrame]:
        """주가 데이터 가져오기"""
        try:
            # 한국 주식인 경우 .KS 추가
            if len(stock_code) == 6 and stock_code.isdigit():
                ticker = f"{stock_code}.KS"
            else:
                ticker = stock_code
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # yfinance를 사용하여 데이터 가져오기
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                # KOSDAQ 시도
                ticker = f"{stock_code}.KQ"
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
            
            if not df.empty:
                # 기술적 지표 추가
                df = self._add_technical_indicators(df)
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"주가 데이터 가져오기 오류 ({stock_code}): {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        try:
            # 이동평균
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # 볼륨 이동평균
            df['VMA20'] = df['Volume'].rolling(window=20).mean()
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # 가격 변화율
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 오류: {e}")
            return df
    
    async def _identify_supply_zones(self, df: pd.DataFrame) -> List[SupplyDemandZone]:
        """공급 존 식별"""
        supply_zones = []
        
        try:
            for i in range(1, len(df) - 1):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                next_candle = df.iloc[i+1] if i+1 < len(df) else current
                
                # 공급 존 조건: 녹색 캔들 후 큰 적색 캔들
                if (prev['Close'] > prev['Open'] and  # 이전 캔들이 상승
                    current['Close'] < current['Open'] and  # 현재 캔들이 하락
                    abs(current['Close'] - current['Open']) >= 
                    self.zone_difference_scale * abs(prev['Close'] - prev['Open']) and
                    current['Volume'] > df['VMA20'].iloc[i] * self.min_volume_multiplier):
                    
                    zone_high = max(prev['High'], current['High'])
                    zone_low = prev['Open']
                    strength = self._calculate_zone_strength(df, i, 'supply')
                    
                    if strength >= self.zone_strength_threshold:
                        supply_zone = SupplyDemandZone(
                            zone_type='supply',
                            high_price=zone_high,
                            low_price=zone_low,
                            volume=current['Volume'],
                            strength=strength,
                            created_at=current.name.to_pydatetime() if hasattr(current.name, 'to_pydatetime') else datetime.now()
                        )
                        supply_zones.append(supply_zone)
            
            # 중복 제거 및 강도 순 정렬
            supply_zones = self._filter_overlapping_zones(supply_zones)
            supply_zones.sort(key=lambda x: x.strength, reverse=True)
            
            return supply_zones[:5]  # 상위 5개만 반환
            
        except Exception as e:
            logger.error(f"공급 존 식별 오류: {e}")
            return []
    
    async def _identify_demand_zones(self, df: pd.DataFrame) -> List[SupplyDemandZone]:
        """수요 존 식별"""
        demand_zones = []
        
        try:
            for i in range(1, len(df) - 1):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                
                # 수요 존 조건: 적색 캔들 후 큰 녹색 캔들
                if (prev['Close'] < prev['Open'] and  # 이전 캔들이 하락
                    current['Close'] > current['Open'] and  # 현재 캔들이 상승
                    abs(current['Close'] - current['Open']) >= 
                    self.zone_difference_scale * abs(prev['Close'] - prev['Open']) and
                    current['Volume'] > df['VMA20'].iloc[i] * self.min_volume_multiplier):
                    
                    zone_high = prev['Open']
                    zone_low = min(prev['Low'], current['Low'])
                    strength = self._calculate_zone_strength(df, i, 'demand')
                    
                    if strength >= self.zone_strength_threshold:
                        demand_zone = SupplyDemandZone(
                            zone_type='demand',
                            high_price=zone_high,
                            low_price=zone_low,
                            volume=current['Volume'],
                            strength=strength,
                            created_at=current.name.to_pydatetime() if hasattr(current.name, 'to_pydatetime') else datetime.now()
                        )
                        demand_zones.append(demand_zone)
            
            # 중복 제거 및 강도 순 정렬
            demand_zones = self._filter_overlapping_zones(demand_zones)
            demand_zones.sort(key=lambda x: x.strength, reverse=True)
            
            return demand_zones[:5]  # 상위 5개만 반환
            
        except Exception as e:
            logger.error(f"수요 존 식별 오류: {e}")
            return []
    
    def _calculate_zone_strength(self, df: pd.DataFrame, index: int, zone_type: str) -> float:
        """존의 강도 계산"""
        try:
            current = df.iloc[index]
            
            # 기본 강도 (볼륨 기반)
            volume_strength = min((current['Volume'] / df['VMA20'].iloc[index]) * 20, 50)
            
            # 가격 움직임 강도
            price_move = abs(current['Close'] - current['Open']) / current['Open'] * 100
            price_strength = min(price_move * 10, 30)
            
            # ATR 대비 움직임
            atr_strength = 0
            if pd.notna(current['ATR']) and current['ATR'] > 0:
                atr_ratio = abs(current['Close'] - current['Open']) / current['ATR']
                atr_strength = min(atr_ratio * 10, 20)
            
            total_strength = volume_strength + price_strength + atr_strength
            return min(total_strength, 100)
            
        except Exception as e:
            logger.error(f"존 강도 계산 오류: {e}")
            return 0
    
    def _filter_overlapping_zones(self, zones: List[SupplyDemandZone]) -> List[SupplyDemandZone]:
        """겹치는 존 필터링"""
        if not zones:
            return zones
        
        filtered_zones = []
        zones_sorted = sorted(zones, key=lambda x: x.strength, reverse=True)
        
        for zone in zones_sorted:
            is_overlapping = False
            for existing_zone in filtered_zones:
                # 존이 겹치는지 확인
                if (zone.low_price <= existing_zone.high_price and 
                    zone.high_price >= existing_zone.low_price):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_zones.append(zone)
        
        return filtered_zones
    
    async def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """볼륨 패턴 분석"""
        try:
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = df['Volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # 볼륨 트렌드 분석
            volume_trend = 'neutral'
            if volume_ratio > 1.5:
                volume_trend = 'increasing'
            elif volume_ratio < 0.7:
                volume_trend = 'decreasing'
            
            # 이상 볼륨 스파이크 감지
            volume_spikes = []
            volume_threshold = df['VMA20'] * 2
            
            for i in range(len(df)):
                if df['Volume'].iloc[i] > volume_threshold.iloc[i]:
                    volume_spikes.append({
                        'date': df.index[i],
                        'volume': df['Volume'].iloc[i],
                        'multiplier': df['Volume'].iloc[i] / df['VMA20'].iloc[i]
                    })
            
            return {
                'current_volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'recent_spikes': volume_spikes[-5:],  # 최근 5개 스파이크
                'avg_daily_volume': avg_volume,
                'volume_volatility': df['Volume'].std() / avg_volume if avg_volume > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"볼륨 패턴 분석 오류: {e}")
            return {'current_volume_ratio': 1.0, 'volume_trend': 'neutral', 'recent_spikes': []}
    
    def _calculate_net_supply_demand(self, current_price: float, 
                                   supply_zones: List[SupplyDemandZone],
                                   demand_zones: List[SupplyDemandZone]) -> float:
        """순 수급 계산"""
        try:
            supply_pressure = 0
            demand_support = 0
            
            # 현재 가격 근처의 존들만 고려 (±5% 범위)
            price_range = current_price * 0.05
            
            for zone in supply_zones:
                if abs(zone.low_price - current_price) <= price_range:
                    distance_factor = 1 - (abs(zone.low_price - current_price) / price_range)
                    supply_pressure += zone.strength * distance_factor
            
            for zone in demand_zones:
                if abs(zone.high_price - current_price) <= price_range:
                    distance_factor = 1 - (abs(zone.high_price - current_price) / price_range)
                    demand_support += zone.strength * distance_factor
            
            # -100 ~ 100 범위로 정규화
            total_pressure = supply_pressure + demand_support
            if total_pressure > 0:
                net_supply_demand = ((demand_support - supply_pressure) / total_pressure) * 100
            else:
                net_supply_demand = 0
            
            return max(-100, min(100, net_supply_demand))
            
        except Exception as e:
            logger.error(f"순 수급 계산 오류: {e}")
            return 0
    
    def _analyze_price_action_strength(self, df: pd.DataFrame) -> float:
        """가격 행동 강도 분석"""
        try:
            # 최근 5일간의 가격 움직임 분석
            recent_data = df.tail(5)
            
            # 가격 변동성
            price_volatility = recent_data['Close'].std() / recent_data['Close'].mean()
            
            # 트렌드 강도 (연속 상승/하락 일수)
            price_changes = recent_data['Close'].diff().dropna()
            trend_strength = 0
            current_trend = 0
            
            for change in price_changes:
                if change > 0:
                    if current_trend >= 0:
                        current_trend += 1
                    else:
                        current_trend = 1
                elif change < 0:
                    if current_trend <= 0:
                        current_trend -= 1
                    else:
                        current_trend = -1
                else:
                    current_trend = 0
                
                trend_strength = max(trend_strength, abs(current_trend))
            
            # 정규화 (0-100)
            strength = min((price_volatility * 1000 + trend_strength * 20), 100)
            return max(0, strength)
            
        except Exception as e:
            logger.error(f"가격 행동 강도 분석 오류: {e}")
            return 50
    
    def _generate_recommendation(self, net_supply_demand: float, 
                               volume_analysis: Dict[str, Any],
                               price_action_strength: float) -> Tuple[str, float]:
        """추천 및 신뢰도 생성"""
        try:
            confidence = 50  # 기본 신뢰도
            
            # 수급 기반 추천
            if net_supply_demand > 30:
                recommendation = "매수"
                confidence += min(net_supply_demand - 30, 30)
            elif net_supply_demand < -30:
                recommendation = "매도"
                confidence += min(abs(net_supply_demand) - 30, 30)
            else:
                recommendation = "관망"
            
            # 볼륨 확인으로 신뢰도 조정
            volume_ratio = volume_analysis.get('current_volume_ratio', 1.0)
            if volume_ratio > 1.5:
                confidence += 10
            elif volume_ratio < 0.7:
                confidence -= 10
            
            # 가격 행동 강도로 신뢰도 조정
            if price_action_strength > 70:
                confidence += 10
            elif price_action_strength < 30:
                confidence -= 10
            
            confidence = max(0, min(100, confidence))
            
            return recommendation, confidence
            
        except Exception as e:
            logger.error(f"추천 생성 오류: {e}")
            return "관망", 50
    
    async def get_supply_demand_ranking(self, stock_codes: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """수급 순위 조회"""
        try:
            tasks = []
            for stock_code in stock_codes[:20]:  # 최대 20개까지만 분석
                tasks.append(self.analyze_supply_demand(stock_code, 30))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            ranking = []
            for result in results:
                if isinstance(result, SupplyDemandData) and result.confidence > 60:
                    ranking.append({
                        'stock_code': result.stock_code,
                        'net_supply_demand': result.net_supply_demand,
                        'recommendation': result.recommendation,
                        'confidence': result.confidence,
                        'current_price': result.current_price
                    })
            
            # 순 수급 순으로 정렬
            ranking.sort(key=lambda x: x['net_supply_demand'], reverse=True)
            return ranking[:limit]
            
        except Exception as e:
            logger.error(f"수급 순위 조회 오류: {e}")
            return []
    
    async def monitor_supply_demand_changes(self, stock_codes: List[str]) -> List[Dict[str, Any]]:
        """수급 변화 모니터링"""
        try:
            alerts = []
            
            for stock_code in stock_codes:
                data = await self.analyze_supply_demand(stock_code, 30)
                
                # 강한 수급 신호 감지
                if abs(data.net_supply_demand) > 50 and data.confidence > 70:
                    alert_type = "강한 매수 신호" if data.net_supply_demand > 0 else "강한 매도 신호"
                    
                    alerts.append({
                        'stock_code': stock_code,
                        'alert_type': alert_type,
                        'net_supply_demand': data.net_supply_demand,
                        'confidence': data.confidence,
                        'current_price': data.current_price,
                        'recommendation': data.recommendation
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"수급 변화 모니터링 오류: {e}")
            return []
    
    def generate_supply_demand_report(self, data_list: List[SupplyDemandData]) -> str:
        """수급 분석 리포트 생성"""
        try:
            report = "📊 수급 분석 리포트\n\n"
            
            for data in data_list:
                report += f"🏢 {data.stock_code}\n"
                report += f"💰 현재가: {data.current_price:,.0f}원\n"
                report += f"📈 순수급: {data.net_supply_demand:+.1f}\n"
                report += f"🎯 추천: {data.recommendation}\n"
                report += f"🔥 신뢰도: {data.confidence:.0f}%\n"
                
                # 주요 존 정보
                if data.supply_zones:
                    strongest_supply = max(data.supply_zones, key=lambda x: x.strength)
                    report += f"🔴 주요 공급존: {strongest_supply.low_price:,.0f}원 (강도: {strongest_supply.strength:.0f})\n"
                
                if data.demand_zones:
                    strongest_demand = max(data.demand_zones, key=lambda x: x.strength)
                    report += f"🟢 주요 수요존: {strongest_demand.high_price:,.0f}원 (강도: {strongest_demand.strength:.0f})\n"
                
                report += "\n" + "="*40 + "\n\n"
            
            return report
            
        except Exception as e:
            logger.error(f"리포트 생성 오류: {e}")
            return "리포트 생성 중 오류가 발생했습니다."
    
    def _create_empty_data(self, stock_code: str) -> SupplyDemandData:
        """빈 데이터 생성"""
        return SupplyDemandData(
            stock_code=stock_code,
            current_price=0,
            supply_zones=[],
            demand_zones=[],
            net_supply_demand=0,
            volume_analysis={'current_volume_ratio': 1.0, 'volume_trend': 'neutral', 'recent_spikes': []},
            price_action_strength=50,
            recommendation="데이터 없음",
            confidence=0,
            analysis_time=datetime.now()
        )
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"수급 분석기 정리 오류: {e}") 