"""
로버트 아놀드 CAT 전략 (Robert Arnold CAT Strategy)
- 모멘텀과 시스템 매매의 대가
- 트렌드 추종과 리스크 관리를 중시
- 기계적 매매 시스템으로 감정 배제
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from ..common import BaseStrategy, StrategyResult, get_stock_value, get_financial_metrics

class ArnoldCatStrategy(BaseStrategy):
    """
    로버트 아놀드 CAT 전략
    
    핵심 원칙:
    1. 모멘텀 추종 - 강한 추세를 따라간다
    2. 시스템 매매 - 감정을 배제한 기계적 매매
    3. 리스크 관리 - 손실 제한과 수익 보호
    4. 다중 시간대 분석 - 단기/중기/장기 추세 확인
    5. 거래량 확인 - 모멘텀의 신뢰성 검증
    """
    
    def __init__(self):
        super().__init__()
        self.strategy_name = "로버트 아놀드 CAT 전략"
        self.description = "모멘텀 기반 시스템 매매 전략"
        
        # 전략 파라미터
        self.momentum_period = 20  # 모멘텀 계산 기간
        self.trend_period = 50     # 트렌드 확인 기간
        self.volume_threshold = 1.5  # 거래량 증가 기준
        self.rsi_overbought = 70   # RSI 과매수 기준
        self.rsi_oversold = 30     # RSI 과매도 기준
        
    def analyze_stock(self, stock_data: Dict[str, Any]) -> StrategyResult:
        """주식 분석 및 투자 추천"""
        try:
            # 기본 데이터 추출
            symbol = get_stock_value(stock_data, 'symbol', 'Unknown')
            price = get_stock_value(stock_data, 'price', 0)
            
            if price <= 0:
                return self._create_error_result(symbol, "가격 정보 없음")
            
            # 재무 지표 추출
            metrics = get_financial_metrics(stock_data)
            
            # 1. 모멘텀 분석
            momentum_score = self._analyze_momentum(stock_data, metrics)
            
            # 2. 트렌드 분석
            trend_score = self._analyze_trend(stock_data, metrics)
            
            # 3. 거래량 분석
            volume_score = self._analyze_volume(stock_data, metrics)
            
            # 4. 기술적 지표 분석
            technical_score = self._analyze_technical(stock_data, metrics)
            
            # 5. 리스크 분석
            risk_score = self._analyze_risk(stock_data, metrics)
            
            # 종합 점수 계산 (가중 평균)
            total_score = (
                momentum_score * 0.30 +  # 모멘텀 30%
                trend_score * 0.25 +     # 트렌드 25%
                volume_score * 0.20 +    # 거래량 20%
                technical_score * 0.15 + # 기술적 지표 15%
                risk_score * 0.10        # 리스크 10%
            )
            
            # 투자 결정
            decision = self._make_investment_decision(total_score, momentum_score, trend_score)
            
            # 목표가 및 손절가 계산
            target_price, stop_loss = self._calculate_price_targets(price, total_score)
            
            return StrategyResult(
                symbol=symbol,
                decision=decision,
                confidence=min(total_score / 100, 0.95),
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=self._generate_reasoning(
                    momentum_score, trend_score, volume_score, 
                    technical_score, risk_score, total_score
                ),
                risk_level=self._calculate_risk_level(risk_score, total_score)
            )
            
        except Exception as e:
            return self._create_error_result(
                get_stock_value(stock_data, 'symbol', 'Unknown'),
                f"분석 중 오류: {str(e)}"
            )
    
    def _analyze_momentum(self, stock_data: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """모멘텀 분석"""
        score = 0
        
        try:
            # 가격 모멘텀 (20일 수익률)
            price_change = metrics.get('price_change_20d', 0)
            if price_change > 15:
                score += 30
            elif price_change > 10:
                score += 25
            elif price_change > 5:
                score += 20
            elif price_change > 0:
                score += 15
            else:
                score += 5
            
            # 상대 강도 (시장 대비 성과)
            market_return = metrics.get('market_return', 0)
            relative_strength = price_change - market_return
            if relative_strength > 5:
                score += 20
            elif relative_strength > 0:
                score += 15
            elif relative_strength > -5:
                score += 10
            else:
                score += 5
            
            # 연속 상승일 체크
            consecutive_up = metrics.get('consecutive_up_days', 0)
            if consecutive_up >= 5:
                score += 15
            elif consecutive_up >= 3:
                score += 10
            elif consecutive_up >= 1:
                score += 5
            
        except Exception:
            score = 30  # 기본값
        
        return min(score, 65)  # 최대 65점
    
    def _analyze_trend(self, stock_data: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """트렌드 분석"""
        score = 0
        
        try:
            price = get_stock_value(stock_data, 'price', 0)
            
            # 이동평균선 위치
            ma20 = metrics.get('ma_20', price)
            ma50 = metrics.get('ma_50', price)
            ma200 = metrics.get('ma_200', price)
            
            # 가격이 이동평균선 위에 있는지 확인
            if price > ma20:
                score += 15
            if price > ma50:
                score += 15
            if price > ma200:
                score += 10
            
            # 이동평균선 정렬 (골든크로스)
            if ma20 > ma50 > ma200:
                score += 20
            elif ma20 > ma50:
                score += 15
            elif ma50 > ma200:
                score += 10
            
            # 트렌드 강도
            trend_strength = metrics.get('trend_strength', 0)
            if trend_strength > 0.8:
                score += 15
            elif trend_strength > 0.6:
                score += 10
            elif trend_strength > 0.4:
                score += 5
            
        except Exception:
            score = 25  # 기본값
        
        return min(score, 75)  # 최대 75점
    
    def _analyze_volume(self, stock_data: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """거래량 분석"""
        score = 0
        
        try:
            # 거래량 증가율
            volume_ratio = metrics.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                score += 25
            elif volume_ratio > 1.5:
                score += 20
            elif volume_ratio > 1.2:
                score += 15
            elif volume_ratio > 1.0:
                score += 10
            else:
                score += 5
            
            # 가격-거래량 관계
            price_change = metrics.get('price_change_1d', 0)
            if price_change > 0 and volume_ratio > 1.2:  # 상승과 함께 거래량 증가
                score += 20
            elif price_change < 0 and volume_ratio < 0.8:  # 하락과 함께 거래량 감소
                score += 15
            
            # OBV (On Balance Volume) 추세
            obv_trend = metrics.get('obv_trend', 0)
            if obv_trend > 0:
                score += 15
            elif obv_trend == 0:
                score += 10
            else:
                score += 5
            
        except Exception:
            score = 40  # 기본값
        
        return min(score, 60)  # 최대 60점
    
    def _analyze_technical(self, stock_data: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """기술적 지표 분석"""
        score = 0
        
        try:
            # RSI 분석
            rsi = metrics.get('rsi', 50)
            if 30 < rsi < 70:  # 정상 범위
                score += 20
            elif rsi > 70:  # 과매수
                score += 10
            elif rsi < 30:  # 과매도
                score += 15
            
            # MACD 분석
            macd_signal = metrics.get('macd_signal', 0)
            if macd_signal > 0:  # 매수 신호
                score += 20
            elif macd_signal == 0:  # 중립
                score += 10
            else:  # 매도 신호
                score += 5
            
            # 볼린저 밴드 위치
            bb_position = metrics.get('bb_position', 0.5)
            if 0.2 < bb_position < 0.8:  # 정상 범위
                score += 15
            elif bb_position > 0.8:  # 상단 근처
                score += 10
            elif bb_position < 0.2:  # 하단 근처
                score += 12
            
            # 스토캐스틱
            stoch_k = metrics.get('stoch_k', 50)
            stoch_d = metrics.get('stoch_d', 50)
            if stoch_k > stoch_d and stoch_k < 80:  # 상승 모멘텀
                score += 15
            elif stoch_k < 20:  # 과매도에서 반등 기대
                score += 12
            
        except Exception:
            score = 35  # 기본값
        
        return min(score, 70)  # 최대 70점
    
    def _analyze_risk(self, stock_data: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """리스크 분석"""
        score = 50  # 기본 점수
        
        try:
            # 변동성 분석
            volatility = metrics.get('volatility_20d', 0.2)
            if volatility < 0.15:  # 낮은 변동성
                score += 20
            elif volatility < 0.25:  # 보통 변동성
                score += 15
            elif volatility < 0.35:  # 높은 변동성
                score += 10
            else:  # 매우 높은 변동성
                score += 5
            
            # 베타 분석
            beta = metrics.get('beta', 1.0)
            if 0.8 <= beta <= 1.2:  # 시장과 유사한 위험
                score += 15
            elif beta < 0.8:  # 낮은 위험
                score += 20
            elif beta > 1.5:  # 높은 위험
                score += 5
            else:
                score += 10
            
            # 최대 손실 분석
            max_drawdown = metrics.get('max_drawdown', 0.1)
            if max_drawdown < 0.1:
                score += 15
            elif max_drawdown < 0.2:
                score += 10
            elif max_drawdown < 0.3:
                score += 5
            else:
                score -= 5
            
        except Exception:
            pass
        
        return max(min(score, 100), 0)  # 0-100점 범위
    
    def _make_investment_decision(self, total_score: float, momentum_score: float, trend_score: float) -> str:
        """투자 결정"""
        # 강력한 매수 조건
        if total_score >= 75 and momentum_score >= 50 and trend_score >= 55:
            return "강력매수"
        
        # 매수 조건
        elif total_score >= 65 and momentum_score >= 40:
            return "매수"
        
        # 관망 조건
        elif total_score >= 45:
            return "관망"
        
        # 매도 조건
        else:
            return "매도"
    
    def _calculate_price_targets(self, current_price: float, total_score: float) -> Tuple[float, float]:
        """목표가 및 손절가 계산"""
        # 점수에 따른 목표 수익률 설정
        if total_score >= 75:
            target_return = 0.20  # 20% 목표
            stop_loss_rate = 0.08  # 8% 손절
        elif total_score >= 65:
            target_return = 0.15  # 15% 목표
            stop_loss_rate = 0.10  # 10% 손절
        elif total_score >= 55:
            target_return = 0.10  # 10% 목표
            stop_loss_rate = 0.12  # 12% 손절
        else:
            target_return = 0.05  # 5% 목표
            stop_loss_rate = 0.15  # 15% 손절
        
        target_price = current_price * (1 + target_return)
        stop_loss = current_price * (1 - stop_loss_rate)
        
        return target_price, stop_loss
    
    def _calculate_risk_level(self, risk_score: float, total_score: float) -> str:
        """위험 수준 계산"""
        if risk_score >= 70 and total_score >= 65:
            return "낮음"
        elif risk_score >= 50 and total_score >= 50:
            return "보통"
        elif risk_score >= 30:
            return "높음"
        else:
            return "매우높음"
    
    def _generate_reasoning(self, momentum_score: float, trend_score: float, 
                          volume_score: float, technical_score: float, 
                          risk_score: float, total_score: float) -> str:
        """투자 근거 생성"""
        reasoning = []
        
        # 모멘텀 분석
        if momentum_score >= 50:
            reasoning.append("🚀 강력한 가격 모멘텀 확인")
        elif momentum_score >= 35:
            reasoning.append("📈 양호한 가격 모멘텀")
        else:
            reasoning.append("📉 모멘텀 부족")
        
        # 트렌드 분석
        if trend_score >= 55:
            reasoning.append("📊 명확한 상승 트렌드")
        elif trend_score >= 40:
            reasoning.append("📈 상승 추세 진행 중")
        else:
            reasoning.append("📉 트렌드 약화")
        
        # 거래량 분석
        if volume_score >= 45:
            reasoning.append("💪 거래량 증가로 신뢰성 높음")
        elif volume_score >= 30:
            reasoning.append("👍 적정 거래량 수준")
        else:
            reasoning.append("⚠️ 거래량 부족")
        
        # 기술적 지표
        if technical_score >= 50:
            reasoning.append("🔧 기술적 지표 양호")
        elif technical_score >= 35:
            reasoning.append("🔧 기술적 지표 보통")
        else:
            reasoning.append("🔧 기술적 지표 부정적")
        
        # 리스크 분석
        if risk_score >= 60:
            reasoning.append("🛡️ 리스크 관리 양호")
        elif risk_score >= 40:
            reasoning.append("⚖️ 적정 위험 수준")
        else:
            reasoning.append("⚠️ 높은 위험 수준")
        
        # 로버트 아놀드 실전 팁 추가
        if total_score >= 70:
            reasoning.append("\n💡 아놀드 전략: 시스템 신호에 따라 기계적 매수")
            reasoning.append("📱 실전 팁: 네이버 증시 '기관/외국인 동향' 확인")
        elif total_score >= 50:
            reasoning.append("\n💡 아놀드 전략: 추가 확인 후 진입 고려")
            reasoning.append("📊 실전 팁: HTS 차트에서 거래량 패턴 재확인")
        else:
            reasoning.append("\n💡 아놀드 전략: 시스템 신호 부정적, 진입 금지")
            reasoning.append("🚫 실전 팁: 감정적 판단 금지, 시스템 규칙 준수")
        
        return " | ".join(reasoning)
    
    def _create_error_result(self, symbol: str, error_msg: str) -> StrategyResult:
        """에러 결과 생성"""
        return StrategyResult(
            symbol=symbol,
            decision="분석불가",
            confidence=0.0,
            target_price=0.0,
            stop_loss=0.0,
            reasoning=f"❌ 오류: {error_msg}",
            risk_level="알수없음"
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        return {
            "name": self.strategy_name,
            "description": self.description,
            "type": "모멘텀/시스템매매",
            "risk_level": "중간",
            "time_horizon": "단기-중기 (1-6개월)",
            "key_indicators": [
                "가격 모멘텀",
                "이동평균선 배열",
                "거래량 증가",
                "RSI/MACD",
                "리스크 지표"
            ],
            "strengths": [
                "감정 배제한 시스템 매매",
                "명확한 진입/청산 규칙",
                "모멘텀 추종으로 큰 수익 가능",
                "리스크 관리 체계적"
            ],
            "weaknesses": [
                "횡보장에서 손실 가능",
                "급격한 시장 변화에 늦은 반응",
                "단기 변동성에 민감"
            ],
            "best_market": "상승 추세장",
            "parameters": {
                "momentum_period": self.momentum_period,
                "trend_period": self.trend_period,
                "volume_threshold": self.volume_threshold,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold
            }
        } 