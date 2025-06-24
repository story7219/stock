#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ML 기반 투자 분석 프로그램 (사용자 시스템 최적화 버전)
RAM 16GB, i5-4460 CPU 환경에 최적화된 경량 버전
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.system_monitor import get_system_monitor
    from src.ml_engine import AdaptiveLearningEngine
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import time
    import logging
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('investment_analysis.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
except ImportError as e:
    print(f"❌ 필수 라이브러리 누락: {e}")
    print("🔧 다음 명령으로 설치하세요:")
    print("pip install yfinance pandas numpy scikit-learn psutil")
    sys.exit(1)

class LightweightInvestmentAnalyzer:
    """💡 경량화된 투자 분석기 (메모리 최적화)"""
    
    def __init__(self):
        self.system_monitor = get_system_monitor()
        self.ml_engine = None
        
        # 시스템 리소스 체크
        if self.system_monitor.is_system_ready_for_ml():
            logger.info("✅ 시스템 준비 완료 - ML 엔진 로드")
            self.ml_engine = AdaptiveLearningEngine()
        else:
            logger.warning("⚠️ 시스템 리소스 부족 - 기본 분석만 실행")
    
    def get_stock_data(self, symbols, period="6mo"):
        """주식 데이터 수집 (메모리 효율적)"""
        
        stock_data = {}
        batch_size = self.system_monitor.get_recommended_batch_size()
        
        logger.info(f"📊 {len(symbols)}개 종목 데이터 수집 (배치크기: {batch_size})")
        
        # 배치별로 처리하여 메모리 절약
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            
            for symbol in batch_symbols:
                try:
                    # 메모리 효율적인 데이터 로드
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if len(hist) > 0:
                        stock_data[symbol] = {
                            'price': hist['Close'].iloc[-1],
                            'volume': hist['Volume'].iloc[-1],
                            'high_52w': hist['High'].max(),
                            'low_52w': hist['Low'].min(),
                            'returns_1w': (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100 if len(hist) >= 5 else 0,
                            'returns_1m': (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100 if len(hist) >= 20 else 0,
                            'volatility': hist['Close'].pct_change().std() * np.sqrt(252) * 100,
                            'trend_score': self._calculate_trend_score(hist),
                            'momentum_score': self._calculate_momentum_score(hist)
                        }
                        
                        logger.debug(f"  ✓ {symbol}: ${stock_data[symbol]['price']:.2f}")
                    else:
                        logger.warning(f"  ✗ {symbol}: 데이터 없음")
                        
                except Exception as e:
                    logger.warning(f"  ✗ {symbol}: 오류 - {e}")
            
            # 메모리 사용률 체크
            if not self.system_monitor.is_system_ready_for_ml():
                logger.warning("⚠️ 메모리 부족 - 배치 처리 일시 중단")
                time.sleep(2)  # 시스템 회복 대기
        
        logger.info(f"✅ {len(stock_data)}개 종목 데이터 수집 완료")
        return stock_data
    
    def _calculate_trend_score(self, hist):
        """추세 점수 계산"""
        if len(hist) < 20:
            return 0
        
        # 단순 이동평균 기반 추세
        ma_short = hist['Close'].rolling(5).mean().iloc[-1]
        ma_long = hist['Close'].rolling(20).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]
        
        trend_score = 0
        if current_price > ma_short > ma_long:
            trend_score = 100
        elif current_price > ma_short:
            trend_score = 70
        elif current_price > ma_long:
            trend_score = 30
        else:
            trend_score = 0
        
        return trend_score
    
    def _calculate_momentum_score(self, hist):
        """모멘텀 점수 계산"""
        if len(hist) < 14:
            return 50
        
        # RSI 기반 모멘텀
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # RSI를 점수로 변환
        if 30 <= current_rsi <= 70:
            momentum_score = 100 - abs(50 - current_rsi)
        elif current_rsi < 30:
            momentum_score = 30  # 과매도
        else:
            momentum_score = 30  # 과매수
        
        return momentum_score
    
    def analyze_stocks(self, stock_data):
        """종목 분석 및 점수 계산"""
        
        logger.info("💡 투자 점수 계산 중...")
        
        analyzed_stocks = []
        
        for symbol, data in stock_data.items():
            try:
                # 기본 점수 계산
                base_score = 0
                reasons = []
                
                # 1. 추세 점수 (30%)
                trend_weight = 0.3
                trend_contribution = data['trend_score'] * trend_weight
                base_score += trend_contribution
                if data['trend_score'] > 70:
                    reasons.append(f"강한 상승추세 ({data['trend_score']:.0f}점)")
                
                # 2. 모멘텀 점수 (25%)
                momentum_weight = 0.25
                momentum_contribution = data['momentum_score'] * momentum_weight
                base_score += momentum_contribution
                if data['momentum_score'] > 70:
                    reasons.append(f"좋은 모멘텀 ({data['momentum_score']:.0f}점)")
                
                # 3. 수익률 점수 (25%)
                returns_score = min(100, max(0, (data['returns_1m'] + 50) * 2))  # -50%~50% -> 0~100점
                returns_weight = 0.25
                returns_contribution = returns_score * returns_weight
                base_score += returns_contribution
                if data['returns_1m'] > 5:
                    reasons.append(f"최근 상승세 (+{data['returns_1m']:.1f}%)")
                
                # 4. 변동성 점수 (20%) - 낮은 변동성이 좋음
                volatility_score = max(0, 100 - data['volatility'])
                volatility_weight = 0.2
                volatility_contribution = volatility_score * volatility_weight
                base_score += volatility_contribution
                if data['volatility'] < 30:
                    reasons.append(f"안정적 변동성 ({data['volatility']:.1f}%)")
                
                # ML 예측 (사용 가능한 경우)
                ml_bonus = 0
                if self.ml_engine:
                    try:
                        ml_features = np.array([[
                            data['trend_score'], data['momentum_score'],
                            data['returns_1w'], data['returns_1m'],
                            data['volatility']
                        ]])
                        
                        prediction = self.ml_engine.predict_price_direction(ml_features)
                        if prediction and hasattr(prediction, 'confidence'):
                            ml_bonus = prediction.confidence * 10  # 최대 10점 보너스
                            if ml_bonus > 5:
                                reasons.append(f"ML 긍정 신호 (+{ml_bonus:.1f}점)")
                    except Exception as e:
                        logger.debug(f"ML 예측 실패 ({symbol}): {e}")
                
                final_score = min(100, base_score + ml_bonus)
                
                analyzed_stocks.append({
                    'symbol': symbol,
                    'score': final_score,
                    'price': data['price'],
                    'returns_1m': data['returns_1m'],
                    'volatility': data['volatility'],
                    'trend_score': data['trend_score'],
                    'momentum_score': data['momentum_score'],
                    'reasons': reasons[:3]  # 상위 3개 이유만
                })
                
            except Exception as e:
                logger.warning(f"분석 실패 ({symbol}): {e}")
        
        # 점수순 정렬
        analyzed_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ {len(analyzed_stocks)}개 종목 분석 완료")
        return analyzed_stocks
    
    def generate_report(self, analyzed_stocks, top_n=5):
        """분석 보고서 생성"""
        
        print("\n" + "="*70)
        print("📈 ML 기반 투자 분석 보고서")
        print(f"⏰ 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # 시스템 상태
        system_status = self.system_monitor.get_system_status_report()
        print(f"\n🖥️ 시스템 상태:")
        print(f"  • CPU: {system_status['current_metrics']['cpu_usage_percent']:.1f}%")
        print(f"  • 메모리: {system_status['current_metrics']['memory_usage_percent']:.1f}%")
        print(f"  • ML 엔진: {'활성화' if self.ml_engine else '비활성화'}")
        print(f"  • 상태: {system_status['status']}")
        
        # Top 종목
        print(f"\n🏆 Top {top_n} 투자 추천 종목:")
        print("-"*70)
        
        for i, stock in enumerate(analyzed_stocks[:top_n], 1):
            print(f"\n{i}. {stock['symbol']}")
            print(f"   점수: {stock['score']:.1f}/100")
            print(f"   현재가: ${stock['price']:.2f}")
            print(f"   1개월 수익률: {stock['returns_1m']:+.1f}%")
            print(f"   변동성: {stock['volatility']:.1f}%")
            print(f"   선정 이유:")
            for reason in stock['reasons']:
                print(f"     • {reason}")
        
        # 통계 요약
        if analyzed_stocks:
            avg_score = sum(s['score'] for s in analyzed_stocks) / len(analyzed_stocks)
            high_score_count = sum(1 for s in analyzed_stocks if s['score'] > 70)
            
            print(f"\n📊 분석 통계:")
            print(f"  • 분석 종목 수: {len(analyzed_stocks)}개")
            print(f"  • 평균 점수: {avg_score:.1f}점")
            print(f"  • 고득점 종목 (70점 이상): {high_score_count}개")
            print(f"  • 최고 점수: {analyzed_stocks[0]['score']:.1f}점")
        
        # 면책 조항
        print(f"\n⚠️ 투자 주의사항:")
        print("  • 이 분석은 참고용이며 투자 조언이 아닙니다")
        print("  • 모든 투자에는 손실 위험이 있습니다")
        print("  • 투자 전 충분한 조사와 전문가 상담을 권장합니다")
        
        print("="*70)

def main():
    """메인 실행 함수"""
    
    print("🚀 ML 기반 투자 분석 프로그램 시작")
    print("💻 사용자 시스템 최적화 버전")
    
    # 분석기 초기화
    analyzer = LightweightInvestmentAnalyzer()
    
    # 샘플 종목 (테스트용)
    sample_symbols = [
        # 주요 미국 주식
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "NFLX", "ADBE", "CRM",
        
        # 주요 ETF
        "SPY", "QQQ", "VTI", "VOO", "IVV"
    ]
    
    print(f"📊 분석 대상: {len(sample_symbols)}개 종목")
    
    try:
        # 데이터 수집
        stock_data = analyzer.get_stock_data(sample_symbols)
        
        if not stock_data:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        # 분석 실행
        analyzed_stocks = analyzer.analyze_stocks(stock_data)
        
        # 보고서 생성
        analyzer.generate_report(analyzed_stocks)
        
        # 결과 저장
        result_file = f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            import json
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analyzed_stocks': analyzed_stocks,
                'system_status': analyzer.system_monitor.get_system_status_report()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {result_file}")
        
    except Exception as e:
        logger.error(f"❌ 분석 실패: {e}")
        print(f"❌ 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 