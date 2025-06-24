 #!/usr/bin/env python3
"""
🚀 초간단 ML 투자 분석기
RAM 16GB, i5-4460 환경 최적화 버전
"""

import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import time
    print("✅ 필수 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ 라이브러리 누락: {e}")
    print("🔧 설치 명령:")
    print("pip install yfinance pandas numpy")
    sys.exit(1)

try:
    import psutil
    SYSTEM_MONITOR = True
    print("✅ 시스템 모니터링 가능")
except ImportError:
    SYSTEM_MONITOR = False
    print("⚠️ psutil 없음 - 기본 모드로 실행")

def get_system_info():
    """시스템 정보 확인"""
    if not SYSTEM_MONITOR:
        return "시스템 모니터링 불가"
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return f"CPU: {cpu_percent:.1f}%, 메모리: {memory.percent:.1f}% (여유: {memory.available/1024**3:.1f}GB)"
    except:
        return "시스템 정보 수집 실패"

def simple_technical_analysis(data):
    """간단한 기술적 분석"""
    if len(data) < 20:
        return 50  # 기본 점수
    
    # 이동평균
    ma5 = data['Close'].rolling(5).mean().iloc[-1]
    ma20 = data['Close'].rolling(20).mean().iloc[-1]
    current_price = data['Close'].iloc[-1]
    
    # 추세 점수
    trend_score = 0
    if current_price > ma5 > ma20:
        trend_score = 90
    elif current_price > ma5:
        trend_score = 70
    elif current_price > ma20:
        trend_score = 50
    else:
        trend_score = 30
    
    # 볼륨 확인
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    recent_volume = data['Volume'].iloc[-1]
    volume_score = 60 if recent_volume > avg_volume else 40
    
    # 변동성
    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
    volatility_score = max(0, 100 - volatility)
    
    # 종합 점수
    final_score = (trend_score * 0.4 + volume_score * 0.3 + volatility_score * 0.3)
    
    return {
        'trend_score': trend_score,
        'volume_score': volume_score,
        'volatility': volatility,
        'final_score': final_score,
        'price': current_price,
        'ma5': ma5,
        'ma20': ma20
    }

def analyze_stock(symbol):
    """개별 종목 분석"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo")
        
        if len(data) == 0:
            return None
        
        analysis = simple_technical_analysis(data)
        
        # 수익률 계산
        returns_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0
        returns_1m = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100 if len(data) >= 20 else 0
        
        analysis.update({
            'symbol': symbol,
            'returns_1w': returns_1w,
            'returns_1m': returns_1m
        })
        
        return analysis
        
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return None

def main():
    """메인 실행"""
    print("🚀 초간단 ML 투자 분석기 시작")
    print("="*50)
    
    # 시스템 상태
    print(f"🖥️ 시스템: {get_system_info()}")
    
    # 분석 대상 종목 (미국 주요 주식)
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "SPY", "QQQ", "VTI"
    ]
    
    print(f"\n📊 {len(symbols)}개 종목 분석 시작...")
    
    results = []
    batch_size = 3  # 메모리 절약
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"\n배치 {i//batch_size + 1}: {batch}")
        
        for symbol in batch:
            print(f"  📈 {symbol} 분석 중...", end=" ")
            result = analyze_stock(symbol)
            
            if result:
                results.append(result)
                print(f"✅ 점수: {result['final_score']:.1f}")
            else:
                print("❌ 실패")
        
        # 메모리 회복을 위한 짧은 대기
        if SYSTEM_MONITOR:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                print("    ⏳ 메모리 회복 대기...")
                time.sleep(3)
    
    # 결과 정렬
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # 보고서 출력
    print("\n" + "="*60)
    print("📈 투자 분석 결과 (Top 5)")
    print("="*60)
    
    for i, stock in enumerate(results[:5], 1):
        print(f"\n{i}. {stock['symbol']}")
        print(f"   📊 종합점수: {stock['final_score']:.1f}/100")
        print(f"   💰 현재가: ${stock['price']:.2f}")
        print(f"   📈 1주일: {stock['returns_1w']:+.1f}%")
        print(f"   📅 1개월: {stock['returns_1m']:+.1f}%")
        print(f"   🎯 추세점수: {stock['trend_score']:.0f}")
        print(f"   📊 변동성: {stock['volatility']:.1f}%")
        
        # 간단한 추천 이유
        reasons = []
        if stock['trend_score'] > 70:
            reasons.append("강한 상승추세")
        if stock['returns_1m'] > 5:
            reasons.append("최근 좋은 성과")
        if stock['volatility'] < 30:
            reasons.append("안정적 변동성")
        
        if reasons:
            print(f"   💡 추천이유: {', '.join(reasons)}")
    
    # 통계
    if results:
        avg_score = sum(r['final_score'] for r in results) / len(results)
        best_score = results[0]['final_score']
        
        print(f"\n📊 분석 통계:")
        print(f"   • 총 종목수: {len(results)}개")
        print(f"   • 평균점수: {avg_score:.1f}점")
        print(f"   • 최고점수: {best_score:.1f}점")
        print(f"   • 70점 이상: {sum(1 for r in results if r['final_score'] > 70)}개")
    
    print(f"\n⚠️ 투자 유의사항:")
    print("   • 이 분석은 참고용이며 투자조언이 아닙니다")
    print("   • 모든 투자에는 손실 위험이 있습니다")
    print("   • 충분한 조사 후 신중하게 투자하세요")
    
    print("\n✅ 분석 완료!")
    print("="*60)

if __name__ == "__main__":
    main()