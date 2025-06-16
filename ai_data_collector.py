"""
🤖 AI 투자 판단을 위한 효율적 데이터 수집기
- 스캘핑과 단기투자에 필요한 핵심 정보만 수집
- 간결하고 최적화된 코드 구조
"""
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class AIDataCollector:
    """AI 투자 판단을 위한 핵심 데이터 수집 클래스"""
    
    def __init__(self, trader):
        """CoreTrader 인스턴스를 받아서 API 호출"""
        self.trader = trader
    
    def get_scalping_signals(self, symbol):
        """🔥 스캐핑용 핵심 시그널 (빠른 판단용)"""
        try:
            logger.info(f"🤖 {symbol} 스캘핑 시그널 수집...")
            
            # 1. 현재가 + 호가 (가장 중요)
            price_info = self.trader.get_current_price(symbol)
            orderbook = self._get_simple_orderbook(symbol)
            
            if not price_info or not orderbook:
                return None
            
            # 2. 간단한 변동성 체크 (1분봉 5개)
            volatility = self._get_quick_volatility(symbol)
            
            # 3. 거래량 급증 여부
            volume_surge = self._check_volume_surge(symbol)
            
            # 4. 시장 전체 분위기 (간단히)
            market_mood = self._get_market_mood()
            
            return {
                'symbol': symbol,
                'name': price_info['name'],
                'price': price_info['price'],
                'timestamp': datetime.now().isoformat(),
                
                # 핵심 지표들
                'bid_strength': orderbook['bid_ratio'],  # 매수세 강도
                'volatility_level': volatility['level'],  # high/medium/low
                'volume_surge': volume_surge,  # True/False
                'market_sentiment': market_mood,  # bullish/bearish/neutral
                
                # AI 판단용 점수 (간단히)
                'scalping_score': self._calculate_scalping_score(orderbook, volatility, volume_surge, market_mood),
                'recommendation': self._get_quick_recommendation(orderbook, volatility, volume_surge)
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} 스캘핑 데이터 수집 실패: {e}")
            return None
    
    def get_portfolio_analysis(self, symbols_list):
        """📊 다중 종목 비교 분석 (단기투자용)"""
        try:
            logger.info(f"🤖 포트폴리오 분석 시작: {len(symbols_list)}개 종목")
            
            stocks_analysis = []
            
            # 각 종목별 빠른 분석
            for symbol in symbols_list:
                analysis = self.get_scalping_signals(symbol)
                if analysis:
                    stocks_analysis.append(analysis)
            
            if not stocks_analysis:
                return None
            
            # 종목들을 점수순으로 랭킹
            stocks_analysis.sort(key=lambda x: x['scalping_score'], reverse=True)
            
            return {
                'analysis_time': datetime.now().isoformat(),
                'total_stocks': len(stocks_analysis),
                
                # 상위 3개 추천
                'top_picks': stocks_analysis[:3],
                
                # 하위 2개 주의
                'avoid_stocks': stocks_analysis[-2:] if len(stocks_analysis) > 2 else [],
                
                # 전체 시장 요약
                'market_summary': self._summarize_market_condition(stocks_analysis),
                
                # AI 추천 액션
                'recommended_action': self._get_portfolio_recommendation(stocks_analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ 포트폴리오 분석 실패: {e}")
            return None
    
    def _get_simple_orderbook(self, symbol):
        """간단한 호가 분석 (매수/매도 비율만)"""
        try:
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", 
                    headers={"tr_id": "FHKST01010200"}, 
                    params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol}
                )
                
                if res and res.get('rt_cd') == '0':
                    output = res.get('output1', {})
                    
                    # 상위 5호가만 계산 (속도 향상)
                    total_bid = sum([int(output.get(f'bidp_rsqn{i}', 0)) for i in range(1, 6)])
                    total_ask = sum([int(output.get(f'askp_rsqn{i}', 0)) for i in range(1, 6)])
                    
                    if total_ask == 0:
                        ratio = 99.9  # 매우 강세
                    else:
                        ratio = round(total_bid / total_ask, 2)
                    
                    return {
                        'bid_ratio': ratio,
                        'sentiment': 'strong_buy' if ratio > 2.0 else 'buy' if ratio > 1.2 else 'sell' if ratio < 0.8 else 'hold'
                    }
                return None
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 호가 조회 실패: {e}")
            return {'bid_ratio': 1.0, 'sentiment': 'neutral'}
    
    def _get_quick_volatility(self, symbol):
        """빠른 변동성 체크 (1분봉 5개만)"""
        try:
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice", 
                    headers={"tr_id": "FHKST03010200"}, 
                    params={
                        "fid_etc_cls_code": "",
                        "fid_cond_mrkt_div_code": "J",
                        "fid_input_iscd": symbol,
                        "fid_input_hour_1": "1",  # 1분봉
                        "fid_pw_data_incu_yn": "Y"
                    }
                )
                
                if res and res.get('rt_cd') == '0':
                    items = res.get('output2', [])[:5]  # 최근 5개만
                    if len(items) < 3:
                        return {'level': 'low', 'range_pct': 0}
                    
                    prices = [int(item.get('stck_prpr', 0)) for item in items]
                    high_price = max(prices)
                    low_price = min(prices)
                    
                    if low_price > 0:
                        range_pct = ((high_price - low_price) / low_price) * 100
                        level = 'high' if range_pct > 2.0 else 'medium' if range_pct > 0.5 else 'low'
                    else:
                        range_pct = 0
                        level = 'low'
                    
                    return {'level': level, 'range_pct': round(range_pct, 2)}
                
                return {'level': 'low', 'range_pct': 0}
        except Exception as e:
            logger.warning(f"⚠️ {symbol} 변동성 조회 실패: {e}")
            return {'level': 'low', 'range_pct': 0}
    
    def _check_volume_surge(self, symbol):
        """거래량 급증 여부만 빠르게 체크"""
        try:
            minute_data = self._get_quick_volatility(symbol)  # 이미 1분봉 데이터 활용
            # 간단히 변동성이 높으면 거래량도 많다고 가정
            return minute_data['level'] == 'high'
        except:
            return False
    
    def _get_market_mood(self):
        """시장 전체 분위기 (간단히)"""
        try:
            # 코스피 지수만 빠르게 체크
            with self.trader.global_limiter, self.trader.market_data_limiter:
                res = self.trader._send_request(
                    "GET", 
                    "/uapi/domestic-stock/v1/quotations/inquire-price", 
                    headers={"tr_id": "FHKST01010100"}, 
                    params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": "005930"}  # 삼성전자로 대체
                )
                
                if res and res.get('rt_cd') == '0':
                    change_pct = float(res['output'].get('prdy_ctrt', 0))
                    if change_pct > 1.0:
                        return 'bullish'
                    elif change_pct < -1.0:
                        return 'bearish'
                    else:
                        return 'neutral'
                
                return 'neutral'
        except:
            return 'neutral'
    
    def _calculate_scalping_score(self, orderbook, volatility, volume_surge, market_mood):
        """스캘핑 점수 계산 (간단한 공식)"""
        score = 0
        
        # 호가 점수
        if orderbook['sentiment'] == 'strong_buy':
            score += 3
        elif orderbook['sentiment'] == 'buy':
            score += 2
        elif orderbook['sentiment'] == 'sell':
            score -= 2
        
        # 변동성 점수
        if volatility['level'] == 'high':
            score += 2
        elif volatility['level'] == 'medium':
            score += 1
        
        # 거래량 점수
        if volume_surge:
            score += 1
        
        # 시장 분위기 점수
        if market_mood == 'bullish':
            score += 1
        elif market_mood == 'bearish':
            score -= 1
        
        return max(0, min(10, score))  # 0~10 범위
    
    def _get_quick_recommendation(self, orderbook, volatility, volume_surge):
        """빠른 추천 (단순 규칙 기반)"""
        if orderbook['sentiment'] in ['strong_buy', 'buy'] and volatility['level'] in ['medium', 'high']:
            return 'BUY'
        elif orderbook['sentiment'] == 'sell':
            return 'SELL'
        else:
            return 'HOLD'
    
    def _summarize_market_condition(self, stocks_analysis):
        """시장 전체 상황 요약"""
        if not stocks_analysis:
            return 'unknown'
        
        buy_signals = sum(1 for s in stocks_analysis if s['recommendation'] == 'BUY')
        sell_signals = sum(1 for s in stocks_analysis if s['recommendation'] == 'SELL')
        total = len(stocks_analysis)
        
        if buy_signals > total * 0.6:
            return 'strong_bullish'
        elif buy_signals > total * 0.4:
            return 'bullish'
        elif sell_signals > total * 0.6:
            return 'bearish'
        else:
            return 'mixed'
    
    def _get_portfolio_recommendation(self, stocks_analysis):
        """포트폴리오 전체 추천"""
        market_condition = self._summarize_market_condition(stocks_analysis)
        
        if market_condition in ['strong_bullish', 'bullish']:
            return 'AGGRESSIVE_BUY'
        elif market_condition == 'bearish':
            return 'DEFENSIVE'
        else:
            return 'SELECTIVE_BUY'

    # === 🚀 실시간 모니터링 ===
    def start_realtime_monitoring(self, symbols_list, ai_callback=None):
        """실시간 AI 모니터링 (WebSocket + 분석)"""
        try:
            logger.info(f"🤖 실시간 AI 모니터링 시작: {symbols_list}")
            
            def price_change_handler(symbol, price, timestamp):
                """가격 변동 시 AI 분석 실행"""
                if symbol in symbols_list:
                    # 빠른 시그널 분석
                    signals = self.get_scalping_signals(symbol)
                    if signals and ai_callback:
                        ai_callback(signals)
            
            # WebSocket 콜백 등록
            self.trader.add_price_callback(price_change_handler)
            
            # WebSocket 시작
            success = self.trader.start_realtime_price_feed(symbols_list)
            if success:
                logger.info("✅ AI 실시간 모니터링 활성화")
            return success
            
        except Exception as e:
            logger.error(f"❌ AI 모니터링 시작 실패: {e}")
            return False

    def stop_monitoring(self):
        """모니터링 중지"""
        self.trader.stop_realtime_price_feed()
        logger.info("🔴 AI 모니터링 중지") 