"""
시장 분석 모듈
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from data.fetcher import MarketPrism
from utils.logger import log_event

class MarketAnalyzer:
    """시장 분석 및 투자 신호 생성"""
    
    def __init__(self, market_prism: MarketPrism):
        self.market_prism = market_prism
        
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """시장 심리 분석"""
        try:
            # 상승/하락 종목 비율
            gainers = self.market_prism.get_price_ranking("rise", 100)
            losers = self.market_prism.get_price_ranking("fall", 100)
            
            strong_gainers = [s for s in gainers if s['change_rate'] >= 5.0]
            strong_losers = [s for s in losers if s['change_rate'] <= -5.0]
            
            # 외국인/기관 순매수 분석
            foreign_data = self.market_prism.get_foreign_institution_ranking("foreign", 50)
            institution_data = self.market_prism.get_foreign_institution_ranking("institution", 50)
            
            foreign_net_buy = sum(s['net_buy_amount'] for s in foreign_data if s['net_buy_amount'] > 0)
            foreign_net_sell = sum(abs(s['net_buy_amount']) for s in foreign_data if s['net_buy_amount'] < 0)
            
            institution_net_buy = sum(s['net_buy_amount'] for s in institution_data if s['net_buy_amount'] > 0)
            institution_net_sell = sum(abs(s['net_buy_amount']) for s in institution_data if s['net_buy_amount'] < 0)
            
            # 업종 분석
            sectors = self.market_prism.get_sector_ranking(30)
            rising_sectors = [s for s in sectors if s['change_rate'] >= 1.0]
            falling_sectors = [s for s in sectors if s['change_rate'] <= -1.0]
            
            # 종합 심리 점수 계산 (0-100)
            sentiment_score = self._calculate_sentiment_score(
                len(strong_gainers), len(strong_losers),
                foreign_net_buy, foreign_net_sell,
                institution_net_buy, institution_net_sell,
                len(rising_sectors), len(falling_sectors)
            )
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'sentiment_score': sentiment_score,
                'sentiment_level': self._get_sentiment_level(sentiment_score),
                'market_stats': {
                    'strong_gainers': len(strong_gainers),
                    'strong_losers': len(strong_losers),
                    'rising_sectors': len(rising_sectors),
                    'falling_sectors': len(falling_sectors)
                },
                'investor_flow': {
                    'foreign': {
                        'net_buy': foreign_net_buy,
                        'net_sell': foreign_net_sell,
                        'bias': 'buy' if foreign_net_buy > foreign_net_sell else 'sell'
                    },
                    'institution': {
                        'net_buy': institution_net_buy,
                        'net_sell': institution_net_sell,
                        'bias': 'buy' if institution_net_buy > institution_net_sell else 'sell'
                    }
                },
                'hot_sectors': rising_sectors[:5],
                'weak_sectors': falling_sectors[:5]
            }
            
            return analysis
            
        except Exception as e:
            log_event("ERROR", f"시장 심리 분석 실패: {e}")
            return {}
    
    def _calculate_sentiment_score(self, strong_gainers: int, strong_losers: int,
                                 foreign_buy: float, foreign_sell: float,
                                 inst_buy: float, inst_sell: float,
                                 rising_sectors: int, falling_sectors: int) -> float:
        """종합적인 시장 심리 점수 계산"""
        score = 50.0  # 중립부터 시작
        
        # 강한 상승/하락 종목 비율
        if strong_gainers + strong_losers > 0:
            gainer_ratio = strong_gainers / (strong_gainers + strong_losers)
            score += (gainer_ratio - 0.5) * 30  # ±15점
        
        # 외국인 순매수/순매도
        if foreign_buy + foreign_sell > 0:
            foreign_ratio = foreign_buy / (foreign_buy + foreign_sell)
            score += (foreign_ratio - 0.5) * 20  # ±10점
        
        # 기관 순매수/순매도
        if inst_buy + inst_sell > 0:
            inst_ratio = inst_buy / (inst_buy + inst_sell)
            score += (inst_ratio - 0.5) * 20  # ±10점
        
        # 업종별 강약
        if rising_sectors + falling_sectors > 0:
            sector_ratio = rising_sectors / (rising_sectors + falling_sectors)
            score += (sector_ratio - 0.5) * 10  # ±5점
        
        return max(0, min(100, score))
    
    def _get_sentiment_level(self, score: float) -> str:
        """심리 점수를 레벨로 변환"""
        if score >= 80:
            return "매우 강세"
        elif score >= 65:
            return "강세"
        elif score >= 50:
            return "보합 상승"
        elif score >= 35:
            return "보합 하락"
        elif score >= 20:
            return "약세"
        else:
            return "매우 약세"
    
    def find_momentum_stocks(self, min_change_rate: float = 3.0) -> List[Dict[str, Any]]:
        """모멘텀 종목 발굴"""
        candidates = []
        
        # 1. 상승률 상위 종목 중 필터링
        gainers = self.market_prism.get_price_ranking("rise", 50)
        for stock in gainers:
            if stock['change_rate'] >= min_change_rate and stock['volume'] >= 1000000:
                candidates.append({
                    **stock,
                    'signal_type': 'price_momentum',
                    'strength': stock['change_rate']
                })
        
        # 2. 외국인 순매수 + 상승 종목
        foreign_stocks = self.market_prism.get_foreign_institution_ranking("foreign", 30)
        for stock in foreign_stocks:
            if (stock['net_buy_amount'] > 0 and 
                stock['change_rate'] >= 1.0 and
                stock['symbol'] not in [c['symbol'] for c in candidates]):
                candidates.append({
                    **stock,
                    'signal_type': 'foreign_buying',
                    'strength': stock['net_buy_amount'] / 100000000  # 억원 단위
                })
        
        # 3. 기관 순매수 + 상승 종목
        inst_stocks = self.market_prism.get_foreign_institution_ranking("institution", 30)
        for stock in inst_stocks:
            if (stock['net_buy_amount'] > 0 and 
                stock['change_rate'] >= 1.0 and
                stock['symbol'] not in [c['symbol'] for c in candidates]):
                candidates.append({
                    **stock,
                    'signal_type': 'institution_buying',
                    'strength': stock['net_buy_amount'] / 100000000
                })
        
        # 강도 순으로 정렬
        candidates.sort(key=lambda x: x['strength'], reverse=True)
        
        log_event("INFO", f"모멘텀 종목 {len(candidates)}개 발굴")
        return candidates[:20]  # 상위 20개만 반환 