"""
🏆 고급 스캘핑 시스템 (메인 인터페이스)
- 최적화된 스캘핑 모듈들의 통합 인터페이스
- 간편한 설정 및 실행
- 실시간 모니터링 및 제어
"""

import logging
import sys
from typing import List, Dict, Optional
from datetime import datetime

# 스캘핑 모듈들 import
from scalping_modules.optimized_scalping_system import OptimizedScalpingSystem
from scalping_modules.atr_analyzer import ATRAnalyzer
from scalping_modules.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from scalping_modules.momentum_scorer import MomentumScorer

logger = logging.getLogger(__name__)

class AdvancedScalpingSystem:
    """고급 스캘핑 시스템 메인 클래스"""
    
    def __init__(self, core_trader):
        """
        고급 스캘핑 시스템 초기화
        
        Args:
            core_trader: CoreTrader 인스턴스
        """
        self.trader = core_trader
        
        # 최적화된 스캘핑 시스템 초기화
        self.scalping_system = OptimizedScalpingSystem(
            core_trader=core_trader,
            daily_api_limit=5000
        )
        
        # 개별 분석기들 (독립 사용 가능)
        self.atr_analyzer = ATRAnalyzer()
        self.multi_analyzer = MultiTimeframeAnalyzer()
        self.momentum_scorer = MomentumScorer()
        
        # 기본 설정
        self.default_config = {
            'target_symbols': [
                '005930',  # 삼성전자
                '000660',  # SK하이닉스
                '035420',  # NAVER
                '051910',  # LG화학
                '006400',  # 삼성SDI
                '035720',  # 카카오
                '028260',  # 삼성물산
                '066570',  # LG전자
                '105560',  # KB금융
                '055550'   # 신한지주
            ],
            'max_concurrent_positions': 3,
            'risk_per_trade': 0.02,  # 2%
            'min_confidence': 70,
            'analysis_interval': 60,  # 60초
        }
        
        logger.info("🏆 고급 스캘핑 시스템 초기화 완료")
    
    def start_scalping(self, 
                      symbols: Optional[List[str]] = None,
                      max_positions: int = 3,
                      risk_percent: float = 0.02) -> None:
        """
        스캘핑 시작 (간편 인터페이스)
        
        Args:
            symbols: 대상 종목 리스트 (기본값: 인기 종목들)
            max_positions: 최대 동시 포지션 수
            risk_percent: 거래당 리스크 비율
        """
        try:
            # 기본 설정 사용
            target_symbols = symbols or self.default_config['target_symbols']
            
            logger.info("🚀 고급 스캘핑 시스템 시작")
            logger.info(f"📊 대상 종목: {len(target_symbols)}개")
            logger.info(f"💼 최대 포지션: {max_positions}개")
            logger.info(f"⚖️ 리스크: {risk_percent:.1%}")
            
            # 시스템 상태 확인
            if not self._check_system_ready():
                logger.error("❌ 시스템 준비 미완료")
                return
            
            # 스캘핑 시작
            self.scalping_system.start_scalping(
                target_symbols=target_symbols,
                max_concurrent_positions=max_positions,
                risk_per_trade=risk_percent
            )
            
        except Exception as e:
            logger.error(f"❌ 스캘핑 시작 실패: {e}")
    
    def stop_scalping(self) -> None:
        """스캘핑 중지"""
        try:
            logger.info("🛑 스캘핑 시스템 중지 요청")
            self.scalping_system.stop_scalping()
            logger.info("✅ 스캘핑 시스템 중지 완료")
            
        except Exception as e:
            logger.error(f"❌ 스캘핑 중지 실패: {e}")
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        try:
            # 기본 상태 정보
            status = self.scalping_system.get_system_status()
            
            # 추가 정보 병합
            status.update({
                'system_type': '고급 스캘핑 시스템',
                'version': '1.0.0',
                'trader_mode': '모의투자' if self.trader.is_mock else '실전투자',
                'default_symbols': self.default_config['target_symbols']
            })
            
            return status
            
        except Exception as e:
            logger.error(f"❌ 시스템 상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def _check_system_ready(self) -> bool:
        """시스템 준비 상태 확인"""
        try:
            # 1. 인증 상태 확인
            if not self.trader.access_token:
                logger.error("❌ API 인증 토큰 없음")
                return False
            
            # 2. 계좌 잔고 확인
            balance = self.trader.get_balance()
            if not balance:
                logger.error("❌ 계좌 정보 조회 실패")
                return False
            
            if balance.cash < 100000:  # 최소 10만원
                logger.warning(f"⚠️ 계좌 잔고 부족: {balance.cash:,}원")
            
            # 3. API 한도 확인
            remaining = self.trader.daily_counter.get_remaining_calls()
            if isinstance(remaining, int) and remaining < 100:
                logger.warning(f"⚠️ API 호출 한도 부족: {remaining}회")
            
            logger.info("✅ 시스템 준비 상태 확인 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 준비 상태 확인 실패: {e}")
            return False
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        개별 종목 분석 (테스트용)
        
        Args:
            symbol: 종목 코드
            
        Returns:
            분석 결과
        """
        try:
            logger.info(f"📊 {symbol} 개별 분석 시작")
            
            # 현재가 조회
            price_data = self.trader.get_current_price(symbol)
            if not price_data:
                return {'error': '가격 데이터 조회 실패'}
            
            current_price = price_data['price']
            
            # 각 분석기별 분석 (간단 버전)
            analysis_result = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'analysis': {}
            }
            
            # ATR 분석 (모의 데이터로 테스트)
            test_prices = [current_price * (1 + i * 0.001) for i in range(-10, 1)]
            atr_result = self.atr_analyzer.calculate_quick_atr(test_prices)
            analysis_result['analysis']['atr'] = {
                'volatility': atr_result.get('atr_percentage', 0),
                'suitability': atr_result.get('scalping_suitability', 50)
            }
            
            # 모멘텀 분석 (간단 버전)
            momentum_result = self.momentum_scorer.calculate_batch_momentum(
                symbol, test_prices, [1000] * len(test_prices)
            )
            if momentum_result:
                analysis_result['analysis']['momentum'] = {
                    'score': momentum_result.combined_score,
                    'direction': momentum_result.momentum_direction,
                    'strength': momentum_result.momentum_strength
                }
            
            logger.info(f"✅ {symbol} 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ {symbol} 분석 실패: {e}")
            return {'error': str(e)}
    
    def get_top_opportunities(self, count: int = 5) -> List[Dict]:
        """
        상위 기회 종목 조회
        
        Args:
            count: 조회할 종목 수
            
        Returns:
            상위 기회 종목 리스트
        """
        try:
            logger.info(f"🔍 상위 {count}개 기회 종목 조회")
            
            # 랭킹 기반 종목 조회
            top_stocks = self.trader.get_top_ranking_stocks(top_n=count * 2)
            if not top_stocks:
                return []
            
            opportunities = []
            for stock in top_stocks[:count]:
                try:
                    symbol = stock['symbol']
                    
                    # 기본 정보
                    opportunity = {
                        'symbol': symbol,
                        'name': stock.get('name', symbol),
                        'price': stock.get('price', 0),
                        'change_rate': stock.get('change_rate', 0),
                        'volume_rate': stock.get('volume_rate', 0),
                        'rank': len(opportunities) + 1
                    }
                    
                    # 간단한 점수 계산
                    momentum_score = abs(stock.get('change_rate', 0)) * 2
                    volume_score = min(stock.get('volume_rate', 0) / 100, 5) * 20
                    opportunity_score = (momentum_score + volume_score) / 2
                    
                    opportunity['opportunity_score'] = round(opportunity_score, 1)
                    opportunities.append(opportunity)
                    
                except Exception as e:
                    logger.debug(f"⚠️ {stock.get('symbol', 'Unknown')} 처리 실패: {e}")
                    continue
            
            # 기회 점수로 재정렬
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            logger.info(f"✅ {len(opportunities)}개 기회 종목 발견")
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ 기회 종목 조회 실패: {e}")
            return []
    
    def run_quick_scan(self) -> Dict:
        """
        빠른 시장 스캔
        
        Returns:
            시장 스캔 결과
        """
        try:
            logger.info("⚡ 빠른 시장 스캔 시작")
            
            scan_start = datetime.now()
            
            # 1. 상위 기회 종목 조회
            opportunities = self.get_top_opportunities(10)
            
            # 2. 시장 상태 분석
            if opportunities:
                avg_change = sum(op['change_rate'] for op in opportunities) / len(opportunities)
                avg_volume = sum(op['volume_rate'] for op in opportunities) / len(opportunities)
                
                market_mood = '강세' if avg_change > 1 else '약세' if avg_change < -1 else '보합'
                activity_level = '활발' if avg_volume > 150 else '보통' if avg_volume > 100 else '저조'
            else:
                avg_change = 0
                avg_volume = 100
                market_mood = '보합'
                activity_level = '보통'
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            
            scan_result = {
                'scan_time': scan_start.isoformat(),
                'duration_seconds': round(scan_duration, 1),
                'market_analysis': {
                    'mood': market_mood,
                    'activity_level': activity_level,
                    'avg_change_rate': round(avg_change, 2),
                    'avg_volume_rate': round(avg_volume, 1)
                },
                'top_opportunities': opportunities[:5],
                'total_scanned': len(opportunities),
                'recommendations': self._generate_recommendations(opportunities)
            }
            
            logger.info(f"✅ 시장 스캔 완료 ({scan_duration:.1f}초)")
            return scan_result
            
        except Exception as e:
            logger.error(f"❌ 시장 스캔 실패: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """추천 사항 생성"""
        recommendations = []
        
        if not opportunities:
            recommendations.append("📊 현재 뚜렷한 기회가 보이지 않습니다")
            return recommendations
        
        top_opportunity = opportunities[0]
        
        # 상위 종목 기반 추천
        if top_opportunity['opportunity_score'] > 8:
            recommendations.append(f"🎯 {top_opportunity['name']} 주목 (기회점수: {top_opportunity['opportunity_score']})")
        
        # 시장 상황 기반 추천
        strong_movers = [op for op in opportunities if abs(op['change_rate']) > 3]
        if strong_movers:
            recommendations.append(f"🔥 강한 움직임: {len(strong_movers)}개 종목 (변동률 3% 이상)")
        
        # 거래량 급증 종목
        high_volume = [op for op in opportunities if op['volume_rate'] > 200]
        if high_volume:
            recommendations.append(f"📈 거래량 급증: {len(high_volume)}개 종목 (2배 이상)")
        
        # 기본 추천사항
        if not recommendations:
            recommendations.append("📖 관망 추천 - 명확한 신호 대기")
        
        return recommendations


def main():
    """메인 실행 함수 (테스트용)"""
    print("🏆 고급 스캘핑 시스템 테스트")
    print("이 파일은 라이브러리로 사용하도록 설계되었습니다.")
    print("core_trader.py와 함께 사용하세요.")
    
    # 사용 예시 출력
    example_code = '''
사용 예시:

from core_trader import CoreTrader
from advanced_scalping_system import AdvancedScalpingSystem

# 1. CoreTrader 초기화
trader = CoreTrader()

# 2. 고급 스캘핑 시스템 초기화
scalping = AdvancedScalpingSystem(trader)

# 3. 빠른 시장 스캔
scan_result = scalping.run_quick_scan()
print(scan_result)

# 4. 스캘핑 시작 (기본 설정)
scalping.start_scalping()

# 5. 스캘핑 중지
scalping.stop_scalping()
'''
    
    print(example_code)


if __name__ == "__main__":
    main() 