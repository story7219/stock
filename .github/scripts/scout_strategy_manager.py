#!/usr/bin/env python3
"""
🚀 척후병 전략 관리 시스템
투자 대가들의 전략을 활용한 자동 매매 시스템
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
import requests

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScoutStrategyManager:
    """척후병 전략 관리자"""
    
    def __init__(self):
        self.is_mock = os.getenv('IS_MOCK', 'True').lower() == 'true'
        self.kis_app_key = os.getenv('KIS_APP_KEY')
        self.kis_app_secret = os.getenv('KIS_APP_SECRET')
        self.kis_account_no = os.getenv('KIS_ACCOUNT_NO')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        logger.info(f"🚀 척후병 시스템 초기화 (Mock 모드: {self.is_mock})")
    
    def validate_environment(self) -> bool:
        """환경 변수 검증"""
        required_vars = ['KIS_APP_KEY', 'KIS_APP_SECRET', 'KIS_ACCOUNT_NO']
        
        if not self.is_mock:
            for var in required_vars:
                if not os.getenv(var):
                    logger.error(f"❌ 필수 환경 변수 누락: {var}")
                    return False
        
        logger.info("✅ 환경 변수 검증 완료")
        return True
    
    async def analyze_market_conditions(self) -> Dict[str, Any]:
        """시장 상황 분석"""
        logger.info("📊 시장 상황 분석 중...")
        
        # 모의 분석 데이터
        analysis = {
            'market_trend': 'bullish',
            'volatility': 'moderate',
            'volume': 'high',
            'sentiment': 'positive',
            'recommended_strategies': [
                'warren_buffett_value',
                'peter_lynch_growth',
                'benjamin_graham_defensive'
            ]
        }
        
        logger.info(f"📈 시장 분석 완료: {analysis['market_trend']} 트렌드")
        return analysis
    
    async def execute_scout_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """척후병 전략 실행"""
        logger.info(f"🎯 전략 실행: {strategy_name}")
        
        if self.is_mock:
            # 모의 거래 결과
            result = {
                'strategy': strategy_name,
                'status': 'success',
                'positions': [
                    {'symbol': 'AAPL', 'action': 'buy', 'quantity': 10, 'price': 150.0},
                    {'symbol': 'MSFT', 'action': 'buy', 'quantity': 5, 'price': 300.0}
                ],
                'total_value': 3000.0,
                'profit_loss': 150.0,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # 실제 거래 로직 (여기서는 기본 구조만)
            result = {
                'strategy': strategy_name,
                'status': 'executed',
                'message': '실제 거래 시스템 연동 필요',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"✅ 전략 실행 완료: {result['status']}")
        return result
    
    async def send_telegram_notification(self, message: str) -> bool:
        """텔레그램 알림 전송"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("⚠️ 텔레그램 설정 없음")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            logger.info("📱 텔레그램 알림 전송 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 텔레그램 알림 실패: {e}")
            return False
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """매매 사이클 실행"""
        logger.info("🔄 매매 사이클 시작")
        
        try:
            # 1. 시장 분석
            market_analysis = await self.analyze_market_conditions()
            
            # 2. 추천 전략 실행
            results = []
            for strategy in market_analysis['recommended_strategies']:
                result = await self.execute_scout_strategy(strategy)
                results.append(result)
            
            # 3. 결과 집계
            cycle_result = {
                'timestamp': datetime.now().isoformat(),
                'market_analysis': market_analysis,
                'strategy_results': results,
                'total_strategies': len(results),
                'successful_strategies': len([r for r in results if r['status'] == 'success']),
                'status': 'completed'
            }
            
            # 4. 알림 전송
            notification_message = f"""🚀 **척후병 매매 사이클 완료**

📊 **시장 분석**
- 트렌드: {market_analysis['market_trend']}
- 변동성: {market_analysis['volatility']}
- 거래량: {market_analysis['volume']}

🎯 **실행된 전략**: {len(results)}개
✅ **성공한 전략**: {cycle_result['successful_strategies']}개

⏰ **실행 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.send_telegram_notification(notification_message)
            
            logger.info("🎉 매매 사이클 완료")
            return cycle_result
            
        except Exception as e:
            logger.error(f"❌ 매매 사이클 오류: {e}")
            await self.send_telegram_notification(f"❌ 매매 사이클 오류: {str(e)}")
            raise
    
    def run(self):
        """메인 실행 함수"""
        logger.info("🚀 척후병 전략 관리 시스템 시작")
        
        # 환경 검증
        if not self.validate_environment():
            logger.error("❌ 환경 검증 실패")
            sys.exit(1)
        
        # 비동기 실행
        try:
            result = asyncio.run(self.run_trading_cycle())
            logger.info(f"✅ 시스템 실행 완료: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"❌ 시스템 실행 실패: {e}")
            sys.exit(1)

def main():
    """CLI 진입점"""
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        manager = ScoutStrategyManager()
        manager.run()
    else:
        print("사용법: python scout_strategy_manager.py run")

if __name__ == "__main__":
    main() 