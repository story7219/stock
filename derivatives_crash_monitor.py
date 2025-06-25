#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 파생상품 폭락/폭등 전용 모니터링 시스템
=========================================
K200 옵션/선물, 나스닥100, S&P500 파생상품을 실시간으로 모니터링하여
폭락/폭등 신호를 감지하고 Gemini AI가 분석하는 전용 시스템
"""

import asyncio
import argparse
import logging
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/derivatives_monitor.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# 로그 디렉토리 생성
Path('logs').mkdir(exist_ok=True)

try:
    from src.modules.derivatives_monitor import get_derivatives_monitor
    from src.modules.notification_system import NotificationSystem
except ImportError as e:
    logger.error(f"모듈 import 오류: {e}")
    sys.exit(1)

class CrashMonitorApp:
    """파생상품 폭락/폭등 모니터링 앱"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.monitor = get_derivatives_monitor(self.gemini_api_key)
        
        # 알림 시스템 초기화
        self.notification_system = NotificationSystem({
            'telegram_enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
        })
        
        # 모니터링 설정
        self.alert_history = []
        self.is_running = False
        
        logger.info("🚨 파생상품 폭락/폭등 모니터링 앱 초기화 완료")
    
    async def run_continuous_monitoring(self, interval_minutes: int = 5, max_duration_hours: int = 24):
        """연속 모니터링 실행"""
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=max_duration_hours)
        
        logger.info(f"🔄 연속 모니터링 시작 - 간격: {interval_minutes}분, 최대 {max_duration_hours}시간")
        
        try:
            async with self.monitor as monitor:
                while self.is_running and datetime.now() < end_time:
                    try:
                        # 파생상품 데이터 수집
                        derivatives_data = await monitor.collect_all_derivatives()
                        
                        # 시장 신호 분석
                        signals = monitor.analyze_market_signals(derivatives_data)
                        
                        # 고위험 신호 필터링
                        high_risk_signals = [s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']]
                        crash_signals = [s for s in high_risk_signals if s.signal_type == "CRASH_WARNING"]
                        surge_signals = [s for s in high_risk_signals if s.signal_type == "SURGE_WARNING"]
                        
                        # 신호가 있으면 처리
                        if high_risk_signals:
                            await self._process_high_risk_signals(high_risk_signals, derivatives_data, monitor)
                        
                        # 상태 출력
                        await self._print_monitoring_status(derivatives_data, signals, crash_signals, surge_signals)
                        
                        # 대기
                        await asyncio.sleep(interval_minutes * 60)
                        
                    except Exception as e:
                        logger.error(f"모니터링 사이클 오류: {e}")
                        await asyncio.sleep(60)  # 1분 대기 후 재시도
                        
        except KeyboardInterrupt:
            logger.info("🛑 사용자에 의해 모니터링 중단")
        except Exception as e:
            logger.error(f"모니터링 시스템 오류: {e}")
        finally:
            self.is_running = False
            logger.info("✅ 모니터링 종료")
    
    async def _process_high_risk_signals(self, signals, derivatives_data, monitor):
        """고위험 신호 처리"""
        try:
            # Gemini AI 분석
            gemini_analysis = await monitor.get_gemini_analysis(signals, derivatives_data)
            
            # 알림 생성
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'signals_count': len(signals),
                'crash_warnings': len([s for s in signals if s.signal_type == "CRASH_WARNING"]),
                'surge_warnings': len([s for s in signals if s.signal_type == "SURGE_WARNING"]),
                'max_confidence': max(s.confidence for s in signals),
                'gemini_analysis': gemini_analysis,
                'signals': [s.__dict__ for s in signals]
            }
            
            # 히스토리에 추가
            self.alert_history.append(alert_data)
            
            # 중복 알림 방지 (최근 30분 내 유사한 알림 체크)
            if self._should_send_alert(alert_data):
                await self._send_alert_notification(alert_data)
                
                # JSON 파일로 저장
                await self._save_alert_to_file(alert_data)
            
        except Exception as e:
            logger.error(f"고위험 신호 처리 오류: {e}")
    
    def _should_send_alert(self, new_alert):
        """알림 전송 여부 판단 (중복 방지)"""
        if not self.alert_history:
            return True
        
        # 최근 30분 내 알림 체크
        recent_time = datetime.now() - timedelta(minutes=30)
        recent_alerts = [
            alert for alert in self.alert_history[-10:]  # 최근 10개만 체크
            if datetime.fromisoformat(alert['timestamp']) > recent_time
        ]
        
        # 유사한 알림이 있으면 스킵
        for alert in recent_alerts:
            if (abs(alert['max_confidence'] - new_alert['max_confidence']) < 10 and
                alert['crash_warnings'] == new_alert['crash_warnings'] and
                alert['surge_warnings'] == new_alert['surge_warnings']):
                return False
        
        return True
    
    async def _send_alert_notification(self, alert_data):
        """알림 전송"""
        try:
            crash_count = alert_data['crash_warnings']
            surge_count = alert_data['surge_warnings']
            max_confidence = alert_data['max_confidence']
            
            # 알림 타입 결정
            if crash_count > surge_count:
                alert_type = "🔴 폭락 경고"
                emoji = "🚨"
            elif surge_count > crash_count:
                alert_type = "🟢 폭등 신호"
                emoji = "🚀"
            else:
                alert_type = "⚠️ 혼합 신호"
                emoji = "📊"
            
            message = f"""
{emoji} **{alert_type}** {emoji}
📅 감지 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 **신호 요약**:
- 폭락 경고: {crash_count}개
- 폭등 신호: {surge_count}개
- 최대 신뢰도: {max_confidence:.1f}%
- 총 신호: {alert_data['signals_count']}개

🤖 **AI 분석 요약**:
{alert_data['gemini_analysis'][:400]}...

⚡ **권고사항**: 즉시 포트폴리오 점검 필요!
"""
            
            # 텔레그램 전송
            if self.notification_system.config.get('telegram_enabled'):
                await self.notification_system.send_telegram_message(message)
            
            # 콘솔 출력
            print("\n" + "="*80)
            print(f"{emoji} {alert_type} {emoji}")
            print("="*80)
            print(message)
            print("="*80)
            
            logger.critical(f"🚨 {alert_type} 알림 전송 완료")
            
        except Exception as e:
            logger.error(f"알림 전송 오류: {e}")
    
    async def _save_alert_to_file(self, alert_data):
        """알림 데이터를 파일로 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alerts/crash_alert_{timestamp}.json"
            
            # alerts 디렉토리 생성
            Path('alerts').mkdir(exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(alert_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 알림 데이터 저장: {filename}")
            
        except Exception as e:
            logger.error(f"알림 데이터 저장 오류: {e}")
    
    async def _print_monitoring_status(self, derivatives_data, signals, crash_signals, surge_signals):
        """모니터링 상태 출력"""
        total_derivatives = sum(len(d) for d in derivatives_data.values())
        high_risk_count = len([s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']])
        
        status_line = f"📊 {datetime.now().strftime('%H:%M:%S')} | "
        status_line += f"파생상품: {total_derivatives}개 | "
        status_line += f"신호: {len(signals)}개 | "
        status_line += f"고위험: {high_risk_count}개 | "
        status_line += f"폭락: {len(crash_signals)}개 | "
        status_line += f"폭등: {len(surge_signals)}개"
        
        print(status_line)
        
        # 고위험 신호가 있으면 상세 출력
        if high_risk_count > 0:
            print(f"⚠️  고위험 신호 감지: {high_risk_count}개")
            for signal in (crash_signals + surge_signals)[:3]:  # 상위 3개만
                print(f"   - {signal.underlying_asset}: {signal.signal_type} ({signal.confidence:.1f}%)")
    
    async def run_single_scan(self):
        """한 번만 스캔 실행"""
        logger.info("🔍 파생상품 단일 스캔 실행")
        
        try:
            async with self.monitor as monitor:
                # 데이터 수집 및 분석
                derivatives_data = await monitor.collect_all_derivatives()
                signals = monitor.analyze_market_signals(derivatives_data)
                
                # 결과 출력
                print("\n" + "="*60)
                print("📊 파생상품 스캔 결과")
                print("="*60)
                
                total_derivatives = sum(len(d) for d in derivatives_data.values())
                print(f"📈 총 파생상품: {total_derivatives}개")
                print(f"🚨 감지된 신호: {len(signals)}개")
                
                # 시장별 현황
                for market, derivatives in derivatives_data.items():
                    market_signals = [s for s in signals if s.underlying_asset == market]
                    print(f"🌍 {market}: {len(derivatives)}개 파생상품, {len(market_signals)}개 신호")
                
                # 고위험 신호 상세 출력
                high_risk_signals = [s for s in signals if s.risk_level in ['HIGH', 'CRITICAL']]
                if high_risk_signals:
                    print(f"\n⚠️  고위험 신호 {len(high_risk_signals)}개 발견:")
                    for i, signal in enumerate(high_risk_signals, 1):
                        print(f"{i}. {signal.underlying_asset} - {signal.signal_type}")
                        print(f"   신뢰도: {signal.confidence:.1f}% | 위험도: {signal.risk_level}")
                        print(f"   요인: {', '.join(signal.trigger_factors)}")
                    
                    # Gemini 분석
                    gemini_analysis = await monitor.get_gemini_analysis(high_risk_signals, derivatives_data)
                    print(f"\n🤖 Gemini AI 분석:")
                    print("-" * 40)
                    print(gemini_analysis)
                
                else:
                    print("\n✅ 현재 고위험 신호 없음 - 정상 상태")
                
                print("\n" + "="*60)
                
        except Exception as e:
            logger.error(f"단일 스캔 오류: {e}")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_running = False
        logger.info("🛑 모니터링 중지 신호 전송")

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='파생상품 폭락/폭등 모니터링 시스템')
    parser.add_argument('--mode', choices=['continuous', 'single'], default='single',
                       help='실행 모드: continuous(연속 모니터링) 또는 single(단일 스캔)')
    parser.add_argument('--interval', type=int, default=5,
                       help='모니터링 간격 (분, 기본값: 5)')
    parser.add_argument('--duration', type=int, default=24,
                       help='최대 모니터링 시간 (시간, 기본값: 24)')
    
    args = parser.parse_args()
    
    print("🚨 파생상품 폭락/폭등 모니터링 시스템 v1.0")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 실행 모드: {args.mode}")
    if args.mode == 'continuous':
        print(f"⏱️  모니터링 간격: {args.interval}분")
        print(f"⏰ 최대 실행 시간: {args.duration}시간")
    print("=" * 60)
    
    # 앱 초기화
    app = CrashMonitorApp()
    
    try:
        if args.mode == 'continuous':
            # 연속 모니터링
            await app.run_continuous_monitoring(args.interval, args.duration)
        else:
            # 단일 스캔
            await app.run_single_scan()
            
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
        app.stop_monitoring()
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 