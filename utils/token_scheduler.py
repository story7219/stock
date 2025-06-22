#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 한국투자증권 API 토큰 자동 발급 스케줄러
매일 아침 7시에 토큰을 자동으로 발급하고 갱신하는 시스템
"""

import asyncio
import json
import logging
import os
import schedule
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import requests
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

class KISTokenScheduler:
    """한국투자증권 API 토큰 자동 발급 및 관리 클래스"""
    
    def __init__(self):
        """토큰 스케줄러 초기화"""
        self.app_key = os.getenv('KIS_APP_KEY', '')
        self.app_secret = os.getenv('KIS_APP_SECRET', '')
        self.base_url = os.getenv('KIS_BASE_URL', 'https://openapi.koreainvestment.com:9443')
        
        # 토큰 저장 경로
        self.token_file = Path('kis_token.json')
        
        # 현재 토큰 정보
        self.current_token = None
        self.token_expires_at = None
        
        # 스케줄러 상태
        self.scheduler_running = False
        self.scheduler_thread = None
        
        logger.info("🔑 KIS 토큰 스케줄러 초기화 완료")
    
    def get_new_token(self) -> Optional[Dict[str, Any]]:
        """새로운 토큰 발급"""
        try:
            logger.info("🔑 새로운 KIS 토큰 발급 시작...")
            
            if not self.app_key or not self.app_secret:
                logger.error("❌ KIS API 키가 설정되지 않았습니다")
                return None
            
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {
                'Content-Type': 'application/json; charset=utf-8'
            }
            data = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                
                # 토큰 만료 시간 계산 (24시간 후)
                expires_at = datetime.now() + timedelta(hours=24)
                
                # 토큰 정보 저장
                token_info = {
                    'access_token': token_data.get('access_token'),
                    'token_type': token_data.get('token_type', 'Bearer'),
                    'expires_in': token_data.get('expires_in', 86400),  # 24시간
                    'expires_at': expires_at.isoformat(),
                    'issued_at': datetime.now().isoformat()
                }
                
                # 파일로 저장
                self.save_token(token_info)
                
                # 메모리에 저장
                self.current_token = token_info['access_token']
                self.token_expires_at = expires_at
                
                logger.info(f"✅ KIS 토큰 발급 성공! 만료 시간: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                return token_info
                
            else:
                logger.error(f"❌ 토큰 발급 실패: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 토큰 발급 중 오류 발생: {e}")
            return None
    
    def save_token(self, token_info: Dict[str, Any]) -> None:
        """토큰 정보를 파일에 저장"""
        try:
            with open(self.token_file, 'w', encoding='utf-8') as f:
                json.dump(token_info, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 토큰 정보 저장 완료: {self.token_file}")
        except Exception as e:
            logger.error(f"❌ 토큰 저장 실패: {e}")
    
    def load_token(self) -> Optional[Dict[str, Any]]:
        """저장된 토큰 정보 로드"""
        try:
            if not self.token_file.exists():
                logger.info("📄 저장된 토큰 파일이 없습니다")
                return None
            
            with open(self.token_file, 'r', encoding='utf-8') as f:
                token_info = json.load(f)
            
            # 만료 시간 확인
            expires_at = datetime.fromisoformat(token_info['expires_at'])
            if datetime.now() >= expires_at:
                logger.info("⏰ 저장된 토큰이 만료되었습니다")
                return None
            
            # 메모리에 저장
            self.current_token = token_info['access_token']
            self.token_expires_at = expires_at
            
            logger.info(f"✅ 토큰 로드 성공! 만료 시간: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
            return token_info
            
        except Exception as e:
            logger.error(f"❌ 토큰 로드 실패: {e}")
            return None
    
    def is_token_valid(self) -> bool:
        """토큰 유효성 확인"""
        if not self.current_token or not self.token_expires_at:
            return False
        
        # 만료 1시간 전에 갱신하도록 설정
        return datetime.now() < (self.token_expires_at - timedelta(hours=1))
    
    def refresh_token_if_needed(self) -> bool:
        """필요시 토큰 갱신"""
        if self.is_token_valid():
            logger.info("✅ 현재 토큰이 유효합니다")
            return True
        
        logger.info("🔄 토큰 갱신이 필요합니다")
        token_info = self.get_new_token()
        return token_info is not None
    
    def daily_token_refresh(self) -> None:
        """매일 토큰 갱신 작업"""
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"🌅 매일 토큰 갱신 작업 시작 - {current_time}")
            
            # 새 토큰 발급
            token_info = self.get_new_token()
            
            if token_info:
                logger.info("✅ 매일 토큰 갱신 완료!")
                
                # 텔레그램 알림 (선택사항)
                self.send_telegram_notification(
                    f"🔑 KIS 토큰 자동 갱신 완료!\n"
                    f"⏰ 갱신 시간: {current_time}\n"
                    f"⏳ 만료 시간: {self.token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                logger.error("❌ 매일 토큰 갱신 실패!")
                self.send_telegram_notification(
                    f"⚠️ KIS 토큰 자동 갱신 실패!\n"
                    f"⏰ 시도 시간: {current_time}\n"
                    f"🔧 수동 확인이 필요합니다."
                )
                
        except Exception as e:
            logger.error(f"❌ 매일 토큰 갱신 중 오류: {e}")
    
    def send_telegram_notification(self, message: str) -> None:
        """텔레그램 알림 전송 (선택사항)"""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            requests.post(url, json=data, timeout=5)
            logger.info("📱 텔레그램 알림 전송 완료")
            
        except Exception as e:
            logger.warning(f"📱 텔레그램 알림 전송 실패: {e}")
    
    def setup_schedule(self) -> None:
        """매일 아침 7시 스케줄 설정"""
        try:
            # 매일 오전 7시에 토큰 갱신
            schedule.every().day.at("07:00").do(self.daily_token_refresh)
            
            # 추가 안전장치: 매 12시간마다 토큰 유효성 확인
            schedule.every(12).hours.do(self.refresh_token_if_needed)
            
            logger.info("⏰ 토큰 갱신 스케줄 설정 완료:")
            logger.info("   - 매일 오전 7시: 토큰 갱신")
            logger.info("   - 매 12시간: 토큰 유효성 확인")
            
        except Exception as e:
            logger.error(f"❌ 스케줄 설정 실패: {e}")
    
    def start_scheduler(self) -> None:
        """스케줄러 시작"""
        if self.scheduler_running:
            logger.warning("⚠️ 스케줄러가 이미 실행 중입니다")
            return
        
        def run_scheduler():
            """스케줄러 실행 함수"""
            self.scheduler_running = True
            logger.info("🚀 KIS 토큰 스케줄러 시작!")
            
            try:
                while self.scheduler_running:
                    schedule.run_pending()
                    time.sleep(60)  # 1분마다 확인
            except Exception as e:
                logger.error(f"❌ 스케줄러 실행 중 오류: {e}")
            finally:
                self.scheduler_running = False
                logger.info("🛑 KIS 토큰 스케줄러 종료")
        
        # 스케줄 설정
        self.setup_schedule()
        
        # 백그라운드 스레드로 실행
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ 토큰 스케줄러가 백그라운드에서 시작되었습니다")
    
    def stop_scheduler(self) -> None:
        """스케줄러 중지"""
        if not self.scheduler_running:
            logger.info("ℹ️ 스케줄러가 실행되고 있지 않습니다")
            return
        
        self.scheduler_running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("🛑 토큰 스케줄러가 중지되었습니다")
    
    def get_token_status(self) -> Dict[str, Any]:
        """현재 토큰 상태 반환"""
        status = {
            'has_token': self.current_token is not None,
            'is_valid': self.is_token_valid(),
            'expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
            'scheduler_running': self.scheduler_running
        }
        
        if self.token_expires_at:
            remaining_time = self.token_expires_at - datetime.now()
            status['remaining_hours'] = remaining_time.total_seconds() / 3600
        
        return status

def create_token_scheduler() -> KISTokenScheduler:
    """토큰 스케줄러 인스턴스 생성"""
    return KISTokenScheduler()

if __name__ == "__main__":
    # 테스트 실행
    scheduler = create_token_scheduler()
    
    # 기존 토큰 로드 시도
    scheduler.load_token()
    
    # 토큰 상태 확인
    status = scheduler.get_token_status()
    print(f"토큰 상태: {status}")
    
    # 필요시 토큰 갱신
    if not status['is_valid']:
        print("토큰 갱신 중...")
        scheduler.refresh_token_if_needed()
    
    # 스케줄러 시작
    scheduler.start_scheduler()
    
    try:
        print("스케줄러가 실행 중입니다. Ctrl+C로 중지하세요.")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n스케줄러를 중지합니다...")
        scheduler.stop_scheduler() 