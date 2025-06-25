#!/usr/bin/env python3
"""
🛡️ 고급 리스크 관리 시스템
- 손절 (Stop Loss)
- 트레일링 스탑 (Trailing Stop)
- 포지션 크기 관리 (Position Sizing)
- 일일 손실 제한 (Daily Loss Limit)
- 연속 손실 제한 (Consecutive Loss Limit)
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "매수"
    SELL = "매도"
    STOP_LOSS = "손절"
    TRAILING_STOP = "트레일링스탑"
    TAKE_PROFIT = "익절"

class PositionStatus(Enum):
    OPEN = "보유중"
    CLOSED = "청산"
    STOP_LOSS_HIT = "손절됨"
    TRAILING_STOP_HIT = "트레일링스탑됨"
    TAKE_PROFIT_HIT = "익절됨"

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    position_type: str  # "LONG" or "SHORT"
    
    # 리스크 관리 설정
    stop_loss_price: float = 0.0
    trailing_stop_percent: float = 0.0
    trailing_stop_price: float = 0.0
    take_profit_price: float = 0.0
    
    # 현재 상태
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    highest_price: float = 0.0  # 트레일링 스탑용
    lowest_price: float = 0.0   # 숏 포지션용
    
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    
    def update_current_price(self, price: float):
        """현재 가격 업데이트 및 PnL 계산"""
        self.current_price = price
        
        if self.position_type == "LONG":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_percent = (price - self.entry_price) / self.entry_price * 100
            
            # 최고가 업데이트 (트레일링 스탑용)
            if price > self.highest_price:
                self.highest_price = price
                # 트레일링 스탑 가격 업데이트
                if self.trailing_stop_percent > 0:
                    self.trailing_stop_price = price * (1 - self.trailing_stop_percent / 100)
        
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_percent = (self.entry_price - price) / self.entry_price * 100
            
            # 최저가 업데이트 (숏 트레일링 스탑용)
            if self.lowest_price == 0 or price < self.lowest_price:
                self.lowest_price = price
                # 트레일링 스탑 가격 업데이트
                if self.trailing_stop_percent > 0:
                    self.trailing_stop_price = price * (1 + self.trailing_stop_percent / 100)
    
    def should_stop_loss(self) -> bool:
        """손절 조건 확인"""
        if self.stop_loss_price == 0:
            return False
        
        if self.position_type == "LONG":
            return self.current_price <= self.stop_loss_price
        else:  # SHORT
            return self.current_price >= self.stop_loss_price
    
    def should_trailing_stop(self) -> bool:
        """트레일링 스탑 조건 확인"""
        if self.trailing_stop_price == 0:
            return False
        
        if self.position_type == "LONG":
            return self.current_price <= self.trailing_stop_price
        else:  # SHORT
            return self.current_price >= self.trailing_stop_price
    
    def should_take_profit(self) -> bool:
        """익절 조건 확인"""
        if self.take_profit_price == 0:
            return False
        
        if self.position_type == "LONG":
            return self.current_price >= self.take_profit_price
        else:  # SHORT
            return self.current_price <= self.take_profit_price

@dataclass
class RiskSettings:
    """리스크 관리 설정"""
    # 기본 리스크 설정
    max_position_size_percent: float = 10.0  # 계좌 대비 최대 포지션 크기 (%)
    default_stop_loss_percent: float = 3.0   # 기본 손절 비율 (%)
    default_trailing_stop_percent: float = 5.0  # 기본 트레일링 스탑 비율 (%)
    default_take_profit_percent: float = 10.0   # 기본 익절 비율 (%)
    
    # 일일 제한
    daily_loss_limit_percent: float = 5.0    # 일일 최대 손실 (%)
    daily_trade_limit: int = 20              # 일일 최대 거래 횟수
    
    # 연속 손실 제한
    max_consecutive_losses: int = 3          # 최대 연속 손실 횟수
    consecutive_loss_cooldown_minutes: int = 60  # 연속 손실 후 대기 시간 (분)
    
    # 포지션 관리
    max_open_positions: int = 5              # 최대 동시 보유 포지션
    correlation_limit: float = 0.7           # 상관관계 제한 (같은 섹터 중복 방지)

class AdvancedRiskManager:
    def __init__(self, initial_balance: float = 1000000, settings: Optional[RiskSettings] = None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.settings = settings or RiskSettings()
        
        # 포지션 관리
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # 일일 통계
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades_count': 0,
            'daily_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'consecutive_losses': 0,
            'last_loss_time': None
        }
        
        # 거래 히스토리
        self.trade_history: List[Dict] = []
        
        logger.info(f"🛡️ 리스크 관리 시스템 초기화 - 초기 잔고: {initial_balance:,}원")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_percent: float) -> int:
        """포지션 크기 계산 (리스크 기반)"""
        # 1% 리스크 기준으로 포지션 크기 계산
        risk_amount = self.current_balance * (self.settings.max_position_size_percent / 100)
        price_risk = entry_price * (stop_loss_percent / 100)
        
        if price_risk == 0:
            return 0
        
        position_size = int(risk_amount / price_risk)
        
        # 최대 포지션 크기 제한
        max_position_value = self.current_balance * (self.settings.max_position_size_percent / 100)
        max_quantity = int(max_position_value / entry_price)
        
        position_size = min(position_size, max_quantity)
        
        logger.info(f"📊 {symbol} 포지션 크기 계산: {position_size}주 (리스크: {risk_amount:,}원)")
        return position_size
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """새 포지션 개설 가능 여부 확인"""
        # 일일 거래 제한 확인
        if self.daily_stats['trades_count'] >= self.settings.daily_trade_limit:
            return False, f"일일 거래 제한 초과 ({self.settings.daily_trade_limit}회)"
        
        # 일일 손실 제한 확인
        daily_loss_percent = abs(self.daily_stats['daily_pnl']) / self.initial_balance * 100
        if self.daily_stats['daily_pnl'] < 0 and daily_loss_percent >= self.settings.daily_loss_limit_percent:
            return False, f"일일 손실 제한 초과 ({daily_loss_percent:.1f}%)"
        
        # 연속 손실 제한 확인
        if self.daily_stats['consecutive_losses'] >= self.settings.max_consecutive_losses:
            if self.daily_stats['last_loss_time']:
                cooldown_end = self.daily_stats['last_loss_time'] + timedelta(minutes=self.settings.consecutive_loss_cooldown_minutes)
                if datetime.now() < cooldown_end:
                    remaining_minutes = (cooldown_end - datetime.now()).total_seconds() / 60
                    return False, f"연속 손실 후 대기 중 ({remaining_minutes:.0f}분 남음)"
        
        # 최대 포지션 수 확인
        if len(self.open_positions) >= self.settings.max_open_positions:
            return False, f"최대 포지션 수 초과 ({self.settings.max_open_positions}개)"
        
        # 이미 같은 종목 보유 중인지 확인
        if symbol in self.open_positions:
            return False, f"{symbol} 이미 보유 중"
        
        return True, "거래 가능"
    
    async def open_position(self, symbol: str, entry_price: float, position_type: str = "LONG",
                          custom_stop_loss_percent: Optional[float] = None,
                          custom_trailing_stop_percent: Optional[float] = None,
                          custom_take_profit_percent: Optional[float] = None) -> Optional[Position]:
        """포지션 개설"""
        
        # 거래 가능 여부 확인
        can_trade, reason = self.can_open_position(symbol)
        if not can_trade:
            logger.warning(f"❌ {symbol} 포지션 개설 불가: {reason}")
            return None
        
        # 리스크 설정
        stop_loss_percent = custom_stop_loss_percent or self.settings.default_stop_loss_percent
        trailing_stop_percent = custom_trailing_stop_percent or self.settings.default_trailing_stop_percent
        take_profit_percent = custom_take_profit_percent or self.settings.default_take_profit_percent
        
        # 포지션 크기 계산
        quantity = self.calculate_position_size(symbol, entry_price, stop_loss_percent)
        if quantity == 0:
            logger.warning(f"❌ {symbol} 포지션 크기가 0입니다")
            return None
        
        # 포지션 생성
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            position_type=position_type
        )
        
        # 리스크 관리 가격 설정
        if position_type == "LONG":
            position.stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
            position.trailing_stop_percent = trailing_stop_percent
            position.trailing_stop_price = entry_price * (1 - trailing_stop_percent / 100)
            position.take_profit_price = entry_price * (1 + take_profit_percent / 100)
            position.highest_price = entry_price
        else:  # SHORT
            position.stop_loss_price = entry_price * (1 + stop_loss_percent / 100)
            position.trailing_stop_percent = trailing_stop_percent
            position.trailing_stop_price = entry_price * (1 + trailing_stop_percent / 100)
            position.take_profit_price = entry_price * (1 - take_profit_percent / 100)
            position.lowest_price = entry_price
        
        # 포지션 등록
        self.open_positions[symbol] = position
        self.daily_stats['trades_count'] += 1
        
        logger.info(f"✅ {symbol} {position_type} 포지션 개설")
        logger.info(f"   진입가: {entry_price:,}원, 수량: {quantity}주")
        logger.info(f"   손절가: {position.stop_loss_price:,}원 (-{stop_loss_percent}%)")
        logger.info(f"   트레일링: {position.trailing_stop_price:,}원 (-{trailing_stop_percent}%)")
        logger.info(f"   익절가: {position.take_profit_price:,}원 (+{take_profit_percent}%)")
        
        return position
    
    async def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """포지션 업데이트 및 리스크 관리 실행"""
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        position.update_current_price(current_price)
        
        # 손절 확인
        if position.should_stop_loss():
            await self.close_position(symbol, current_price, PositionStatus.STOP_LOSS_HIT)
            return "STOP_LOSS"
        
        # 트레일링 스탑 확인
        if position.should_trailing_stop():
            await self.close_position(symbol, current_price, PositionStatus.TRAILING_STOP_HIT)
            return "TRAILING_STOP"
        
        # 익절 확인
        if position.should_take_profit():
            await self.close_position(symbol, current_price, PositionStatus.TAKE_PROFIT_HIT)
            return "TAKE_PROFIT"
        
        return None
    
    async def close_position(self, symbol: str, exit_price: float, 
                           status: PositionStatus = PositionStatus.CLOSED) -> Optional[Position]:
        """포지션 청산"""
        if symbol not in self.open_positions:
            logger.warning(f"❌ {symbol} 포지션을 찾을 수 없습니다")
            return None
        
        position = self.open_positions[symbol]
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.status = status
        
        # 실현 손익 계산
        if position.position_type == "LONG":
            position.realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            position.realized_pnl = (position.entry_price - exit_price) * position.quantity
        
        # 잔고 업데이트
        self.current_balance += position.realized_pnl
        self.daily_stats['daily_pnl'] += position.realized_pnl
        
        # 승패 기록
        if position.realized_pnl > 0:
            self.daily_stats['wins'] += 1
            self.daily_stats['consecutive_losses'] = 0  # 연속 손실 리셋
        else:
            self.daily_stats['losses'] += 1
            self.daily_stats['consecutive_losses'] += 1
            self.daily_stats['last_loss_time'] = datetime.now()
        
        # 거래 히스토리 저장
        trade_record = {
            'symbol': symbol,
            'position_type': position.position_type,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': position.exit_time.isoformat(),
            'realized_pnl': position.realized_pnl,
            'pnl_percent': position.realized_pnl / (position.entry_price * position.quantity) * 100,
            'status': status.value,
            'holding_minutes': (position.exit_time - position.entry_time).total_seconds() / 60
        }
        self.trade_history.append(trade_record)
        
        # 포지션 이동
        self.closed_positions.append(position)
        del self.open_positions[symbol]
        
        logger.info(f"🔄 {symbol} 포지션 청산 ({status.value})")
        logger.info(f"   진입가: {position.entry_price:,}원 → 청산가: {exit_price:,}원")
        logger.info(f"   실현손익: {position.realized_pnl:+,.0f}원 ({position.realized_pnl/(position.entry_price * position.quantity)*100:+.1f}%)")
        logger.info(f"   현재잔고: {self.current_balance:,.0f}원")
        
        return position
    
    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.open_positions.values())
        total_position_value = sum(pos.current_price * pos.quantity for pos in self.open_positions.values())
        
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance * 100,
            'open_positions_count': len(self.open_positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_position_value': total_position_value,
            'daily_stats': self.daily_stats,
            'win_rate': self.daily_stats['wins'] / max(1, self.daily_stats['wins'] + self.daily_stats['losses']) * 100
        }
    
    def print_portfolio_status(self):
        """포트폴리오 현황 출력"""
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*70)
        print("🛡️ 리스크 관리 포트폴리오 현황")
        print("="*70)
        
        print(f"💰 현재 잔고: {summary['current_balance']:,.0f}원")
        print(f"📈 총 수익률: {summary['total_return']:+.2f}%")
        print(f"📊 보유 포지션: {summary['open_positions_count']}개")
        print(f"💎 평가손익: {summary['total_unrealized_pnl']:+,.0f}원")
        
        print(f"\n📅 일일 통계:")
        print(f"   거래 횟수: {self.daily_stats['trades_count']}회")
        print(f"   일일 손익: {self.daily_stats['daily_pnl']:+,.0f}원")
        print(f"   승률: {summary['win_rate']:.1f}% ({self.daily_stats['wins']}승 {self.daily_stats['losses']}패)")
        print(f"   연속 손실: {self.daily_stats['consecutive_losses']}회")
        
        if self.open_positions:
            print(f"\n📋 보유 포지션:")
            for symbol, pos in self.open_positions.items():
                print(f"   {symbol}: {pos.unrealized_pnl:+,.0f}원 ({pos.unrealized_pnl_percent:+.1f}%)")
                print(f"      진입: {pos.entry_price:,}원 → 현재: {pos.current_price:,}원")
                print(f"      손절: {pos.stop_loss_price:,}원 | 트레일링: {pos.trailing_stop_price:,}원")

# 테스트 시뮬레이션
async def test_risk_management():
    """리스크 관리 시스템 테스트"""
    print("🧪 리스크 관리 시스템 테스트 시작")
    
    # 리스크 관리자 초기화
    risk_manager = AdvancedRiskManager(initial_balance=10000000)  # 1천만원
    
    # 테스트 시나리오 1: 정상 거래
    print("\n📈 시나리오 1: 정상 거래")
    position1 = await risk_manager.open_position("005930", 73000, "LONG")  # 삼성전자
    
    # 가격 상승 시뮬레이션
    await risk_manager.update_position("005930", 75000)  # +2.7%
    await risk_manager.update_position("005930", 77000)  # +5.5%
    
    risk_manager.print_portfolio_status()
    
    # 익절 시뮬레이션
    result = await risk_manager.update_position("005930", 80300)  # +10% (익절)
    print(f"결과: {result}")
    
    # 테스트 시나리오 2: 손절 시뮬레이션
    print("\n📉 시나리오 2: 손절 시뮬레이션")
    position2 = await risk_manager.open_position("000660", 125000, "LONG")  # SK하이닉스
    
    # 가격 하락 시뮬레이션
    await risk_manager.update_position("000660", 122000)  # -2.4%
    result = await risk_manager.update_position("000660", 121250)  # -3% (손절)
    print(f"결과: {result}")
    
    # 테스트 시나리오 3: 트레일링 스탑 시뮬레이션
    print("\n🔄 시나리오 3: 트레일링 스탑 시뮬레이션")
    position3 = await risk_manager.open_position("035420", 195000, "LONG")  # NAVER
    
    # 가격 상승 후 하락
    await risk_manager.update_position("035420", 205000)  # +5.1%
    await risk_manager.update_position("035420", 210000)  # +7.7% (최고점)
    await risk_manager.update_position("035420", 208000)  # +6.7%
    result = await risk_manager.update_position("035420", 199500)  # 트레일링 스탑 발동
    print(f"결과: {result}")
    
    # 최종 결과
    print("\n" + "="*70)
    print("🏁 테스트 완료")
    risk_manager.print_portfolio_status()
    
    return risk_manager

if __name__ == "__main__":
    asyncio.run(test_risk_management()) 