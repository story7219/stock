import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from kis_api_client import KISAPIClient, StockPrice, OrderRequest

@dataclass
class Position:
    symbol: str
    name: str
    quantity: int
    entry_price: int
    entry_time: datetime
    target_profit_rate: float = 1.5  # 1.5% 목표
    stop_loss_rate: float = 0.5      # 0.5% 손절
    max_hold_minutes: int = 120      # 최대 2시간

@dataclass
class TradeResult:
    symbol: str
    name: str
    action: str
    quantity: int
    price: int
    timestamp: datetime
    profit_rate: float = 0.0
    profit_amount: int = 0

class ScalpingTrader:
    def __init__(self):
        self.client = KISAPIClient()
        self.positions: List[Position] = []
        self.daily_trades: List[TradeResult] = []
        self.max_daily_trades = 5  # 하루 최대 5번 거래
        self.max_position_size = 1000000  # 최대 포지션 크기 100만원
        
    def find_scalping_candidates(self) -> List[StockPrice]:
        """스캘핑 후보 종목 선별"""
        print("📊 스캘핑 후보 종목 검색 중...")
        
        # 거래량 상위 종목 조회
        volume_stocks = self.client.get_trading_volume_ranking(50)
        
        candidates = []
        for stock in volume_stocks:
            # 필터링 조건
            if (stock.current_price >= 5000 and  # 최소 5천원 이상
                stock.current_price <= 100000 and  # 최대 10만원 이하
                stock.volume >= 1000000 and  # 최소 100만주 거래량
                abs(stock.change_rate) >= 1.0):  # 최소 1% 변동률
                
                candidates.append(stock)
        
        # 변동률 순으로 정렬
        candidates.sort(key=lambda x: abs(x.change_rate), reverse=True)
        
        print(f"✅ {len(candidates)}개 후보 종목 발견")
        return candidates[:10]  # 상위 10개만 선택
    
    def analyze_entry_signal(self, stock: StockPrice) -> bool:
        """매수 신호 분석"""
        # 간단한 모멘텀 기반 진입 신호
        if (stock.change_rate > 2.0 and  # 2% 이상 상승
            stock.volume > 2000000):     # 충분한 거래량
            return True
        return False
    
    def calculate_position_size(self, price: int) -> int:
        """포지션 크기 계산"""
        # 최대 포지션 크기 내에서 계산
        max_shares = self.max_position_size // price
        return min(max_shares, 100)  # 최대 100주로 제한
    
    def enter_position(self, stock: StockPrice) -> bool:
        """포지션 진입"""
        if len([t for t in self.daily_trades if t.action == "매수"]) >= self.max_daily_trades:
            print("⚠️ 일일 거래 한도 도달")
            return False
        
        if len(self.positions) >= 3:  # 최대 3개 포지션 동시 보유
            print("⚠️ 최대 포지션 수 도달")
            return False
        
        quantity = self.calculate_position_size(stock.current_price)
        
        # 시장가 매수 주문
        order = OrderRequest(
            symbol=stock.symbol,
            order_type="01",  # 시장가
            quantity=quantity,
            action="buy"
        )
        
        try:
            result = self.client.place_order(order)
            
            if result.get('rt_cd') == '0':
                # 포지션 기록
                position = Position(
                    symbol=stock.symbol,
                    name=stock.name,
                    quantity=quantity,
                    entry_price=stock.current_price,
                    entry_time=datetime.now()
                )
                self.positions.append(position)
                
                # 거래 기록
                trade = TradeResult(
                    symbol=stock.symbol,
                    name=stock.name,
                    action="매수",
                    quantity=quantity,
                    price=stock.current_price,
                    timestamp=datetime.now()
                )
                self.daily_trades.append(trade)
                
                print(f"✅ 매수 체결: {stock.name} {quantity}주 @ {stock.current_price:,}원")
                return True
                
        except Exception as e:
            print(f"❌ 매수 주문 실패: {e}")
        
        return False
    
    def check_exit_conditions(self, position: Position) -> Optional[str]:
        """청산 조건 체크"""
        try:
            current_stock = self.client.get_current_price(position.symbol)
            current_price = current_stock.current_price
            
            # 수익률 계산
            profit_rate = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # 익절 조건
            if profit_rate >= position.target_profit_rate:
                return "익절"
            
            # 손절 조건
            if profit_rate <= -position.stop_loss_rate:
                return "손절"
            
            # 시간 기반 청산
            elapsed_time = datetime.now() - position.entry_time
            if elapsed_time.total_seconds() > position.max_hold_minutes * 60:
                return "시간청산"
                
        except Exception as e:
            print(f"❌ 포지션 체크 실패: {e}")
        
        return None
    
    def exit_position(self, position: Position, reason: str) -> bool:
        """포지션 청산"""
        try:
            current_stock = self.client.get_current_price(position.symbol)
            current_price = current_stock.current_price
            
            # 시장가 매도 주문
            order = OrderRequest(
                symbol=position.symbol,
                order_type="01",  # 시장가
                quantity=position.quantity,
                action="sell"
            )
            
            result = self.client.place_order(order)
            
            if result.get('rt_cd') == '0':
                # 수익률 계산
                profit_rate = ((current_price - position.entry_price) / position.entry_price) * 100
                profit_amount = (current_price - position.entry_price) * position.quantity
                
                # 거래 기록
                trade = TradeResult(
                    symbol=position.symbol,
                    name=position.name,
                    action=f"매도({reason})",
                    quantity=position.quantity,
                    price=current_price,
                    timestamp=datetime.now(),
                    profit_rate=profit_rate,
                    profit_amount=profit_amount
                )
                self.daily_trades.append(trade)
                
                # 포지션 제거
                self.positions.remove(position)
                
                print(f"✅ 매도 체결({reason}): {position.name} {position.quantity}주 @ {current_price:,}원")
                print(f"   💰 수익률: {profit_rate:+.2f}%, 수익금: {profit_amount:+,}원")
                return True
                
        except Exception as e:
            print(f"❌ 매도 주문 실패: {e}")
        
        return False
    
    def run_trading_session(self):
        """트레이딩 세션 실행"""
        print("🚀 스캘핑 트레이딩 시작")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        while True:
            try:
                current_time = datetime.now()
                
                # 장 시간 체크 (9:00 ~ 15:20)
                if current_time.hour < 9 or current_time.hour >= 15 and current_time.minute >= 20:
                    print("⏰ 장 마감 시간")
                    break
                
                # 기존 포지션 체크
                for position in self.positions.copy():
                    exit_reason = self.check_exit_conditions(position)
                    if exit_reason:
                        self.exit_position(position, exit_reason)
                
                # 새로운 진입 기회 탐색 (10분마다)
                if current_time.minute % 10 == 0:
                    candidates = self.find_scalping_candidates()
                    
                    for stock in candidates:
                        if self.analyze_entry_signal(stock):
                            # 이미 보유 중인 종목은 스킵
                            if any(p.symbol == stock.symbol for p in self.positions):
                                continue
                                
                            if self.enter_position(stock):
                                break  # 한 번에 하나씩만 진입
                
                # 현재 상태 출력
                self.print_status()
                
                # 30초 대기
                time.sleep(30)
                
            except KeyboardInterrupt:
                print("\n⏹️ 트레이딩 중단됨")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                time.sleep(60)  # 1분 대기 후 재시도
        
        # 세션 종료 시 모든 포지션 청산
        print("📊 세션 종료 - 모든 포지션 청산")
        for position in self.positions.copy():
            self.exit_position(position, "장마감")
        
        self.print_daily_summary()
    
    def print_status(self):
        """현재 상태 출력"""
        print(f"\n📊 현재 상태 ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   보유 포지션: {len(self.positions)}개")
        print(f"   오늘 거래: {len([t for t in self.daily_trades if t.action == '매수'])}회")
        
        if self.positions:
            print("   📈 보유 종목:")
            for pos in self.positions:
                try:
                    current = self.client.get_current_price(pos.symbol)
                    profit_rate = ((current.current_price - pos.entry_price) / pos.entry_price) * 100
                    elapsed = datetime.now() - pos.entry_time
                    print(f"      {pos.name}: {profit_rate:+.2f}% ({elapsed.total_seconds()//60:.0f}분 경과)")
                except:
                    pass
    
    def print_daily_summary(self):
        """일일 요약 출력"""
        print("\n📈 일일 거래 요약")
        print("=" * 50)
        
        total_trades = len([t for t in self.daily_trades if t.action == "매수"])
        total_profit = sum(t.profit_amount for t in self.daily_trades if hasattr(t, 'profit_amount'))
        
        print(f"총 거래 횟수: {total_trades}회")
        print(f"총 수익금: {total_profit:+,}원")
        
        if self.daily_trades:
            print("\n상세 거래 내역:")
            for trade in self.daily_trades:
                if hasattr(trade, 'profit_rate') and trade.profit_rate != 0:
                    print(f"{trade.timestamp.strftime('%H:%M')} | {trade.action} | "
                          f"{trade.name} | {trade.quantity}주 @ {trade.price:,}원 | "
                          f"{trade.profit_rate:+.2f}% ({trade.profit_amount:+,}원)")
                else:
                    print(f"{trade.timestamp.strftime('%H:%M')} | {trade.action} | "
                          f"{trade.name} | {trade.quantity}주 @ {trade.price:,}원")

if __name__ == "__main__":
    trader = ScalpingTrader()
    trader.run_trading_session() 