# -*- coding: utf-8 -*-
# test_improved.py - 개선된 한국투자증권 API 테스트 (구글시트 연동)
import requests
import random
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pickle
import time

# 구글 시트 연동을 위한 추가 import
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("[WARNING] 구글 시트 연동을 위해 gspread 설치가 필요합니다: pip install gspread google-auth")

# 제미나이 API 연동
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARNING] 제미나이 연동을 위해 google-generativeai 설치가 필요합니다: pip install google-generativeai")

# .env 파일 로드
load_dotenv()

@dataclass
class TokenRequestHeader:
    """토큰 요청 헤더"""
    content_type: str = "application/json; charset=utf-8"

@dataclass
class TokenRequestBody:
    """토큰 요청 바디"""
    grant_type: str
    appkey: str
    appsecret: str

@dataclass
class TokenResponseBody:
    """토큰 응답 바디"""
    access_token: str
    token_type: str
    expires_in: int
    access_token_token_expired: str

@dataclass
class StockPriceInfo:
    """주식 가격 정보"""
    name: str
    current_price: int
    change_amount: int
    change_rate: float

@dataclass
class TokenInfo:
    """토큰 정보 관리 클래스"""
    access_token: str
    token_type: str
    expires_in: int
    expired_datetime: str
    issued_at: datetime  # 발급 시간 추가
    
    def is_expired(self) -> bool:
        """토큰이 만료되었는지 확인"""
        try:
            # expired_datetime 파싱 (예: "2023-12-25 15:30:45")
            expire_time = datetime.strptime(self.expired_datetime, "%Y-%m-%d %H:%M:%S")
            return datetime.now() >= expire_time
        except:
            # 파싱 실패 시 발급 시간 기준으로 24시간 후 만료
            return datetime.now() >= self.issued_at + timedelta(hours=24)
    
    def should_refresh(self) -> bool:
        """토큰을 갱신해야 하는지 확인 (6시간 기준)"""
        return datetime.now() >= self.issued_at + timedelta(hours=6)

class TokenManager:
    """토큰 관리 클래스"""
    
    def __init__(self, cache_file="token_cache.pkl"):
        self.cache_file = cache_file
        
    def save_token(self, token_info: TokenInfo):
        """토큰을 파일에 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(token_info, f)
            print(f"[CACHE] 토큰을 {self.cache_file}에 저장했습니다.")
        except Exception as e:
            print(f"[ERROR] 토큰 저장 실패: {str(e)}")
    
    def load_token(self) -> Optional[TokenInfo]:
        """파일에서 토큰 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    token_info = pickle.load(f)
                print(f"[CACHE] 기존 토큰을 {self.cache_file}에서 로드했습니다.")
                return token_info
        except Exception as e:
            print(f"[ERROR] 토큰 로드 실패: {str(e)}")
        return None
    
    def clear_cache(self):
        """토큰 캐시 삭제"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                print("[CACHE] 토큰 캐시를 삭제했습니다.")
        except Exception as e:
            print(f"[ERROR] 캐시 삭제 실패: {str(e)}")

class SmartMockKISAPI:
    """개선된 KIS API 모의 클래스"""
    
    def __init__(self):
        self.access_token = "mock_token"
        self.balance = 1000000  # 모의 잔고 100만원
        self.holdings = {}  # 보유 주식
        
    def get_access_token(self):
        print("[TOKEN] 모의 토큰 발급 완료")
        return "mock_access_token_12345"
    
    def get_stock_price(self, stock_code: str) -> StockPriceInfo:
        """실제와 유사한 주식 가격 (더 현실적인 가격)"""
        mock_prices = {
            "005930": {"name": "삼성전자", "current_price": 71000 + random.randint(-2000, 2000)},
            "000660": {"name": "SK하이닉스", "current_price": 126000 + random.randint(-3000, 3000)},
            "035420": {"name": "NAVER", "current_price": 194000 + random.randint(-5000, 5000)},
            "207940": {"name": "삼성바이오로직스", "current_price": 780000 + random.randint(-20000, 20000)},
            "005380": {"name": "현대차", "current_price": 205000 + random.randint(-5000, 5000)},
            # 저가 종목 추가 (테스트용)
            "003550": {"name": "LG", "current_price": 85000 + random.randint(-5000, 5000)},
            "017670": {"name": "SK텔레콤", "current_price": 52000 + random.randint(-2000, 2000)},
        }
        
        if stock_code in mock_prices:
            stock = mock_prices[stock_code]
            change = random.uniform(-5, 5)  # -5% ~ +5% 변동
            change_rate = round(change, 2)
            change_amount = int(stock["current_price"] * change / 100)
            
            return StockPriceInfo(
                name=stock["name"],
                current_price=stock["current_price"],
                change_rate=change_rate,
                change_amount=change_amount
            )
        else:
            return StockPriceInfo(
                name="알 수 없는 종목",
                current_price=50000 + random.randint(-5000, 5000),
                change_rate=round(random.uniform(-2, 2), 2),
                change_amount=random.randint(-1000, 1000)
            )
    
    def calculate_max_quantity(self, stock_code):
        """잔고로 살 수 있는 최대 수량 계산"""
        stock_info = self.get_stock_price(stock_code)
        max_qty = int(self.balance / stock_info.current_price)
        return max_qty
    
    def smart_buy(self, stock_code, target_quantity=None):
        """스마트 매수 - 잔고에 맞춰 수량 조절"""
        stock_info = self.get_stock_price(stock_code)
        max_qty = self.calculate_max_quantity(stock_code)
        
        if target_quantity is None:
            # 잔고의 30% 정도로 매수
            target_quantity = max(1, int(max_qty * 0.3))
        
        if max_qty == 0:
            print(f"[ERROR] {stock_info.name} 매수 불가: 잔고 부족")
            return {"status": "failed", "reason": "insufficient_balance"}
        
        # 실제 매수할 수량 (목표 수량과 최대 수량 중 작은 값)
        buy_quantity = min(target_quantity, max_qty)
        
        return self.buy_stock(stock_code, buy_quantity)
    
    def buy_stock(self, stock_code, quantity):
        """매수 주문"""
        stock_info = self.get_stock_price(stock_code)
        total_cost = stock_info.current_price * quantity
        
        if self.balance >= total_cost:
            self.balance -= total_cost
            if stock_code in self.holdings:
                self.holdings[stock_code] += quantity
            else:
                self.holdings[stock_code] = quantity
                
            print(f"[BUY] 매수 성공: {stock_info.name} {quantity}주 (단가: {stock_info.current_price:,}원)")
            print(f"[BALANCE] 잔고: {self.balance:,}원")
            return {"order_id": f"buy_{random.randint(1000, 9999)}", "status": "success"}
        else:
            print(f"[ERROR] 매수 실패: 잔고 부족")
            return {"status": "failed", "reason": "insufficient_balance"}
    
    def sell_stock(self, stock_code, quantity):
        """매도 주문"""
        if stock_code in self.holdings and self.holdings[stock_code] >= quantity:
            stock_info = self.get_stock_price(stock_code)
            total_amount = stock_info.current_price * quantity
            
            self.balance += total_amount
            self.holdings[stock_code] -= quantity
            
            if self.holdings[stock_code] == 0:
                del self.holdings[stock_code]
            
            print(f"[SELL] 매도 성공: {stock_info.name} {quantity}주 (단가: {stock_info.current_price:,}원)")
            print(f"[BALANCE] 잔고: {self.balance:,}원")
            return {"order_id": f"sell_{random.randint(1000, 9999)}", "status": "success"}
        else:
            print(f"[ERROR] 매도 실패: 보유 수량 부족")
            return {"status": "failed", "reason": "insufficient_stock"}
    
    def get_portfolio(self):
        """포트폴리오 조회"""
        print("\n" + "="*50)
        print("[PORTFOLIO] 현재 포트폴리오")
        print("="*50)
        print(f"[CASH] 현금 잔고: {self.balance:,}원")
        
        if not self.holdings:
            print("[STOCKS] 보유 주식: 없음")
            total_value = self.balance
        else:
            total_value = self.balance
            print("[STOCKS] 보유 주식:")
            for stock_code, quantity in self.holdings.items():
                stock_info = self.get_stock_price(stock_code)
                value = stock_info.current_price * quantity
                total_value += value
                profit_per_stock = stock_info.change_amount * quantity
                print(f"   - {stock_info.name} ({stock_code}): {quantity}주")
                print(f"     현재가: {stock_info.current_price:,}원 | 총 가치: {value:,}원")
                print(f"     평가손익: {profit_per_stock:,}원 ({stock_info.change_rate:+.2f}%)")
        
        print(f"[TOTAL] 총 자산: {total_value:,}원")
        profit = total_value - 1000000
        profit_rate = (profit / 1000000) * 100
        print(f"[PROFIT] 총 수익: {profit:,}원 ({profit_rate:+.2f}%)")
        print("="*50)

@dataclass
class TradingRecord:
    """매매 기록"""
    timestamp: str          # 거래 시간
    trade_type: str         # 매수/매도
    stock_code: str         # 종목코드
    stock_name: str         # 종목명
    quantity: int           # 수량
    price: int              # 체결가
    total_amount: int       # 총 금액
    commission: int         # 수수료
    tax: int                # 세금
    net_amount: int         # 실수령액
    profit_loss: int        # 손익 (매도시만)
    profit_rate: float      # 수익률 (매도시만)
    balance_after: int      # 거래 후 잔고
    order_no: str           # 주문번호
    strategy: str           # 사용한 전략
    note: str               # 메모

class GoogleSheetsManager:
    """구글 시트 매매 기록 관리 클래스"""
    
    def __init__(self):
        self.gc = None
        self.sheet = None
        self.is_connected = False
        self.service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        self.spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID", "")
        self.worksheet_name = os.getenv("GOOGLE_WORKSHEET_NAME", "매매기록")
        
        if GOOGLE_SHEETS_AVAILABLE:
            self.connect_to_google_sheets()
        else:
            print("[WARNING] 구글 시트 기능이 비활성화되었습니다.")
    
    def connect_to_google_sheets(self):
        """구글 시트 연결"""
        try:
            # 서비스 계정 파일 확인
            if not os.path.exists(self.service_account_file):
                print(f"[ERROR] 서비스 계정 파일을 찾을 수 없습니다: {self.service_account_file}")
                print("[INFO] 구글 클라우드 콘솔에서 서비스 계정 키를 다운로드하세요.")
                return
            
            # 인증 설정
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                self.service_account_file, 
                scopes=scope
            )
            
            self.gc = gspread.authorize(credentials)
            
            # 스프레드시트 열기
            if self.spreadsheet_id:
                try:
                    spreadsheet = self.gc.open_by_key(self.spreadsheet_id)
                except:
                    print(f"[ERROR] 스프레드시트 ID로 열기 실패: {self.spreadsheet_id}")
                    print("[INFO] 새로운 시트를 생성합니다.")
                    spreadsheet = self.gc.create("주식매매기록")
                    print(f"[INFO] 새 시트 ID: {spreadsheet.id}")
            else:
                # 새 시트 생성
                spreadsheet = self.gc.create("주식매매기록")
                print(f"[INFO] 새 시트 생성됨. ID: {spreadsheet.id}")
                print(f"[INFO] .env 파일에 GOOGLE_SPREADSHEET_ID={spreadsheet.id} 추가하세요.")
            
            # 워크시트 선택 또는 생성
            try:
                self.sheet = spreadsheet.worksheet(self.worksheet_name)
            except:
                self.sheet = spreadsheet.add_worksheet(
                    title=self.worksheet_name, 
                    rows="1000", 
                    cols="20"
                )
                # 헤더 설정
                self.setup_headers()
            
            self.is_connected = True
            print(f"[SUCCESS] 구글 시트 연결 성공: {self.worksheet_name}")
            
        except Exception as e:
            print(f"[ERROR] 구글 시트 연결 실패: {str(e)}")
            self.is_connected = False
    
    def setup_headers(self):
        """매매 기록 시트 헤더 설정"""
        headers = [
            "거래일시", "구분", "종목코드", "종목명", "수량", 
            "체결가", "총금액", "수수료", "세금", "실수령액",
            "손익", "수익률(%)", "거래후잔고", "주문번호", 
            "전략", "메모"
        ]
        
        try:
            self.sheet.insert_row(headers, 1)
            # 헤더 스타일링
            self.sheet.format('A1:P1', {
                'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.8},
                'textFormat': {'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}, 'bold': True}
            })
            print("[INFO] 헤더 설정 완료")
        except Exception as e:
            print(f"[ERROR] 헤더 설정 실패: {str(e)}")
    
    def save_trading_record(self, record: TradingRecord):
        """매매 기록 저장"""
        if not self.is_connected:
            print("[WARNING] 구글 시트가 연결되지 않았습니다. 로컬에 저장합니다.")
            self.save_to_local_file(record)
            return
        
        try:
            row_data = [
                record.timestamp,
                record.trade_type,
                record.stock_code,
                record.stock_name,
                record.quantity,
                record.price,
                record.total_amount,
                record.commission,
                record.tax,
                record.net_amount,
                record.profit_loss if record.profit_loss != 0 else "",
                f"{record.profit_rate:.2f}" if record.profit_rate != 0 else "",
                record.balance_after,
                record.order_no,
                record.strategy,
                record.note
            ]
            
            self.sheet.append_row(row_data)
            
            # 최근 거래 행에 색상 적용 (매수: 파란색, 매도: 빨간색)
            last_row = len(self.sheet.get_all_values())
            
            if record.trade_type == "매수":
                color = {'red': 0.8, 'green': 0.9, 'blue': 1.0}
            else:  # 매도
                color = {'red': 1.0, 'green': 0.8, 'blue': 0.8}
            
            self.sheet.format(f'A{last_row}:P{last_row}', {
                'backgroundColor': color
            })
            
            print(f"[SAVE] 구글 시트에 매매 기록 저장 완료: {record.trade_type} {record.stock_name}")
            
        except Exception as e:
            print(f"[ERROR] 구글 시트 저장 실패: {str(e)}")
            print("[FALLBACK] 로컬 파일로 저장합니다.")
            self.save_to_local_file(record)
    
    def save_to_local_file(self, record: TradingRecord):
        """로컬 파일에 백업 저장"""
        try:
            import csv
            
            filename = f"trading_records_{datetime.now().strftime('%Y%m')}.csv"
            file_exists = os.path.exists(filename)
            
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 파일이 없으면 헤더 추가
                if not file_exists:
                    headers = [
                        "거래일시", "구분", "종목코드", "종목명", "수량", 
                        "체결가", "총금액", "수수료", "세금", "실수령액",
                        "손익", "수익률", "거래후잔고", "주문번호", 
                        "전략", "메모"
                    ]
                    writer.writerow(headers)
                
                # 데이터 추가
                row_data = [
                    record.timestamp, record.trade_type, record.stock_code,
                    record.stock_name, record.quantity, record.price,
                    record.total_amount, record.commission, record.tax,
                    record.net_amount, record.profit_loss, record.profit_rate,
                    record.balance_after, record.order_no, record.strategy,
                    record.note
                ]
                writer.writerow(row_data)
            
            print(f"[BACKUP] 로컬 파일에 저장: {filename}")
            
        except Exception as e:
            print(f"[ERROR] 로컬 파일 저장 실패: {str(e)}")
    
    def get_trading_summary(self, days: int = 30) -> Dict[str, Any]:
        """매매 요약 정보 조회"""
        if not self.is_connected:
            return {"error": "구글 시트가 연결되지 않았습니다."}
        
        try:
            records = self.sheet.get_all_records()
            
            # 최근 N일 데이터 필터링
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_records = []
            
            for record in records:
                try:
                    trade_date = datetime.strptime(record['거래일시'][:10], '%Y-%m-%d')
                    if trade_date >= cutoff_date:
                        recent_records.append(record)
                except:
                    continue
            
            # 요약 계산
            total_trades = len(recent_records)
            buy_count = sum(1 for r in recent_records if r['구분'] == '매수')
            sell_count = sum(1 for r in recent_records if r['구분'] == '매도')
            
            total_profit = sum(
                float(r['손익']) for r in recent_records 
                if r['손익'] and str(r['손익']).replace('-', '').replace('.', '').isdigit()
            )
            
            summary = {
                "period_days": days,
                "total_trades": total_trades,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "total_profit": total_profit,
                "avg_profit_per_trade": total_profit / max(sell_count, 1),
                "win_rate": 0  # 승률 계산은 추가 로직 필요
            }
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] 매매 요약 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def show_recent_trades(self, limit: int = 10):
        """최근 매매 기록 조회"""
        if not self.is_connected:
            print("[WARNING] 구글 시트가 연결되지 않았습니다.")
            return
        
        try:
            records = self.sheet.get_all_records()
            recent_records = records[-limit:] if len(records) >= limit else records
            
            print(f"\n📊 최근 {len(recent_records)}건 매매 기록:")
            print("="*80)
            
            for record in reversed(recent_records):  # 최신순 정렬
                trade_type = record['구분']
                stock_name = record['종목명']
                quantity = record['수량']
                price = record['체결가']
                timestamp = record['거래일시']
                profit = record['손익'] if record['손익'] else 0
                
                emoji = "📈" if trade_type == "매수" else "📉"
                print(f"{emoji} {timestamp} | {trade_type} {stock_name} {quantity}주 @{price:,}원")
                
                if profit:
                    profit_emoji = "💰" if float(profit) > 0 else "💸"
                    print(f"    {profit_emoji} 손익: {profit:,}원")
            
            print("="*80)
            
        except Exception as e:
            print(f"[ERROR] 최근 매매 기록 조회 실패: {str(e)}")

class TradingSystemWithGoogleSheets(SmartMockKISAPI):
    """구글 시트 연동된 트레이딩 시스템"""
    
    def __init__(self):
        super().__init__()
        self.sheets_manager = GoogleSheetsManager()
        self.buy_prices = {}  # 매수가 기록 (손익 계산용)
    
    def buy_stock_with_record(self, stock_code: str, quantity: int, strategy: str = "기본전략", note: str = ""):
        """매수 + 구글시트 기록"""
        # 매수 실행
        result = self.buy_stock(stock_code, quantity)
        
        if result.get("status") == "success":
            stock_info = self.get_stock_price(stock_code)
            
            # 매수가 기록 (손익 계산을 위해)
            if stock_code not in self.buy_prices:
                self.buy_prices[stock_code] = []
            self.buy_prices[stock_code].extend([stock_info.current_price] * quantity)
            
            # 매매 기록 생성
            record = TradingRecord(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trade_type="매수",
                stock_code=stock_code,
                stock_name=stock_info.name,
                quantity=quantity,
                price=stock_info.current_price,
                total_amount=stock_info.current_price * quantity,
                commission=int(stock_info.current_price * quantity * 0.00015),  # 수수료 0.015%
                tax=0,  # 매수시 세금 없음
                net_amount=stock_info.current_price * quantity + int(stock_info.current_price * quantity * 0.00015),
                profit_loss=0,  # 매수시 손익 없음
                profit_rate=0.0,
                balance_after=self.balance,
                order_no=result.get("order_id", ""),
                strategy=strategy,
                note=note
            )
            
            # 구글 시트에 저장
            self.sheets_manager.save_trading_record(record)
            
        return result
    
    def sell_stock_with_record(self, stock_code: str, quantity: int, strategy: str = "기본전략", note: str = ""):
        """매도 + 구글시트 기록"""
        # 매도 실행
        result = self.sell_stock(stock_code, quantity)
        
        if result.get("status") == "success":
            stock_info = self.get_stock_price(stock_code)
            
            # 손익 계산
            profit_loss = 0
            profit_rate = 0.0
            
            if stock_code in self.buy_prices and self.buy_prices[stock_code]:
                # FIFO 방식으로 매수가 계산
                buy_prices_for_sale = self.buy_prices[stock_code][:quantity]
                avg_buy_price = sum(buy_prices_for_sale) / len(buy_prices_for_sale)
                
                profit_loss = (stock_info.current_price - avg_buy_price) * quantity
                profit_rate = ((stock_info.current_price - avg_buy_price) / avg_buy_price) * 100
                
                # 매도된 만큼 매수가 기록에서 제거
                self.buy_prices[stock_code] = self.buy_prices[stock_code][quantity:]
            
            # 세금 및 수수료 계산
            commission = int(stock_info.current_price * quantity * 0.00015)  # 수수료 0.015%
            tax = int(stock_info.current_price * quantity * 0.0023)  # 증권거래세 0.23%
            
            # 매매 기록 생성
            record = TradingRecord(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trade_type="매도",
                stock_code=stock_code,
                stock_name=stock_info.name,
                quantity=quantity,
                price=stock_info.current_price,
                total_amount=stock_info.current_price * quantity,
                commission=commission,
                tax=tax,
                net_amount=stock_info.current_price * quantity - commission - tax,
                profit_loss=int(profit_loss),
                profit_rate=round(profit_rate, 2),
                balance_after=self.balance,
                order_no=result.get("order_id", ""),
                strategy=strategy,
                note=note
            )
            
            # 구글 시트에 저장
            self.sheets_manager.save_trading_record(record)
            
        return result
    
    def show_trading_dashboard(self):
        """매매 대시보드 표시"""
        print("\n" + "="*60)
        print("📊 매매 대시보드 (구글 시트 연동)")
        print("="*60)
        
        # 포트폴리오 정보
        self.get_portfolio()
        
        # 최근 매매 기록
        self.sheets_manager.show_recent_trades(5)
        
        # 월간 요약
        summary = self.sheets_manager.get_trading_summary(30)
        if "error" not in summary:
            print(f"\n📈 월간 매매 요약 (최근 30일):")
            print(f"   총 거래: {summary['total_trades']}건")
            print(f"   매수: {summary['buy_count']}건 | 매도: {summary['sell_count']}건")
            print(f"   총 손익: {summary['total_profit']:,.0f}원")
            print(f"   거래당 평균: {summary['avg_profit_per_trade']:,.0f}원")
        
        print("="*60)

class GeminiAnalyzer:
    """제미나이 AI 시장 분석 클래스"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_available = GEMINI_AVAILABLE and bool(self.api_key)
        
        if self.is_available:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print("[SUCCESS] 제미나이 AI 분석기 초기화 완료")
        else:
            print("[WARNING] 제미나이 API 키가 없거나 라이브러리가 설치되지 않았습니다.")
    
    def analyze_market_condition(self, market_data: Dict[str, Any]) -> str:
        """시장 상황 분석"""
        if not self.is_available:
            return "제미나이 AI 분석을 사용할 수 없습니다."
        
        try:
            prompt = f"""
            다음 한국 주식시장 데이터를 분석하고 단기 투자 전략을 제안해주세요:
            
            시장 데이터:
            {json.dumps(market_data, ensure_ascii=False, indent=2)}
            
            분석해주세요:
            1. 현재 시장 상황 (상승/하락/횡보)
            2. 주목할 만한 종목과 이유
            3. 스캘핑에 적합한 종목 추천
            4. 위험 요소 및 주의사항
            5. 오늘의 투자 전략
            
            간결하고 실용적인 답변을 부탁합니다.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"제미나이 분석 중 오류 발생: {str(e)}"
    
    def analyze_stock_signal(self, stock_code: str, stock_data: Dict[str, Any]) -> str:
        """개별 종목 매매 신호 분석"""
        if not self.is_available:
            return "제미나이 AI 분석을 사용할 수 없습니다."
        
        try:
            prompt = f"""
            종목코드 {stock_code}에 대한 다음 데이터를 분석해주세요:
            
            {json.dumps(stock_data, ensure_ascii=False, indent=2)}
            
            다음 관점에서 분석해주세요:
            1. 매수/매도/관망 신호
            2. 신호 강도 (1-5점)
            3. 진입가격 제안
            4. 목표가격 및 손절가격
            5. 보유 기간 추천
            
            스캘핑 관점에서 분석해주세요.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"종목 분석 중 오류 발생: {str(e)}"

class AIEnhancedTradingSystem(TradingSystemWithGoogleSheets):
    """AI 강화된 트레이딩 시스템"""
    
    def __init__(self):
        super().__init__()
        self.gemini_analyzer = GeminiAnalyzer()
    
    def get_ai_market_briefing(self):
        """AI 시장 브리핑"""
        print("\n" + "="*60)
        print("🤖 AI 시장 분석 브리핑 (제미나이 기반)")
        print("="*60)
        
        if not self.gemini_analyzer.is_available:
            print("❌ 제미나이 API를 사용할 수 없습니다.")
            print("💡 .env 파일에 GEMINI_API_KEY를 추가하세요.")
            return
        
        # 시장 데이터 수집 (모의 데이터)
        market_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "kospi_change": round(random.uniform(-2, 2), 2),
            "kosdaq_change": round(random.uniform(-3, 3), 2),
            "volume_leaders": [
                {"code": "005930", "name": "삼성전자", "change_rate": 1.5},
                {"code": "000660", "name": "SK하이닉스", "change_rate": -0.8},
                {"code": "035420", "name": "NAVER", "change_rate": 2.3}
            ],
            "trading_value_leaders": [
                {"code": "207940", "name": "삼성바이오로직스", "trading_value": 500000000},
                {"code": "005380", "name": "현대차", "trading_value": 450000000}
            ]
        }
        
        print("📊 AI 분석 중...")
        analysis = self.gemini_analyzer.analyze_market_condition(market_data)
        
        print("\n🎯 제미나이 AI 분석 결과:")
        print("-" * 50)
        print(analysis)
    print("="*60)
    
    def get_ai_stock_recommendation(self, stock_codes: List[str]):
        """AI 종목 추천"""
        print(f"\n🔍 AI 종목 분석 ({len(stock_codes)}개 종목)")
        print("="*50)
        
        for stock_code in stock_codes:
            stock_info = self.get_stock_price(stock_code)
            
            stock_data = {
                "code": stock_code,
                "name": stock_info.name,
                "current_price": stock_info.current_price,
                "change_rate": stock_info.change_rate,
                "change_amount": stock_info.change_amount,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"\n📈 {stock_info.name} ({stock_code}) 분석:")
            
            if self.gemini_analyzer.is_available:
                ai_analysis = self.gemini_analyzer.analyze_stock_signal(stock_code, stock_data)
                print(ai_analysis)
            else:
                print("❌ AI 분석 불가 - 제미나이 API 키가 필요합니다.")
            
            print("-" * 30)

def test_ai_enhanced_trading():
    """AI 강화 트레이딩 시스템 테스트"""
    print("=== 🤖 AI 강화 트레이딩 시스템 테스트 ===")
    
    # AI 강화 시스템 초기화
    ai_system = AIEnhancedTradingSystem()
    
    print("\n[STEP 1] AI 시장 브리핑")
    ai_system.get_ai_market_briefing()
    
    print("\n[STEP 2] AI 종목 추천")
    test_stocks = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER
    ai_system.get_ai_stock_recommendation(test_stocks)
    
    print("\n[STEP 3] AI 기반 매매 테스트")
    confirm = input("AI 추천에 따라 테스트 매매를 실행하시겠습니까? (y/N): ").strip().lower()
    
    if confirm == 'y':
        # AI 추천 종목으로 테스트 매매
        ai_system.buy_stock_with_record(
            stock_code="005930",
            quantity=5,
            strategy="AI추천전략",
            note="제미나이 AI 분석 기반 매수"
        )
        
        time.sleep(2)
        
        ai_system.sell_stock_with_record(
            stock_code="005930",
            quantity=2,
            strategy="AI추천전략",
            note="제미나이 AI 분석 기반 부분매도"
        )
    
    print("\n✅ AI 강화 트레이딩 시스템 테스트 완료!")

def test_google_sheets_trading():
    """구글 시트 연동 매매 테스트"""
    print("=== 📊 구글 시트 연동 매매 테스트 ===")
    
    # 구글 시트 연동 시스템 초기화
    trading_system = TradingSystemWithGoogleSheets()
    
    # 연결 상태 확인
    print(f"\n[연결 상태] 구글 시트 연결: {'✅ 성공' if trading_system.sheets_manager.is_connected else '❌ 실패'}")
    
    if not trading_system.sheets_manager.is_connected:
        print("[ERROR] 구글 시트 연결에 실패했습니다.")
        print("확인사항:")
        print("1. service_account.json 파일이 있는지 확인")
        print("2. .env 파일의 GOOGLE_SPREADSHEET_ID 확인")
        print("3. 구글 시트에 서비스 계정 권한 부여 확인")
        return
    
    print("\n[STEP 1] 테스트 매수 실행")
    result1 = trading_system.buy_stock_with_record(
        stock_code="005930",
        quantity=3,
        strategy="구글시트연동테스트",
        note="시트 연동 테스트용 매수"
    )
    
    print(f"매수 결과: {result1}")
    
    print("\n[STEP 2] 포트폴리오 확인")
    trading_system.get_portfolio()
    
    print("\n[STEP 3] 테스트 매도 실행")
    result2 = trading_system.sell_stock_with_record(
        stock_code="005930",
        quantity=1,
        strategy="구글시트연동테스트",
        note="시트 연동 테스트용 매도"
    )
    
    print(f"매도 결과: {result2}")
    
    print("\n[STEP 4] 구글 시트 기록 확인")
    trading_system.sheets_manager.show_recent_trades(3)
    
    print("\n[STEP 5] 매매 대시보드")
    trading_system.show_trading_dashboard()
    
    print("\n✅ 구글 시트 연동 테스트 완료!")
    print("📊 구글 시트를 확인해서 매매 기록이 저장되었는지 확인해주세요.")

def main():
    """메인 함수 - AI 기능 추가"""
    print("=== 🚀 한국투자증권 API 실제 매매 시스템 (AI+구글시트 연동) 🚀 ===\n")
    
    while True:
        print("\n" + "="*70)
        print("💰 AI 강화 트레이딩 시스템 메뉴:")
        print("="*70)
        print("1. 토큰 관리 테스트")
        print("2. 모의투자 API 테스트") 
        print("3. 시장 데이터 분석")
        print("4. 🌟 궁극의 스캘핑 시스템 (Full MCP)")
        print("5. MCP 상태 확인")
        print("6. 🎯 일일 궁극 브리핑")
        print("7. 💸 실제 매매 테스트 (모의투자)")
        print("8. 계좌 잔고 조회")
        print("9. 📊 구글 시트 연동 매매 테스트")
        print("10. 📈 매매 대시보드 (구글시트)")
        print("11. 🤖 AI 시장 브리핑 (제미나이)")
        print("12. 🧠 AI 강화 트레이딩 테스트")
        print("13. 종료")
        print("="*70)
        print("⚠️  실제 매매는 모의투자 환경에서 먼저 테스트하세요!")
        print("📊 구글 시트: service_account.json + GOOGLE_SPREADSHEET_ID 필요")
        print("🤖 AI 분석: GEMINI_API_KEY 필요")
        
        choice = input("선택 (1-13): ").strip()
        
        if choice == "1":
            test_token_management()
        elif choice == "2":
            test_real_api()
        elif choice == "3":
            short_term_trading_strategy()
        elif choice == "4":
            # 궁극의 스캘핑 시스템
            api = SmartMockKISAPI()
            ultimate_system = UltimateScalpingSystem(api)
            ultimate_system.daily_ultimate_briefing()
        elif choice == "5":
            # MCP 상태 확인
            mcp = FullPowerMCPIntegration()
            mcp.check_mcp_status()
        elif choice == "6":
            # 일일 궁극 브리핑
            api = SmartMockKISAPI()
            ultimate_system = UltimateScalpingSystem(api)
            ultimate_system.daily_ultimate_briefing()
        elif choice == "7":
            # 실제 매매 테스트
            test_real_trading()
        elif choice == "8":
            # 계좌 잔고만 조회
            APP_KEY = os.getenv("KIS_APP_KEY")
            APP_SECRET = os.getenv("KIS_APP_SECRET")
            ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")
            
            if all([APP_KEY, APP_SECRET, ACCOUNT_NO]):
                api = RealTradingAPI(APP_KEY, APP_SECRET, ACCOUNT_NO, is_real=False)
                token = api.get_access_token()
                if token:
                    api.get_account_balance()
            else:
                print("[ERROR] .env 파일 설정이 필요합니다.")
        elif choice == "9":
            # 구글 시트 연동 매매 테스트
            test_google_sheets_trading()
        elif choice == "10":
            # 매매 대시보드
            trading_system = TradingSystemWithGoogleSheets()
            trading_system.show_trading_dashboard()
        elif choice == "11":
            # AI 시장 브리핑
            ai_system = AIEnhancedTradingSystem()
            ai_system.get_ai_market_briefing()
        elif choice == "12":
            # AI 강화 트레이딩 테스트
            test_ai_enhanced_trading()
        elif choice == "13":
            print("🚀 트레이딩 시스템을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1-13 중에서 선택해주세요.")

if __name__ == "__main__":
    main()