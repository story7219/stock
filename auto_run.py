"""
매일 자동 실행 (Windows 작업 스케줄러용)
"""

from simple_trading import SimpleTrading
from datetime import datetime

def main():
    """매일 실행할 함수"""
    try:
        trader = SimpleTrading()
        total_value, return_rate = trader.daily_trading()
        
        # 결과를 파일로 저장
        with open("trading_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()}: 자산 {total_value:,}원 ({return_rate:+.1%})\n")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 