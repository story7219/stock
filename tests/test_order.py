"""
실제 매수 주문 테스트 스크립트
- CoreTrader를 사용한 주문 테스트
- 모의투자/실전투자 환경 확인
"""
import sys
from core_trader import CoreTrader
import config

def run_test_order():
    """
    실제 매수 주문을 테스트하기 위한 간단한 스크립트.
    config.py에 설정된 값으로 주문을 실행합니다.
    """
    # 설정 파일에 따라 실전/모의 환경이 결정됩니다.
    if config.IS_MOCK:
        print("⚠️ 현재 모의투자 환경으로 설정되어 있습니다.")
        print(f"   - 서버: {config.KIS_BASE_URL}")
    else:
        print("🔥 현재 실전투자 환경으로 설정되어 있습니다.")
        print(f"   - 서버: {config.KIS_BASE_URL}")

    print("="*50)
    print("🚀 주문 테스트를 시작합니다.")
    print(f"   - 계좌번호: {config.KIS_ACCOUNT_NO}")
    print("="*50)

    try:
        # CoreTrader 인스턴스 생성
        trader = CoreTrader()

        # 테스트용 종목 설정 (삼성전자)
        stock_code = "005930"
        quantity = 1
        
        print(f"\n[주문 실행] '{stock_code}' 종목 {quantity}주 시장가 매수 주문을 전송합니다...")

        # 현재가 확인
        current_price_info = trader.get_current_price(stock_code)
        if current_price_info:
            print(f"   - 현재가: {current_price_info['price']:,}원")
            print(f"   - 종목명: {current_price_info['name']}")
        
        # 잔고 확인
        balance = trader.get_balance()
        if balance and balance.get('rt_cd') == '0':
            cash = int(balance['output2'][0]['dnca_tot_amt'])
            print(f"   - 보유 현금: {cash:,}원")
            
            if current_price_info and cash < current_price_info['price']:
                print("❌ 현금이 부족하여 주문을 실행할 수 없습니다.")
                return
        else:
            print("⚠️ 잔고 조회에 실패했습니다. 주문을 계속 진행합니다.")

        # 주문 실행 (CoreTrader의 execute_order 메서드 사용)
        success = trader.execute_order(
            symbol=stock_code,
            side='buy',
            quantity=quantity,
            price=0,  # 시장가
            log_payload={'status': 'test_order'}
        )

        if success:
            print("\n✅ 주문이 성공적으로 접수되었습니다.")
            print("   잠시 후 HTS나 MTS의 '미체결 내역' 또는 '체결 내역'을 확인해주세요.")
        else:
            print("\n❌ 주문 접수에 실패했습니다.")
            print("   로그를 확인하여 실패 사유를 파악해주세요.")

    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 예외가 발생했습니다: {e}")
        print("  - core_trader.py의 CoreTrader 클래스나 execute_order 메서드를 확인해주세요.")
        print("  - 또는 API 서버와의 통신에 문제가 있을 수 있습니다.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_test_order() 