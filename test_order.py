import sys
from core_trader import KISTrader
import config

def run_test_order():
    """
    실제 매수 주문을 테스트하기 위한 간단한 스크립트.
    config.py에 설정된 값으로 주문을 실행합니다.
    """
    # 설정 파일에 따라 실전/모의 환경이 결정됩니다.
    # config.py에서 IS_MOCK_TRADING = False 인지 다시 한번 확인해주세요.
    if config.IS_MOCK_TRADING:
        print("❌ 테스트 오류: 현재 모의투자 환경으로 설정되어 있습니다.")
        print("실제 매매를 테스트하려면 config.py 파일에서 IS_MOCK_TRADING을 False로 변경해주세요.")
        sys.exit(1)

    print("="*50)
    print("🚀 실제 매수 주문 테스트를 시작합니다.")
    print(f"   - 계좌번호: {config.KIS_ACCOUNT_NO}")
    print("="*50)

    try:
        # KISTrader 인스턴스 생성
        trader = KISTrader()

        # config.py 파일에서 테스트 주문 설정을 읽어옵니다.
        stock_code = config.TEST_ORDER_STOCK_CODE
        quantity = config.TEST_ORDER_QUANTITY
        order_type = config.TEST_ORDER_TYPE

        order_type_name = "시장가" if order_type == "01" else "지정가"
        
        print(f"\n[주문 실행] '{stock_code}' 종목 {quantity}주 {order_type_name} 매수 주문을 전송합니다...")

        # 주문 실행
        response = trader.place_order(
            stock_code=stock_code,
            quantity=quantity,
            order_type=order_type
        )

        print("\n[API 응답 결과]")
        print(response)

        # 한국투자증권 API의 실제 응답 코드는 'rt_cd' 입니다. '0'이 성공입니다.
        if response.get('rt_cd') == '0':
            print("\n✅ 주문이 성공적으로 접수되었습니다.")
            print("   잠시 후 HTS나 MTS의 '미체결 내역' 또는 '체결 내역'을 확인해주세요.")
            print(f"   - 메시지: {response.get('msg1')}")
        else:
            print("\n❌ 주문 접수에 실패했습니다.")
            print(f"   - 실패 사유: {response.get('msg1')}")

    except Exception as e:
        print(f"\n스크립트 실행 중 예외가 발생했습니다: {e}")
        print("  - core_trader.py의 KISTrader 클래스나 place_order 함수를 확인해주세요.")
        print("  - 또는 API 서버와의 통신에 문제가 있을 수 있습니다.")


if __name__ == "__main__":
    run_test_order() 