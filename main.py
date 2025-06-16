"""
AI 트레이딩 시스템 메인 실행 파일
- AI 시장 분석 실행
- 자동 매매 실행
- 시스템 설정 확인
"""
import argparse
import logging
import config
from analysis_engine import MarketAnalyzer
from core_trader import CoreTrader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_market_analysis(args):
    """(업그레이드) AI가 자동으로 시장 주도주를 찾아 분석합니다."""
    print("\n📈 AI 기반 시장 브리핑을 시작합니다.")
    try:
        analyzer = MarketAnalyzer()
        trader = CoreTrader()  # trader 변수를 함수 내에서 정의
        result = analyzer.get_trading_insights(args.image)
        print("\n--- 🤖 AI 분석 결과 ---")
        print(result)
        if trader.telegram_bot:
            trader.telegram_bot.send_message(f"--- 🤖 AI 분석 결과 ---\n{result}")
    except Exception as e:
        print(f"❌ 분석 실행 실패: {e}")

def run_auto_trading(args):
    print("\n🤖 자동 매매 시스템을 시작합니다.")
    try:
        trader = CoreTrader()
        # access_token 대신 token_manager 확인
        if not trader.token_manager.get_valid_token(): 
            print("❌ API 토큰을 가져올 수 없습니다.")
            return
        
        if args.action == 'balance':
            balance = trader.get_balance()
            if balance and balance.get('rt_cd') == '0':
                print("✅ 잔고 조회 성공!")
                print(f" - 예수금: {int(balance['output2'][0]['dnca_tot_amt']):,} 원")
                for stock in balance.get('output1', []):
                    print(f" - {stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주")
            else:
                print(f"❌ 잔고 조회 실패: {balance.get('msg1', '알 수 없는 오류') if balance else '응답 없음'}")

        elif args.action in ['buy', 'sell']:
            if not args.stock_code or not args.quantity:
                print("❌ 종목코드(-s)와 수량(-q)을 입력해주세요.")
                return
            trader.place_order(args.stock_code, args.action, args.quantity, args.price)
            
        elif args.action == 'report':
            # generate_daily_report 메서드가 없으므로 간단한 보고서 생성
            balance = trader.get_balance()
            if balance and balance.get('rt_cd') == '0':
                report = f"📊 일일 보고서\n예수금: {int(balance['output2'][0]['dnca_tot_amt']):,} 원"
                print(report)
                if trader.telegram_bot:
                    trader.telegram_bot.send_message(report)
            else:
                print("❌ 보고서 생성 실패: 잔고 조회 오류")
            
    except Exception as e:
        print(f"❌ 트레이딩 시스템 오류: {e}")
        
def show_config(args=None):  # args를 선택적 매개변수로 변경
    print("\n⚙️ 현재 시스템 설정")
    print(f" - 모의투자: {'Yes' if config.IS_MOCK_TRADING else 'No'}")
    print(f" - KIS APP KEY: {'설정됨' if config.KIS_APP_KEY else '설정 필요'}")
    print(f" - KIS APP SECRET: {'설정됨' if config.KIS_APP_SECRET else '설정 필요'}")
    print(f" - KIS 계좌번호: {'설정됨' if config.KIS_ACCOUNT_NUMBER else '설정 필요'}")
    print(f" - Gemini API KEY: {'설정됨' if config.GEMINI_API_KEY else '설정 필요'}")
    print(f" - 텔레그램 토큰: {'설정됨' if config.TELEGRAM_BOT_TOKEN else '설정 필요'}")
    print(f" - 구글 시트 파일: {'설정됨' if config.GOOGLE_SERVICE_ACCOUNT_FILE else '설정 필요'}")
    print(f" - 구글 시트 ID: {'설정됨' if config.GOOGLE_SPREADSHEET_ID else '설정 필요'}")

def run_default():
    """기본 실행 함수 - 설정 확인 후 잔고 조회"""
    print("🚀 AI 트레이딩 시스템을 시작합니다!")
    
    # 1. 설정 확인
    show_config()
    
    # 2. 잔고 조회 시도
    try:
        print("\n💰 잔고 조회를 시작합니다...")
        trader = CoreTrader()
        
        if not trader.token_manager.get_valid_token():
            print("❌ API 토큰을 가져올 수 없습니다. .env 파일을 확인해주세요.")
        return

        balance = trader.get_balance()
        if balance and balance.get('rt_cd') == '0':
            print("✅ 잔고 조회 성공!")
            print(f" - 예수금: {int(balance['output2'][0]['dnca_tot_amt']):,} 원")
            holdings = balance.get('output1', [])
            if holdings:
                print(" - 보유 종목:")
                for stock in holdings:
                    if int(stock['hldg_qty']) > 0:  # 보유 수량이 0보다 큰 경우만
                        print(f"   • {stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주")
            else:
                print(" - 보유 종목: 없음")
        else:
            print(f"❌ 잔고 조회 실패: {balance.get('msg1', '알 수 없는 오류') if balance else '응답 없음'}")
            
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
    
    print("\n📋 사용 가능한 명령어:")
    print(" - python main.py config          : 설정 확인")
    print(" - python main.py trade balance   : 잔고 조회")
    print(" - python main.py trade report    : 일일 보고서")
    print(" - python main.py analyze 이미지파일 : AI 분석")

def run_strategy_scan():
    """투자 전략 스캔 및 실행"""
    print("\n🎯 투자 전략 스캔을 시작합니다...")
    try:
        from strategy_engine import run_strategy
        run_strategy()
    except Exception as e:
        logger.error(f"전략 실행 실패: {e}")

def run_ai_trading():
    """AI 자동 매매 실행"""
    print("\n🤖 AI 자동 매매를 시작합니다...")
    try:
        from strategy_engine import run_ai_strategy
        run_ai_strategy()
    except Exception as e:
        logger.error(f"AI 매매 실행 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="AI 트레이딩 시스템")
    subparsers = parser.add_subparsers(dest='command', required=False)

    ana_p = subparsers.add_parser('analyze', help='AI가 시장 주도주를 자동 분석합니다.')
    ana_p.add_argument('image', type=str, help='분석에 참고할 차트 이미지 파일 경로')
    ana_p.set_defaults(func=run_market_analysis)

    trd_p = subparsers.add_parser('trade', help='자동 매매를 실행합니다.')
    trd_p.add_argument('action', choices=['balance', 'buy', 'sell', 'report'], help='수행할 동작')
    trd_p.add_argument('-s', '--stock_code', type=str, help='종목코드')
    trd_p.add_argument('-q', '--quantity', type=int, help='수량')
    trd_p.add_argument('-p', '--price', type=int, default=0, help='가격 (지정가, 0이면 시장가)')
    trd_p.set_defaults(func=run_auto_trading)
    
    cfg_p = subparsers.add_parser('config', help='현재 설정을 확인합니다.')
    cfg_p.set_defaults(func=show_config)

    # AI 매매 명령어 추가
    ai_parser = subparsers.add_parser('ai', help='AI 자동 매매 실행')
    
    # 전략 실행 명령어 추가
    strategy_parser = subparsers.add_parser('strategy', help='투자 전략 실행')
    
    args = parser.parse_args()
    
    if not args.command:
        show_config()
        return
    
    # 명령어가 없으면 기본 실행
    if args.command == 'ai':
        run_ai_trading()
    elif args.command == 'strategy':
        run_strategy_scan()
    else:
        args.func(args)

if __name__ == "__main__":
    main() 