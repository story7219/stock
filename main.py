"""
AI 트레이딩 시스템 메인 실행 파일
- AI 시장 분석 실행
- 자동 매매 실행
- 시스템 설정 확인
"""
import argparse
import logging
from PIL import Image, ImageDraw, ImageFont

import config
from analysis_engine import MarketAnalyzer
from core_trader import KisTrader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_market_analysis(args):
    print("\n📈 AI 기반 시장 브리핑을 시작합니다.")
    try:
        analyzer = MarketAnalyzer()
        result = analyzer.get_trading_insights(args.text, args.image)
        print("\n--- 🤖 AI 분석 결과 ---")
        print(result)
    except Exception as e:
        print(f"❌ 분석 실행 실패: {e}")

def run_auto_trading(args):
    print("\n🤖 자동 매매 시스템을 시작합니다.")
    try:
        trader = KisTrader()
        if not trader.access_token: return
        
        if args.action == 'balance':
            balance = trader.get_balance()
            if balance and balance.get('rt_cd') == '0':
                print("✅ 잔고 조회 성공!")
                print(f" - 예수금: {int(balance['output2'][0]['dnca_tot_amt']):,} 원")
                for stock in balance.get('output1', []):
                    print(f" - {stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주")
            else:
                print(f"❌ 잔고 조회 실패: {balance.get('msg1', '알 수 없는 오류')}")

        elif args.action in ['buy', 'sell']:
            trader.place_order(args.stock_code, args.action, args.quantity, args.price)
            
    except Exception as e:
        print(f"❌ 트레이딩 시스템 오류: {e}")
        
def show_config(args):
    print("\n⚙️ 현재 시스템 설정")
    print(f" - 모의투자: {'Yes' if config.IS_MOCK_TRADING else 'No'}")
    print(f" - 총 투자금: {config.TOTAL_CAPITAL:,} 원")
    print(f" - KIS APP KEY: {'설정됨' if config.KIS_APP_KEY != 'YOUR_APP_KEY' else '설정 필요'}")
    print(f" - Google API KEY: {'설정됨' if config.GOOGLE_API_KEY != 'YOUR_GOOGLE_API_KEY' else '설정 필요'}")

def main():
    parser = argparse.ArgumentParser(description="AI 트레이딩 시스템")
    subparsers = parser.add_subparsers(dest='command', required=True)

    ana_p = subparsers.add_parser('analyze', help='AI로 시장 데이터와 차트를 분석합니다.')
    ana_p.add_argument('text', type=str, help='분석할 시장 데이터 텍스트')
    ana_p.add_argument('image', type=str, help='분석할 차트 이미지 파일 경로')
    ana_p.set_defaults(func=run_market_analysis)

    trd_p = subparsers.add_parser('trade', help='자동 매매를 실행합니다.')
    trd_p.add_argument('action', choices=['balance', 'buy', 'sell'], help='수행할 동작')
    trd_p.add_argument('-s', '--stock_code', type=str, help='종목코드')
    trd_p.add_argument('-q', '--quantity', type=int, help='수량')
    trd_p.add_argument('-p', '--price', type=int, default=0, help='가격 (지정가, 0이면 시장가)')
    trd_p.set_defaults(func=run_auto_trading)
    
    cfg_p = subparsers.add_parser('config', help='현재 설정을 확인합니다.')
    cfg_p.set_defaults(func=show_config)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main() 