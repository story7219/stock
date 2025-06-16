"""
한국투자증권 API MCP 서버
"""
import asyncio
import json
from datetime import datetime
from kis_api_client import KISAPIClient
from scalping_trader import ScalpingTrader

class KISTradingMCPServer:
    def __init__(self):
        self.client = KISAPIClient()
        self.trader = ScalpingTrader()
    
    async def handle_request(self, method: str, params: dict) -> dict:
        """MCP 요청 처리"""
        try:
            if method == "get_current_price":
                symbol = params.get("symbol")
                result = self.client.get_current_price(symbol)
                return {
                    "symbol": result.symbol,
                    "name": result.name,
                    "current_price": result.current_price,
                    "change_rate": result.change_rate,
                    "volume": result.volume
                }
            
            elif method == "get_volume_ranking":
                limit = params.get("limit", 20)
                stocks = self.client.get_trading_volume_ranking(limit)
                return [
                    {
                        "symbol": stock.symbol,
                        "name": stock.name,
                        "current_price": stock.current_price,
                        "change_rate": stock.change_rate,
                        "volume": stock.volume
                    }
                    for stock in stocks
                ]
            
            elif method == "start_scalping":
                # 비동기로 스캘핑 시작
                asyncio.create_task(self._run_scalping_async())
                return {"status": "스캘핑 트레이딩 시작"}
            
            elif method == "get_trading_status":
                return {
                    "positions": len(self.trader.positions),
                    "daily_trades": len(self.trader.daily_trades),
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {"error": f"알 수 없는 메서드: {method}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_scalping_async(self):
        """비동기 스캘핑 실행"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.trader.run_trading_session)

if __name__ == "__main__":
    server = KISTradingMCPServer()
    
    # 간단한 테스트
    print("🔄 한국투자증권 API 테스트")
    
    try:
        # 현재가 조회 테스트
        samsung = server.client.get_current_price("005930")
        print(f"✅ 삼성전자: {samsung.current_price:,}원 ({samsung.change_rate:+.2f}%)")
        
        # 거래량 순위 테스트
        volume_stocks = server.client.get_trading_volume_ranking(5)
        print("\n📊 거래량 상위 5종목:")
        for i, stock in enumerate(volume_stocks, 1):
            print(f"{i}. {stock.name}: {stock.current_price:,}원 ({stock.change_rate:+.2f}%)")
        
        print("\n✅ API 연결 성공!")
        
    except Exception as e:
        print(f"❌ API 연결 실패: {e}") 