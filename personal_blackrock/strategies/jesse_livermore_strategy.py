import json
from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class JesseLivermoreStrategy(BaseStrategy):
    """
    제시 리버모어의 추세 추종 단기 매매 전략입니다.
    """
    
    @property
    def style_name(self) -> str:
        return "livermore"

    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        제시 리버모어 추세 추종 전략으로 주식을 분석합니다.
        """
        try:
            stock_code = stock_data.get('stock_code', 'Unknown')
            company_name = stock_data.get('company_name', 'Unknown')
            
            analysis_result = self._calculate_trend_score(stock_data)
            
            return {
                "strategy": "jesse_livermore",
                "stock_code": stock_code,
                "company_name": company_name,
                "score": analysis_result["score"],
                "recommendation": analysis_result["recommendation"],
                "analysis": analysis_result["analysis"],
                "timestamp": stock_data.get("timestamp", "")
            }
            
        except Exception as e:
            return {
                "strategy": "jesse_livermore",
                "stock_code": stock_data.get('stock_code', 'Unknown'),
                "score": 50,
                "recommendation": "보류",
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _calculate_trend_score(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """추세 추종 기준으로 점수를 계산합니다."""
        try:
            score = 50
            analysis_details = []
            
            technical_indicators = stock_data.get('technical_indicators', {})
            current_price = stock_data.get('current_price', 0)
            
            # 이동평균선 분석
            ma20 = technical_indicators.get('ma20', 0)
            ma60 = technical_indicators.get('ma60', 0)
            
            if current_price > ma20 > ma60:
                score += 20
                analysis_details.append("상승 추세 확인 (+20점)")
            elif current_price > ma20:
                score += 10
                analysis_details.append("단기 상승 추세 (+10점)")
            
            # 거래량 분석
            volume_ratio = technical_indicators.get('volume_ratio', 1)
            if volume_ratio > 2:
                score += 15
                analysis_details.append("거래량 폭증 (+15점)")
            elif volume_ratio > 1.5:
                score += 10
                analysis_details.append("거래량 증가 (+10점)")
            
            # 모멘텀 분석
            rsi = technical_indicators.get('rsi', 50)
            if 50 < rsi < 70:
                score += 10
                analysis_details.append("모멘텀 양호 (+10점)")
            elif rsi > 70:
                score -= 5
                analysis_details.append("과매수 구간 (-5점)")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "매수"
                analysis = f"추세 추종 전략 기준 매수 신호입니다. {', '.join(analysis_details)}"
            elif score >= 50:
                recommendation = "보유"
                analysis = f"추세 추종 전략 기준 관망 추천입니다. {', '.join(analysis_details)}"
            else:
                recommendation = "매도"
                analysis = f"추세 추종 전략 기준 매도 신호입니다. {', '.join(analysis_details)}"
            
            return {"score": score, "recommendation": recommendation, "analysis": analysis}
            
        except Exception as e:
            return {"score": 50, "recommendation": "보류", "analysis": f"점수 계산 중 오류: {str(e)}"}

    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        제시 리버모어 스타일의 추세 추종 분석을 위한 프롬프트를 생성합니다.
        """
        company_name = comprehensive_data.get('company_name', 'N/A')
        sector = comprehensive_data.get('sector', 'N/A')
        financial_ratios = json.dumps(comprehensive_data.get('financial_ratios', {}), ensure_ascii=False)
        price_metrics = json.dumps(comprehensive_data.get('price_metrics', {}), ensure_ascii=False)
        trading_data = json.dumps(comprehensive_data.get('trading_data', {}), ensure_ascii=False)
        technical_indicators = json.dumps(comprehensive_data.get('technical_indicators', {}), ensure_ascii=False)

        return f"""
        [SYSTEM]
        You are an expert trader following Jesse Livermore's trend-following and momentum-based approach. Your analysis must focus on price action, volume patterns, and market timing for short to medium-term trades. Provide your final analysis ONLY in Korean JSON format as specified below.

        [Company Information]
        - Company Name: {company_name}
        - Sector: {sector}
        - Financial Ratios: {financial_ratios}
        - Price Metrics: {price_metrics}
        - Supply/Demand Data (Recent 1 Month): {trading_data}
        - Technical Indicators: {technical_indicators}

        [Recent News & Disclosures]
        {news_data}

        [Analysis Task]
        Analyze the stock based on Jesse Livermore's principles:
        1. Trend identification and momentum analysis
        2. Volume confirmation and accumulation patterns
        3. Breakout and breakdown signals
        4. Risk management and stop-loss levels
        5. Market timing and entry/exit points

        Key Livermore Concepts to Evaluate:
        - Price Action: Is the stock in a clear uptrend/downtrend?
        - Volume Confirmation: Does volume support the price movement?
        - Breakout Patterns: Any significant resistance/support breaks?
        - Market Sentiment: Is the overall market supportive?

        [JSON Output Format]
        {{
          "total_score": "A score from 0 to 100 evaluating the overall trend-following opportunity.",
          "component_scores": {{
            "trend_strength": "Score (0-100) for the strength and clarity of the current trend.",
            "volume_confirmation": "Score (0-100) for volume supporting the price trend.",
            "breakout_potential": "Score (0-100) for proximity to significant breakout levels.",
            "momentum_indicators": "Score (0-100) based on RSI, MACD, and other momentum indicators.",
            "market_timing": "Score (0-100) for current market timing and entry opportunity."
          }},
          "trading_signals": {{
            "primary_signal": "Buy, Sell, or Hold based on trend analysis.",
            "entry_price": "Recommended entry price level.",
            "stop_loss": "Suggested stop-loss level for risk management.",
            "target_price": "Initial profit target based on technical analysis.",
            "position_sizing": "Recommended position size (Small, Medium, Large)."
          }},
          "risk_assessment": {{
            "risk_level": "Low, Medium, or High based on volatility and market conditions.",
            "key_risks": ["List of primary trading risks."],
            "risk_reward_ratio": "Expected risk-to-reward ratio for the trade."
          }},
          "rationale": "A comprehensive trading thesis based on Livermore's trend-following methodology, explaining the technical setup and timing rationale."
        }}
        """

    @property
    def required_keys(self) -> List[str]:
        return ['trading_signals', 'risk_assessment']

    @property
    def component_keys(self) -> List[str]:
        return ['trend_strength', 'volume_confirmation', 'breakout_potential', 'momentum_indicators', 'market_timing'] 