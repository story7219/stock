import json
from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class IchimokuStrategy(BaseStrategy):
    """
    일목산인의 일목균형표 분석 전략입니다.
    시간론, 파동론, 수준론의 3대 이론을 바탕으로 한 종합적 차트 분석을 수행하며,
    전환선, 기준선, 선행스팬, 후행스팬, 구름대를 통한 균형점 분석을 제시합니다.
    """
    
    @property
    def style_name(self) -> str:
        return "ichimoku"

    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """일목균형표 전략으로 주식을 분석합니다."""
        try:
            stock_code = stock_data.get('stock_code', 'Unknown')
            company_name = stock_data.get('company_name', 'Unknown')
            
            analysis_result = self._calculate_ichimoku_score(stock_data)
            
            return {
                "strategy": "ichimoku",
                "stock_code": stock_code,
                "company_name": company_name,
                "score": analysis_result["score"],
                "recommendation": analysis_result["recommendation"],
                "analysis": analysis_result["analysis"],
                "timestamp": stock_data.get("timestamp", "")
            }
            
        except Exception as e:
            return {
                "strategy": "ichimoku",
                "stock_code": stock_data.get('stock_code', 'Unknown'),
                "score": 50,
                "recommendation": "보류",
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _calculate_ichimoku_score(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """일목균형표 기준으로 점수를 계산합니다."""
        try:
            score = 50
            analysis_details = []
            
            current_price = stock_data.get('current_price', 0)
            technical_indicators = stock_data.get('technical_indicators', {})
            
            # 기준선과 전환선 분석
            ma20 = technical_indicators.get('ma20', 0)  # 전환선 대용
            ma60 = technical_indicators.get('ma60', 0)  # 기준선 대용
            
            if current_price > ma20 > ma60:
                score += 20
                analysis_details.append("가격이 구름대 위에서 상승 추세 (+20점)")
            elif current_price > ma20:
                score += 10
                analysis_details.append("단기 상승 신호 (+10점)")
            
            # RSI로 균형점 분석
            rsi = technical_indicators.get('rsi', 50)
            if 40 <= rsi <= 60:
                score += 15
                analysis_details.append("균형점 근처에서 안정적 (+15점)")
            elif 30 <= rsi <= 70:
                score += 10
                analysis_details.append("적정 균형 범위 (+10점)")
            
            # 거래량으로 힘의 균형 분석
            volume_ratio = technical_indicators.get('volume_ratio', 1)
            if volume_ratio > 1.3:
                score += 10
                analysis_details.append("거래량 증가로 힘의 균형 변화 (+10점)")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "매수"
                analysis = f"일목균형표 기준 매수 신호입니다. {', '.join(analysis_details)}"
            elif score >= 50:
                recommendation = "보유"
                analysis = f"일목균형표 기준 균형 상태입니다. {', '.join(analysis_details)}"
            else:
                recommendation = "매도"
                analysis = f"일목균형표 기준 약세 신호입니다. {', '.join(analysis_details)}"
            
            return {"score": score, "recommendation": recommendation, "analysis": analysis}
            
        except Exception as e:
            return {"score": 50, "recommendation": "보류", "analysis": f"점수 계산 중 오류: {str(e)}"}

    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        일목산인 스타일의 일목균형표 분석을 위한 프롬프트를 생성합니다.
        """
        company_name = comprehensive_data.get('company_name', 'N/A')
        sector = comprehensive_data.get('sector', 'N/A')
        financial_ratios = json.dumps(comprehensive_data.get('financial_ratios', {}), ensure_ascii=False)
        price_metrics = json.dumps(comprehensive_data.get('price_metrics', {}), ensure_ascii=False)
        trading_data = json.dumps(comprehensive_data.get('trading_data', {}), ensure_ascii=False)
        technical_indicators = json.dumps(comprehensive_data.get('technical_indicators', {}), ensure_ascii=False)
        
        return f"""
        [SYSTEM]
        You are a master technical analyst specializing in Ichimoku Kinko Hyo (일목균형표) developed by Goichi Hosoda (일목산인). Your analysis is based on the three pillars: Time Theory (시간론), Wave Theory (파동론), and Price Level Theory (수준론). Focus on balance, harmony, and the interplay of time and price. Provide your final analysis ONLY in Korean JSON format as specified below.

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
        Analyze the stock based on Ichimoku Kinko Hyo principles:
        1. Tenkan-sen (전환선): 9-period conversion line analysis
        2. Kijun-sen (기준선): 26-period base line analysis  
        3. Senkou Span A & B (선행스팬 A, B): Leading spans forming the cloud
        4. Chikou Span (후행스팬): Lagging span analysis
        5. Kumo (구름대): Cloud analysis for support/resistance
        6. Time Theory (시간론): Basic, variation, and corresponding numbers
        7. Wave Theory (파동론): I, V, N wave patterns
        8. Price Level Theory (수준론): E, V, N calculations

        Key Ichimoku Concepts to Evaluate:
        - Balance Point (균형점): Where price finds equilibrium
        - Cloud Thickness: Strength of support/resistance
        - Span Relationships: Bullish/bearish configurations
        - Time Cycles: 9, 17, 26, 33, 42, 65, 76 periods
        - Wave Patterns: Trend continuation vs reversal signals
        - Future Cloud: 26-period forward projection

        [JSON Output Format]
        {{
          "total_score": "A score from 0 to 100 evaluating the overall Ichimoku analysis alignment.",
          "component_scores": {{
            "tenkan_kijun_relationship": "Score (0-100) for Tenkan-sen vs Kijun-sen relationship and crossovers.",
            "price_vs_cloud": "Score (0-100) for price position relative to the cloud (above/in/below).",
            "cloud_analysis": "Score (0-100) for cloud thickness, color, and future projection.",
            "chikou_span": "Score (0-100) for Chikou span position and clearance from price action.",
            "time_theory": "Score (0-100) for time cycle analysis and key turning points.",
            "wave_theory": "Score (0-100) for wave pattern identification and trend analysis.",
            "price_level_theory": "Score (0-100) for price target calculations and support/resistance levels."
          }},
          "ichimoku_analysis": {{
            "current_trend": "Overall trend direction (Strong Bullish, Bullish, Neutral, Bearish, Strong Bearish)",
            "tenkan_kijun": {{
              "tenkan_value": "Current Tenkan-sen value if calculable",
              "kijun_value": "Current Kijun-sen value if calculable", 
              "relationship": "Tenkan vs Kijun relationship (Bullish Cross, Bearish Cross, Above, Below, Converging)"
            }},
            "cloud_status": {{
              "position": "Price position relative to cloud (Above Cloud, In Cloud, Below Cloud)",
              "cloud_color": "Current cloud color (Green/Bullish, Red/Bearish, Transitioning)",
              "thickness": "Cloud thickness assessment (Thick, Medium, Thin)",
              "future_cloud": "Future cloud projection (26 periods ahead)"
            }},
            "chikou_analysis": "Chikou span position and significance",
            "time_cycles": {{
              "current_cycle": "Current position in major time cycles",
              "next_key_date": "Next important time cycle turning point",
              "cycle_strength": "Strength of current time cycle (Strong, Medium, Weak)"
            }},
            "wave_pattern": {{
              "current_wave": "Current wave type (I-wave, V-wave, N-wave)",
              "wave_position": "Position within the wave structure",
              "next_target": "Next wave target based on pattern"
            }}
          }},
          "trading_decision": {{
            "action": "'BUY', 'SELL', 'HOLD', or 'WAIT'. Based on Ichimoku balance analysis.",
            "entry_point": "Specific entry price based on Ichimoku levels",
            "stop_loss": "Stop-loss based on key Ichimoku support levels",
            "profit_target": "Profit target based on wave theory and price level calculations",
            "position_size": "Recommended position size (Small, Medium, Large) based on signal strength",
            "timing": "Entry timing based on time theory (Immediate, Wait for Cycle, Wait for Confirmation)"
          }},
          "rationale": "A comprehensive analysis integrating all three Ichimoku theories. Explain the balance point, time cycle position, wave structure, and price level significance. Include specific reasoning for entry/exit strategy based on Ichimoku harmony principles."
        }}
        """

    @property
    def required_keys(self) -> List[str]:
        return ['trading_decision', 'ichimoku_analysis']

    @property
    def component_keys(self) -> List[str]:
        return [
            "tenkan_kijun_relationship", "price_vs_cloud", "cloud_analysis",
            "chikou_span", "time_theory", "wave_theory", "price_level_theory"
        ] 