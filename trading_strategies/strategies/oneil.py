import json
from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class WilliamOneilStrategy(BaseStrategy):
    """
    윌리엄 오닐의 CAN SLIM 투자 전략입니다.
    강력한 성장 모멘텀을 가진 주식을 발굴하며, 
    차트 패턴 분석을 통한 최적 매수 타이밍을 제시합니다.
    """
    
    @property
    def style_name(self) -> str:
        return "oneil"

    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        윌리엄 오닐 CAN SLIM 전략으로 주식을 분석합니다.
        
        Args:
            stock_data: 주식 데이터
            
        Returns:
            Dict: 분석 결과
        """
        try:
            # 기본 정보 추출
            stock_code = stock_data.get('stock_code', 'Unknown')
            company_name = stock_data.get('company_name', 'Unknown')
            
            # 뉴스 데이터 준비
            news_data = stock_data.get('news_data', '뉴스 정보 없음')
            if isinstance(news_data, list):
                news_data = '\n'.join([str(item) for item in news_data])
            
            # CAN SLIM 점수 계산
            analysis_result = self._calculate_canslim_score(stock_data)
            
            return {
                "strategy": "william_oneil",
                "stock_code": stock_code,
                "company_name": company_name,
                "score": analysis_result["score"],
                "recommendation": analysis_result["recommendation"],
                "analysis": analysis_result["analysis"],
                "timestamp": stock_data.get("timestamp", "")
            }
            
        except Exception as e:
            return {
                "strategy": "william_oneil",
                "stock_code": stock_data.get('stock_code', 'Unknown'),
                "score": 50,
                "recommendation": "보류",
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _calculate_canslim_score(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        CAN SLIM 기준으로 점수를 계산합니다.
        """
        try:
            score = 0
            analysis_details = []
            
            financial_ratios = stock_data.get('financial_ratios', {})
            technical_indicators = stock_data.get('technical_indicators', {})
            
            # C - Current Earnings (현재 실적)
            eps_growth = financial_ratios.get('eps_growth', 0)
            if eps_growth > 25:
                score += 20
                analysis_details.append("현재 실적 우수 (+20점)")
            elif eps_growth > 10:
                score += 10
                analysis_details.append("현재 실적 양호 (+10점)")
            
            # A - Annual Earnings (연간 실적)
            roe = financial_ratios.get('roe', 0)
            if roe > 15:
                score += 15
                analysis_details.append("연간 실적 우수 (+15점)")
            elif roe > 10:
                score += 8
                analysis_details.append("연간 실적 양호 (+8점)")
            
            # N - New (신제품, 새로운 최고가)
            current_price = stock_data.get('current_price', 0)
            high_52w = stock_data.get('high_52w', 0)
            if current_price > high_52w * 0.95:  # 52주 최고가 근처
                score += 15
                analysis_details.append("52주 최고가 근처 (+15점)")
            elif current_price > high_52w * 0.85:
                score += 8
                analysis_details.append("52주 최고가 권 (+8점)")
            
            # S - Supply and Demand (수급)
            volume_ratio = technical_indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                score += 15
                analysis_details.append("거래량 급증 (+15점)")
            elif volume_ratio > 1.2:
                score += 8
                analysis_details.append("거래량 증가 (+8점)")
            
            # L - Leader (업계 선도주)
            market_cap = stock_data.get('market_cap', 0)
            if market_cap > 1000000000000:  # 1조원 이상
                score += 10
                analysis_details.append("대형주 리더십 (+10점)")
            elif market_cap > 500000000000:  # 5천억원 이상
                score += 5
                analysis_details.append("중대형주 (+5점)")
            
            # I - Institutional Sponsorship (기관 투자)
            # 기관 보유 비율이 높으면 가점 (데이터가 있다면)
            score += 5  # 기본 점수
            analysis_details.append("기관 투자 기본 점수 (+5점)")
            
            # M - Market Direction (시장 방향)
            rsi = technical_indicators.get('rsi', 50)
            if 40 <= rsi <= 60:
                score += 10
                analysis_details.append("시장 방향 양호 (+10점)")
            elif 30 <= rsi <= 70:
                score += 5
                analysis_details.append("시장 방향 보통 (+5점)")
            
            # 점수 범위 제한
            score = max(0, min(100, score))
            
            # 추천 결정
            if score >= 70:
                recommendation = "매수"
                analysis = f"CAN SLIM 전략 기준 강력한 매수 추천입니다. {', '.join(analysis_details)}"
            elif score >= 50:
                recommendation = "보유"
                analysis = f"CAN SLIM 전략 기준 보유 추천입니다. {', '.join(analysis_details)}"
            else:
                recommendation = "매도"
                analysis = f"CAN SLIM 전략 기준 매수 매력도가 낮습니다. {', '.join(analysis_details)}"
            
            return {
                "score": score,
                "recommendation": recommendation,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "score": 50,
                "recommendation": "보류",
                "analysis": f"CAN SLIM 점수 계산 중 오류: {str(e)}"
            }

    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        윌리엄 오닐 스타일의 CAN SLIM + 차트 패턴 분석을 위한 프롬프트를 생성합니다.
        """
        company_name = comprehensive_data.get('company_name', 'N/A')
        sector = comprehensive_data.get('sector', 'N/A')
        financial_ratios = json.dumps(comprehensive_data.get('financial_ratios', {}), ensure_ascii=False)
        price_metrics = json.dumps(comprehensive_data.get('price_metrics', {}), ensure_ascii=False)
        trading_data = json.dumps(comprehensive_data.get('trading_data', {}), ensure_ascii=False)
        technical_indicators = json.dumps(comprehensive_data.get('technical_indicators', {}), ensure_ascii=False)

        return f"""
        [SYSTEM]
        You are an expert analyst specializing in William O'Neil's CAN SLIM methodology combined with his advanced chart pattern analysis from "How to Make Money in Stocks". Your analysis must identify winning chart patterns and optimal buy points. Provide your final analysis ONLY in Korean JSON format as specified below.

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
        Analyze the company based on CAN SLIM criteria AND O'Neil's chart pattern methodology. Focus on:
        1. CAN SLIM fundamentals (7 criteria)
        2. Chart pattern recognition (Cup with Handle, Flat Base, Double Bottom, etc.)
        3. Base building analysis (7-week minimum consolidation)
        4. Volume characteristics during breakouts
        5. Relative strength vs market
        6. Proper buy point identification

        Key O'Neil Chart Concepts to Evaluate:
        - Base Formation: Has the stock built a proper base (7+ weeks)?
        - Volume Surge: 40-50% above average on breakout?
        - Relative Strength: Is RS line at/near highs?
        - Market Leadership: Is this a market leader breaking out?

        [JSON Output Format]
        {{
          "total_score": "A score from 0 to 100 evaluating the overall CAN SLIM + Chart Pattern alignment.",
          "component_scores": {{
            "C_current_earnings": "Score (0-100) for Current Big or Accelerating Quarterly Earnings and Sales per Share.",
            "A_annual_earnings": "Score (0-100) for Big Annual Earnings Growth.",
            "N_new": "Score (0-100) for New Products, New Management, New Highs.",
            "S_supply_demand": "Score (0-100) for Supply and Demand. Look for heavy-volume accumulation by institutions.",
            "L_leader_laggard": "Score (0-100) for Leader or Laggard? Is this the leading stock in its industry?",
            "I_institutional_sponsorship": "Score (0-100) for Institutional Sponsorship.",
            "M_market_direction": "Score (0-100) for Market Direction.",
            "chart_pattern_score": "Score (0-100) for chart pattern quality and base formation.",
            "breakout_potential": "Score (0-100) for proximity to proper buy point and breakout potential."
          }},
          "chart_analysis": {{
            "pattern_type": "Identified chart pattern (Cup with Handle, Flat Base, Double Bottom, Ascending Base, etc.)",
            "base_length": "Length of current base formation in weeks",
            "buy_point": "Specific price level for proper buy point entry",
            "volume_requirement": "Required volume increase % for valid breakout",
            "relative_strength": "Assessment of relative strength vs market",
            "pattern_quality": "Quality rating of the chart pattern (Excellent, Good, Fair, Poor)"
          }},
          "investment_decision": {{
            "confidence": "High, Medium, or Low, representing the conviction level for the investment.",
            "buy_timing": "Immediate, Wait for Breakout, or Avoid",
            "position_size": "Recommended position size (Small, Medium, Large) based on pattern quality",
            "summary": "A summary of the CAN SLIM + Chart Pattern analysis."
          }},
          "rationale": "A comprehensive investment thesis integrating CAN SLIM fundamentals with O'Neil's chart pattern methodology. Explain why this stock is or is not a winning growth stock candidate with proper chart setup."
        }}
        """

    @property
    def required_keys(self) -> List[str]:
        return ['chart_analysis']

    @property
    def component_keys(self) -> List[str]:
        return [
            "C_current_earnings", "A_annual_earnings", "N_new", 
            "S_supply_demand", "L_leader_laggard", "I_institutional_sponsorship", 
            "M_market_direction", "chart_pattern_score", "breakout_potential"
        ] 