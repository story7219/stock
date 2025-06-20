import json
from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class PeterLynchStrategy(BaseStrategy):
    """
    피터 린치의 '성장주를 합리적인 가격에' (GARP) 투자 전략입니다.
    개인 투자자가 이해하기 쉬운 비즈니스를 가진 성장 기업을 발굴합니다.
    """
    
    @property
    def style_name(self) -> str:
        return "peter_lynch"

    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """피터 린치 GARP 전략으로 주식을 분석합니다."""
        try:
            stock_code = stock_data.get('stock_code', 'Unknown')
            company_name = stock_data.get('company_name', 'Unknown')
            
            analysis_result = self._calculate_garp_score(stock_data)
            
            return {
                "strategy": "peter_lynch",
                "stock_code": stock_code,
                "company_name": company_name,
                "score": analysis_result["score"],
                "recommendation": analysis_result["recommendation"],
                "analysis": analysis_result["analysis"],
                "timestamp": stock_data.get("timestamp", "")
            }
            
        except Exception as e:
            return {
                "strategy": "peter_lynch",
                "stock_code": stock_data.get('stock_code', 'Unknown'),
                "score": 50,
                "recommendation": "보류",
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _calculate_garp_score(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """GARP 기준으로 점수를 계산합니다."""
        try:
            score = 50
            analysis_details = []
            
            financial_ratios = stock_data.get('financial_ratios', {})
            
            # 성장률 분석
            eps_growth = financial_ratios.get('eps_growth', 0)
            if eps_growth > 15:
                score += 20
                analysis_details.append("높은 성장률 (+20점)")
            elif eps_growth > 10:
                score += 15
                analysis_details.append("양호한 성장률 (+15점)")
            elif eps_growth > 5:
                score += 10
                analysis_details.append("적정 성장률 (+10점)")
            
            # PEG 비율 분석 (PER/성장률)
            per = financial_ratios.get('per', 0)
            if eps_growth > 0 and per > 0:
                peg = per / eps_growth
                if peg < 1:
                    score += 15
                    analysis_details.append("PEG 비율 우수 (+15점)")
                elif peg < 1.5:
                    score += 10
                    analysis_details.append("PEG 비율 양호 (+10점)")
            
            # ROE 분석
            roe = financial_ratios.get('roe', 0)
            if roe > 12:
                score += 10
                analysis_details.append("ROE 우수 (+10점)")
            elif roe > 8:
                score += 5
                analysis_details.append("ROE 양호 (+5점)")
            
            # 부채비율 분석
            debt_ratio = financial_ratios.get('debt_ratio', 100)
            if debt_ratio < 40:
                score += 5
                analysis_details.append("부채비율 양호 (+5점)")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "매수"
                analysis = f"GARP 전략 기준 우수한 성장주입니다. {', '.join(analysis_details)}"
            elif score >= 50:
                recommendation = "보유"
                analysis = f"GARP 전략 기준 보통 수준입니다. {', '.join(analysis_details)}"
            else:
                recommendation = "매도"
                analysis = f"GARP 전략 기준 투자 매력도가 낮습니다. {', '.join(analysis_details)}"
            
            return {"score": score, "recommendation": recommendation, "analysis": analysis}
            
        except Exception as e:
            return {"score": 50, "recommendation": "보류", "analysis": f"점수 계산 중 오류: {str(e)}"}

    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        피터 린치 스타일의 GARP 분석을 위한 프롬프트를 생성합니다.
        """
        company_name = comprehensive_data.get('company_name', 'N/A')
        sector = comprehensive_data.get('sector', 'N/A')
        financial_ratios = json.dumps(comprehensive_data.get('financial_ratios', {}), ensure_ascii=False)
        price_metrics = json.dumps(comprehensive_data.get('price_metrics', {}), ensure_ascii=False)
        trading_data = json.dumps(comprehensive_data.get('trading_data', {}), ensure_ascii=False)
        technical_indicators = json.dumps(comprehensive_data.get('technical_indicators', {}), ensure_ascii=False)

        return f"""
        [SYSTEM]
        You are an analyst following Peter Lynch's investment philosophy. You are looking for "ten-baggers" by finding growth companies at a reasonable price (GARP). Your analysis must be practical and grounded in common sense. Provide your final analysis ONLY in Korean JSON format as specified below.

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
        Analyze the company based on Peter Lynch's key principles. For each principle, provide a score from 0 to 100 and a brief rationale.
        The 'story_score' must be critically evaluated using the 'Recent News & Disclosures' to understand the current narrative, product developments, and competitive landscape.

        [JSON Output Format]
        {{
          "total_score": "A score from 0 to 100 evaluating the overall alignment with Lynch's GARP strategy.",
          "company_category": "One of Lynch's six categories: Slow Grower, Stalwart, Fast Grower, Cyclical, Turnaround, or Asset Play.",
          "component_scores": {{
            "GARP_score": "Score (0-100) for Growth at a Reasonable Price. Focus on the PEG ratio. Is it below 1.0?",
            "story_score": "Score (0-100) for the company's story. Is it a simple, understandable business? Is there something 'dull' or 'unfashionable' about it?",
            "fundamentals_score": "Score (0-100) for financial health. Check for low debt-to-equity, and that inventory growth is not outpacing sales growth.",
            "insider_activity_score": "Score (0-100) based on insider buying and share buybacks."
          }},
          "investment_decision": {{
            "confidence": "High, Medium, or Low, representing the conviction level for the investment.",
            "summary": "A summary of why this company is or is not a good Peter Lynch-style investment."
          }},
          "rationale": "A comprehensive, final investment thesis explaining the company's story, its growth prospects relative to its price, and why it has the potential to be a multi-bagger."
        }}
        """

    @property
    def required_keys(self) -> List[str]:
        return ['company_category']

    @property
    def component_keys(self) -> List[str]:
        return [
            "GARP_score", "story_score", 
            "fundamentals_score", "insider_activity_score"
        ] 