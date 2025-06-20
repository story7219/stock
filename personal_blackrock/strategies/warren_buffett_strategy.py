import json
from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class WarrenBuffettStrategy(BaseStrategy):
    """
    워렌 버핏의 가치 투자 전략입니다.
    강력한 경제적 해자를 가진 훌륭한 기업을 합리적인 가격에 매수합니다.
    """
    
    @property
    def style_name(self) -> str:
        return "buffett"

    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """워렌 버핏 가치 투자 전략으로 주식을 분석합니다."""
        try:
            stock_code = stock_data.get('stock_code', 'Unknown')
            company_name = stock_data.get('company_name', 'Unknown')
            
            analysis_result = self._calculate_value_score(stock_data)
            
            return {
                "strategy": "warren_buffett",
                "stock_code": stock_code,
                "company_name": company_name,
                "score": analysis_result["score"],
                "recommendation": analysis_result["recommendation"],
                "analysis": analysis_result["analysis"],
                "timestamp": stock_data.get("timestamp", "")
            }
            
        except Exception as e:
            return {
                "strategy": "warren_buffett",
                "stock_code": stock_data.get('stock_code', 'Unknown'),
                "score": 50,
                "recommendation": "보류",
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _calculate_value_score(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """가치 투자 기준으로 점수를 계산합니다."""
        try:
            score = 50
            analysis_details = []
            
            financial_ratios = stock_data.get('financial_ratios', {})
            
            # ROE 분석
            roe = financial_ratios.get('roe', 0)
            if roe > 15:
                score += 20
                analysis_details.append("ROE 우수 (+20점)")
            elif roe > 10:
                score += 10
                analysis_details.append("ROE 양호 (+10점)")
            
            # PER 분석 (저평가 선호)
            per = financial_ratios.get('per', 0)
            if 5 <= per <= 15:
                score += 15
                analysis_details.append("적정 PER 범위 (+15점)")
            elif per <= 20:
                score += 8
                analysis_details.append("PER 양호 (+8점)")
            
            # 부채비율 분석 (낮을수록 좋음)
            debt_ratio = financial_ratios.get('debt_ratio', 100)
            if debt_ratio < 30:
                score += 15
                analysis_details.append("부채비율 우수 (+15점)")
            elif debt_ratio < 50:
                score += 10
                analysis_details.append("부채비율 양호 (+10점)")
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "매수"
                analysis = f"가치 투자 기준 우수한 투자 대상입니다. {', '.join(analysis_details)}"
            elif score >= 50:
                recommendation = "보유"
                analysis = f"가치 투자 기준 보통 수준입니다. {', '.join(analysis_details)}"
            else:
                recommendation = "매도"
                analysis = f"가치 투자 기준 투자 매력도가 낮습니다. {', '.join(analysis_details)}"
            
            return {"score": score, "recommendation": recommendation, "analysis": analysis}
            
        except Exception as e:
            return {"score": 50, "recommendation": "보류", "analysis": f"점수 계산 중 오류: {str(e)}"}

    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        워렌 버핏 스타일의 가치 투자 분석을 위한 프롬프트를 생성합니다.
        """
        company_name = comprehensive_data.get('company_name', 'N/A')
        sector = comprehensive_data.get('sector', 'N/A')
        financial_ratios = json.dumps(comprehensive_data.get('financial_ratios', {}), ensure_ascii=False)
        price_metrics = json.dumps(comprehensive_data.get('price_metrics', {}), ensure_ascii=False)
        trading_data = json.dumps(comprehensive_data.get('trading_data', {}), ensure_ascii=False)
        technical_indicators = json.dumps(comprehensive_data.get('technical_indicators', {}), ensure_ascii=False)

        return f"""
        [SYSTEM]
        You are an analyst embodying Warren Buffett's value investing philosophy. Your focus is on long-term ownership of wonderful businesses at a fair price. Provide your final analysis ONLY in Korean JSON format as specified below.

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
        Analyze the company based on Warren Buffett's core tenets. For each tenet, provide a score from 0 to 100 and a brief rationale.
        The 'management_tenets' score should be carefully evaluated based on the 'Recent News & Disclosures' for any information regarding management's actions, capital allocation, or shareholder communications.

        [JSON Output Format]
        {{
          "total_score": "A score from 0 to 100 evaluating the overall alignment with Buffett's principles.",
          "component_scores": {{
            "business_tenets": "Score (0-100) for understanding the business (is it simple and understandable?).",
            "management_tenets": "Score (0-100) for rational, candid, and owner-focused management.",
            "financial_tenets": "Score (0-100) for strong financial performance (high return on equity, consistent earnings, low debt).",
            "value_tenets": "Score (0-100) for the attractiveness of the price (is it selling at a significant discount to its intrinsic value?)."
          }},
          "investment_decision": {{
            "confidence": "High, Medium, or Low, representing the conviction level for a long-term investment.",
            "intrinsic_value_estimate": "An estimated range for the company's intrinsic value per share.",
            "margin_of_safety": "The percentage discount of the current price to the estimated intrinsic value."
          }},
          "rationale": "A comprehensive, final investment thesis in the style of a letter to shareholders, explaining why this company is (or is not) a 'wonderful business' worth owning for the long term."
        }}
        """

    @property
    def required_keys(self) -> List[str]:
        return ['intrinsic_value_estimate', 'margin_of_safety']

    @property
    def component_keys(self) -> List[str]:
        return [
            "business_tenets", "management_tenets", 
            "financial_tenets", "value_tenets"
        ] 