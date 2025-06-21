import json
from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class BlackRockStrategy(BaseStrategy):
    """
    기관 투자자 관점의 블랙록 스타일 종합 분석 전략입니다.
    """
    @property
    def style_name(self) -> str:
        return "blackrock"

    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        블랙록 전략으로 주식을 분석합니다.
        
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
            
            # AI 분석 프롬프트 생성
            prompt = self.create_prompt(stock_data, news_data)
            
            # 실제 AI 분석은 여기서 수행되어야 하지만, 
            # 현재는 기본적인 점수 계산으로 대체
            analysis_result = self._calculate_basic_score(stock_data)
            
            return {
                "strategy": "blackrock",
                "stock_code": stock_code,
                "company_name": company_name,
                "score": analysis_result["score"],
                "recommendation": analysis_result["recommendation"],
                "analysis": analysis_result["analysis"],
                "timestamp": stock_data.get("timestamp", ""),
                "prompt": prompt  # 디버깅용
            }
            
        except Exception as e:
            return {
                "strategy": "blackrock",
                "stock_code": stock_data.get('stock_code', 'Unknown'),
                "score": 50,
                "recommendation": "보류",
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "error": str(e)
            }

    def _calculate_basic_score(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        기본적인 점수 계산을 수행합니다.
        
        Args:
            stock_data: 주식 데이터
            
        Returns:
            Dict: 점수 및 추천 정보
        """
        try:
            score = 50  # 기본 점수
            
            # 재무 비율 평가
            financial_ratios = stock_data.get('financial_ratios', {})
            if financial_ratios:
                # ROE가 10% 이상이면 가점
                roe = financial_ratios.get('roe', 0)
                if roe > 10:
                    score += 10
                elif roe > 5:
                    score += 5
                
                # PER이 적정 범위면 가점
                per = financial_ratios.get('per', 0)
                if 10 <= per <= 20:
                    score += 10
                elif 5 <= per <= 30:
                    score += 5
                
                # 부채비율이 낮으면 가점
                debt_ratio = financial_ratios.get('debt_ratio', 100)
                if debt_ratio < 30:
                    score += 10
                elif debt_ratio < 50:
                    score += 5
            
            # 기술적 지표 평가
            technical_indicators = stock_data.get('technical_indicators', {})
            if technical_indicators:
                # RSI가 적정 범위면 가점
                rsi = technical_indicators.get('rsi', 50)
                if 30 <= rsi <= 70:
                    score += 5
                
                # 이동평균선 위에 있으면 가점
                current_price = stock_data.get('current_price', 0)
                ma20 = technical_indicators.get('ma20', 0)
                if current_price > ma20 and ma20 > 0:
                    score += 5
            
            # ESG 관련 평가 (섹터 기반)
            sector = stock_data.get('sector', '')
            if any(keyword in sector.lower() for keyword in ['기술', '바이오', '헬스케어', '신재생']):
                score += 10  # ESG 친화적 섹터 가점
            
            # 점수 범위 제한
            score = max(0, min(100, score))
            
            # 추천 결정
            if score >= 70:
                recommendation = "매수"
                analysis = f"블랙록 전략 기준 우수한 투자 대상입니다. (점수: {score}점)"
            elif score >= 50:
                recommendation = "보유"
                analysis = f"블랙록 전략 기준 보통 수준의 투자 대상입니다. (점수: {score}점)"
            else:
                recommendation = "매도"
                analysis = f"블랙록 전략 기준 투자 매력도가 낮습니다. (점수: {score}점)"
            
            return {
                "score": score,
                "recommendation": recommendation,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "score": 50,
                "recommendation": "보류",
                "analysis": f"점수 계산 중 오류: {str(e)}"
            }

    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        블랙록 스타일의 종합 분석을 위한 프롬프트를 생성합니다.
        """
        company_name = comprehensive_data.get('company_name', 'N/A')
        sector = comprehensive_data.get('sector', 'N/A')
        market_cap = f"{comprehensive_data.get('market_cap', 0):,}"
        financial_ratios = json.dumps(comprehensive_data.get('financial_ratios', {}), ensure_ascii=False)
        price_metrics = json.dumps(comprehensive_data.get('price_metrics', {}), ensure_ascii=False)
        trading_data = json.dumps(comprehensive_data.get('trading_data', {}), ensure_ascii=False)
        technical_indicators = json.dumps(comprehensive_data.get('technical_indicators', {}), ensure_ascii=False)
        
        return f"""
        [SYSTEM]
        You are a top-tier analyst at BlackRock, the world's largest asset manager. Your analysis must be comprehensive, data-driven, and institutional-grade. Provide your final analysis ONLY in Korean JSON format as specified below.

        [Company Information]
        - Company Name: {company_name}
        - Sector: {sector}
        - Market Cap: {market_cap} KRW
        - Financial Ratios: {financial_ratios}
        - Price Metrics: {price_metrics}
        - Supply/Demand Data (Recent 1 Month): {trading_data}
        - Technical Indicators: {technical_indicators}

        [Recent News & Disclosures]
        {news_data}

        [Analysis Task]
        Based on the provided data, perform a comprehensive analysis from a BlackRock perspective. Evaluate the stock's investment attractiveness for a large-scale, long-term fund.
        The 'market_sentiment_score' should be critically evaluated based on the 'Recent News & Disclosures'.

        [JSON Output Format]
        {{
          "total_score": "A score from 0 to 100 evaluating the overall investment attractiveness. Higher is better.",
          "component_scores": {{
            "fundamental_score": "Score (0-100) for financial health, profitability, and valuation.",
            "technical_score": "Score (0-100) based on chart patterns, trends, and momentum.",
            "quantitative_score": "Score (0-100) based on quantitative factors like supply/demand and volatility.",
            "market_sentiment_score": "Score (0-100) considering news, industry trends, and overall market sentiment."
          }},
          "investment_decision": {{
            "confidence": "High, Medium, or Low, representing the conviction level for the investment.",
            "target_weight_percent": "Recommended portfolio weight (0-10%).",
            "target_return_percent": "Expected annual return percentage.",
            "holding_period": "Recommended holding period (e.g., '12-18 months')."
          }},
          "chart_analysis": {{
            "timing_score": "Score (0-10) for the current entry timing attractiveness.",
            "entry_price_band": "Recommended price range for initiating a position (e.g., '85,000 - 90,000 KRW').",
            "support_levels": ["List of key support price levels."],
            "resistance_levels": ["List of key resistance price levels."],
            "timing_rationale": "Brief rationale for the timing score, considering chart patterns and indicators."
          }},
          "risk_metrics": {{
            "overall_risk_rating": "Low, Medium, or High.",
            "key_risks": ["List of primary risks (e.g., 'regulatory changes', 'competitive pressure')."],
            "risk_mitigation_strategy": "Brief strategy to manage identified risks."
          }},
          "monitoring": {{
            "key_monitoring_indicators": ["List of crucial metrics to watch post-investment (e.g., 'quarterly earnings', 'competitor product launches')."]
          }},
          "rationale": "A comprehensive, final investment thesis summarizing why this stock is or is not recommended for a BlackRock fund, integrating all scored components."
        }}
        """

    @property
    def required_keys(self) -> List[str]:
        return ['chart_analysis', 'risk_metrics', 'monitoring']

    @property
    def component_keys(self) -> List[str]:
        return ['fundamental_score', 'technical_score', 'quantitative_score', 'market_sentiment_score'] 