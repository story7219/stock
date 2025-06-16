class AdvancedTrader:
    def __init__(self, kis_api, market_analyzer):
        # ... existing code ...
        
        # 오닐 투자철학 통합
        self.oneil_analyzer = OneilAnalyzer()
        
    async def make_trading_decision(self, symbol: str) -> Dict:
        """오닐 원칙이 통합된 매매 결정"""
        try:
            # 기존 분석
            market_analysis = await self.market_analyzer.get_comprehensive_analysis()
            stock_data = await self._get_comprehensive_stock_data(symbol)
            
            # 오닐 원칙 분석 추가
            oneil_analysis = await self.oneil_analyzer.analyze_stock(symbol, stock_data)
            
            # 통합 분석 프롬프트
            integrated_prompt = f"""
당신은 윌리엄 오닐의 CAN SLIM 방법론을 완벽히 마스터한 AI 트레이더입니다.

=== 시장 전체 분석 ===
{json.dumps(market_analysis, ensure_ascii=False, indent=2)}

=== 종목 데이터: {symbol} ===
{json.dumps(stock_data, ensure_ascii=False, indent=2)}

=== 오닐 CAN SLIM 분석 ===
{json.dumps(oneil_analysis, ensure_ascii=False, indent=2)}

오닐의 투자 원칙을 바탕으로 매매 결정을 내려주세요:

**핵심 체크리스트:**
1. C - 최근 분기 EPS 25% 이상 증가?
2. A - 연간 EPS 3년간 25% 이상 성장?
3. N - 신제품/서비스/경영진 변화?
4. S - 주식 수급 (유통주식 감소, 기관 매수)?
5. L - 업종 1-2위 리더 종목?
6. I - 기관 투자자 3분기 연속 순매수?
7. M - 시장 전체 상승 추세?

**차트 패턴 확인:**
- 컵앤핸들 패턴
- 플랫베이스 패턴  
- 더블보텀 패턴
- 하이타이트베이스 패턴

**매수 타이밍:**
- 베이스 돌파 시점
- 거래량 급증 확인
- 상대강도 80 이상

JSON 형식으로 응답:
{{
    "can_slim_score": 0-7,
    "individual_scores": {{
        "C": 0-1, "A": 0-1, "N": 0-1, "S": 0-1,
        "L": 0-1, "I": 0-1, "M": 0-1
    }},
    "chart_pattern": "패턴명",
    "pattern_stage": "베이스구축중/돌파임박/돌파완료",
    "buy_signal": true/false,
    "buy_point": 매수가격,
    "stop_loss": 손절가격,
    "target_price": 목표가격,
    "risk_reward_ratio": 비율,
    "confidence": 0.0-1.0,
    "reasoning": "상세 분석 근거"
}}
"""
            
            response = await self.model.generate_content_async(integrated_prompt)
            return self._parse_ai_response(response.text)
            
        except Exception as e:
            self.logger.error(f"통합 매매 결정 오류: {e}")
            return {} 