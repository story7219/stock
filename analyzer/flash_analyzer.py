# -*- coding: utf-8 -*-
"""
FlashStockAI: AI 주식 분석 통합 엔진 (v3.0)
- Gemini 1.5 Flash를 활용한 고속/고품질 멀티모달 분석 (텍스트, 차트 이미지)
- 상세한 프롬프트 엔지니어링을 통한 심층 분석 및 구체적인 투자 전략 제시
- 비동기 데이터 처리를 통한 성능 최적화
- 분석 결과의 구조화된 데이터(JSON) 및 음성 브리핑 제공
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
from PIL import Image
import io

# Rich 라이브러리 추가 (향상된 터미널 출력용)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Gemini API는 핵심 의존성이므로 바깥으로 이동
import google.generativeai as genai

try:
    import config
    from core.core_trader import CoreTrader
    from data_providers.dart_api import DartApiHandler
    from news_collector import NewsCollector, NewsItem
    from database_manager import DatabaseManager
except ImportError as e:
    print(f"모듈 임포트 중 오류 발생: {e}. 경로 설정을 확인하세요.")
    # 웹 앱 환경에서는 아래 클래스들이 정상적으로 주입되므로, 직접 실행 시 오류 방지를 위함
    class CoreTrader: pass
    class DartApiHandler: pass
    class NewsCollector: pass
    class DatabaseManager: pass

logger = logging.getLogger(__name__)

class FlashStockAIAnalyzer:
    """
    Gemini 1.5 Flash AI를 사용하여 주식의 다각적, 멀티모달 분석을 수행하는 통합 분석 클래스.
    """
    def __init__(self, trader: CoreTrader, dart_handler: DartApiHandler, news_collector: NewsCollector, db_manager: DatabaseManager):
        self.trader = trader
        self.dart_handler = dart_handler
        self.news_collector = news_collector
        self.db_manager = db_manager
        self.console = Console()
        
        if not config.GEMINI_API_KEY:
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        # 멀티모달 분석이 가능한 gemini-1.5-flash 모델을 명시적으로 사용
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ FlashStockAIAnalyzer 초기화 완료. AI 분석 준비 완료.")

    async def analyze_stock_from_text(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        단일 종목에 대한 텍스트 기반 심층 분석을 수행합니다.
        """
        logger.info(f"--- 📈 종목 [{stock_code}] 텍스트 기반 분석 시작 ---")
        
        data = await self._gather_individual_data(stock_code)
        if not data:
            logger.error(f"[{stock_code}] 데이터 수집 실패로 분석 중단.")
            return None
            
        prompt = self._generate_analysis_prompt(stock_code, data)
        
        try:
            logger.info(f"[{stock_code}] Gemini API로 텍스트 분석 요청...")
            response = await self.gemini_model.generate_content_async(prompt)
            
            analysis_result = self._parse_analysis_response(stock_code, response.text)
            await self._save_analysis_to_db(stock_code, "text_analysis", analysis_result)
            
            logger.info(f"--- ✅ 종목 [{stock_code}] 텍스트 분석 완료 (종합 점수: {analysis_result.get('overall_score', 'N/A')}/100) ---")
            self.print_analysis_to_terminal(analysis_result)
            return analysis_result

        except Exception as e:
            logger.error(f"❌ [{stock_code}] Gemini 텍스트 분석 중 오류: {e}", exc_info=True)
            return None

    async def analyze_stock_from_image(self, stock_code: str, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        단일 종목에 대한 차트 이미지 기반 심층 분석을 수행합니다.
        """
        logger.info(f"--- 🖼️ 종목 [{stock_code}] 이미지 기반 분석 시작 ---")
        
        # 이미지와 함께 분석할 최소한의 텍스트 데이터 수집
        data = await self._gather_individual_data(stock_code, include_financials=False, include_news=False)
        if not data:
            logger.error(f"[{stock_code}] 이미지 분석을 위한 기본 데이터 수집 실패.")
            return None
        
        prompt = self._generate_analysis_prompt(stock_code, data, is_image_analysis=True)
        
        try:
            # Pillow를 사용하여 이미지 열기
            chart_image = Image.open(io.BytesIO(image_bytes))
            
            logger.info(f"[{stock_code}] Gemini API로 이미지+텍스트 멀티모달 분석 요청...")
            response = await self.gemini_model.generate_content_async([prompt, chart_image])
            
            analysis_result = self._parse_analysis_response(stock_code, response.text)
            await self._save_analysis_to_db(stock_code, "image_analysis", analysis_result)

            logger.info(f"--- ✅ 종목 [{stock_code}] 이미지 분석 완료 (종합 점수: {analysis_result.get('overall_score', 'N/A')}/100) ---")
            self.print_analysis_to_terminal(analysis_result)
            return analysis_result

        except Exception as e:
            logger.error(f"❌ [{stock_code}] Gemini 이미지 분석 중 오류: {e}", exc_info=True)
            return None

    async def _gather_individual_data(self, stock_code: str, include_financials: bool = True, include_news: bool = True) -> Optional[Dict[str, Any]]:
        """
        단일 종목 분석에 필요한 모든 데이터를 비동기적으로 수집합니다.
        분석 유형에 따라 데이터 수집 범위를 조절합니다.
        """
        tasks = {
            "price": self.trader.get_current_price(stock_code),
            "tech_indicators": self.trader.get_technical_indicators(stock_code)
        }
        if include_financials:
            tasks["financials"] = asyncio.to_thread(self.dart_handler.get_financials_for_last_quarters, stock_code)
        if include_news:
            tasks["news"] = asyncio.to_thread(self.news_collector.get_realtime_news, keywords=[stock_code], limit=10)
            
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        data = dict(zip(tasks.keys(), results))
        
        for key, value in data.items():
            if isinstance(value, Exception):
                logger.warning(f"⚠️ [{stock_code}] '{key}' 데이터 수집 중 오류: {value}")
                data[key] = None
        return data if any(v is not None for v in data.values()) else None

    def _generate_analysis_prompt(self, stock_code: str, data: Dict[str, Any], is_image_analysis: bool = False) -> str:
        """
        텍스트 및 이미지 분석을 위한 통합 프롬프트를 생성합니다.
        """
        # 데이터 전처리
        price_info = data.get('price') or {}
        tech_indicators_df = data.get('tech_indicators')
        tech_indicators = tech_indicators_df.iloc[-1].to_dict() if tech_indicators_df is not None and not tech_indicators_df.empty else {}
        
        # Prompt Body 구성
        prompt_body = f"""
        **종목 코드:** {stock_code}
        **현재 주가 정보:** {json.dumps(price_info, ensure_ascii=False)}
        **핵심 기술적 지표:** {json.dumps(tech_indicators, ensure_ascii=False, default=str)}
        """

        if not is_image_analysis:
            financials = data.get('financials') or {}
            latest_quarter_key = next(iter(financials.keys()), "N/A")
            latest_financials = financials.get(latest_quarter_key, [])
            news_list: List[NewsItem] = data.get('news', [])
            
            prompt_body += f"""
            **최신 재무제표 요약 ({latest_quarter_key}):** {json.dumps(latest_financials[:3], indent=2, ensure_ascii=False)}
            **최신 관련 뉴스 (긍정/부정 스코어 포함):** {json.dumps([{"title": n.title, "sentiment_score": round(n.sentiment_score, 2)} for n in news_list], ensure_ascii=False)}
            """
        
        analysis_instructions = """
        **[기술적 분석]** (차트 이미지 또는 제공된 지표 기반)
         - 현재 추세: (상승/하락/횡보 및 근거)
         - 주요 지지/저항선: (가격대 명시)
         - 기술적 종합 의견 및 점수: (0-100점)
        """
        if not is_image_analysis:
            analysis_instructions += """
            **[기본적 분석 (펀더멘탈)]**
             - 재무 건전성: (성장성, 수익성, 안정성 종합 평가)
             - 기업 모멘텀: (최신 뉴스와 공시를 바탕으로 한 성장 동력 평가)
             - 펀더멘탈 종합 의견 및 점수: (0-100점)
            """

        return f"""
        당신은 대한민국 최고의 AI 주식 투자 전략가 'FlashStockAI'입니다. 제시된 모든 데이터를 종합하여, 다음 종목에 대한 명확하고 실행 가능한 투자 전략을 제시해주세요. 반드시 지정된 JSON 형식으로만 응답해야 합니다.

        ### 1. 분석 대상 정보
        {prompt_body}
        
        {"**[참고]** 제공된 차트 이미지를 주의 깊게 분석하여 기술적 분석에 반영해주세요." if is_image_analysis else ""}

        ### 2. 분석 및 평가 지시사항
        {analysis_instructions}

        ### 3. 최종 투자 전략 수립 (JSON 형식으로 출력)
        아래 명시된 키(key)를 사용하여 JSON 객체를 생성해주세요. 각 값은 명확하고 간결하게 작성해주세요.

        ```json
        {{
          "stock_code": "{stock_code}",
          "analysis_type": "{"Image-based" if is_image_analysis else "Text-based"}",
          "technical_score": <기술적 분석 점수 (정수)>,
          "fundamental_score": <{"N/A" if is_image_analysis else "펀더멘탈 분석 점수 (정수)"}>,
          "overall_score": <종합 점수 (정수, 기술적/펀더멘탈 점수를 가중 평균)>,
          "investment_opinion": "<매수/중립/매도 등 명확한 의견>",
          "strategy": {{
            "summary": "<투자 전략 한 줄 요약>",
            "entry_price": "<추천 진입 가격대>",
            "target_price": "<목표 수익 실현 가격대>",
            "stop_loss": "<손절 가격대>"
          }},
          "reasoning": {{
            "positive_factors": [
              "<긍정적 요인 1>",
              "<긍정적 요인 2>"
            ],
            "negative_factors": [
              "<부정적 요인 1>",
              "<부정적 요인 2>"
            ]
          }}
        }}
        ```
        """.strip()

    def _parse_analysis_response(self, stock_code: str, raw_text: str) -> Dict:
        """Gemini 응답 텍스트에서 JSON 블록을 추출하고 파싱합니다."""
        logger.debug(f"[{stock_code}] 파싱할 원본 응답: {raw_text}")
        json_match = re.search(r'```json\n({.*?})\n```', raw_text, re.DOTALL)
        
        if not json_match:
            logger.error(f"❌ [{stock_code}] 분석 결과에서 유효한 JSON 블록을 찾지 못했습니다.")
            # JSON을 찾지 못했을 경우, 텍스트라도 반환 시도
            return {"error": "Failed to parse AI response.", "raw_text": raw_text}
            
        try:
            parsed_json = json.loads(json_match.group(1))
            logger.info(f"✅ [{stock_code}] AI 응답 파싱 성공.")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"❌ [{stock_code}] JSON 파싱 오류: {e}\n원본 텍스트: {json_match.group(1)}")
            return {"error": "JSON Decode Error", "raw_text": json_match.group(1)}

    async def _save_analysis_to_db(self, stock_code: str, analysis_type: str, result: Dict):
        """분석 결과를 데이터베이스에 저장합니다."""
        if "error" in result:
            logger.warning(f"[{stock_code}] 오류가 포함된 분석 결과는 DB에 저장하지 않습니다.")
            return

        await self.db_manager.save_analysis_result(
            stock_code=stock_code,
            analysis_type=analysis_type,
            overall_score=result.get('overall_score'),
            investment_opinion=result.get('investment_opinion'),
            target_price=result.get('strategy', {}).get('target_price'),
            stop_loss=result.get('strategy', {}).get('stop_loss'),
            raw_response=json.dumps(result, ensure_ascii=False)
        )
        logger.info(f"💾 [{stock_code}] {analysis_type} 분석 결과를 데이터베이스에 저장했습니다.")
    
    def print_analysis_to_terminal(self, result: Optional[Dict[str, Any]]):
        """분석 결과를 보기 좋은 표 형태로 터미널에 출력합니다."""
        if not result or "error" in result:
            self.console.print(Panel("[bold red]AI 분석 결과를 처리하는 중 오류가 발생했습니다.[/bold red]", title="오류", border_style="red"))
            return

        title = f"FlashStockAI 분석 결과: [bold cyan]{result.get('stock_code')}[/bold cyan] ({result.get('analysis_type')})"
        
        table = Table(title=title, show_header=True, header_style="bold magenta", box=None)
        table.add_column("항목", style="dim", width=20)
        table.add_column("내용", style="bold")

        table.add_row("종합 점수", f"[bold yellow]{result.get('overall_score', 'N/A')} / 100[/bold yellow]")
        table.add_row("투자 의견", f"[bold green]{result.get('investment_opinion', 'N/A')}[/bold green]")
        table.add_row("-" * 20, "-" * 50)
        
        strategy = result.get('strategy', {})
        table.add_row("전략 요약", strategy.get('summary', 'N/A'))
        table.add_row("진입 가격", strategy.get('entry_price', 'N/A'))
        table.add_row("목표 가격", f"[green]{strategy.get('target_price', 'N/A')}[/green]")
        table.add_row("손절 가격", f"[red]{strategy.get('stop_loss', 'N/A')}[/red]")
        table.add_row("-" * 20, "-" * 50)

        reasoning = result.get('reasoning', {})
        table.add_row("[green]긍정적 요인[/green]", "\n".join(f"- {item}" for item in reasoning.get('positive_factors', [])))
        table.add_row("[red]부정적 요인[/red]", "\n".join(f"- {item}" for item in reasoning.get('negative_factors', [])))
        
        self.console.print(table)

# 모듈 직접 실행 시 테스트를 위한 main 함수 (예시)
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 의존성 모듈 초기화 (실제 환경에서는 외부에서 주입)
    # 테스트를 위해 Mock 객체나 실제 객체를 설정해야 합니다.
    # 예: trader = CoreTrader(...) 등
    # 여기서는 실행되지 않도록 pass 처리
    pass

if __name__ == '__main__':
    # asyncio.run(main())
    print("FlashStockAIAnalyzer 모듈이 로드되었습니다.")
    print("웹 애플리케이션(app.py)을 통해 실행해주세요.") 