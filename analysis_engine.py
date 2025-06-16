"""
AI 기반 시장 분석 엔진
- Gemini를 활용하여 시장 데이터와 차트 이미지를 종합적으로 분석
- 단기 매매에 대한 투자 의견 제시
"""
import google.generativeai as genai
from PIL import Image
import logging
import config

logger = logging.getLogger(__name__)

try:
    if config.GOOGLE_API_KEY and config.GOOGLE_API_KEY != 'YOUR_GOOGLE_API_KEY':
        genai.configure(api_key=config.GOOGLE_API_KEY)
        logger.info("✅ Google Gemini API가 설정되었습니다.")
    else:
        logger.warning("⚠️ Google Gemini API 키가 설정되지 않았습니다. config.py를 확인해주세요.")
except Exception as e:
    logger.error(f"❌ Google Gemini API 키 설정 중 오류 발생: {e}")

class MarketAnalyzer:
    """Gemini를 활용한 시장 데이터 및 차트 분석기"""
    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        if not getattr(genai, '_client', None):
            raise ValueError("Gemini API 키가 없어 MarketAnalyzer를 초기화할 수 없습니다.")
        self.model = genai.GenerativeModel(model_name)

    def get_trading_insights(self, market_data_text: str, chart_image_path: str) -> str:
        logger.info(f"시장 데이터와 차트({chart_image_path}) 분석을 시작합니다...")
        try:
            img = Image.open(chart_image_path)
        except Exception as e:
            return f"오류: 이미지 파일({chart_image_path})을 열 수 없습니다. {e}"

        prompt = f"""
        당신은 20년 경력의 베테랑 펀드매니저입니다. 주어진 최신 시장 데이터와 차트 이미지를 종합 분석하여, 단기(1~3일) 트레이딩 관점에서 투자 전략을 제시하세요.

        **분석 가이드라인:**
        1. **종합 분석:** 모든 데이터(등락률, 거래량, 수급)와 차트를 함께 고려하세요.
        2. **단기 관점:** 1~3일 내의 움직임에 초점을 맞춥니다.
        3. **명확한 판단:** '매수', '매도', '관망' 중 하나로 명확히 제시하세요.
        4. **핵심 근거:** 판단의 핵심 이유를 1~2가지로 간결하게 요약합니다.
        5. **차트 활용:** 차트 패턴, 캔들, 이동평균선 등을 적극 활용하세요.

        **입력 데이터:**
        ---
        {market_data_text}
        ---

        **분석 요청:**
        위 데이터와 첨부된 차트 이미지를 바탕으로, 아래 형식에 맞춰 주요 종목에 대한 투자 판단을 표로 정리해주세요.

        | 종목명 | 판단(매수/매도/관망) | 근거 요약 |
        |---|---|---|
        """
        try:
            response = self.model.generate_content([prompt, img])
            logger.info("✅ AI 분석이 완료되었습니다.")
            return response.text
        except Exception as e:
            return f"오류: AI 분석 중 문제가 발생했습니다. ({e})" 