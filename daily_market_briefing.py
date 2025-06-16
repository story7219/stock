"""
데일리 마켓 브리핑 및 단기 매매 판단 시스템
- Gemini 1.5 Flash (멀티모달) 모델을 활용하여 주식 시장 데이터와 차트 이미지를 동시 분석
- 단기(1~3일) 관점의 매수/매도/관망 의견 제시 및 근거 요약
- 사용자가 제공한 데이터를 기반으로 작동
"""
import os
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
# 사용자의 구글 AI 스튜디오 API 키를 환경변수 'GOOGLE_API_KEY'에 설정해주세요.
# 예: set GOOGLE_API_KEY=your_api_key
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    logger.error("환경변수 'GOOGLE_API_KEY'가 설정되지 않았습니다. API 키를 설정해주세요.")
    GOOGLE_API_KEY = None
except Exception as e:
    logger.error(f"API 키 설정 중 오류 발생: {e}")
    GOOGLE_API_KEY = None

class DailyMarketAnalyzer:
    """
    시장 데이터와 차트를 분석하여 매매 의견을 생성하는 클래스
    """
    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        """
        초기화 함수
        Args:
            model_name (str): 사용할 Gemini 모델 이름
        """
        if not GOOGLE_API_KEY:
            raise ValueError("Google API 키가 설정되지 않았습니다. 시스템을 실행할 수 없습니다.")
        
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"✅ Gemini 모델 '{model_name}'이(가) 성공적으로 로드되었습니다.")

    def analyze_market_data(self, market_data_text: str, chart_image_path: str) -> str:
        """
        텍스트 데이터와 차트 이미지를 분석하여 매매 판단을 내립니다.

        Args:
            market_data_text (str): 등락률, 거래량, 거래대금, 수급 데이터가 포함된 텍스트
            chart_image_path (str): 분석할 차트 이미지 파일의 경로

        Returns:
            str: AI가 생성한 분석 결과 (마크다운 테이블 형식)
        """
        logger.info("시장 데이터와 차트 이미지 분석을 시작합니다...")
        
        try:
            img = Image.open(chart_image_path)
        except FileNotFoundError:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {chart_image_path}")
            return "오류: 차트 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요."
        except Exception as e:
            logger.error(f"이미지 로딩 중 오류 발생: {e}")
            return "오류: 차트 이미지를 처리하는 중 문제가 발생했습니다."

        # 프롬프트 구성
        prompt = f"""
        당신은 20년 경력의 베테랑 펀드매니저입니다. 당신의 임무는 주어진 최신 시장 데이터와 차트 이미지를 종합적으로 분석하여, 단기(1~3일) 트레이딩 관점에서 투자 전략을 제시하는 것입니다.

        **분석 가이드라인:**
        1.  **종합적 분석:** 제공된 모든 데이터(등락률, 거래량, 거래대금, 외국인/기관 수급)와 차트 이미지를 함께 고려하여 분석합니다.
        2.  **단기 관점:** 1~3일 이내의 단기적인 주가 움직임에 초점을 맞춥니다. 장기적인 가치 평가는 제외합니다.
        3.  **명확한 판단:** 각 종목에 대해 '매수', '매도', '관망' 중 하나의 명확한 의견을 제시해야 합니다.
        4.  **핵심 근거:** 판단의 핵심이 된 이유를 1~2가지로 간결하게 요약합니다. (예: '거래량 급증 및 전고점 돌파 시도', '외국인 대량 순매도 및 추세 이탈')
        5.  **차트 활용:** 차트의 패턴(예: 정배열, 골든크로스, 데드크로스, 지지/저항선), 캔들 모양, 이동평균선의 위치 등을 분석에 적극적으로 활용하세요.

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
            logger.info("Gemini API 호출 중...")
            response = self.model.generate_content([prompt, img])
            logger.info("✅ 분석이 완료되었습니다.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 호출 중 오류 발생: {e}")
            return f"오류: AI 분석 중 문제가 발생했습니다. ({e})"

def get_sample_data_and_image():
    """분석을 위한 샘플 데이터를 반환합니다."""
    
    market_data = """
    ## 📆 오늘의 한국 주식시장 주요 데이터 (샘플)

    ### 📈 등락률 순위 (상위 10)
    1. 미래반도체: +29.8%, 15,200원 (▲3,500) - AI 반도체 신규 계약 발표
    2. 대한바이오: +21.5%, 8,800원 (▲1,550) - 임상 3상 성공 기대감
    3. 한국모빌리티: +15.2%, 110,500원 (▲14,500) - 정부 자율주행 사업 참여

    ### 📊 거래량 순위 (상위 10)
    1. 미래반도체: 50,000,000주 (+29.8%)
    2. 태평양중공업: 35,000,000주 (+5.1%)
    3. 제일화학: 28,000,000주 (-2.3%)

    ### 💰 거래대금 순위 (상위 10)
    1. 미래반도체: 7,000억원 (+29.8%)
    2. 삼성전자: 6,500억원 (+1.2%)
    3. 한국모빌리티: 5,800억원 (+15.2%)

    ### 👨‍💼 외국인/기관 순매수 (금액 기준, 상위)
    1. 삼성전자: 외국인 +1,200억 / 기관 +500억
    2. 한국모빌리티: 외국인 +800억 / 기관 +950억
    3. 미래반도체: 외국인 -50억 / 기관 +300억 (차익 실현)
    """

    sample_image_path = "sample_chart.png"
    try:
        img = Image.new('RGB', (800, 400), color = 'white')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("malgun.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), "샘플 차트 이미지 (미래반도체)\n- 장대양봉 및 거래량 폭발\n- 전고점 돌파", fill=(0,0,0), font=font)
        img.save(sample_image_path)
        logger.info(f"'{sample_image_path}'에 샘플 차트 이미지를 생성했습니다.")
    except Exception as e:
        logger.warning(f"샘플 이미지 생성 중 오류 (Pillow 라이브러리 필요): {e}")
        sample_image_path = None

    return market_data, sample_image_path

if __name__ == "__main__":
    # --- 사용법 ---
    # 1. (최초 1회) 터미널에 `set GOOGLE_API_KEY=your_api_key_here` 입력하여 API 키 설정
    # 2. 아래 `market_data_text`에 실제 시장 데이터를 복사하여 붙여넣습니다.
    # 3. `chart_image_path`에 분석하고 싶은 종목의 차트 이미지 파일 경로를 입력합니다.
    # 4. 터미널에서 `python daily_market_briefing.py`를 실행합니다.
    
    if GOOGLE_API_KEY:
        # 샘플 데이터 가져오기 (실제 사용 시 이 부분을 실제 데이터로 교체)
        market_data_text, chart_image_path = get_sample_data_and_image()
        
        if chart_image_path:
            analyzer = DailyMarketAnalyzer()
            analysis_result = analyzer.analyze_market_data(market_data_text, chart_image_path)
            
            # 결과 출력
            print("\n" + "="*80)
            print("📈 AI 기반 데일리 마켓 브리핑")
            print("="*80 + "\n")
            print(analysis_result)
        else:
            logger.error("차트 이미지가 없어 분석을 진행할 수 없습니다.")
    else:
        print("\n[오류] GOOGLE_API_KEY가 설정되지 않았습니다. 스크립트 상단 또는 환경변수를 확인해주세요.") 