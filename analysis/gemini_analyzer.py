# analysis/gemini_analyzer.py
import sys
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 시스템 경로 설정 ---
# 이 파일은 analysis 폴더 안에 있으므로, 프로젝트 루트는 부모 디렉토리의 부모 디렉토리임
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from PIL import Image
from utils.logger import log_event
import config
import json
from typing import Literal
import io

# --- Gemini API 설정 ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY가 환경 변수에 설정되지 않았습니다.")
    
    genai.configure(api_key=API_KEY)

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 1024,
    }
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    GEMINI_INITIALIZED = True
    log_event("INFO", "Gemini AI 분석기 초기화 성공.")

except (ValueError, Exception) as e:
    log_event("CRITICAL", f"Gemini AI 초기화 실패: {e}")
    GEMINI_INITIALIZED = False

# --- 프롬프트 정의 ---
SYSTEM_PROMPT = """
당신은 대한민국 주식 시장의 트레이딩 전문가입니다. 당신의 임무는 주어진 주식 차트 이미지를 분석하여, 단기 트레이딩 관점에서 지금이 '매수(BUY)'하기 좋은 시점인지, 아니면 '보류(HOLD)'해야 하는 시점인지 판단하는 것입니다.

분석 시 다음 사항들을 종합적으로 고려하세요:
1.  **캔들 패턴**: 장대양봉, 망치형, 상승장악형 등 긍정적 신호가 있는지 확인합니다.
2.  **이동평균선(MA)**: 5일(녹색), 20일(주황색), 60일(보라색) 이동평균선이 정배열인지, 골든크로스가 발생했는지, 주가가 주요 이평선 위에 안착했는지 확인합니다.
3.  **거래량(Volume)**: 최근 주가 상승 시 거래량이 동반되었는지 확인합니다. 의미 있는 거래량 증가는 신뢰도를 높입니다.
4.  **추세**: 전반적인 주가 추세가 상승 추세인지, 박스권 횡보 후 상단 돌파 시점인지 평가합니다.

분석 결과는 반드시 아래의 JSON 형식 중 하나로만 답변해야 합니다. 그 외의 설명은 절대 추가하지 마세요.

1.  **매수 추천 시**:
    ```json
    {
      "decision": "BUY",
      "reason": "장대양봉과 함께 거래량이 급증하며 전고점을 돌파하는 강력한 상승 신호가 포착되었습니다."
    }
    ```

2.  **보류 추천 시**:
    ```json
    {
      "decision": "HOLD",
      "reason": "주가가 20일 이동평균선 아래에 있고, 거래량 없이 하락하는 등 추세 전환 신호가 명확하지 않아 관망이 필요합니다."
    }
    ```
"""

def analyze_chart_with_gemini(chart_path: str) -> dict | None:
    """
    주어진 차트 이미지를 Gemini AI로 분석하여 매수/보류 결정을 반환합니다.

    Args:
        chart_path (str): 분석할 차트 이미지 파일의 경로

    Returns:
        dict | None: AI의 분석 결과 (예: {'decision': 'BUY', 'reason': '...'}) 또는 실패 시 None
    """
    if not GEMINI_INITIALIZED:
        log_event("ERROR", "Gemini AI가 초기화되지 않아 분석을 건너뜁니다.")
        return None
        
    if not os.path.exists(chart_path):
        log_event("ERROR", f"차트 파일이 존재하지 않습니다: {chart_path}")
        return None

    try:
        log_event("INFO", f"Gemini AI 차트 분석 시작: {chart_path}")
        img = Image.open(chart_path)
        
        # PIL 이미지를 바이트로 변환
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        
        image_part = {
            "mime_type": "image/png",
            "data": byte_arr.getvalue()
        }
        
        prompt_parts = [SYSTEM_PROMPT, image_part]
        
        response = model.generate_content(prompt_parts)
        
        # 응답 텍스트에서 JSON 부분만 추출
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        analysis_result = json.loads(cleaned_response)

        decision = analysis_result.get("decision")
        if decision not in ["BUY", "HOLD"]:
            raise ValueError(f"예상치 못한 결정 값: {decision}")

        log_event("SUCCESS", f"Gemini 분석 완료: {analysis_result}")
        return analysis_result

    except (json.JSONDecodeError, ValueError) as e:
        log_event("ERROR", f"Gemini 응답 파싱 실패: {e}\n원본 응답: {response.text}")
        return None
    except Exception as e:
        log_event("ERROR", f"Gemini 분석 중 예기치 않은 오류 발생: {e}")
        return None

if __name__ == '__main__':
    # 이 파일을 직접 실행할 경우, 테스트를 위해 프로젝트 루트에 'test_chart.png'가 있다고 가정합니다.
    # 실제로는 chart_generator.py를 통해 생성된 파일을 사용해야 합니다.
    
    # 테스트를 위한 가짜 차트 파일 생성
    from utils.chart_generator import generate_stock_chart
    test_ticker = "005930" # 삼성전자
    print(f"[{test_ticker}] 테스트용 차트 생성 시도...")
    test_chart_path = generate_stock_chart(test_ticker)
    
    if test_chart_path:
        print(f"테스트 차트 생성 성공: {test_chart_path}")
        # Gemini 분석 실행
        result = analyze_chart_with_gemini(test_chart_path)
        if result:
            print("\n[Gemini 분석 결과]")
            print(f"  - 결정: {result.get('decision')}")
            print(f"  - 근거: {result.get('reason')}")
        else:
            print("\nGemini 분석에 실패했습니다.")
    else:
        print("테스트용 차트 생성에 실패하여 분석을 진행할 수 없습니다.") 