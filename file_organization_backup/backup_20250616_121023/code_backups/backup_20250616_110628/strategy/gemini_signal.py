# gemini_signal.py
# Gemini API를 이용해 트레이딩 신호를 생성하는 모듈

import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils.parser import parse_gemini_signal_response # 새로 만든 파서 import
from utils.logger import log_event # log_event import 추가
import pandas as pd
import config # config 파일 import
from utils.chart_generator import create_chart_image
import base64
from strategy.prompts import get_prompt, PromptType # 팩토리 함수와 타입 임포트

# .env 파일에서 API 키를 로드합니다.
# 이 코드가 실행되는 위치의 상위 디렉토리에 .env 파일이 있어야 합니다.
load_dotenv() 

# Google Generative AI 설정
try:
    # GEMINI_API_KEY 환경 변수를 찾아서 설정하도록 수정
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    genai.configure(api_key=api_key)
except Exception as e:
    log_event("CRITICAL", f"Gemini API 설정 실패: {e}")
    # 시스템의 다른 부분은 계속 작동해야 하므로, 여기서 프로그램을 종료하지는 않습니다.

def get_gemini_trading_signal(df: pd.DataFrame, ticker: str):
    """
    주어진 데이터프레임과 차트 이미지로 Gemini를 호출하여 매매 신호를 받습니다.
    """
    if not genai:
        log_event("ERROR", "Gemini API가 초기화되지 않아 신호 생성을 건너뜁니다.")
        return None

    # 1. 차트 이미지 생성
    chart_file = f"temp_chart_{ticker}.png"
    if not create_chart_image(df, ticker, file_path=chart_file):
        log_event("ERROR", f"[{ticker}] 차트 이미지 생성 실패로 Gemini 분석을 건너뜁니다.")
        return None

    try:
        # 2. 모델 설정 (멀티모달 지원 모델)
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        
        # 3. 프롬프트 생성 (팩토리 이용)
        prompt_text = get_prompt(
            prompt_type=PromptType.TRADING_SIGNAL, # 프롬프트 종류 지정
            ticker=ticker,
            df_string=df.tail(10).to_string()
        )
        
        # 4. 이미지 파일 준비
        with open(chart_file, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_part = {
            "mime_type": "image/png",
            "data": image_data
        }

        # 5. Gemini API 호출 (텍스트 + 이미지)
        contents = [prompt_text, image_part]
        response = model.generate_content(contents)
        
        # 6. 결과 파싱
        return parse_gemini_signal_response(response.text)

    except Exception as e:
        log_event("ERROR", f"[{ticker}] Gemini API 호출 중 오류 발생: {e}")
        return None
    finally:
        # 7. 임시 차트 파일 삭제
        if os.path.exists(chart_file):
            os.remove(chart_file)
            log_event("INFO", f"임시 차트 파일 삭제: {chart_file}") 