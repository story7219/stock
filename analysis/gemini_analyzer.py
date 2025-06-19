#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Gemini AI 기반 주식 분석 및 추천 시스템
- 기술적 지표를 바탕으로 Gemini AI가 직접 투자 의견을 제시합니다.
"""

import os
import sys
import pandas as pd
import google.generativeai as genai
import asyncio

# 프로젝트 루트 경로 설정
# analysis/gemini_analyzer.py -> ../
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import GEMINI_API_KEY, DATA_PATH, REPORT_PATH
from utils.logger import get_logger

# 로거 설정
logger = get_logger(__name__)

# Gemini API 키 설정 및 검증
try:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Gemini API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    genai.configure(api_key=GEMINI_API_KEY)
except ValueError as e:
    logger.error(e)
    sys.exit(1)


def create_analysis_prompt(ticker, name, financial_data, price_data):
    """Gemini AI에게 보낼 분석 프롬프트를 생성합니다."""
    
    financial_str = financial_data.to_string() if not financial_data.empty else "재무 데이터가 없습니다."
    price_str = price_data.to_string() if not price_data.empty else "주가 데이터가 없습니다."

    # 데이터가 너무 길 경우 잘라내기 (토큰 제한 고려)
    financial_str = financial_str[:4000]
    price_str = price_str[:4000]

    return f"""
### 기업 분석 요청 (System analsysis request)

**Your Role:** You are a top-tier financial analyst in South Korea. Your task is to analyze the provided company data and generate a clear, concise, and insightful report for investors.

**1. Company to Analyze:**
*   Company Name: {name}
*   Ticker Symbol: {ticker}

**2. Provided Data:**
*   **Key Financial Statements (Recent Years):**
{financial_str}

*   **Recent Stock Prices & Technical Indicators:**
{price_str}

**3. Report Generation Guidelines:**
Please generate the report in **KOREAN** following the structure below. Your analysis must be based *only* on the data provided. Conclude with a final investment rating from one of the following: "강력 매수 (Strong Buy)", "매수 (Buy)", "중립 (Neutral)", "매도 (Sell)", "강력 매도 (Strong Sell)".

---
### **{name}({ticker}) 기업 분석 리포트**

**1. 핵심 요약 (Executive Summary):**
*   (Summarize the company's core business, recent performance, and stock trends in 3-4 sentences.)

**2. 재무 상태 분석 (Financial Health Analysis):**
*   (Evaluate the company's growth, profitability, and stability based on the financial statements. Mention key metrics like revenue, operating profit, and debt-to-equity ratio.)

**3. 기술적 분석 (Technical Analysis):**
*   (Analyze the current stock trend and momentum based on the provided price data, including moving averages and Ichimoku Cloud. Provide short-term and mid-to-long-term technical outlooks.)

**4. 종합 투자 의견 (Overall Investment Opinion):**
*   (Synthesize the financial and technical analyses to provide a final investment opinion. State 2-3 key reasons for your rating.)
---
"""

async def analyze_stock_with_gemini(ticker, name):
    """
    수집된 데이터를 사용하여 Gemini AI로 종목을 분석하고 리포트를 저장합니다.
    재무 데이터가 없어도 주가 데이터만 있으면 분석을 시도합니다.
    """
    logger.info(f"[{ticker}:{name}] Gemini AI 분석을 시작합니다.")
    
    financial_path = os.path.join(DATA_PATH, f"{ticker}_financials.csv")
    price_path = os.path.join(DATA_PATH, f"{ticker}_price_data.csv")

    # 최소한 주가 데이터는 있어야 분석이 의미가 있음
    if not os.path.exists(price_path):
        logger.warning(f"[{ticker}:{name}] 주가 데이터 파일이 없어 분석을 건너뜁니다.")
        return

    financial_df = pd.DataFrame() # 기본 빈 데이터프레임
    if os.path.exists(financial_path):
        try:
            financial_df = pd.read_csv(financial_path)
        except Exception as e:
            logger.error(f"[{ticker}:{name}] 재무 데이터 파일 로딩 중 오류 발생: {e}")
            # 재무 데이터 로딩에 실패해도 일단 진행
    
    try:
        # 기술적 지표가 많은 후반부 데이터를 사용
        price_df = pd.read_csv(price_path).tail(60)
    except Exception as e:
        logger.error(f"[{ticker}:{name}] 주가 데이터 파일 로딩 중 오류 발생: {e}")
        return # 주가 데이터 로딩 실패는 치명적이므로 분석 중단

    prompt = create_analysis_prompt(ticker, name, financial_df, price_df)
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        loop = asyncio.get_running_loop()
        
        # functools.partial을 사용하여 동기 함수 호출을 준비
        import functools
        blocking_call = functools.partial(
            model.generate_content, 
            prompt, 
            request_options={"timeout": 240}
        )
        
        response = await loop.run_in_executor(None, blocking_call)
        
        report_content = response.text
        report_filename = f"{ticker}_{name}_분석_리포트.md"
        report_path = os.path.join(REPORT_PATH, report_filename)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        logger.info(f"[{ticker}:{name}] 분석 리포트 저장이 완료되었습니다. -> {report_filename}")

    except Exception as e:
        logger.error(f"[{ticker}:{name}] Gemini AI 분석 중 오류 발생: {e}")
        error_file = os.path.join(REPORT_PATH, "_error_log.txt")
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(f"Error analyzing {name}({ticker}): {str(e)}\n")

if __name__ == '__main__':
    # 이 파일을 직접 실행할 때를 위한 테스트 코드
    async def test_module():
        logger.info("--- Gemini 분석 모듈 테스트 시작 ---")
        # 테스트 전 '005930' (삼성전자) 데이터가 data 폴더에 있는지 확인하세요.
        # main.py를 실행하여 데이터를 먼저 수집해야 합니다.
        sample_ticker = "005930"
        sample_name = "삼성전자"
        
        if os.path.exists(os.path.join(DATA_PATH, f"{sample_ticker}_financials.csv")):
            logger.info(f"테스트 종목 '{sample_name}'에 대한 분석을 시작합니다.")
            await analyze_stock_with_gemini(sample_ticker, sample_name)
        else:
            logger.warning(f"테스트를 위한 '{sample_name}' 데이터 파일이 'data' 폴더에 없습니다.")
            logger.warning("main.py를 먼저 실행하여 데이터를 수집해주세요.")
        logger.info("--- Gemini 분석 모듈 테스트 종료 ---")

    asyncio.run(test_module()) 