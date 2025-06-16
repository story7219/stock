#!/usr/bin/env python3
"""
🔍 AI 코드 품질 검사 도구
GitHub Actions와 로컬 환경 모두 지원
"""

import os
import sys
import glob
import google.generativeai as genai
from utils.telegram_bot import TelegramNotifier
import ast
import subprocess
import logging
from pathlib import Path

# GitHub Actions 환경에서 안전한 config 로드
try:
    import config
    CONFIG_LOADED = True
    print("✅ config 모듈 로드 성공")
except Exception as e:
    CONFIG_LOADED = False
    print(f"⚠️ config 모듈 로드 실패: {e}")
    print("📋 기본 로깅 설정으로 진행")
    
    # 기본 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def gather_codebase():
    """프로젝트 내 모든 .py 파일의 내용을 취합합니다."""
    code_contents = []
    # 현재 디렉터리 및 모든 하위 디렉터리의 .py 파일 검색
    for filepath in glob.glob('**/*.py', recursive=True):
        # 특정 파일/디렉터리 제외
        if '.venv' in filepath or '.github' in filepath or 'service_account.json' in filepath:
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                code_contents.append(f"# --- File: {filepath} ---\n\n{content}")
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
    return "\n\n".join(code_contents)

def get_ai_code_review(codebase: str):
    """Gemini를 통해 코드 리뷰를 요청합니다."""
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        당신은 Python 코드 리뷰를 전문으로 하는 시니어 개발자입니다.
        아래에 제공된 전체 코드 베이스를 분석하고, 다음 관점에서 상세한 리뷰를 작성해 주세요.
        결과는 Markdown 형식으로 정리해 주세요.

        **1. 잠재적 버그:** 논리적 오류나 예외 처리가 미흡한 부분이 있나요?
        **2. 코드 스멜(Code Smells):** 불필하게 복잡하거나, 비효율적이거나, 이해하기 어려운 코드가 있나요? (예: 너무 긴 함수, 중복 코드)
        **3. 가독성 및 유지보수:** 변수명, 함수명이 명확한가요? 주석은 필요한 곳에 잘 작성되었나요?
        **4. 성능 개선점:** 더 빠르거나 효율적으로 개선할 수 있는 부분이 있나요?
        **5. 보안 취약점:** 외부 입력 처리나 API 키 관리 등에서 보안상 위험한 부분이 있나요?

        **전체 코드 베이스:**
        ```python
        {codebase}
        ```

        **리뷰 형식:**
        ### 🤖 AI 코드 분석 데일리 리포트

        **종합 평가:** (코드 전체에 대한 간략한 총평)

        **👍 잘된 점:**
        - (예: 역할 분리가 명확한 클래스 설계가 돋보입니다.)

        **💡 개선 제안:**
        1. **[파일명:줄번호]** [개선할 내용 요약]
           - **문제점:** (현재 코드의 문제점 상세 설명)
           - **개선안:** (수정 코드 예시 또는 구체적인 개선 방향 제시)
        2. **[파일명:줄번호]** ...

        **결론 및 다음 단계:** (리뷰 요약 및 권장하는 다음 작업)
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"AI 코드 리뷰 생성 중 오류가 발생했습니다: {e}"

def main():
    """메인 함수 - config 모듈 사용"""
    print("=== AI 코드 리뷰 시스템 시작 ===")
    
    # config 모듈을 통한 설정 검증
    missing_configs, optional_configs = config.validate_config()
    
    # 코드 리뷰에 필요한 필수 설정 확인
    required_for_review = []
    if not config.GEMINI_API_KEY:
        required_for_review.append('GEMINI_API_KEY')
    if not config.TELEGRAM_BOT_TOKEN:
        required_for_review.append('TELEGRAM_BOT_TOKEN')
    if not config.TELEGRAM_CHAT_ID:
        required_for_review.append('TELEGRAM_CHAT_ID')
    
    if required_for_review:
        print("❌ 코드 리뷰에 필요한 환경변수가 설정되지 않았습니다:")
        for var in required_for_review:
            print(f"  - {var}")
        sys.exit(1)
    
    print("✅ 모든 필수 환경변수가 설정되었습니다.")
    print("\n🚀 AI 코드 리뷰를 시작합니다...")
    
    # 코드베이스 수집
    codebase = gather_codebase()
    if not codebase:
        print("❌ 리뷰할 코드를 찾지 못했습니다.")
        return

    print("🤖 Gemini에게 코드 리뷰를 요청합니다...")
    review_report = get_ai_code_review(codebase)

    print("📋 리뷰 리포트 생성 완료!")
    print("=" * 50)
    print(review_report)
    print("=" * 50)

    # 텔레그램으로 결과 전송
    try:
        telegram_bot = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        telegram_bot.send_message(review_report)
        print("✅ 텔레그램으로 리뷰 리포트를 성공적으로 전송했습니다.")
    except Exception as e:
        print(f"❌ 텔레그램 전송 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 