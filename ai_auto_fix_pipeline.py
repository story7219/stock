#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: ai_auto_fix_pipeline.py
모듈: 중급 수준 AI 자동수정 파이프라인
목적: Gemini AI를 사용한 Python 코드 자동수정

Author: AI Code Fixer
Created: 2025-07-08
Version: 1.0.0

Features:
    - Gemini AI 기반 코드 수정
    - 백업 생성
    - 오류 처리
    - 진행률 표시

Dependencies:
    - Python 3.11+
    - google-generativeai
    - aiofiles
    - tqdm
"""

from tqdm import tqdm
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict, Any
import aiofiles
import asyncio
import logging
import os
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 백업 디렉토리
BACKUP_DIR = Path("backup_ai_fixes")
BACKUP_DIR.mkdir(exist_ok=True)

async def ai_auto_fix_pipeline(failed_files: List[str], num_fixers: int = 2) -> List[Dict[str, Any]]:
    """AI 자동수정 파이프라인"""

    # Gemini API 사용 가능 여부
    try:
        _ = genai
        GEMINI_AVAILABLE = True
    except Exception:
        GEMINI_AVAILABLE = False
    if not GEMINI_AVAILABLE:
        logger.error("❌ Gemini AI를 사용할 수 없습니다.")
        return []

    # API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash-8b")

    if not api_key:
        logger.error("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        return []

    # Gemini 설정 (최신 버전은 configure 불필요)
    # genai.configure(api_key=api_key)  # 필요시만 사용

    results = []

    # 진행률 표시
    if tqdm:
        file_iterator = tqdm(failed_files, desc="🤖 AI 자동수정", unit="files")
    else:
        file_iterator = failed_files
        logger.info(f"🚀 AI 자동수정 시작: {len(failed_files)}개 파일")

    for file_path in file_iterator:
        try:
            # 파일 읽기 (다양한 인코딩 시도)
            original_code = None
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                        original_code = await f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if original_code is None:
                logger.error(f"❌ 파일 읽기 실패: {file_path} - 인코딩 문제")
                results.append({
                    'file': file_path,
                    'fixed': False,
                    'error': 'Encoding error'
                })
                continue

            # AI 프롬프트 생성
            prompt = (
                "다음 Python 코드의 오류를 수정해주세요.\n\n"
                "수정 요구사항:\n"
                "1. 문법 오류 수정\n"
                "2. 들여쓰기 오류 수정\n"
                "3. 후행 공백 제거\n"
                "4. 누락된 콜론(:) 추가\n"
                "5. 괄호 균형 맞추기\n\n"
                "수정된 전체 코드만 반환해주세요. 설명은 포함하지 마세요.\n\n"
                f"원본 코드:\n```python\n{original_code}\n```"
            )

            # Gemini AI 호출
            try:
                model_instance = genai.GenerativeModel(model_name)
                response = model_instance.generate_content([prompt])
                # 최신 버전은 candidates[0].content.parts[0].text 구조일 수 있음
                try:
                    fixed_code = response.candidates[0].content.parts[0].text.strip()
                except Exception:
                    fixed_code = response.text.strip() if hasattr(response, 'text') else ''
                # 코드 블록 마커 제거
                if fixed_code.startswith('```python'):
                    fixed_code = fixed_code[9:]
                if fixed_code.endswith('```'):
                    fixed_code = fixed_code[:-3]
                fixed_code = fixed_code.strip()
            except Exception as api_error:
                logger.error(f"❌ Gemini API 호출 실패: {file_path} - {api_error}")
                results.append({
                    'file': file_path,
                    'fixed': False,
                    'error': f'API call failed: {api_error}'
                })
                continue

            # 코드가 실제로 변경되었는지 확인
            if fixed_code and fixed_code != original_code:
                # 백업 생성
                backup_file = BACKUP_DIR / f"{Path(file_path).name}.backup_{int(asyncio.get_event_loop().time())}"
                async with aiofiles.open(backup_file, 'w', encoding='utf-8') as f:
                    await f.write(original_code)
                # 파일 덮어쓰기
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(fixed_code)
                logger.info(f"✅ AI 수정 성공: {file_path}")
                results.append({
                    'file': file_path,
                    'fixed': True,
                    'backup': str(backup_file)
                })
            else:
                results.append({
                    'file': file_path,
                    'fixed': False,
                    'reason': 'No changes needed'
                })
        except Exception as e:
            logger.error(f"❌ AI 수정 실패: {file_path} - {e}")
            results.append({
                'file': file_path,
                'fixed': False,
                'error': str(e)
            })
    # 결과 요약
    fixed_count = sum(1 for r in results if r['fixed'])
    failed_count = len(results) - fixed_count
    logger.info(f"📊 AI 자동수정 완료: 성공 {fixed_count}개, 실패 {failed_count}개")
    return results

async def main():
    """테스트용 메인 함수"""
    # 테스트 파일 목록
    test_files = ["test_file.py"]  # 실제 파일 경로로 변경
    if not test_files:
        print("❌ 수정할 파일이 없습니다.")
        return
    results = await ai_auto_fix_pipeline(test_files)
    print(f"\n📊 결과:")
    for result in results:
        status = "✅ 성공" if result['fixed'] else "❌ 실패"
        print(f"  - {result['file']}: {status}")

if __name__ == "__main__":
    asyncio.run(main())

