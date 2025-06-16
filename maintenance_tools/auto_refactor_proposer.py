"""
AI 기반 자동 리팩토링 제안기
- 지정된 파이썬 파일들을 분석하여 리팩토링 제안을 생성합니다.
- Gemini AI를 사용하여 전체 파일 내용을 기반으로 수정안을 만듭니다.
"""
import logging
import re
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가하여 config.py를 임포트할 수 있도록 함
sys.path.append(str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai
import config

logger = logging.getLogger(__name__)

# --- 데이터 클래스 정의 ---
from dataclasses import dataclass

@dataclass
class RefactoringProposal:
    """리팩토링 제안 정보를 담는 데이터 클래스"""
    file_path: str
    refactored_code: str
    explanation: str

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "refactored_code": self.refactored_code,
            "explanation": self.explanation,
        }

# --- AI 리팩토링 제안기 클래스 ---
class AutoRefactorProposer:
    """Gemini AI를 사용하여 코드 리팩토링을 제안하는 클래스"""
    def __init__(self):
        try:
            # .env를 통해 로드된 API 키 사용
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("✅ Gemini AI 모델 초기화 성공")
        except Exception as e:
            logger.error(f"❌ Gemini AI 모델 초기화 실패: {e}")
            self.model = None

    def _get_target_files(self):
        """분석할 핵심 대상 파일 목록을 반환합니다."""
        root = Path('.')
        return [
            root / "core_trader.py",
            root / "analysis_engine.py",
            root / "main.py",
        ]

    def propose_refactoring_for_file(self, file_path: Path):
        """단일 파일에 대한 리팩토링을 제안합니다."""
        if not self.model or not file_path.exists():
            return None

        logger.info(f"📄 '{file_path}' 파일 분석 시작...")
        original_code = file_path.read_text(encoding='utf-8')

        prompt = f"""
        당신은 파이썬 리팩토링 전문가입니다. 아래 코드를 분석하고, 더 나은 코드 품질(가독성, 성능, 보안, 모듈화)을 위해 리팩토링을 제안해주세요.

        **지시사항:**
        1. 코드의 핵심 로직과 기능은 절대 변경하지 마세요.
        2. 현대적인 파이썬 스타일(type hints, f-strings, 예외 처리 등)을 적용해주세요.
        3. 코드를 변경할 필요가 없다면, 다른 말 없이 "변경할 필요 없음" 이라고만 답변해주세요.
        4. 코드를 변경해야 한다면, 반드시 전체 파일 내용을 수정된 버전으로 제공해야 합니다. 서론이나 부가 설명 없이 코드 블록만 반환해주세요.
        5. 변경 사항에 대한 간단한 설명을 코드 블록 뒤에 `[설명]` 태그를 사용하여 한두 문장으로 요약해주세요.

        --- 분석 대상 코드: {file_path.name} ---
        ```python
        {original_code}
        ```
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            if "변경할 필요 없음" in text:
                logger.info(f"👍 '{file_path}' 파일은 리팩토링이 필요 없습니다.")
                return None
            
            # 정규표현식을 사용하여 코드 블록과 설명 추출
            code_match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
            if not code_match:
                logger.warning(f"⚠️ '{file_path}' 파일의 리팩토링된 코드 블록을 찾지 못했습니다.")
                return None
            
            refactored_code = code_match.group(1).strip()
            
            explanation_match = re.search(r"\[설명\]\s*(.*)", text)
            explanation = explanation_match.group(1).strip() if explanation_match else "AI가 코드 개선을 제안했습니다."

            logger.info(f"✨ '{file_path}' 파일에 대한 리팩토링 제안 생성 완료.")
            return RefactoringProposal(
                file_path=str(file_path),
                refactored_code=refactored_code,
                explanation=explanation
            )
        except Exception as e:
            logger.error(f"❌ '{file_path}' 파일 분석 중 Gemini API 오류 발생: {e}")
            return None

    def run(self):
        """대상 파일 전체에 대해 리팩토링 제안을 실행하고 리스트를 반환합니다."""
        proposals = []
        for file_path in self._get_target_files():
            if proposal := self.propose_refactoring_for_file(file_path):
                proposals.append(proposal)
        return proposals 