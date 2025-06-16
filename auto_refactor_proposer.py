"""
AI 리팩토링 제안 모듈
- 품질 분석 리포트를 바탕으로 구체적인 리팩토링 제안 생성
"""
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

class RefactorProposer:
    def __init__(self):
        if not getattr(genai, '_client', None):
            raise ValueError("Gemini가 설정되지 않아 리팩토링 제안을 생성할 수 없습니다.")
        self.model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

    def generate_proposals(self, quality_report):
        proposals = {}
        for file_path, report in quality_report.items():
            if report['complexity'] > 10 or report['maintainability'] < 70:
                logger.info(f"'{file_path}'에 대한 리팩토링 제안 생성 중...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    proposal_text = self._get_ai_proposal(file_path, code, report)
                    proposals[file_path] = proposal_text
                except Exception as e:
                    logger.error(f"'{file_path}' 제안 생성 실패: {e}")
        return proposals
    
    def _get_ai_proposal(self, file_path, code, report):
        prompt = f"""
        파일 경로: {file_path}
        기존 코드 분석 결과:
        - 복잡도 점수: {report['complexity']:.2f} (높을수록 복잡)
        - 유지보수성 점수: {report['maintainability']:.2f} (낮을수록 문제)
        - AI 코멘트: {report['ai_comment']}

        위 분석 결과를 바탕으로, 아래의 Python 코드를 더 읽기 쉽고, 효율적이며, 유지보수하기 좋게 리팩토링하는 구체적인 방법을 제안해줘.
        단순히 "개선하세요"가 아니라, 실제 수정할 코드 예시(`diff` 형식)를 포함하여 제안해줘.

        기존 코드:
        ```python
        {code}
        ```
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"AI 제안 API 호출 실패: {e}")
            return "AI 제안 생성 중 오류가 발생했습니다." 