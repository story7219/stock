"""
AI 코드 품질 분석 모듈
- Radon 라이브러리를 사용해 코드 복잡도, 유지보수성 등 정적 분석 수행
- Gemini AI를 통해 코드의 잠재적 문제점 및 개선 방안에 대한 정성적 분석 수행
"""
import os
import glob
import logging
from radon.complexity import cc_visit
from radon.metrics import mi_visit
import google.generativeai as genai

logger = logging.getLogger(__name__)

class CodeQualityAnalyzer:
    def __init__(self):
        if not getattr(genai, '_client', None):
            logger.warning("Gemini가 설정되지 않아 AI 분석을 건너뜁니다.")
            self.ai_enabled = False
        else:
            self.model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
            self.ai_enabled = True

    def analyze_directory(self, path="."):
        report = {}
        py_files = [f for f in glob.glob(f"{path}/**/*.py", recursive=True) if "maintenance_tools" not in f]
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # 정적 분석
                cc_results = cc_visit(code)
                mi_score = mi_visit(code)
                
                # AI 분석
                ai_comment = self.get_ai_analysis(code) if self.ai_enabled else "AI 분석 비활성화"
                
                report[file_path] = {
                    "complexity": sum(c.complexity for c in cc_results) / len(cc_results) if cc_results else 0,
                    "maintainability": mi_score,
                    "ai_comment": ai_comment
                }
                logger.info(f"✅ '{file_path}' 분석 완료")
            except Exception as e:
                logger.error(f"❌ '{file_path}' 분석 실패: {e}")
        return report

    def get_ai_analysis(self, code):
        prompt = f"""
        다음 Python 코드를 전문가의 입장에서 리뷰하고, 잠재적인 문제점, 버그 가능성, 보안 취약점, 개선할 점을 간결하게 요약해줘.
        코드가 매우 훌륭하다면 칭찬해줘. 핵심만 간단히 언급해줘.

        ```python
        {code}
        ```
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"AI 분석 API 호출 실패: {e}")
            return "AI 분석 중 오류가 발생했습니다." 