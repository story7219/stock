def _get_ai_proposal(self, file_path, code, report):
    prompt = f"""
    파일 경로: {file_path}
    기존 코드 분석 결과:
    - 복잡도 점수: {report['complexity']:.2f} (높을수록 복잡)
    - 유지보수성 점수: {report['maintainability']:.2f} (낮을수록 문제)
    - AI 코멘트: {report['ai_comment']}

    위 분석 결과를 바탕으로, 아래의 Python 코드를 더 읽기 쉽고, 효율적이며, 유지보수하기 좋은 코드로 **완전히 재작성**해줘.
    **다른 설명이나 주석, ```python ... ``` 마크다운 없이, 오직 리팩토링된 최종 Python 코드 내용만 응답으로 제공해야 해.**

    기존 코드:
    ```python
    {code}
    ```
    """
    try:
        # ... (기존과 동일) ... 