import re

def parse_gemini_signal_response(response_text: str) -> dict:
    """
    Gemini API의 텍스트 응답을 파싱하여 'signal'과 'reason'을 포함하는 딕셔셔너리로 변환합니다.

    Args:
        response_text (str): Gemini API로부터 받은 원시 텍스트 응답.

    Returns:
        dict: 'signal' (매수/매도/보류)과 'reason' (판단 이유)을 포함하는 딕셔너리.
              파싱에 실패하면 기본값{'signal': '보류', 'reason': '파싱 실패'}을 반환합니다.
    """
    try:
        # 'signal' 또는 'Signal' 뒤에 오는 단어(매수, 매도, 보류)를 찾습니다.
        signal_match = re.search(r"[Ss]ignal\s*:\s*(\S+)", response_text)
        signal = signal_match.group(1).strip() if signal_match else "보류"

        # 'reason' 또는 'Reason' 뒤에 오는 나머지 텍스트를 찾습니다.
        reason_match = re.search(r"[Rr]eason\s*:\s*(.*)", response_text, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "판단 이유를 찾을 수 없습니다."

        # '매수', '매도', '보류' 외의 값이 나올 경우를 대비한 처리
        if signal not in ['매수', '매도', '보류']:
            signal = '보류'
            reason = f"알 수 없는 신호({signal})가 감지되었습니다. 원본: {response_text}"

        return {"signal": signal, "reason": reason}
    except Exception as e:
        return {"signal": "보류", "reason": f"응답 파싱 중 오류 발생: {e}\n원본 응답: {response_text}"} 