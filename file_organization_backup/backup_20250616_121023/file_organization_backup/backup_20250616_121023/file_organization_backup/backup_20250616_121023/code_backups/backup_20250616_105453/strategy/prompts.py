# strategy/prompts.py
# Gemini API에 사용될 프롬프트 템플릿을 관리하는 모듈 (팩토리 패턴 적용)

from enum import Enum, auto

class PromptType(Enum):
    """프롬프트의 종류를 정의하는 열거형"""
    TRADING_SIGNAL = auto()
    # 예: MARKET_ANALYSIS = auto()
    # 예: LONG_TERM_INVESTMENT = auto()

def get_prompt(prompt_type: PromptType, **kwargs) -> str:
    """
    요청된 타입에 맞는 프롬프트 템플릿을 반환하는 팩토리 함수.
    """
    if prompt_type == PromptType.TRADING_SIGNAL:
        # TRADING_SIGNAL에 필요한 ticker와 df_string 인자를 kwargs에서 추출
        return _get_trading_signal_prompt(
            ticker=kwargs.get('ticker'),
            df_string=kwargs.get('df_string')
        )
    # --- 향후 다른 프롬프트 타입을 여기에 추가 ---
    # elif prompt_type == PromptType.MARKET_ANALYSIS:
    #     return _get_market_analysis_prompt(...)
    
    raise ValueError(f"지원하지 않는 프롬프트 타입입니다: {prompt_type}")

def _get_trading_signal_prompt(ticker: str, df_string: str) -> str:
    """
    (내부용) 단기 매매 신호 분석을 위한 프롬프트 템플릿.
    차트 이미지와 함께 사용되는 것을 전제로 합니다.
    """
    return f"""
    전문 주식 트레이더로서 다음 정보를 종합적으로 분석해줘.

    1.  **종목 데이터**:
        - 종목 코드: {ticker}
        - 최근 5일간 5분봉 OHLCV 데이터 (최신 10개):
        {df_string}

    2.  **첨부된 차트 이미지**:
        - 위 데이터로 생성된 캔들스틱, 이동평균선(20, 60), 거래량 차트.

    **분석 요청**:
    첨부된 차트 이미지의 패턴(예: 캔들 모양, 이동평균선 배열, 거래량 변화)을 최우선으로 고려하여, 현재 시점이 단기적으로 '매수'하기에 적절한지 아니면 '보류'해야 하는지 판단해줘.

    **응답 형식** (반드시 JSON 형식으로 응답):
    {{
      "signal": "매수" 또는 "보류",
      "reason": "판단에 대한 구체적인 이유를 차트 패턴에 근거하여 3~4문장으로 설명"
    }}
    """

# 향후 다른 상황에 맞는 프롬프트를 _(언더스코어)로 시작하는 내부용 함수로 추가합니다.
# def _get_market_condition_prompt(...): ...

# 향후 다른 상황에 맞는 프롬프트를 함수로 추가할 수 있습니다.
# 예: def get_long_term_analysis_prompt(...): ...
# 예: def get_market_condition_prompt(...): ... 