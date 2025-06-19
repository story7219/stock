# -*- coding: utf-8 -*-
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

import os
import sys
import json
import io
from PIL import Image
from datetime import datetime

# --- 프로젝트 루트 경로 설정 ---
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 테스트 대상 모듈 임포트
from analyzer.flash_analyzer import FlashStockAIAnalyzer
from news_collector import NewsItem

# --- 테스트 데이터 ---
MOCK_STOCK_CODE = "005930"
MOCK_GEMINI_RESPONSE = {
    "stock_code": MOCK_STOCK_CODE,
    "analysis_type": "Text-based",
    "technical_score": 75,
    "fundamental_score": 85,
    "overall_score": 80,
    "investment_opinion": "매수",
    "strategy": {
        "summary": "기술적 반등 및 펀더멘탈 개선 기대",
        "entry_price": "75,000원 ~ 76,000원",
        "target_price": "85,000원",
        "stop_loss": "72,000원"
    },
    "reasoning": {
        "positive_factors": ["글로벌 반도체 업황 회복 신호", "기관의 꾸준한 순매수 유입"],
        "negative_factors": ["환율 변동성 확대", "단기 기술적 과열 가능성"]
    }
}

# --- Mock 객체 설정 ---
@pytest.fixture
def mock_services():
    """테스트에 필요한 Mock 서비스 객체들을 생성합니다."""
    # CoreTrader Mock
    mock_trader = MagicMock()
    mock_trader.get_current_price = AsyncMock(return_value={"stck_prpr": "76000", "prdy_vrss_sign": "2"})
    # 기술적 지표 분석이 실패하는 경우도 테스트하기 위함
    mock_trader.get_technical_indicators = AsyncMock(return_value=None) 
    
    # DartApiHandler Mock
    mock_dart = MagicMock()
    mock_dart.get_financials_for_last_quarters.return_value = {"2023.12": [{"account_nm": "매출액", "thstrm_amount": "10000"}]}
    
    # NewsCollector Mock
    mock_news = MagicMock()
    mock_news.get_realtime_news.return_value = [
        NewsItem(
            title="삼성전자, 신규 AI 칩 발표",
            content="차세대 AI 칩을 공개하며 시장의 기대를 모으고 있다.",
            url="http://test.com/news/1",
            timestamp=datetime.now(),
            source="테스트뉴스",
            sentiment='positive',
            sentiment_score=0.8,
            related_stocks=[MOCK_STOCK_CODE]
        )
    ]
    
    # DatabaseManager Mock
    mock_db = MagicMock()
    mock_db.save_analysis_result = AsyncMock()
    
    # Gemini Model Mock
    mock_gemini_model = MagicMock()
    # generate_content_async는 awaitable이어야 하므로 AsyncMock으로 설정
    mock_gemini_response_obj = MagicMock()
    mock_gemini_response_obj.text = f"```json\n{json.dumps(MOCK_GEMINI_RESPONSE, ensure_ascii=False)}\n```"
    mock_gemini_model.generate_content_async = AsyncMock(return_value=mock_gemini_response_obj)
    
    # FlashStockAIAnalyzer 초기화 및 Gemini 모델 패치
    analyzer = FlashStockAIAnalyzer(mock_trader, mock_dart, mock_news, mock_db)
    analyzer.gemini_model = mock_gemini_model # 실제 API 호출 대신 Mock 모델 사용
    
    return {
        "analyzer": analyzer,
        "trader": mock_trader,
        "dart": mock_dart,
        "news": mock_news,
        "db": mock_db,
        "gemini": mock_gemini_model
    }

# --- 테스트 케이스 ---
@pytest.mark.asyncio
async def test_analyze_stock_from_text_success(mock_services):
    """
    텍스트 기반 종목 분석이 성공적으로 수행되는지 테스트합니다.
    - 데이터 수집, 프롬프트 생성, API 호출, 응답 파싱, DB 저장의 전체 흐름을 검증합니다.
    """
    # given
    analyzer = mock_services["analyzer"]
    
    # when
    result = await analyzer.analyze_stock_from_text(MOCK_STOCK_CODE)
    
    # then
    # 1. 분석 결과가 정상적으로 파싱되었는지 확인
    assert result is not None
    assert "error" not in result
    assert result["stock_code"] == MOCK_STOCK_CODE
    assert result["overall_score"] == 80
    assert result["investment_opinion"] == "매수"
    assert "target_price" in result["strategy"]

    # 2. 각 서비스의 메서드가 예상대로 호출되었는지 확인
    mock_services["trader"].get_current_price.assert_awaited_once_with(MOCK_STOCK_CODE)
    mock_services["dart"].get_financials_for_last_quarters.assert_called_once()
    mock_services["news"].get_realtime_news.assert_called_once()
    mock_services["gemini"].generate_content_async.assert_awaited_once()
    mock_services["db"].save_analysis_result.assert_awaited_once()

@pytest.mark.asyncio
async def test_parse_invalid_json_response(mock_services):
    """
    AI가 유효하지 않은 JSON 형식으로 응답했을 때, 오류 처리가 정상적으로 되는지 테스트합니다.
    """
    # given
    analyzer = mock_services["analyzer"]
    
    # AI 응답을 깨진 JSON으로 설정
    invalid_response_obj = MagicMock()
    invalid_response_obj.text = "```json\n{\"key\": \"value\",,}\n```" # 잘못된 JSON
    analyzer.gemini_model.generate_content_async.return_value = invalid_response_obj

    # when
    result = await analyzer.analyze_stock_from_text(MOCK_STOCK_CODE)

    # then
    assert result is not None
    assert "error" in result
    assert result["error"] == "JSON Decode Error"
    
    # 실패한 결과는 DB에 저장되지 않아야 함
    mock_services["db"].save_analysis_result.assert_not_awaited()

@pytest.mark.asyncio
async def test_image_analysis_flow(mock_services):
    """
    이미지 기반 분석 요청 시, 데이터 수집 범위가 달라지는지(재무/뉴스 제외) 테스트합니다.
    """
    # given
    analyzer = mock_services["analyzer"]
    
    # 이 테스트 케이스를 위한 이미지 분석용 Mock 응답 설정
    image_mock_response = MOCK_GEMINI_RESPONSE.copy()
    image_mock_response["analysis_type"] = "Image-based"
    
    mock_gemini_response_obj = MagicMock()
    mock_gemini_response_obj.text = f"```json\n{json.dumps(image_mock_response, ensure_ascii=False)}\n```"
    analyzer.gemini_model.generate_content_async.return_value = mock_gemini_response_obj

    # Pillow를 사용하여 유효한 1x1 픽셀의 검은색 PNG 이미지 생성
    img = Image.new('RGB', (1, 1), color = 'black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    mock_image_bytes = img_byte_arr.getvalue()
    
    # when
    # analyze_stock_from_image가 예외 없이 실행되어야 함
    result = await analyzer.analyze_stock_from_image(MOCK_STOCK_CODE, mock_image_bytes)
    
    # then
    # 분석 결과가 정상적으로 반환되었는지 확인 (AI 응답은 동일한 mock을 사용)
    assert result is not None
    assert result["stock_code"] == MOCK_STOCK_CODE
    assert result["analysis_type"] == "Image-based"

    # 이미지 분석 시에는 재무(DART)와 뉴스 데이터를 수집하지 않아야 함
    mock_services["trader"].get_current_price.assert_awaited_once_with(MOCK_STOCK_CODE)
    mock_services["trader"].get_technical_indicators.assert_awaited_once_with(MOCK_STOCK_CODE)
    mock_services["dart"].get_financials_for_last_quarters.assert_not_called()
    mock_services["news"].get_realtime_news.assert_not_called()
    
    # Gemini API는 이미지와 함께 호출되어야 함 (prompt, image)
    # generate_content_async.call_args[0][0]은 첫번째 위치 인자(리스트)
    args, _ = mock_services["gemini"].generate_content_async.call_args
    assert len(args[0]) == 2 # [prompt, image]
    assert "차트 이미지를 주의 깊게 분석" in args[0][0] # 프롬프트에 이미지 분석 지시가 포함되었는지 확인
    # 두 번째 인자가 Image 객체인지 확인
    assert isinstance(args[0][1], Image.Image)

def test_generate_prompt_content():
    """프롬프트 생성 로직이 의도대로 동작하는지 간단히 확인합니다."""
    # given
    analyzer = FlashStockAIAnalyzer(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # when
    # 간단한 데이터로 텍스트/이미지 프롬프트 생성
    sample_data = {"price": {"stck_prpr": "10000"}, "tech_indicators": None}
    text_prompt = analyzer._generate_analysis_prompt("000000", sample_data, is_image_analysis=False)
    image_prompt = analyzer._generate_analysis_prompt("000000", sample_data, is_image_analysis=True)
    
    # then
    # 텍스트 프롬프트에는 펀더멘탈 분석 지시가 포함되어야 함
    assert "[기본적 분석 (펀더멘탈)]" in text_prompt
    assert "최신 재무제표 요약" in text_prompt
    # 이미지 프롬프트에는 해당 내용이 없어야 함
    assert "[기본적 분석 (펀더멘탈)]" not in image_prompt
    assert "최신 재무제표 요약" not in image_prompt
    assert "차트 이미지를 주의 깊게 분석" in image_prompt 