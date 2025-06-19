# -*- coding: utf-8 -*-
import streamlit as st
import asyncio
import os
import sys
import logging
from datetime import datetime

# --- 프로젝트 루트 경로 설정 ---
# 이 파일(app.py)의 실제 위치를 기준으로 프로젝트 루트를 찾습니다.
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_path 
# sys.path에 프로젝트 루트를 추가하여 다른 모듈들을 임포트할 수 있도록 합니다.
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 내부 모듈 임포트 ---
# sys.path 설정 후에 임포트해야 합니다.
try:
    import config
    from core.core_trader import CoreTrader
    from data_providers.dart_api import DartApiHandler
    from news_collector import NewsCollector
    from database_manager import DatabaseManager
    from analyzer.flash_analyzer import FlashStockAIAnalyzer
    from voice_synthesizer import VoiceSynthesizer
except ImportError as e:
    st.error(f"⚠️ 필수 모듈을 로드하는 데 실패했습니다: {e}")
    st.info("프로젝트 구조가 올바른지, 필요한 모든 파일이 존재하는지 확인해주세요.")
    st.stop()


# --- Streamlit 캐시를 사용하여 백엔드 서비스 초기화 ---
@st.cache_resource
def initialize_services():
    """
    애플리케이션에 필요한 모든 백엔드 서비스를 초기화하고 캐싱합니다.
    이 함수는 앱 실행 중 한 번만 호출됩니다.
    """
    try:
        logger.info("백엔드 서비스 초기화를 시작합니다...")
        trader = CoreTrader()
        dart_handler = DartApiHandler(api_key=config.DART_API_KEY)
        news_collector = NewsCollector()
        db_manager = DatabaseManager()
        # db_manager의 초기화 메서드(비동기)를 이벤트 루프에서 실행
        asyncio.run(db_manager.initialize()) 
        analyzer = FlashStockAIAnalyzer(trader, dart_handler, news_collector, db_manager)
        voice_synthesizer = VoiceSynthesizer()
        logger.info("✅ 모든 백엔드 서비스가 성공적으로 초기화되었습니다.")
        return {
            "analyzer": analyzer,
            "voice_synthesizer": voice_synthesizer
        }
    except Exception as e:
        logger.error(f"❌ 서비스 초기화 중 심각한 오류 발생: {e}", exc_info=True)
        st.error(f"서비스 초기화에 실패했습니다: {e}. 'config.py' 파일의 API 키 등을 확인해주세요.")
        st.stop()

# --- 메인 애플리케이션 로직 ---
def main():
    st.set_page_config(
        page_title="FlashStockAI ⚡",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 서비스 로드
    services = initialize_services()
    analyzer = services["analyzer"]
    voice_synth = services["voice_synthesizer"]

    # --- 사이드바 UI ---
    with st.sidebar:
        st.image("https://storage.googleapis.com/aipi_images/flashstock_logo.png", width=250)
        st.header("종목 분석")

        stock_code = st.text_input("종목 코드 입력", placeholder="예: 005930")
        uploaded_file = st.file_uploader("차트 이미지 업로드 (선택 사항)", type=['png', 'jpg', 'jpeg'])
        
        analyze_button = st.button("🚀 분석 시작", use_container_width=True, type="primary")

        st.markdown("---")
        st.info("💡 종목 코드를 입력하면 텍스트 기반으로, 차트 이미지를 업로드하면 이미지 기반으로 분석합니다.")

    # --- 메인 패널 UI ---
    st.title("⚡ FlashStockAI: AI 주식 분석 리포트")

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'voice_file' not in st.session_state:
        st.session_state.voice_file = None

    if analyze_button:
        # 이전 결과 초기화
        st.session_state.analysis_result = None
        st.session_state.voice_file = None
        
        # 입력 값 유효성 검사
        if not stock_code and not uploaded_file:
            st.warning("종목 코드 또는 차트 이미지를 입력해주세요.")
            return
        if not stock_code and uploaded_file:
            st.warning("차트 이미지 분석 시에도 종목 코드는 필수입니다.")
            return

        with st.spinner(f"종목 코드 [{stock_code}] 분석 중... AI가 데이터를 처리하고 있습니다. 잠시만 기다려주세요."):
            try:
                result = None
                if uploaded_file is not None:
                    # 이미지 분석
                    image_bytes = uploaded_file.getvalue()
                    result = asyncio.run(analyzer.analyze_stock_from_image(stock_code, image_bytes))
                else:
                    # 텍스트 분석
                    result = asyncio.run(analyzer.analyze_stock_from_text(stock_code))
                
                st.session_state.analysis_result = result
                
                if result and "error" not in result:
                    # 음성 생성
                    summary_for_voice = f"종목코드 {result.get('stock_code')}에 대한 플래시스탁 AI 분석 결과입니다. " \
                                        f"투자 의견은 {result.get('investment_opinion')}이며, 종합 점수는 100점 만점에 {result.get('overall_score')}점입니다. " \
                                        f"전략 요약: {result.get('strategy', {}).get('summary', '정보 없음')}"
                    
                    voice_file_path = voice_synth.speak(summary_for_voice, save_to_file=True)
                    st.session_state.voice_file = voice_file_path

            except Exception as e:
                logger.error(f"❌ 분석 실행 중 오류 발생: {e}", exc_info=True)
                st.error(f"분석 중 오류가 발생했습니다: {e}")


    # --- 결과 출력 ---
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        if "error" in result:
            st.error(f"AI 분석 실패: {result.get('error')}")
            st.code(result.get('raw_text', '상세 정보 없음'))
        else:
            st.success(f"**{result.get('stock_code')}**에 대한 AI 분석이 완료되었습니다.")

            # 음성 브리핑
            if st.session_state.voice_file and os.path.exists(st.session_state.voice_file):
                st.audio(st.session_state.voice_file, format='audio/mp3')
            
            # Rich를 사용한 터미널 스타일 출력을 Streamlit에 맞게 재구성
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="종합 점수", value=f"{result.get('overall_score', 'N/A')} / 100")
            with col2:
                st.metric(label="투자 의견", value=result.get('investment_opinion', 'N/A'))

            st.markdown("---")
            
            strategy = result.get('strategy', {})
            st.subheader("📈 투자 전략")
            strat_cols = st.columns(3)
            strat_cols[0].metric("진입 가격", strategy.get('entry_price', 'N/A'))
            strat_cols[1].metric("목표 가격", strategy.get('target_price', 'N/A'))
            strat_cols[2].metric("손절 가격", strategy.get('stop_loss', 'N/A'))
            st.info(f"**전략 요약:** {strategy.get('summary', 'N/A')}")
            
            st.markdown("---")

            st.subheader("🔍 분석 근거")
            reasoning = result.get('reasoning', {})
            pos_col, neg_col = st.columns(2)
            with pos_col:
                st.success("**👍 긍정적 요인**")
                for factor in reasoning.get('positive_factors', []):
                    st.markdown(f"- {factor}")
            
            with neg_col:
                st.error("**👎 부정적 요인**")
                for factor in reasoning.get('negative_factors', []):
                    st.markdown(f"- {factor}")


if __name__ == "__main__":
    main() 