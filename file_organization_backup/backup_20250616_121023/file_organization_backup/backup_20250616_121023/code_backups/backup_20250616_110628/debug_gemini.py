"""
Gemini API 연결 테스트 도구
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_connection():
    """Gemini API 연결 테스트"""
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        
        # 사용 가능한 모델 목록 확인
        print("🔍 사용 가능한 모델들:")
        models = list(genai.list_models())
        for model in models:
            print(f"  - {model.name}")
        
        # 각 모델 테스트
        test_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        for model_name in test_models:
            try:
                print(f"\n🧪 {model_name} 테스트 중...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("안녕하세요! 간단한 테스트입니다.")
                print(f"✅ {model_name}: 성공")
                print(f"   응답: {response.text[:50]}...")
                return True
            except Exception as e:
                print(f"❌ {model_name}: 실패 - {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ API 설정 실패: {e}")
        return False

if __name__ == "__main__":
    test_gemini_connection() 