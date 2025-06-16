"""
빠른 연결 테스트
"""

def quick_connection_test():
    """빠른 연결 테스트"""
    print("⚡ 빠른 연결 테스트 시작...")
    
    try:
        # 1. 환경 변수 확인
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            print("❌ GEMINI_API_KEY가 설정되지 않았습니다")
            return False
        
        print("✅ Gemini API 키 확인됨")
        
        # 2. 구글 인증 파일 확인
        google_creds = os.getenv('GOOGLE_CREDENTIALS_PATH', 'google_credentials.json')
        if not os.path.exists(google_creds):
            print(f"❌ 구글 인증 파일이 없습니다: {google_creds}")
            return False
        
        print("✅ 구글 인증 파일 확인됨")
        
        # 3. 구글 시트 API 테스트
        import gspread
        from google.oauth2.service_account import Credentials
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        credentials = Credentials.from_service_account_file(google_creds, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        print("✅ 구글 시트 API 연결 성공")
        
        # 4. Gemini API 테스트
        import google.generativeai as genai
        
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("안녕하세요")
        if response and response.text:
            print("✅ Gemini API 연결 성공")
        else:
            print("❌ Gemini API 응답 없음")
            return False
        
        print("\n🎉 모든 연결 테스트 성공!")
        print("✨ 구글 시트 통합 시스템을 사용할 준비가 되었습니다")
        
        return True
        
    except Exception as e:
        print(f"❌ 연결 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    quick_connection_test() 