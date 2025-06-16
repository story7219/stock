@echo off
chcp 65001 > nul
echo 🔧 자동 코드 품질 검사 서비스 설치
echo =====================================

echo.
echo 📦 Python 패키지 설치 중...
python -m pip install --upgrade pip
python -m pip install google-generativeai schedule python-dotenv win10toast

echo.
echo 🔧 서비스 설치 중...
python auto_quality_service.py install

echo.
echo ✅ 설치 완료!
echo 📅 매일 오전 7시에 자동으로 코드 품질 검사가 실행됩니다.
echo 📁 결과는 quality_reports 폴더에 저장됩니다.
echo.
echo 💡 추가 명령어:
echo    python auto_quality_service.py now      - 지금 바로 분석
echo    python auto_quality_service.py status   - 서비스 상태 확인
echo    python auto_quality_service.py uninstall - 서비스 제거
echo.
pause 