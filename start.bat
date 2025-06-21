@echo off
chcp 65001 > nul
title 🚀 주식 분석 시스템 with Enhanced Token Manager

echo ==========================================
echo 🚀 주식 분석 시스템 시작
echo ==========================================
echo.
echo 📊 시스템 기능:
echo   ✅ 매일 오전 7시 자동 토큰 발행
echo   ✅ 코드 품질 자동 검사
echo   ✅ 코스피 200 전체 종목 분석
echo   ✅ 실시간 모니터링
echo   ✅ 텔레그램 알림
echo.
echo 🔧 시작 중...
echo.

REM 가상환경 활성화 (있다면)
if exist "venv\Scripts\activate.bat" (
    echo 📦 가상환경 활성화 중...
    call venv\Scripts\activate.bat
)

REM 의존성 설치
echo 📦 의존성 확인 및 설치 중...
pip install -r requirements.txt --quiet

REM 메인 시스템 실행
echo ✅ 시스템 실행 중...
echo.
python main.py

echo.
echo 👋 시스템이 종료되었습니다.
pause 