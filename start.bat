@echo off
chcp 65001 > nul
title 🚀 Ultra Stock Analysis System v5.0

echo.
echo ===============================================
echo  🚀 Ultra Stock Analysis System v5.0
echo  코스피200·나스닥100·S&P500 AI 분석 시스템
echo ===============================================
echo.

REM 가상환경 확인 및 활성화
if exist ".venv\Scripts\activate.bat" (
    echo 🔧 가상환경 활성화 중...
    call .venv\Scripts\activate.bat
    echo ✅ 가상환경 활성화 완료
) else (
    echo ⚠️ 가상환경이 없습니다. 전역 Python을 사용합니다.
)

echo.
echo 🎨 GUI 애플리케이션을 시작합니다...
echo.

REM Python 스크립트 실행
python start.py

REM 오류 발생 시 대기
if errorlevel 1 (
    echo.
    echo ❌ 실행 중 오류가 발생했습니다.
    pause
)

echo.
echo 👋 프로그램이 종료되었습니다.
pause 