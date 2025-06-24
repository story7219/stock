@echo off
chcp 65001 > nul
title 🚀 Premium Stock Analysis System - 통합 런처

echo.
echo ===============================================
echo  🚀 Premium Stock Analysis System
echo  🤖 Gemini AI 100%% 활용 투자 분석 시스템
echo  📊 코스피200·나스닥100·S&P500 완전 분석
echo  🎯 마스터 런처로 모든 기능 접근 가능
echo ===============================================
echo.

cd /d "%~dp0.."

echo 📁 프로젝트 디렉토리: %CD%
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
echo 🔍 Python 환경 확인 중...
python --version
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았거나 PATH에 없습니다.
    pause
    exit /b 1
)

echo.
echo 🚀 마스터 런처 실행 중...
echo 🤖 Gemini AI 투자 분석 시스템 + 파일 관리 시스템
echo.

REM 마스터 런처 실행
python MASTER_LAUNCHER.py

REM 실행 결과 확인
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 실행이 성공적으로 완료되었습니다!
) else (
    echo.
    echo ❌ 실행 중 오류가 발생했습니다.
    echo 💡 .env 파일의 API 키 설정을 확인해주세요.
)

echo.
echo 👋 프로그램을 종료합니다.
pause