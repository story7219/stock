@echo off
chcp 65001 > nul
title 🚀 Ultra Premium HTS v5.0 - AI 투자 분석 시스템

echo.
echo ===============================================
echo  🚀 Ultra Premium HTS v5.0
echo  🤖 Gemini AI 100%% 활용 투자 분석 시스템
echo  📊 코스피200·나스닥100·S&P500 완전 분석
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
echo 🤖 Gemini AI 투자 분석 시스템을 시작합니다...
echo 🧹 고급 결측치 보정 및 데이터 정제 자동화 포함
echo.

REM 메인 시스템 실행
python main.py

REM 실행 결과 확인
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 분석이 성공적으로 완료되었습니다!
) else (
    echo.
    echo ❌ 실행 중 오류가 발생했습니다.
    echo 💡 .env 파일의 API 키 설정을 확인해주세요.
)

echo.
echo �� 프로그램을 종료합니다.
pause 