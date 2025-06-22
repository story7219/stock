@echo off
chcp 65001 > nul
title 고성능 HTS (Home Trading System)

echo ================================================================
echo                고성능 HTS (Home Trading System) v1.0
echo                비동기 처리 ^| 멀티레벨 캐싱 ^| 성능 최적화
echo ================================================================
echo.

REM 가상환경 활성화 (존재하는 경우)
if exist "venv\Scripts\activate.bat" (
    echo 🔧 가상환경 활성화 중...
    call venv\Scripts\activate.bat
)

REM Python 버전 확인
echo 🐍 Python 버전 확인:
python --version
echo.

REM 필수 패키지 설치 확인
echo 📦 필수 패키지 설치 확인:
if exist "requirements.txt" (
    echo 패키지 설치 중... (시간이 걸릴 수 있습니다)
    pip install -r requirements.txt --quiet
    if %errorlevel% neq 0 (
        echo ❌ 패키지 설치 실패! 수동으로 설치해주세요:
        echo pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✅ 패키지 설치 완료
) else (
    echo ⚠️ requirements.txt 파일이 없습니다.
)
echo.

REM 로그 디렉토리 생성
if not exist "logs" mkdir logs

REM 캐시 디렉토리 생성
if not exist "cache" mkdir cache

REM 데이터 디렉토리 생성
if not exist "data" mkdir data

echo 🚀 HTS 시스템 시작...
echo.

REM HTS 애플리케이션 실행
python main.py

REM 실행 결과 확인
if %errorlevel% neq 0 (
    echo.
    echo ❌ HTS 시스템 실행 중 오류가 발생했습니다.
    echo 로그 파일을 확인해주세요: hts_application.log
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ HTS 시스템이 정상적으로 종료되었습니다.
echo.
pause 