@echo off
chcp 65001 > nul
title 🚀 투자분석시스템 - 빠른시작

echo.
echo ╔══════════════════════════════════════════╗
echo ║     🚀 투자 분석 시스템 빠른 시작          ║
echo ║     Quick Start Investment Analysis       ║
echo ╚══════════════════════════════════════════╝
echo.

cd /d "%~dp0.."

echo 📁 프로젝트 디렉토리: %CD%
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
python MASTER_LAUNCHER.py

echo.
echo ✅ 실행 완료!
pause