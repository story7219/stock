@echo off
chcp 65001 > nul
echo 🗑️ 자동 코드 품질 검사 서비스 제거
echo =====================================

python auto_quality_service.py uninstall

echo.
echo ✅ 제거 완료!
pause 