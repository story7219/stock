@echo off
REM PowerShell/Python 환경에서 안정적 실행을 위한 배치 파일
chcp 65001 > nul

REM (선택) 가상환경 활성화
IF EXIST .venv\Scripts\activate (
    call .venv\Scripts\activate
)

REM .env 환경변수 로드 (python-dotenv가 자동 처리)
REM 필요시 set 명령으로 직접 지정 가능
REM set GEMINI_API_KEY=your_api_key

REM 메인 실행
python main.py

REM 오류 발생 시 일시정지
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [실행 오류 발생 - 로그를 확인하세요]
    pause
) 