# cleanup.ps1 - 프로젝트 자동 정리 스크립트
# 한글 깨짐 방지를 위해 출력 인코딩을 UTF-8로 설정
[System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# --- 정리 후 남겨둘 필수 파일 및 폴더 목록 ---
$itemsToKeep = @(
    ".git",              # Git 히스토리 (필수)
    ".github",           # GitHub 워크플로우 (필수)
    "maintenance_tools", # 코드 관리 도구 폴더
    ".env",              # 모든 API 키 (필수)
    ".gitignore",        # Git 무시 목록
    "main.py",           # 핵심 파일 1
    "config.py",         # 핵심 파일 2
    "core_trader.py",    # 핵심 파일 3
    "analysis_engine.py",# 핵심 파일 4
    "requirements.txt",  # 라이브러리 목록
    "cleanup.ps1"        # 자기 자신을 지우지 않도록 추가
)

# 현재 디렉토리의 모든 항목 (숨김 파일 포함)
$allItems = Get-ChildItem -Path . -Force | Select-Object -ExpandProperty Name

# 삭제할 항목 목록 생성
$itemsToDelete = @()
foreach ($item in $allItems) {
    if ($itemsToKeep -notcontains $item) {
        $itemsToDelete += $item
    }
}

if ($itemsToDelete.Count -eq 0) {
    Write-Host "✅ 모든 파일이 이미 정리되어 있습니다. 삭제할 파일이 없습니다." -ForegroundColor Green
    exit
}

# --- 사용자에게 최종 확인 받기 ---
Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "🔥 아래의 파일 및 폴더들이 영구적으로 삭제될 예정입니다 🔥" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Yellow
$itemsToDelete | ForEach-Object { Write-Host "- $_" }
Write-Host "==================================================" -ForegroundColor Yellow
$confirmation = Read-Host "❓ 정말로 위의 모든 항목을 삭제하시겠습니까? (y/n)"

if ($confirmation -ne 'y') {
    Write-Host "작업이 취소되었습니다." -ForegroundColor Red
    exit
}

# --- 확인 후 삭제 실행 ---
Write-Host "삭제 작업을 시작합니다..."
foreach ($item in $itemsToDelete) {
    $itemPath = Join-Path -Path . -ChildPath $item
    try {
        Remove-Item -Path $itemPath -Recurse -Force -ErrorAction Stop
        Write-Host "✅ 삭제됨: $item" -ForegroundColor Green
    } catch {
        Write-Host "❌ 오류: $item 삭제 실패. $_" -ForegroundColor Red
    }
}

Write-Host "🎉 모든 불필요한 파일 정리가 성공적으로 완료되었습니다!" 