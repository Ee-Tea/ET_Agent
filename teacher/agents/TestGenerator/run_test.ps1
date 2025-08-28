# RAGAS 시스템 테스트 실행 스크립트
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RAGAS 시스템 테스트 실행" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "현재 디렉토리: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

Write-Host "1. uv 환경에서 RAGAS 테스트 실행..." -ForegroundColor Green
Write-Host ""

try {
    # uv 환경에서 Python 스크립트 실행
    uv run python run_test.py
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "테스트 완료" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
} catch {
    Write-Host "오류 발생: $_" -ForegroundColor Red
    Write-Host "uv 환경을 확인해주세요." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "아무 키나 누르면 종료됩니다..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
