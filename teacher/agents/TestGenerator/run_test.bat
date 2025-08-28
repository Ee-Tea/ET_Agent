@echo off
echo ========================================
echo RAGAS 시스템 테스트 실행
echo ========================================
echo.

echo 현재 디렉토리: %CD%
echo.

echo 1. uv 환경에서 RAGAS 테스트 실행...
echo.

uv run python run_test.py

echo.
echo ========================================
echo 테스트 완료
echo ========================================
pause
