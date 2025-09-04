#!/usr/bin/env python3
"""
ET-Agent FastAPI 서버 실행 스크립트
"""

import os
import sys
import uvicorn
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """FastAPI 서버 실행"""
    
    # 환경 변수 설정
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6380")
    os.environ.setdefault("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
    os.environ.setdefault("LLM_TEMPERATURE", "0.2")
    
    # 서버 설정
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("API_LOG_LEVEL", "info")
    
    print(f"🚀 ET-Agent FastAPI 서버 시작")
    print(f"📍 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"🔍 헬스 체크: http://{host}:{port}/health")
    print(f"🔄 리로드: {'활성화' if reload else '비활성화'}")
    print("-" * 50)
    
    # FastAPI 서버 실행
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()


