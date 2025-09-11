# auth/main.py
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import os, json
from .auth_routes import router as auth_router  # prefix="/auth"

def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if not raw:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://172.29.208.1:3000",
        ]
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return [v.strip() for v in raw.split(",") if v.strip()]

app = FastAPI(title="Auth API")

# CORS는 반드시 최종 app 인스턴스에 1회만 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (CORS 전에/후에 어느쪽이든 OK, 중요한 건 app 재생성 금지)
app.include_router(auth_router)

# 헬스 체크
@app.get("/auth/health")
async def auth_health():
    return {"status": "healthy", "service": "authentication"}

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def _chrome_probe():
    return Response(status_code=204)

@app.get("/favicon.ico", include_in_schema=False)
async def _favicon():
    return Response(status_code=204)
