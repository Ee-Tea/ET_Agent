# auth/main.py
import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware  # 리버스프록시 대비
from .auth_routes import router as auth_router
from .db import init_db

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
ENV = os.getenv("ENV", "dev")  # dev / prod
IS_DEV = ENV != "prod"

app = FastAPI(title="Auth API")

# (선택) 프록시 뒤에 있을 때 X-Forwarded-* 신뢰
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],   # 와일드카드 금지
    allow_credentials=True,            # 쿠키 전달 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

def set_session_cookie(resp: Response, key: str, value: str):
    """
    개발/운영에 맞춰 세션 쿠키 속성 설정
    """
    if IS_DEV:
        resp.set_cookie(
            key=key,
            value=value,
            httponly=True,
            samesite="lax",   # 로컬 HTTP에서 안전
            secure=False,     # HTTP이므로 False
            path="/",
        )
    else:
        resp.set_cookie(
            key=key,
            value=value,
            httponly=True,
            samesite="none",  # 크로스사이트 요청 허용
            secure=True,      # HTTPS 필수
            path="/",
            # domain=".example.com",  # 필요할 때만 지정
        )

app.include_router(auth_router)

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/auth/health")
async def auth_health():
    return {"status": "healthy", "service": "authentication"}

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def _chrome_probe():
    return Response(status_code=204)
