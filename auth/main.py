# auth/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .auth_routes import router as auth_router  # 이미 prefix="/auth" 붙어있음

app = FastAPI(title="Auth API")

# 프론트(3000)에서 /auth/google를 fetch 하고 싶다면 CORS 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 라우터 등록 (/auth/*)
app.include_router(auth_router)

# 헬스 체크
@app.get("/auth/health")
async def auth_health():
    return {"status": "healthy", "service": "authentication"}

