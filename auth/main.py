# auth/main.py
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from .auth_routes import router as auth_router  # prefix="/auth"

FRONTEND_ORIGIN = "http://localhost:3000"  # 필요하면 .env로 관리

app = FastAPI(title="Auth API")

# CORS는 반드시 최종 app 인스턴스에 1회만 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],  # "*" 금지, 정확한 오리진만
    allow_credentials=True,           # 쿠키 전달 허용
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
