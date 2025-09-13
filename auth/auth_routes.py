import json
from fastapi import APIRouter, Cookie, Depends, HTTPException, Header, status, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from urllib.parse import urlparse
from typing import Optional

from .db import get_db
from .auth_service import auth_service, GoogleOAuthError
from .schemas import AuthResponse, User, GoogleAuthRequest
from .schemas import UserPublic 
from .config import settings
from fastapi.responses import RedirectResponse
from .schemas import AuthResponse, User as UserSchema  # <- 스키마 이름 충돌 방지
from .models.user import User as UserModel   
from secrets import token_urlsafe
from .auth_service import naver_oauth_service, kakao_oauth_service

router = APIRouter(prefix="/auth", tags=["authentication"])

# --- (추가) 디버그용 마스킹/직렬화 헬퍼 ---
def _mask(val: Optional[str]) -> Optional[str]:
    if not val or not isinstance(val, str):
        return val
    if len(val) <= 12:
        return "***"
    return f"{val[:6]}...{val[-6:]}"

def _sanitize_auth_response(ar) -> dict:
    """
    AuthResponse(pydantic v2) 기준:
    {
      "user": {...},
      "token": {"access_token": "...", "refresh_token": "...", "expires_in": 3600},
      "is_new": bool
    }
    """
    try:
        data = ar.model_dump() if hasattr(ar, "model_dump") else (
            ar.dict() if hasattr(ar, "dict") else dict(ar) if isinstance(ar, dict) else {}
        )
    except Exception:
        data = {}

    token = data.get("token", {}) or {}
    if isinstance(token, dict):
        if "access_token" in token:
            token["access_token"] = _mask(token["access_token"])
        if "refresh_token" in token:
            token["refresh_token"] = _mask(token.get("refresh_token"))
    data["token"] = token
    return data

def _get_cookie_from_header(request: Request, name: str) -> Optional[str]:
    raw = request.headers.get("cookie") or ""
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k.strip() == name:
            return v
    return None

def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(default=None),
    # 표준 세션 쿠키 (HTTPOnly)
    auth_session: Optional[str] = Cookie(default=None),
    # 하위 호환/폴백
    access_token: Optional[str] = Cookie(default=None),
    access_token_web: Optional[str] = Cookie(default=None),
) -> UserModel:
    # 디버그(임시): 실제로 쿠키/헤더가 들어오는지 로그
    print("[AUTH] dep Cookie exists:", bool(access_token))
    print("[AUTH] raw Cookie header:", request.headers.get("cookie"))

    # 1) 표준 세션 쿠키 우선
    token = auth_session
    # 2) 하위 호환: access_token(HTTPOnly)
    if not token:
        token = access_token
    if not token:
        token = _get_cookie_from_header(request, "access_token")
    if not token:
        token = request.cookies.get("auth_session")
    if not token:
        token = _get_cookie_from_header(request, "auth_session")
    # fallback: JS 접근 가능한 토큰도 허용 (동일한 JWT를 복제 저장)
    if not token and access_token_web:
        token = access_token_web
    if not token:
        token = _get_cookie_from_header(request, "access_token_web")
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})

    token_data = auth_service.verify_token(token)
    if not token_data or not token_data.user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials", headers={"WWW-Authenticate": "Bearer"})

    user = db.query(UserModel).filter(UserModel.user_id == token_data.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found", headers={"WWW-Authenticate": "Bearer"})
    return user


@router.get("/google")
async def google_auth():
    """Initiate Google OAuth flow"""
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={settings.GOOGLE_CLIENT_ID}&"
        f"redirect_uri={settings.GOOGLE_REDIRECT_URI}&"
        "response_type=code&"
        "scope=openid%20email%20profile&"  # ✅ 반드시 이렇게
        "access_type=offline&"
        "include_granted_scopes=true&"
        "prompt=consent"
    )
    return RedirectResponse(url=google_auth_url, status_code=302)

from fastapi.responses import HTMLResponse

@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db),
):
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")

    auth_response = await auth_service.authenticate_google(db, code)

    # 1) 여기서 8124 도메인에 쿠키 'access_token'을 확실히 심음
    html = f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Login OK</title></head>
  <body>
    <p>로그인 완료! 잠시 후 이동합니다...</p>
    <script>
      // 2) 쿠키가 박힌 뒤 프론트(3000)로 이동
      window.location.replace("{settings.FRONTEND_BASE_URL}/?login=ok");
    </script>
  </body>
</html>"""

    resp = HTMLResponse(content=html, status_code=200)
    # 표준 세션 쿠키
    resp.set_cookie(
        key="auth_session",
        value=auth_response.token.access_token,
        httponly=True,
        secure=False,     # 로컬 http (운영 True)
        samesite="lax",
        path="/",
        max_age=auth_response.token.expires_in,
    )
    # 하위 호환: access_token 유지(HTTPOnly)
    try:
        resp.set_cookie(
            key="access_token",
            value=auth_response.token.access_token,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
            max_age=auth_response.token.expires_in,
        )
    except Exception:
        pass
    # 프론트 디버깅/폴백용(로컬 개발): JS에서 읽을 수 있는 토큰 복제 쿠키
    try:
        resp.set_cookie(
            key="access_token_web",
            value=auth_response.token.access_token,
            httponly=False,
            secure=False,
            samesite="lax",
            path="/",
            max_age=auth_response.token.expires_in,
        )
    except Exception:
        pass
    return resp


# ================= KAKAO OAuth =================
@router.get("/kakao")
async def kakao_auth():
    if not (settings.KAKAO_CLIENT_ID and settings.KAKAO_REDIRECT_URI):
        raise HTTPException(status_code=500, detail="KAKAO OAuth not configured")
    state = token_urlsafe(16)
    auth_url = (
        "https://kauth.kakao.com/oauth/authorize?"
        f"response_type=code&client_id={settings.KAKAO_CLIENT_ID}&"
        f"redirect_uri={settings.KAKAO_REDIRECT_URI}&state={state}"
    )
    resp = RedirectResponse(url=auth_url, status_code=302)
    resp.set_cookie("kakao_oauth_state", state, httponly=True, samesite="lax", path="/")
    return resp

@router.get("/kakao/callback")
async def kakao_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db),
):
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")

    cookie_state = request.cookies.get("kakao_oauth_state")
    if cookie_state and state and cookie_state != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    auth_response = await kakao_oauth_service.authenticate(db, code)

    html = f"""<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Login OK</title></head>
  <body>
    <p>카카오 로그인 완료! 잠시 후 이동합니다...</p>
    <script>
      window.location.replace(\"{settings.FRONTEND_BASE_URL}/?login=ok\");
    </script>
  </body>
</html>"""

    resp = HTMLResponse(content=html, status_code=200)
    resp.delete_cookie("kakao_oauth_state", path="/")
    resp.set_cookie(
        key="auth_session",
        value=auth_response.token.access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
        max_age=auth_response.token.expires_in,
    )
    try:
        resp.set_cookie(
            key="access_token",
            value=auth_response.token.access_token,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
            max_age=auth_response.token.expires_in,
        )
    except Exception:
        pass
    try:
        resp.set_cookie(
            key="access_token_web",
            value=auth_response.token.access_token,
            httponly=False,
            secure=False,
            samesite="lax",
            path="/",
            max_age=auth_response.token.expires_in,
        )
    except Exception:
        pass
    return resp

@router.get("/me", response_model=UserPublic)
async def get_current_user_info(current_user: UserModel = Depends(get_current_user)):
    return UserPublic(
        id=str(current_user.user_id),
        name=(current_user.name or ""),
        email=current_user.email,
        picture=getattr(current_user, "picture", None),
        provider=(current_user.provider or "google"),
    )

@router.post("/refresh")
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token"""
    from datetime import timedelta
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": current_user.user_id}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/logout")
async def logout():
    """Logout user: delete auth cookies for both with/without domain."""
    from urllib.parse import urlparse
    resp = JSONResponse({"message": "Logged out"})
    frontend_host = None
    try:
        frontend_host = urlparse(settings.FRONTEND_BASE_URL).hostname or None
    except Exception:
        frontend_host = None

    # delete with and without domain to cover how cookie was set
    for dom in (None, frontend_host):
        try:
            resp.delete_cookie("auth_session", path="/", domain=dom)
        except Exception:
            pass
        try:
            resp.delete_cookie("access_token", path="/", domain=dom)
        except Exception:
            pass
        try:
            resp.delete_cookie("access_token_web", path="/", domain=dom)
        except Exception:
            pass
        try:
            resp.delete_cookie("kakao_oauth_state", path="/", domain=dom)
        except Exception:
            pass
        try:
            resp.delete_cookie("naver_oauth_state", path="/", domain=dom)
        except Exception:
            pass
    return resp

# Health check endpoint
@router.get("/health")
async def auth_health():
    """Authentication service health check"""
    return {"status": "healthy", "service": "authentication"}

@router.get("/new")
async def auth_info():
    """인증 API 정보"""
    return {
        "message": "ET Agent Authentication API",
        "version": "1.0.0",
        "docs": "/docs",  # FastAPI에서 기본적으로 제공하는 문서 경로
        "auth_endpoints": {
            "google_auth": "/auth/google",
            "google_callback": "/auth/google/callback",
            "me": "/auth/me",
            "refresh": "/auth/refresh",
            "logout": "/auth/logout"
        }
    }

@router.get("/favicon.ico", include_in_schema=False)
async def favicon_noop():
    from fastapi import Response
    return Response(status_code=204)

@router.get("/debug/cookies")
def debug_cookies(request: Request):
    # 브라우저가 서버로 보낸 쿠키를 그대로 보여줌
    return {"cookies": dict(request.cookies)}

@router.get("/debug/headers")
def debug_headers(request: Request):
    # Authorization 헤더도 같이 확인
    return {
        "headers": dict(request.headers),
        "cookies": dict(request.cookies),
    }

# ===== Debug helpers: set/clear cookie to test proxy & browser behavior =====
@router.get("/debug/set-cookie")
def debug_set_cookie():
    # 실제 JWT 발급으로 즉시 /auth/me 검증 가능
    token = auth_service.create_access_token({"sub": "debug-user"})
    resp = JSONResponse({"ok": True, "set": "auth_session"})
    resp.set_cookie(
        key="auth_session",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
        max_age=300,
    )
    try:
        resp.set_cookie(
            key="access_token_web",
            value=token,
            httponly=False,
            secure=False,
            samesite="lax",
            path="/",
            max_age=300,
        )
    except Exception:
        pass
    return resp

@router.get("/debug/clear-cookie")
def debug_clear_cookie():
    resp = JSONResponse({"ok": True, "cleared": True})
    for name in ("auth_session", "access_token", "access_token_web"):
        try:
            resp.delete_cookie(name, path="/")
        except Exception:
            pass
    return resp

@router.get("/debug/whoami")
def debug_whoami(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    auth_session: Optional[str] = Cookie(default=None),
    access_token: Optional[str] = Cookie(default=None),
    access_token_web: Optional[str] = Cookie(default=None),
):
    token = auth_session or access_token or access_token_web
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    data = auth_service.verify_token(token) if token else None
    return {"token_user_id": getattr(data, "user_id", None), "has_token": bool(token)}

@router.post("/debug/upsert-user")
def debug_upsert_user(
    request: Request,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(default=None),
    auth_session: Optional[str] = Cookie(default=None),
    access_token: Optional[str] = Cookie(default=None),
    access_token_web: Optional[str] = Cookie(default=None),
):
    token = auth_session or access_token or access_token_web
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    data = auth_service.verify_token(token) if token else None
    user_id = getattr(data, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="No token or invalid token")
    existing = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if existing:
        return {"ok": True, "user_id": user_id, "created": False}
    u = UserModel(
        user_id=str(user_id),
        name="Debug User",
        is_active=True,
        provider="debug",
        provider_account_id=str(user_id),
        scope="debug",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"ok": True, "user_id": u.user_id, "created": True}

# ================= NAVER OAuth =================
@router.get("/naver")
async def naver_auth(request: Request):
    if not (settings.NAVER_CLIENT_ID and settings.NAVER_REDIRECT_URI):
        raise HTTPException(status_code=500, detail="NAVER OAuth not configured")

    state = token_urlsafe(16)
    # 간단히 쿠키에 저장 (실서비스는 서버세션/스토리지 권장)
    auth_url = (
        "https://nid.naver.com/oauth2.0/authorize?"
        f"response_type=code&client_id={settings.NAVER_CLIENT_ID}&"
        f"redirect_uri={settings.NAVER_REDIRECT_URI}&state={state}"
    )
    resp = RedirectResponse(url=auth_url, status_code=302)
    resp.set_cookie("naver_oauth_state", state, httponly=True, samesite="lax", path="/")
    return resp

@router.get("/naver/callback")
async def naver_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db),
):
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")

    # CSRF state 확인
    cookie_state = request.cookies.get("naver_oauth_state")
    if cookie_state and state and cookie_state != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    auth_response = await naver_oauth_service.authenticate(db, code, state)

    html = f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Login OK</title></head>
  <body>
    <p>네이버 로그인 완료! 잠시 후 이동합니다...</p>
    <script>
      window.location.replace("{settings.FRONTEND_BASE_URL}/?login=ok");
    </script>
  </body>
</html>"""

    resp = HTMLResponse(content=html, status_code=200)
    resp.delete_cookie("naver_oauth_state", path="/")
    # 표준 세션 쿠키 설정
    resp.set_cookie(
        key="auth_session",
        value=auth_response.token.access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
        max_age=auth_response.token.expires_in,
    )
    # 하위 호환: access_token(HTTPOnly)도 유지
    try:
        resp.set_cookie(
            key="access_token",
            value=auth_response.token.access_token,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
            max_age=auth_response.token.expires_in,
        )
    except Exception:
        pass
    try:
        resp.set_cookie(
            key="access_token_web",
            value=auth_response.token.access_token,
            httponly=False,
            secure=False,
            samesite="lax",
            path="/",
            max_age=auth_response.token.expires_in,
        )
    except Exception:
        pass
    return resp