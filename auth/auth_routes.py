from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from .db import get_db
from .auth_service import auth_service, GoogleOAuthError
from .schemas import AuthResponse, User, GoogleAuthRequest
from .config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])

def get_current_user(db: Session = Depends(get_db), token: str = Depends(lambda: None)):
    """Dependency to get current user from Authorization header"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = auth_service.get_current_user(db, token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@router.get("/google")
async def google_auth():
    """Initiate Google OAuth flow"""
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={settings.GOOGLE_CLIENT_ID=REDACTED"redirect_uri={settings.GOOGLE_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=openid email profile&"
        f"access_type=offline"
    )
    return {"auth_url": google_auth_url}

@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db)
):
    print("[OAUTH] settings.GOOGLE_REDIRECT_URI =", settings.GOOGLE_REDIRECT_URI)
    print("[OAUTH] query_params:", dict(request.query_params))
    print("[OAUTH] code:", code, "error:", error)
    print("[OAUTH] settings.GOOGLE_REDIRECT_URI:", settings.GOOGLE_REDIRECT_URI)
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")

    try:
        auth_response = await auth_service.authenticate_google(db, code)
        return auth_response
    except GoogleOAuthError as e:
        # 개발 중엔 400으로 상세 원인 그대로 노출
        raise HTTPException(
            status_code=400,
            detail={
                "message": e.args[0],
                "status": e.status,
                "payload": e.payload,  # 예: {"error":"invalid_grant","error_description":"..."}
            },
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return current_user

@router.post("/refresh")
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token"""
    from datetime import timedelta
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": current_user.id}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/logout")
async def logout():
    """Logout user (client should discard token)"""
    return {"message": "Successfully logged out"}

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
