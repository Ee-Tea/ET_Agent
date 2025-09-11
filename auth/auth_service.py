# auth/auth_service.py
import httpx
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .config import settings
from .models.user import User, OAuthAccount
from .schemas import GoogleUserInfo, Token, TokenData, AuthResponse

class GoogleOAuthError(Exception):
    def __init__(self, message: str, status: int | None = None, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}

class AuthService:
    # __init__ 없어도 됩니다. (프로퍼티로 직접 참조)

    @property
    def google_client_id(self) -> str:
        return settings.GOOGLE_CLIENT_ID=REDACTED
    def google_client_secret(self) -> str:
        return settings.GOOGLE_CLIENT_SECRET=REDACTED
    def google_redirect_uri(self) -> str:
        return settings.GOOGLE_REDIRECT_URI

    @property
    def secret_key(self) -> str:
        return settings.SECRET_KEY

    @property
    def algorithm(self) -> str:
        return settings.ALGORITHM

    @property
    def access_token_expire_minutes(self) -> int:
        return settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[TokenData]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            return TokenData(user_id=user_id) if user_id else None
        except JWTError:
            return None

    async def get_google_user_info(self, code: str) -> GoogleUserInfo:
        # 필수 설정 검증 (에러 조기 발견)
        if not (self.google_client_id and self.google_redirect_uri and self.google_client_secret):
            raise GoogleOAuthError("Missing GOOGLE_* settings (client_id/secret/redirect_uri)")

        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": self.google_client_id,
            "client_secret": self.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.google_redirect_uri,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            t_res = await client.post(token_url, data=token_data)
            if t_res.status_code != 200:
                try:
                    payload = t_res.json()
                except Exception:
                    payload = {"raw": t_res.text}
                raise GoogleOAuthError("Google token exchange failed", status=t_res.status_code, payload=payload)

            token_json = t_res.json()
            access_token = token_json.get("access_token")
            if not access_token:
                raise GoogleOAuthError("No access_token in token response", payload=token_json)

            ui_res = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if ui_res.status_code != 200:
                try:
                    payload = ui_res.json()
                except Exception:
                    payload = {"raw": ui_res.text}
                raise GoogleOAuthError("Google userinfo fetch failed", status=ui_res.status_code, payload=payload)

            ud = ui_res.json()
            return GoogleUserInfo(
                id=ud["id"],
                email=ud["email"],
                name=ud["name"],
                picture=ud.get("picture"),
                verified_email=ud.get("verified_email", True),
            )

    def get_or_create_user(self, db: Session, google_user: GoogleUserInfo) -> tuple[User, bool]:
        oauth = db.query(OAuthAccount).filter(
            and_(OAuthAccount.provider == "google", OAuthAccount.provider_account_id == google_user.id)
        ).first()
        if oauth:
            user = oauth.user
            user.email = google_user.email
            user.name = google_user.name
            user.updated_at = datetime.utcnow()
            db.commit(); db.refresh(user)
            return user, False

        existing = db.query(User).filter(User.email == google_user.email).first()
        if existing:
            db.add(OAuthAccount(
                user_id=existing.id, provider="google", provider_account_id=google_user.id,
                access_token=None, refresh_token=None, expires_at=None, scope="openid email profile"
            ))
            db.commit()
            return existing, False

        new_user = User(email=google_user.email, name=google_user.name, is_active=True)
        db.add(new_user); db.flush()
        db.add(OAuthAccount(
            user_id=new_user.id, provider="google", provider_account_id=google_user.id,
            access_token=None, refresh_token=None, expires_at=None, scope="openid email profile"
        ))
        db.commit(); db.refresh(new_user)
        return new_user, True

    async def authenticate_google(self, db: Session, code: str) -> AuthResponse:
        google_user = await self.get_google_user_info(code)
        user, is_new = self.get_or_create_user(db, google_user)
        token = self.create_access_token(data={"sub": user.id}, expires_delta=timedelta(minutes=self.access_token_expire_minutes))
        return AuthResponse(
            user=user,
            token=Token(access_token=token, token_type="bearer", expires_in=self.access_token_expire_minutes * 60),
            is_new_user=is_new
        )

# 전역 인스턴스
auth_service = AuthService()
__all__ = ["AuthService", "auth_service", "GoogleOAuthError"]
