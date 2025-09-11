# auth/auth_service.py
import base64
import json
import httpx
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .config import settings
from .models.user import User
from .schemas import GoogleUserInfo, NaverUserInfo, Token, TokenData, AuthResponse

class GoogleOAuthError(Exception):
    def __init__(self, message: str, status: int | None = None, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


class NaverOAuthError(Exception):
    def __init__(self, message: str, status: int | None = None, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}

class AuthService:
    # __init__ 없어도 됩니다. (프로퍼티로 직접 참조)

    @property
    def google_client_id(self) -> str:
        return settings.GOOGLE_CLIENT_ID

    @property
    def google_client_secret(self) -> str:
        return settings.GOOGLE_CLIENT_SECRET

    @property
    def google_redirect_uri(self) -> str:
        return settings.GOOGLE_REDIRECT_URI

    # NAVER
    @property
    def naver_client_id(self) -> Optional[str]:
        return settings.NAVER_CLIENT_ID

    @property
    def naver_client_secret(self) -> Optional[str]:
        return settings.NAVER_CLIENT_SECRET

    @property
    def naver_redirect_uri(self) -> Optional[str]:
        return settings.NAVER_REDIRECT_URI

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
            sub = payload.get("sub")
            if sub is None:
                return None
            return TokenData(user_id=str(sub))
        except JWTError:
            return None

    async def get_google_user_info(self, code: str) -> GoogleUserInfo:
        """Google OAuth code를 access_token/id_token으로 교환하고, 사용자 정보를 안전하게 반환.
        - userinfo v1(oidc) → v2(legacy) 순서로 조회
        - email이 빠진 경우 id_token(payload)에서 보조로 추출
        - 상세 디버그 로그 포함 (운영 전 로거로 변경 권장)
        """
        # 0) 필수 설정 확인
        if not (self.google_client_id and self.google_client_secret and self.google_redirect_uri):
            raise GoogleOAuthError("Missing GOOGLE_* settings (client_id/secret/redirect_uri)")

        print("[OAUTH] start get_google_user_info")
        print("[OAUTH] redirect_uri =", self.google_redirect_uri)

        # 1) code → token 교환
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": self.google_client_id,
            "client_secret": self.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.google_redirect_uri,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            t_res = await client.post(
                token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if t_res.status_code != 200:
                try:
                    payload = t_res.json()
                except Exception:
                    payload = {"raw": t_res.text}
                print("[OAUTH] token exchange failed:", payload)
                raise GoogleOAuthError(
                    "Google token exchange failed",
                    status=t_res.status_code,
                    payload=payload,
                )

            token_json = t_res.json()
            print("[OAUTH] token_json keys:", list(token_json.keys()))
            access_token = token_json.get("access_token")
            id_token = token_json.get("id_token")
            if not access_token:
                print("[OAUTH] missing access_token:", token_json)
                raise GoogleOAuthError("No access_token in token response", payload=token_json)

            # 2) userinfo 조회 (v1 → v2 순서)
            async def _fetch_userinfo(url: str):
                r = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if r.status_code == 200:
                    try:
                        return r.json(), None
                    except Exception as e:
                        return None, {"parse_error": str(e), "raw": r.text}
                else:
                    try:
                        return None, r.json()
                    except Exception:
                        return None, {"raw": r.text, "status": r.status_code}

            ud, err_v1 = await _fetch_userinfo("https://openidconnect.googleapis.com/v1/userinfo")
            if ud is None:
                print("[OAUTH] userinfo v1 failed:", err_v1)
                ud, err_v2 = await _fetch_userinfo("https://www.googleapis.com/oauth2/v2/userinfo")
                if ud is None:
                    print("[OAUTH] userinfo v2 also failed:", err_v2)
                    raise GoogleOAuthError(
                        "Google userinfo fetch failed",
                        status=400,
                        payload={"v1_error": err_v1, "v2_error": err_v2},
                    )

            # 3) 안전 파싱 (KeyError 방지)
            #    - email이 없을 수 있으니 .get 사용
            #    - id_token payload에서 보조 추출 시도
            def _b64url_decode(s: str) -> bytes:
                # base64url 패딩 보정
                s += "=" * ((4 - len(s) % 4) % 4)
                return base64.urlsafe_b64decode(s.encode("utf-8"))

            email = ud.get("email")
            email_verified = ud.get("email_verified")
            sub_or_id = ud.get("sub") or ud.get("id")
            name = ud.get("name") or ud.get("given_name") or ""
            picture = ud.get("picture")

            print("[OAUTH] userinfo sample:", {
                "keys": list(ud.keys()),
                "sub_or_id": sub_or_id,
                "email_present": bool(email),
                "email_verified": email_verified,
                "name": name,
                "has_picture": bool(picture),
            })

            # userinfo에 email이 없다면 id_token payload에서 보조 추출
            if not email and id_token:
                try:
                    parts = id_token.split(".")
                    if len(parts) >= 2:
                        payload_bytes = _b64url_decode(parts[1])
                        idp = json.loads(payload_bytes.decode("utf-8"))

                        print("[OAUTH] id_token payload keys:", list(idp.keys()))
                        email = idp.get("email") or email
                        if email_verified is None:
                            email_verified = idp.get("email_verified")
                        if sub_or_id is None:
                            sub_or_id = idp.get("sub")
                        if not name:
                            name = idp.get("name") or idp.get("given_name") or name
                        if not picture:
                            picture = idp.get("picture") or picture
                except Exception as e:
                    print("[OAUTH] id_token decode error:", repr(e))

            # 최종적으로 email이 없으면 명확한 가이드와 함께 에러
            if not email:
                raise GoogleOAuthError(
                    "Google userinfo has no 'email'. Ensure scope includes 'openid email profile'.",
                    status=400,
                    payload={
                        "userinfo_keys": list(ud.keys()),
                        "token_keys": list(token_json.keys()),
                    },
                )

            # 4) 결과 조립 (verified_email 기본 True로 보정)
            if email_verified is None:
                email_verified = True

            return GoogleUserInfo(
                id=sub_or_id or "",
                email=email,
                name=name,
                picture=picture,
                verified_email=bool(email_verified),
            )

    def get_or_create_user(self, db: Session, google_user: GoogleUserInfo) -> tuple[User, bool]:
        # OAuth 계정으로 사용자 찾기
        existing = db.query(User).filter(
            and_(User.provider == "google", User.provider_account_id == google_user.id)
        ).first()
        if existing:
            existing.email = google_user.email
            existing.name = google_user.name
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing, False

        # 이메일로 기존 사용자 찾기
        existing_by_email = db.query(User).filter(User.email == google_user.email).first()
        if existing_by_email:
            existing_by_email.provider = "google"
            existing_by_email.provider_account_id = google_user.id
            existing_by_email.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing_by_email)
            return existing_by_email, False

        new_user = User(
            email=google_user.email, 
            name=google_user.name, 
            is_active=True,
            provider="google",
            provider_account_id=google_user.id,
            scope="openid email profile"
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user, True

    async def authenticate_google(self, db: Session, code: str) -> AuthResponse:
        google_user = await self.get_google_user_info(code)
        user, is_new = self.get_or_create_user(db, google_user)
        token = self.create_access_token(data={"sub": str(user.user_id)})
        
        
        return AuthResponse(
            user=user,
            token=Token(access_token=token, token_type="bearer", expires_in=self.access_token_expire_minutes * 60),
            is_new_user=is_new
        )


class NaverOAuthService:
    @property
    def client_id(self) -> Optional[str]:
        return settings.NAVER_CLIENT_ID

    @property
    def client_secret(self) -> Optional[str]:
        return settings.NAVER_CLIENT_SECRET

    @property
    def redirect_uri(self) -> Optional[str]:
        return settings.NAVER_REDIRECT_URI

    async def exchange_token(self, code: str, state: Optional[str]) -> dict:
        if not (self.client_id and self.client_secret and self.redirect_uri):
            raise NaverOAuthError("Missing NAVER_* settings (client_id/secret/redirect_uri)")
        token_url = "https://nid.naver.com/oauth2.0/token"
        # Naver 권장: x-www-form-urlencoded 로 POST 전송 (GET도 지원하나 환경에 따라 실패 사례 존재)
        form = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "state": state or "",
            "redirect_uri": self.redirect_uri,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(
                token_url,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            try:
                payload = res.json()
            except Exception:
                payload = {"raw": res.text}
            # access_token 미포함 또는 오류 필드가 있으면 명확히 에러 반환
            if res.status_code != 200 or (isinstance(payload, dict) and payload.get("error")):
                # 디버그에 도움 되도록 전체 payload 동봉
                raise NaverOAuthError("Naver token exchange failed", status=res.status_code, payload=payload)
            return payload

    async def fetch_userinfo(self, access_token: str) -> NaverUserInfo:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(
                "https://openapi.naver.com/v1/nid/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if res.status_code != 200:
                try:
                    payload = res.json()
                except Exception:
                    payload = {"raw": res.text}
                raise NaverOAuthError("Naver userinfo fetch failed", status=res.status_code, payload=payload)
            ud = res.json() or {}
            response = (ud or {}).get("response", {})
            return NaverUserInfo(
                id=str(response.get("id", "")),
                email=response.get("email"),
                name=response.get("name"),
                nickname=response.get("nickname"),
                profile_image=response.get("profile_image"),
                mobile=response.get("mobile"),
            )

    def get_or_create_user(self, db: Session, naver_user: NaverUserInfo) -> tuple[User, bool]:
        # OAuth 계정으로 사용자 찾기
        existing = db.query(User).filter(
            and_(User.provider == "naver", User.provider_account_id == naver_user.id)
        ).first()
        if existing:
            if naver_user.email:
                existing.email = naver_user.email
            existing.name = naver_user.name or naver_user.nickname or existing.name
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing, False

        email = naver_user.email
        if email:
            existing_by_email = db.query(User).filter(User.email == email).first()
            if existing_by_email:
                existing_by_email.provider = "naver"
                existing_by_email.provider_account_id = naver_user.id
                existing_by_email.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing_by_email)
                return existing_by_email, False

        new_user = User(
            email=email, 
            name=naver_user.name or naver_user.nickname or "", 
            is_active=True,
            provider="naver",
            provider_account_id=naver_user.id,
            scope="profile"
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user, True

    async def authenticate(self, db: Session, code: str, state: Optional[str]) -> AuthResponse:
        token_json = await self.exchange_token(code, state)
        access_token = token_json.get("access_token")
        if not access_token:
            raise NaverOAuthError("No access_token in Naver token response", payload=token_json)
        userinfo = await self.fetch_userinfo(access_token)
        user, is_new = self.get_or_create_user(db, userinfo)
        jwt_token = auth_service.create_access_token(data={"sub": str(user.user_id)})
        return AuthResponse(
            user=user,
            token=Token(access_token=jwt_token, token_type="bearer", expires_in=auth_service.access_token_expire_minutes * 60),
            is_new_user=is_new,
        )


# 전역 인스턴스
auth_service = AuthService()
naver_oauth_service = NaverOAuthService()
__all__ = [
    "AuthService",
    "auth_service",
    "GoogleOAuthError",
    "NaverOAuthError",
    "NaverOAuthService",
    "naver_oauth_service",
]
