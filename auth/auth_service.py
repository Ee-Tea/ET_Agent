import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .config import settings
from .models.user import User, OAuthAccount
from .schemas import GoogleUserInfo, Token, TokenData, AuthResponse

# Password hashing (for future use)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.google_client_id = settings.GOOGLE_CLIENT_ID=REDACTED = settings.GOOGLE_CLIENT_SECRET=REDACTED = settings.GOOGLE_REDIRECT_URI
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token and return token data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            return TokenData(user_id=user_id)
        except JWTError:
            return None

    async def get_google_user_info(self, code: str) -> Optional[GoogleUserInfo]:
        """Exchange Google OAuth code for user info"""
        try:
            # Exchange code for access token
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.google_redirect_uri,
            }
            
            async with httpx.AsyncClient() as client:
                token_response = await client.post(token_url, data=token_data)
                token_response.raise_for_status()
                token_json = token_response.json()
                
                access_token = token_json.get("access_token")
                if not access_token:
                    return None
                
                # Get user info from Google
                user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
                headers = {"Authorization": f"Bearer {access_token}"}
                
                user_response = await client.get(user_info_url, headers=headers)
                user_response.raise_for_status()
                user_data = user_response.json()
                
                return GoogleUserInfo(
                    id=user_data["id"],
                    email=user_data["email"],
                    name=user_data["name"],
                    picture=user_data.get("picture"),
                    verified_email=user_data.get("verified_email", True)
                )
                
        except Exception as e:
            print(f"Error getting Google user info: {e}")
            return None

    def get_or_create_user(self, db: Session, google_user: GoogleUserInfo) -> tuple[User, bool]:
        """Get or create user from Google OAuth info"""
        # Check if OAuth account exists
        oauth_account = db.query(OAuthAccount).filter(
            and_(
                OAuthAccount.provider == "google",
                OAuthAccount.provider_account_id == google_user.id
            )
        ).first()
        
        if oauth_account:
            # User exists, update info
            user = oauth_account.user
            user.email = google_user.email
            user.name = google_user.name
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            return user, False
        
        # Check if user exists by email
        existing_user = db.query(User).filter(User.email == google_user.email).first()
        
        if existing_user:
            # User exists but no OAuth account, create OAuth account
            oauth_account = OAuthAccount(
                user_id=existing_user.id,
                provider="google",
                provider_account_id=google_user.id,
                access_token=None,  # We don't store access tokens for security
                refresh_token=None,
                expires_at=None,
                scope="openid email profile"
            )
            db.add(oauth_account)
            db.commit()
            return existing_user, False
        
        # Create new user and OAuth account
        new_user = User(
            email=google_user.email,
            name=google_user.name,
            is_active=True
        )
        db.add(new_user)
        db.flush()  # Get the user ID
        
        oauth_account = OAuthAccount(
            user_id=new_user.id,
            provider="google",
            provider_account_id=google_user.id,
            access_token=None,
            refresh_token=None,
            expires_at=None,
            scope="openid email profile"
        )
        db.add(oauth_account)
        db.commit()
        db.refresh(new_user)
        
        return new_user, True

    async def authenticate_google(self, db: Session, code: str) -> Optional[AuthResponse]:
        """Complete Google OAuth authentication flow"""
        # Get user info from Google
        google_user = await self.get_google_user_info(code)
        if not google_user:
            return None
        
        # Get or create user
        user, is_new_user = self.get_or_create_user(db, google_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"sub": user.id}, expires_delta=access_token_expires
        )
        
        return AuthResponse(
            user=user,
            token=Token(
                access_token=access_token,
                token_type="bearer",
                expires_in=self.access_token_expire_minutes * 60
            ),
            is_new_user=is_new_user
        )

    def get_user_by_id(self, db: Session, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()

    def get_current_user(self, db: Session, token: str) -> Optional[User]:
        """Get current user from JWT token"""
        token_data = self.verify_token(token)
        if token_data is None:
            return None
        return self.get_user_by_id(db, token_data.user_id)

# Global auth service instance
auth_service = AuthService()

