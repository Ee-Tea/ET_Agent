from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from typing import Literal

class UserPublic(BaseModel):
    id: str
    name: str
    email: EmailStr
    picture: str | None = None
    provider: Literal["google", "kakao", "naver"] = "google"

# User schemas
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    pass

class User(UserBase):
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# OAuth schemas
class OAuthAccountBase(BaseModel):
    provider: str
    provider_account_id: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    scope: Optional[str] = None

class OAuthAccountCreate(OAuthAccountBase):
    user_id: str

class OAuthAccount(OAuthAccountBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    user_id: str | None = None

class GoogleUserInfo(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    verified_email: bool = True

class GoogleAuthRequest(BaseModel):
    code: str
    state: Optional[str] = None

# Naver schemas
class NaverUserInfo(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    nickname: Optional[str] = None
    profile_image: Optional[str] = None
    mobile: Optional[str] = None

# Response schemas
class AuthResponse(BaseModel):
    user: User
    token: Token
    is_new_user: bool = False

