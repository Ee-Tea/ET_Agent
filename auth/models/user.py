from sqlalchemy import String, DateTime, func, Boolean, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base
import uuid

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str | None] = mapped_column(String(320), unique=True, index=True, nullable=True)
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # OAuth fields
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 'google', 'naver', etc.
    provider_account_id: Mapped[str | None] = mapped_column(String(255), nullable=True)  # Provider user ID
    access_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[str | None] = mapped_column(String(20), nullable=True)  # Unix timestamp
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)

