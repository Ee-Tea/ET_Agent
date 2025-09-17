import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import settings

# DATABASE_URL 환경변수 우선, 기본값은 컨테이너 네트워크 서비스명 사용
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    settings.DATABASE_URL if settings.DATABASE_URL else "postgresql+psycopg://postgres:postgres@langgraph-postgres:5432/postgres",
)

# SQLAlchemy 2.0 스타일
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db():
    """FastAPI Depends용 세션 생성기"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
