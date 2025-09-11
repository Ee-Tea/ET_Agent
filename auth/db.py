from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import settings

from sqlalchemy.engine import make_url
print("[DB] URL =", settings.DATABASE_URL, "driver=", make_url(settings.DATABASE_URL).drivername)


# SQLAlchemy 2.0 스타일
engine = create_engine(
    settings.DATABASE_URL,
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
