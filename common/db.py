from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import settings

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
