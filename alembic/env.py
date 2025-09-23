# alembic/env.py
from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# --- [PATH 설정] 프로젝트 루트 추가 (alembic/ 상위의 상위 경로) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- [모델/메타데이터 import] ---
# Base = declarative_base()가 정의된 곳
from auth.models.user import Base  # Base 선언을 여기서 가져옴

# autogenerate가 먹도록 실제 모델 모듈을 import해서 mapper 등록을 보장
# (중요: Base만 가져오면 테이블이 비어있을 수 있음)
import auth.models.user  # noqa: F401  # User, OAuthAccount 등 테이블 선언 모듈

# --- Alembic 기본 설정 로드 ---
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_db_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        x = context.get_x_argument(as_dictionary=True)
        url = x.get("dburl")
    if not url:
        url = config.get_main_option("sqlalchemy.url", "")
    if not url:
        raise RuntimeError("DATABASE_URL or -x dburl or sqlalchemy.url must be set")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,      # 컬럼 타입 변경 감지
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # alembic.ini 로드 후 URL 주입
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _get_db_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
