import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

# ---------------------------------------------------------------------------
# 1. Ensure backend directory is in sys.path
# ---------------------------------------------------------------------------
# alembic/env.py -> alembic -> backend
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# 2. Import Base and database engine/URL from app.core.database
# ---------------------------------------------------------------------------
# Base contains the DeclarativeBase metadata which tracks all registered models.
from app.core.database import Base, SQLALCHEMY_DATABASE_URL, engine
import app.models  # Ensures all models are imported and registered with Base.metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Dynamically inject the database URL loaded securely from .env
config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

# ---------------------------------------------------------------------------
# 3. Target metadata for 'autogenerate' support
# ---------------------------------------------------------------------------
# Alembic inspects target_metadata to compare the Python ORM models with
# the live database schema when generating migrations.
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Reuse our pre-configured SQLAlchemy engine which has psycopg 3 compatibility
    # and connection health checks (pool_pre_ping=True).
    connectable = engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
