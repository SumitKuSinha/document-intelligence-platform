"""
Database Connection Layer

This module configures the PostgreSQL database connection using SQLAlchemy 2.x.
It handles:
1. Loading the DATABASE_URL environment variable securely from the root .env file.
2. Adapting the URL scheme for compatibility with the modern psycopg 3 driver (postgresql+psycopg://).
3. Initializing the SQLAlchemy Engine.
4. Setting up SessionLocal (Sessionmaker / session factory) for creating database sessions.
5. Declaring the Base class for ORM models.
6. Providing a get_db dependency helper to manage session lifecycles in FastAPI.
"""

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# 1. Environment Configuration
# ---------------------------------------------------------------------------
# Resolve the project root directory (document-intelligence-platform/)
# database.py -> core -> app -> backend -> document-intelligence-platform
ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"

# Load variables from the root .env file if it exists, otherwise fallback to system environment
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is missing. "
        "Please provide a valid connection string in the root .env file."
    )

# ---------------------------------------------------------------------------
# 2. PostgreSQL Driver URL Compatibility
# ---------------------------------------------------------------------------
# SQLAlchemy 2.x defaults `postgresql://` to psycopg2.
# Since psycopg 3 (`psycopg`) is installed, we ensure the scheme uses `postgresql+psycopg://`.
# Cloud providers (e.g., Render, Supabase) typically supply `postgresql://` or `postgres://`.
if DATABASE_URL.startswith("postgresql://"):
    SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
else:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL

# ---------------------------------------------------------------------------
# 3. SQLAlchemy Engine
# ---------------------------------------------------------------------------
# The Engine is the core interface to the database. It manages a pool of
# database connections and translates SQL statements.
# `pool_pre_ping=True` checks if a connection is alive before handing it out,
# avoiding errors caused by closed idle connections (common with hosted DBs).
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# 4. Session Factory (SessionLocal)
# ---------------------------------------------------------------------------
# SessionLocal is a factory class that generates new database Session objects.
# Each session represents a "workspace" for database operations within a unit of work.
# - autocommit=False: Transactions must be committed explicitly.
# - autoflush=False: Queries won't prematurely flush uncommitted changes to the DB.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ---------------------------------------------------------------------------
# 5. Declarative Base (SQLAlchemy 2.x style)
# ---------------------------------------------------------------------------
# In SQLAlchemy 2.0+, models inherit from a class derived from DeclarativeBase.
# It maintains the registry and metadata for all ORM models.
class Base(DeclarativeBase):
    pass

# ---------------------------------------------------------------------------
# 6. Database Session Dependency
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session per request
    and safely closes it when the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
