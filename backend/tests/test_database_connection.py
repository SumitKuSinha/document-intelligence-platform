"""
Database Connection Test

This test verifies that the backend application can successfully establish
a network connection with the configured PostgreSQL database instance and
execute a query using SQLAlchemy and the psycopg driver.
"""

from pathlib import Path
import sys

# Ensure backend root is in sys.path so app imports resolve regardless of working directory
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import pytest
except ImportError:
    pytest = None

from sqlalchemy import text

from app.core.database import engine


def test_database_connection():
    """
    Test connectivity to the PostgreSQL database.

    Executes a simple 'SELECT 1' query to verify:
    1. Network reachability and DNS resolution of the database host.
    2. Successful authentication with database credentials.
    3. Proper initialization of the SQLAlchemy engine and psycopg driver.
    4. Successful execution of SQL statements and retrieval of results.
    """
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1")).scalar()
        assert result == 1, f"Expected query result 1, but received {result}"


if __name__ == "__main__":
    test_database_connection()
    print("Database connection test passed successfully!")
