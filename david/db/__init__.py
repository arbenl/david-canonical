"""PostgreSQL data layer for DAVID.

Usage:
    # Start Postgres (first time):
    docker compose up -d

    # Create schema:
    david db init

    # Check status:
    david db status

Connection pool is lazy-initialized on first call to get_conn().
DATABASE_URL defaults to postgresql://david:david@localhost:5432/david
and can be overridden via the DATABASE_URL environment variable.
"""

from .connection import get_conn, ping  # noqa: F401
from .schema import create_all          # noqa: F401
