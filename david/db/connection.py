"""Connection pool management for the DAVID Postgres database.

Thread-safe ThreadedConnectionPool backed by DATABASE_URL from config.
All callers use the `get_conn()` context manager which auto-commits on
success and rolls back on exception — never leaves an open transaction.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

from ..config import DATABASE_URL

_pool: ThreadedConnectionPool | None = None
_MIN_CONN = 1
_MAX_CONN = 10


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(_MIN_CONN, _MAX_CONN, DATABASE_URL)
    return _pool


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a Postgres connection from the pool.

    Auto-commits on clean exit; rolls back and re-raises on exception.
    Connection is returned to the pool in both cases.

    Example::

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def ping() -> bool:
    """Return True if Postgres is reachable, False otherwise."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def close_all() -> None:
    """Close all pooled connections (call on process shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
