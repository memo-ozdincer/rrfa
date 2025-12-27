"""Database utilities for data formatting scripts."""

import sqlite3
from pathlib import Path
from typing import Optional


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get connection to the unified database.

    Args:
        db_path: Path to database (default: data/db/unified.db)

    Returns:
        sqlite3.Connection object
    """
    if db_path is None:
        db_path = Path('data/db/unified.db')

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    # Enable foreign keys and other pragmas
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def get_table_count(conn: sqlite3.Connection, table_name: str) -> int:
    """
    Get row count for a table.

    Args:
        conn: Database connection
        table_name: Name of table

    Returns:
        Number of rows
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]


def verify_tables_exist(conn: sqlite3.Connection, tables: list) -> dict:
    """
    Verify that required tables exist in the database.

    Args:
        conn: Database connection
        tables: List of required table names

    Returns:
        Dict mapping table name to boolean (exists or not)
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}

    return {table: table in existing_tables for table in tables}
