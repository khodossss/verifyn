"""Migration 001: drop query_hash, add embedding column.

Idempotent. Run with: python -m migrations.001_drop_query_hash_add_embedding
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path


def migrate(db_path: str) -> None:
    path = Path(db_path)
    if not path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(query_history)")
    columns = {row[1] for row in cursor.fetchall()}

    if "embedding" in columns and "query_hash" not in columns:
        print("Migration 001 already applied. Skipping.")
        conn.close()
        return

    if "query_history" not in {
        row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }:
        print("Table query_history does not exist. Skipping.")
        conn.close()
        return

    backup_path = f"{db_path}.pre-001.bak"
    shutil.copy2(db_path, backup_path)
    print(f"Backup created: {backup_path}")

    cursor.execute(
        """
        CREATE TABLE query_history_new (
            id INTEGER NOT NULL PRIMARY KEY,
            query TEXT NOT NULL,
            embedding TEXT,
            mode VARCHAR,
            result TEXT NOT NULL,
            reputation_updated INTEGER,
            created_at DATETIME
        )
        """
    )

    cursor.execute(
        """
        INSERT INTO query_history_new (id, query, embedding, mode, result, reputation_updated, created_at)
        SELECT id, query, NULL, mode, result, reputation_updated, created_at
        FROM query_history
        """
    )

    cursor.execute("DROP INDEX IF EXISTS ix_query_history_query_hash")
    cursor.execute("DROP TABLE query_history")
    cursor.execute("ALTER TABLE query_history_new RENAME TO query_history")

    conn.commit()

    cursor.execute("PRAGMA table_info(query_history)")
    new_columns = [row[1] for row in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) FROM query_history")
    row_count = cursor.fetchone()[0]

    conn.close()

    print("Migration 001 applied successfully.")
    print(f"  Columns: {new_columns}")
    print(f"  Rows preserved: {row_count}")
    print("  Removed: query_hash")
    print("  Added: embedding (TEXT, nullable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migration 001: drop query_hash, add embedding")
    parser.add_argument("--db", default="data/verifyn.db", help="Path to SQLite database")
    args = parser.parse_args()
    migrate(args.db)
