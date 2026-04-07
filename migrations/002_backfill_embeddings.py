"""Migration 002: backfill OpenAI embeddings for query_history rows.

Idempotent and resumable. Processes rows where embedding IS NULL.
Requires OPENAI_API_KEY. Run with: python -m migrations.002_backfill_embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


def backfill(db_path: str, batch_size: int = 50) -> None:
    path = Path(db_path)
    if not path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.")
        sys.exit(1)

    from openai import OpenAI

    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=api_key)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(query_history)")
    columns = {row[1] for row in cursor.fetchall()}
    if "embedding" not in columns:
        print("Error: embedding column not found. Run migration 001 first.")
        conn.close()
        sys.exit(1)

    cursor.execute("SELECT id, query FROM query_history WHERE embedding IS NULL ORDER BY id")
    rows = cursor.fetchall()

    if not rows:
        print("All rows already have embeddings. Nothing to do.")
        conn.close()
        return

    print(f"Computing embeddings for {len(rows)} rows using {model}...")
    total_updated = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        try:
            response = client.embeddings.create(model=model, input=texts)
        except Exception as exc:
            print(f"  Error at batch {i // batch_size + 1}: {exc}")
            print(f"  Committing {total_updated} rows processed so far. Re-run to continue.")
            conn.commit()
            conn.close()
            sys.exit(1)

        for j, emb_data in enumerate(response.data):
            embedding_json = json.dumps(emb_data.embedding)
            cursor.execute("UPDATE query_history SET embedding = ? WHERE id = ?", (embedding_json, ids[j]))
            total_updated += 1

        batch_num = i // batch_size + 1
        total_batches = (len(rows) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} embeddings computed")

        # Commit per batch so the migration is resumable
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM query_history WHERE embedding IS NOT NULL")
    with_emb = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM query_history WHERE embedding IS NULL")
    without_emb = cursor.fetchone()[0]

    conn.close()

    print(f"\nDone. {total_updated} embeddings computed and saved.")
    print(f"  Rows with embedding: {with_emb}")
    print(f"  Rows without embedding: {without_emb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migration 002: backfill embeddings for existing queries")
    parser.add_argument("--db", default="data/verifyn.db", help="Path to SQLite database")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for OpenAI API calls")
    args = parser.parse_args()
    backfill(args.db, args.batch_size)
