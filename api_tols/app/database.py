"""
SQLite database management.
Three tables: datasets, training_jobs, models.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH, KNOWN_MODELS, STORAGE_DIR


def _get_conn(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """Create tables if they don't exist and register known models."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = _get_conn(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                image_dir TEXT,
                label_dir TEXT,
                num_images INTEGER DEFAULT 0,
                num_annotations INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT DEFAULT 'fine_tune',
                dataset_id INTEGER REFERENCES datasets(id),
                model_id INTEGER,
                status TEXT DEFAULT 'pending',
                pid INTEGER,
                config TEXT DEFAULT '{}',
                mlflow_run_id TEXT,
                output_model_path TEXT,
                error_message TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                started_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT DEFAULT '',
                file_path TEXT NOT NULL,
                environment TEXT DEFAULT 'staging',
                training_job_id INTEGER REFERENCES training_jobs(id),
                mlflow_run_id TEXT,
                metrics TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                promoted_at TEXT
            );
        """)
        # Auto-register known models if table is empty
        row = conn.execute("SELECT COUNT(*) as cnt FROM models").fetchone()
        if row["cnt"] == 0:
            for m in KNOWN_MODELS:
                if Path(m["file_path"]).exists():
                    conn.execute(
                        "INSERT INTO models (name, version, file_path, environment, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (m["name"], m["version"], m["file_path"], m["environment"],
                         datetime.now(timezone.utc).isoformat()),
                    )
        conn.commit()
    finally:
        conn.close()


# ── Generic helpers ────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Parse JSON fields
    for key in ("config", "metrics"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# ── Datasets ───────────────────────────────────────────────────────────

def create_dataset(name: str, description: str, image_dir: str, label_dir: str,
                   num_images: int, num_annotations: int) -> dict:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO datasets (name, description, image_dir, label_dir, num_images, num_annotations) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, description, image_dir, label_dir, num_images, num_annotations),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def update_dataset(dataset_id: int, **kwargs) -> dict | None:
    conn = _get_conn()
    try:
        allowed = {"name", "description", "image_dir", "label_dir",
                    "num_images", "num_annotations", "status"}
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            return _row_to_dict(row)
        vals.append(dataset_id)
        conn.execute(f"UPDATE datasets SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_datasets(status: str = "active") -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ── Training Jobs ──────────────────────────────────────────────────────

def create_job(job_type: str, dataset_id: int, model_id: int | None,
               config: dict) -> dict:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO training_jobs (job_type, dataset_id, model_id, config) VALUES (?, ?, ?, ?)",
            (job_type, dataset_id, model_id, json.dumps(config)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def update_job(job_id: int, **kwargs) -> dict | None:
    conn = _get_conn()
    try:
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k == "config":
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return get_job(job_id)
        vals.append(job_id)
        conn.execute(f"UPDATE training_jobs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_job(job_id: int, db_path: str | Path | None = None) -> dict | None:
    conn = _get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_jobs(status: str | None = None) -> list[dict]:
    conn = _get_conn()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM training_jobs WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ── Models ─────────────────────────────────────────────────────────────

def create_model(name: str, version: str, file_path: str,
                 environment: str = "staging", training_job_id: int | None = None,
                 mlflow_run_id: str | None = None, metrics: dict | None = None,
                 db_path: str | Path | None = None) -> dict:
    conn = _get_conn(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO models (name, version, file_path, environment, training_job_id, "
            "mlflow_run_id, metrics) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, version, file_path, environment, training_job_id,
             mlflow_run_id, json.dumps(metrics or {})),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_model(model_id: int) -> dict | None:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_models(environment: str | None = None) -> list[dict]:
    conn = _get_conn()
    try:
        if environment:
            rows = conn.execute(
                "SELECT * FROM models WHERE environment = ? ORDER BY created_at DESC",
                (environment,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM models ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def promote_model(model_id: int) -> dict | None:
    conn = _get_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        # Demote current production to archived
        conn.execute(
            "UPDATE models SET environment = 'archived' WHERE environment = 'production'"
        )
        # Promote target
        conn.execute(
            "UPDATE models SET environment = 'production', promoted_at = ? WHERE id = ?",
            (now, model_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_production_model() -> dict | None:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM models WHERE environment = 'production' LIMIT 1"
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()
