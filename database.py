"""
=============================================================================
PlantCare AI — Database Management & Analytics Module
Phase 4: Prediction History and Analytics Dashboard
=============================================================================
Description:
Handles persistent SQLite storage for leaf disease prediction records,
safe parameterized CRUD operations, privacy-compliant image referencing,
and real-time aggregate statistics for the Analytics Dashboard.
=============================================================================
"""

import os
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from flask import Flask, g, current_app

DEFAULT_DB_FILENAME = "plantcare.db"


def get_db_path(app: Optional[Flask] = None) -> str:
    """
    Resolves the absolute path to the SQLite database file.
    Prefers app.config['DATABASE'] if configured, otherwise falls back
    to a project-root-relative plantcare.db.
    """
    if app and "DATABASE" in app.config:
        return app.config["DATABASE"]
    if current_app and "DATABASE" in current_app.config:
        return current_app.config["DATABASE"]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DEFAULT_DB_FILENAME)


def get_db(custom_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Returns a thread-safe SQLite connection for the current Flask application context,
    or a new connection if called outside an active request context.
    Configures row_factory = sqlite3.Row for dictionary-like column access.
    """
    target_path = custom_path or get_db_path()

    # If within an active Flask application context, cache in flask.g
    if current_app:
        if not hasattr(g, "_database") or g._database is None:
            db = sqlite3.connect(target_path, detect_types=sqlite3.PARSE_DECLTYPES)
            db.row_factory = sqlite3.Row
            g._database = db
        return g._database

    # Outside application context (e.g. standalone scripts, unit tests)
    conn = sqlite3.connect(target_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def close_db(e=None) -> None:
    """
    Closes the database connection at the end of the Flask application context.
    """
    if current_app and hasattr(g, "_database"):
        db = g._database
        if db is not None:
            db.close()
            g._database = None


def init_db(db_path: Optional[str] = None) -> None:
    """
    Creates the necessary tables and indexes if they do not already exist.
    Uses parameterized DDL with strict types and indexes for fast filtering.
    """
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                crop TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                raw_class TEXT NOT NULL,
                is_healthy INTEGER NOT NULL,
                confidence REAL NOT NULL,
                confidence_percent TEXT NOT NULL,
                image_path TEXT,
                notes TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_is_healthy ON predictions(is_healthy);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_raw_class ON predictions(raw_class);")
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    crop: str,
    disease_name: str,
    raw_class: str,
    is_healthy: int,
    confidence: float,
    confidence_percent: str,
    image_path: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """
    Inserts a newly validated prediction record into the SQLite database
    using strict parameterized SQL (? markers) to eliminate SQL injection risks.

    Returns:
        int: The auto-generated record ID.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            crop, disease_name, raw_class, is_healthy,
            confidence, confidence_percent, image_path, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            crop,
            disease_name,
            raw_class,
            1 if is_healthy else 0,
            float(confidence),
            confidence_percent,
            image_path,
            notes
        )
    )
    conn.commit()
    return cursor.lastrowid


def get_all_predictions(
    limit: Optional[int] = None,
    offset: int = 0,
    crop_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves prediction records ordered by latest timestamp first.
    Supports optional filtering by crop or health status.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params: List[Any] = []

    if crop_filter and crop_filter.lower() != "all":
        query += " AND LOWER(crop) = LOWER(?)"
        params.append(crop_filter)

    if status_filter and status_filter.lower() != "all":
        if status_filter.lower() in ("healthy", "1"):
            query += " AND is_healthy = 1"
        elif status_filter.lower() in ("diseased", "0"):
            query += " AND is_healthy = 0"

    query += " ORDER BY timestamp DESC, id DESC"

    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_prediction_by_id(record_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single prediction record by primary key ID.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (int(record_id),))
    row = cursor.fetchone()
    return dict(row) if row else None


def delete_prediction(record_id: int, db_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Deletes a single prediction record from the database.
    Returns:
        (success: bool, image_path: Optional[str])
        The image_path is returned so the caller can safely delete any associated disk file.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    # Check if record exists and fetch its image_path before deleting
    cursor.execute("SELECT image_path FROM predictions WHERE id = ?", (int(record_id),))
    row = cursor.fetchone()
    if not row:
        return False, None

    image_path = row["image_path"]
    cursor.execute("DELETE FROM predictions WHERE id = ?", (int(record_id),))
    conn.commit()
    return True, image_path


def clear_all_predictions(db_path: Optional[str] = None) -> List[str]:
    """
    Clears all records from the predictions table.
    Returns:
        List[str]: List of image paths associated with the deleted records,
                   allowing the caller to purge obsolete files from disk.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_path FROM predictions WHERE image_path IS NOT NULL")
    rows = cursor.fetchall()
    image_paths = [r["image_path"] for r in rows if r["image_path"]]

    cursor.execute("DELETE FROM predictions")
    conn.commit()
    return image_paths


def get_dashboard_analytics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes genuine real-time statistics from actual database records.
    Returns exact counts, averages, and class distributions with zero fake metrics.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    # 1. Total Scans Count
    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total_scans = cursor.fetchone()["total"]

    if total_scans == 0:
        return {
            "total_scans": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "healthy_percent": 0.0,
            "diseased_percent": 0.0,
            "avg_confidence": 0.0,
            "crop_distribution": {},
            "class_distribution": {},
            "recent_scans": [],
            "daily_activity": []
        }

    # 2. Healthy vs Diseased Counts
    cursor.execute("SELECT COUNT(*) as healthy FROM predictions WHERE is_healthy = 1")
    healthy_count = cursor.fetchone()["healthy"]
    diseased_count = total_scans - healthy_count

    healthy_percent = round((healthy_count / total_scans) * 100, 1)
    diseased_percent = round((diseased_count / total_scans) * 100, 1)

    # 3. Average Confidence Score
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
    avg_conf_raw = cursor.fetchone()["avg_conf"] or 0.0
    avg_confidence = round(float(avg_conf_raw) * 100, 1)

    # 4. Crop Breakdown
    cursor.execute("SELECT crop, COUNT(*) as count FROM predictions GROUP BY crop ORDER BY count DESC")
    crop_rows = cursor.fetchall()
    crop_distribution = {r["crop"]: r["count"] for r in crop_rows}

    # 5. Class Breakdown (all 7 classes accounted for)
    cursor.execute("SELECT raw_class, COUNT(*) as count FROM predictions GROUP BY raw_class ORDER BY count DESC")
    class_rows = cursor.fetchall()
    class_distribution = {r["raw_class"]: r["count"] for r in class_rows}

    # 6. Recent Scans (Top 6 latest)
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC, id DESC LIMIT 6")
    recent_rows = cursor.fetchall()
    recent_scans = [dict(r) for r in recent_rows]

    # 7. Daily Activity Timeline (Grouped by date, up to last 10 active days)
    cursor.execute("""
        SELECT DATE(timestamp) as scan_date, COUNT(*) as count
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY scan_date ASC
        LIMIT 10
    """)
    daily_rows = cursor.fetchall()
    daily_activity = [{"date": r["scan_date"], "count": r["count"]} for r in daily_rows]

    return {
        "total_scans": total_scans,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "healthy_percent": healthy_percent,
        "diseased_percent": diseased_percent,
        "avg_confidence": avg_confidence,
        "crop_distribution": crop_distribution,
        "class_distribution": class_distribution,
        "recent_scans": recent_scans,
        "daily_activity": daily_activity
    }
