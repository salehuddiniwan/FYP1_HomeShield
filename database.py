"""
HomeShield Database — SQLite event logging, camera config, zone storage.

Optimisations over previous version:
  * Added indexes on events(created_at), events(event_type),
    events(camera_id) and zones(camera_id) — the event table grows
    every alert and these queries are used on every dashboard refresh.
  * `PRAGMA journal_mode=WAL` lets readers and writers run concurrently
    (dashboard polls while the processing loop is logging).
  * `PRAGMA synchronous=NORMAL` trades a tiny durability window for a
    big write-throughput win on the alert path.
"""
import sqlite3
import json
import threading
from datetime import datetime
from config import Config


class Database:
    _local = threading.local()

    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self._init_db()

    def _get_conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Performance pragmas — applied per connection
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                url         TEXT NOT NULL,
                location    TEXT DEFAULT '',
                active      INTEGER DEFAULT 1,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS zones (
                zone_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name   TEXT NOT NULL,
                camera_id   INTEGER NOT NULL,
                polygon     TEXT NOT NULL,
                zone_type   TEXT DEFAULT 'danger',
                active      INTEGER DEFAULT 1,
                FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT NOT NULL,
                camera_id   INTEGER,
                camera_name TEXT,
                person_category TEXT DEFAULT 'unknown',
                confidence  REAL DEFAULT 0.0,
                snapshot_path TEXT DEFAULT '',
                alert_sent  INTEGER DEFAULT 0,
                details     TEXT DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
            );

            CREATE TABLE IF NOT EXISTS persons (
                person_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                category    TEXT NOT NULL DEFAULT 'adult',
                embedding   BLOB,
                photo_path  TEXT DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS intruders (
                intruder_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id    INTEGER,
                camera_name  TEXT DEFAULT '',
                category     TEXT DEFAULT 'adult',
                photo_path   TEXT DEFAULT '',
                detected_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                dismissed    INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            -- Indexes — event queries dominate dashboard load time
            CREATE INDEX IF NOT EXISTS idx_events_created_at
                ON events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_camera
                ON events(camera_id);
            CREATE INDEX IF NOT EXISTS idx_zones_camera
                ON zones(camera_id, active);
            CREATE INDEX IF NOT EXISTS idx_intruders_time
                ON intruders(detected_at DESC);
        """)

        # Migration: add zone_type column to existing installs that pre-date it
        cols = {r[1] for r in conn.execute("PRAGMA table_info(zones)").fetchall()}
        if "zone_type" not in cols:
            conn.execute("ALTER TABLE zones ADD COLUMN zone_type TEXT DEFAULT 'danger'")

        # Migration: add photo_path to persons for existing installs
        pcols = {r[1] for r in conn.execute("PRAGMA table_info(persons)").fetchall()}
        if "photo_path" not in pcols:
            conn.execute("ALTER TABLE persons ADD COLUMN photo_path TEXT DEFAULT ''")

        conn.commit()

    # ── Cameras ──────────────────────────────────────────────
    def add_camera(self, name, url, location=""):
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO cameras (name, url, location) VALUES (?, ?, ?)",
            (name, url, location),
        )
        conn.commit()
        return cur.lastrowid

    def get_cameras(self, active_only=False):
        conn = self._get_conn()
        q = "SELECT * FROM cameras"
        if active_only:
            q += " WHERE active = 1"
        return [dict(r) for r in conn.execute(q).fetchall()]

    def update_camera(self, camera_id, **kwargs):
        if not kwargs:
            return
        conn = self._get_conn()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [camera_id]
        conn.execute(f"UPDATE cameras SET {sets} WHERE camera_id = ?", vals)
        conn.commit()

    def delete_camera(self, camera_id):
        conn = self._get_conn()
        conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
        conn.execute("DELETE FROM zones   WHERE camera_id = ?", (camera_id,))
        conn.commit()

    # ── Zones ────────────────────────────────────────────────
    def add_zone(self, zone_name, camera_id, polygon, zone_type="danger"):
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO zones (zone_name, camera_id, polygon, zone_type) "
            "VALUES (?, ?, ?, ?)",
            (zone_name, camera_id, json.dumps(polygon), zone_type),
        )
        conn.commit()
        return cur.lastrowid

    def get_zones(self, camera_id=None, zone_type=None):
        conn = self._get_conn()
        clauses = ["active = 1"]
        params  = []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if zone_type is not None:
            clauses.append("zone_type = ?")
            params.append(zone_type)
        q = "SELECT * FROM zones WHERE " + " AND ".join(clauses)
        rows = conn.execute(q, params).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["polygon"]   = json.loads(d["polygon"])
            d["zone_type"] = d.get("zone_type") or "danger"
            result.append(d)
        return result

    def delete_zone(self, zone_id):
        conn = self._get_conn()
        conn.execute("DELETE FROM zones WHERE zone_id = ?", (zone_id,))
        conn.commit()

    # ── Persons (face registry) ──────────────────────────────
    def add_person(self, name, category, embedding_bytes):
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO persons (name, category, embedding) VALUES (?, ?, ?)",
            (name, category, embedding_bytes),
        )
        conn.commit()
        return cur.lastrowid

    def get_persons(self, include_embeddings=True):
        conn = self._get_conn()
        if include_embeddings:
            rows = conn.execute(
                "SELECT person_id, name, category, embedding, photo_path, created_at "
                "FROM persons ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT person_id, name, category, photo_path, created_at "
                "FROM persons ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_person(self, person_id):
        conn = self._get_conn()
        conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
        conn.commit()

    def update_person(self, person_id, **kwargs):
        """Allowed fields: name, category, photo_path."""
        allowed = {"name", "category", "photo_path"}
        fields  = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        conn = self._get_conn()
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [person_id]
        conn.execute(f"UPDATE persons SET {sets} WHERE person_id = ?", vals)
        conn.commit()

    # ── Intruders (seen-unknown log) ─────────────────────────
    def add_intruder(self, camera_id, camera_name, category, photo_path):
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO intruders (camera_id, camera_name, category, photo_path)
               VALUES (?, ?, ?, ?)""",
            (camera_id, camera_name, category, photo_path),
        )
        conn.commit()
        return cur.lastrowid

    def get_intruders(self, include_dismissed=False, limit=100):
        conn = self._get_conn()
        q = "SELECT * FROM intruders"
        if not include_dismissed:
            q += " WHERE dismissed = 0"
        q += " ORDER BY detected_at DESC LIMIT ?"
        return [dict(r) for r in conn.execute(q, (limit,)).fetchall()]

    def get_intruder(self, intruder_id):
        """Direct primary-key lookup — O(1), unlike get_intruders + linear scan."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM intruders WHERE intruder_id = ?",
            (intruder_id,),
        ).fetchone()
        return dict(row) if row else None

    def dismiss_intruder(self, intruder_id):
        conn = self._get_conn()
        conn.execute(
            "UPDATE intruders SET dismissed = 1 WHERE intruder_id = ?",
            (intruder_id,),
        )
        conn.commit()

    def delete_intruder(self, intruder_id):
        conn = self._get_conn()
        row = conn.execute(
            "SELECT photo_path FROM intruders WHERE intruder_id = ?",
            (intruder_id,),
        ).fetchone()
        conn.execute("DELETE FROM intruders WHERE intruder_id = ?", (intruder_id,))
        conn.commit()
        return dict(row) if row else None

    # ── Events ───────────────────────────────────────────────
    def log_event(self, event_type, camera_id=None, camera_name="",
                  person_category="unknown", confidence=0.0,
                  snapshot_path="", alert_sent=False, details=""):
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO events
               (event_type, camera_id, camera_name, person_category, confidence,
                snapshot_path, alert_sent, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_type, camera_id, camera_name, person_category,
             round(confidence, 3), snapshot_path, int(alert_sent), details),
        )
        conn.commit()
        return cur.lastrowid

    def get_events(self, limit=100, event_type=None, camera_id=None):
        conn = self._get_conn()
        clauses, params = [], []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)

        q = "SELECT * FROM events"
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        return [dict(r) for r in conn.execute(q, params).fetchall()]

    def clear_events(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM events")
        conn.commit()

    def get_today_alert_count(self):
        conn = self._get_conn()
        today = datetime.now().strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM events "
            "WHERE event_type != 'normal' AND date(created_at) = ?",
            (today,),
        ).fetchone()
        return row["cnt"] if row else 0

    # ── Settings ─────────────────────────────────────────────
    def get_setting(self, key, default=None):
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        conn.commit()
