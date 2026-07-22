import sqlite3
from datetime import datetime

DB_PATH = 'study.db'


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name instead of index
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            subject     TEXT NOT NULL,
            date        TEXT NOT NULL,
            start_time  TEXT NOT NULL,
            duration    REAL NOT NULL,
            mood        INTEGER NOT NULL,
            focus       INTEGER NOT NULL,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def add_session(subject, date, start_time, duration, mood, focus):
    conn = get_connection()
    conn.execute('''
        INSERT INTO sessions (subject, date, start_time, duration, mood, focus)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (subject, date, start_time, duration, mood, focus))
    conn.commit()
    conn.close()


def get_all_sessions():
    conn = get_connection()
    rows = conn.execute('SELECT * FROM sessions ORDER BY date DESC, start_time DESC').fetchall()
    conn.close()
    return rows


def get_session_count():
    conn = get_connection()
    count = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
    conn.close()
    return count


def delete_session(session_id):
    conn = get_connection()
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()


def seed_sample_data():
    """
    Inserts some sample sessions so the dashboard isn't empty on first run.
    Call this manually from the shell if needed: python -c "from database import *; seed_sample_data()"
    """
    sample = [
        ('Calculus',     '2024-11-01', '09:00', 1.5, 3, 4),
        ('Calculus',     '2024-11-02', '08:30', 2.0, 4, 5),
        ('Calculus',     '2024-11-04', '10:00', 1.0, 2, 3),
        ('Programming',  '2024-11-01', '19:00', 2.5, 5, 5),
        ('Programming',  '2024-11-03', '20:00', 2.0, 4, 4),
        ('Programming',  '2024-11-05', '18:30', 3.0, 5, 5),
        ('Physics',      '2024-11-02', '14:00', 1.5, 3, 3),
        ('Physics',      '2024-11-04', '15:00', 1.0, 2, 2),
        ('English',      '2024-11-03', '11:00', 1.0, 4, 4),
        ('English',      '2024-11-05', '21:00', 0.5, 3, 3),
        ('Calculus',     '2024-11-06', '09:30', 2.0, 4, 4),
        ('Programming',  '2024-11-06', '19:30', 1.5, 5, 5),
        ('Physics',      '2024-11-07', '13:00', 2.0, 3, 3),
        ('English',      '2024-11-07', '10:00', 1.0, 4, 4),
        ('Calculus',     '2024-11-08', '08:00', 1.5, 3, 3),
    ]
    conn = get_connection()
    conn.executemany('''
        INSERT INTO sessions (subject, date, start_time, duration, mood, focus)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', sample)
    conn.commit()
    conn.close()
    print(f"Seeded {len(sample)} sample sessions.")
