"""
database.py
===========
Phase 7/9: Database Connector and History Storage Module.

Provides dual support:
1. MySQL (Production / College lab requirement)
2. SQLite fallback (database/ecg_history.db for zero-config local operation)

Stores:
- Analysis ID
- Timestamp
- Filename / Sample Source
- Prediction Label
- Primary Class
- Confidence Score
- Heart Rate (BPM)
- Analysis Status Text
- Abnormal Flag (0 or 1)
- Beat Breakdown Summary (JSON)
"""

import os
import json
import sqlite3
from datetime import datetime

# MySQL configuration from environment (with defaults)
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
MYSQL_DB = os.environ.get('MYSQL_DB', 'ecg_arrhythmia_db')

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'ecg_history.db')


class ECGDatabase:
    """Handles persistent storage of ECG screening records across MySQL or SQLite."""
    
    def __init__(self):
        self.use_mysql = False
        self._init_db()
        
    def _init_db(self):
        """Attempts MySQL connection, falls back to SQLite if MySQL is unavailable."""
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        
        # 1. Try MySQL
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}")
            conn.database = MYSQL_DB
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ecg_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    analysis_date DATETIME NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    prediction VARCHAR(255) NOT NULL,
                    primary_class VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL,
                    heart_rate FLOAT NOT NULL,
                    status VARCHAR(255) NOT NULL,
                    is_abnormal TINYINT NOT NULL,
                    total_beats INT NOT NULL,
                    breakdown_json TEXT
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
            self.use_mysql = True
            print(f"[+] Connected to MySQL database '{MYSQL_DB}' at {MYSQL_HOST}")
            return
        except Exception as e:
            # MySQL unavailable, fallback to SQLite
            self.use_mysql = False
            
        # 2. SQLite Fallback
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ecg_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                primary_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                heart_rate REAL NOT NULL,
                status TEXT NOT NULL,
                is_abnormal INTEGER NOT NULL,
                total_beats INTEGER NOT NULL,
                breakdown_json TEXT
            )
        """)
        conn.commit()
        conn.close()
        print(f"[+] Initialized SQLite database fallback at '{SQLITE_DB_PATH}'")

    def save_analysis(self, filename, result_dict):
        """Saves an inference result dictionary to the active database."""
        now_dt = datetime.now()
        dt_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        prediction = result_dict.get('prediction', 'Unknown')
        primary_class = result_dict.get('primary_class', 'Unknown')
        confidence = float(result_dict.get('confidence', 0.0))
        heart_rate = float(result_dict.get('heart_rate_bpm', 0.0))
        status = result_dict.get('status', 'Analyzed')
        is_abnormal = 1 if result_dict.get('is_abnormal', False) else 0
        total_beats = int(result_dict.get('total_beats_analyzed', 0))
        breakdown_str = json.dumps(result_dict.get('breakdown', {}))
        
        record_id = None
        
        if self.use_mysql:
            try:
                import mysql.connector
                conn = mysql.connector.connect(
                    host=MYSQL_HOST,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    database=MYSQL_DB
                )
                cursor = conn.cursor()
                query = """
                    INSERT INTO ecg_history 
                    (analysis_date, filename, prediction, primary_class, confidence, heart_rate, status, is_abnormal, total_beats, breakdown_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (dt_str, filename, prediction, primary_class, confidence, heart_rate, status, is_abnormal, total_beats, breakdown_str))
                conn.commit()
                record_id = cursor.lastrowid
                cursor.close()
                conn.close()
                return record_id
            except Exception:
                pass  # Fallback to SQLite on temporary MySQL failure
                
        # SQLite
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        query = """
            INSERT INTO ecg_history 
            (analysis_date, filename, prediction, primary_class, confidence, heart_rate, status, is_abnormal, total_beats, breakdown_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (dt_str, filename, prediction, primary_class, confidence, heart_rate, status, is_abnormal, total_beats, breakdown_str))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        return record_id

    def get_all_history(self, limit=50):
        """Retrieves past analysis records ordered by newest first."""
        records = []
        if self.use_mysql:
            try:
                import mysql.connector
                conn = mysql.connector.connect(
                    host=MYSQL_HOST,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    database=MYSQL_DB
                )
                cursor = conn.cursor(dictionary=True)
                cursor.execute(f"SELECT * FROM ecg_history ORDER BY id DESC LIMIT {limit}")
                rows = cursor.fetchall()
                for r in rows:
                    records.append({
                        'id': r['id'],
                        'date': str(r['analysis_date']),
                        'filename': r['filename'],
                        'prediction': r['prediction'],
                        'primary_class': r['primary_class'],
                        'confidence': round(float(r['confidence']), 1),
                        'heart_rate': round(float(r['heart_rate']), 1),
                        'status': r['status'],
                        'is_abnormal': bool(r['is_abnormal']),
                        'total_beats': r['total_beats']
                    })
                cursor.close()
                conn.close()
                return records
            except Exception:
                pass
                
        # SQLite
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM ecg_history ORDER BY id DESC LIMIT {limit}")
        rows = cursor.fetchall()
        for r in rows:
            records.append({
                'id': r['id'],
                'date': r['analysis_date'],
                'filename': r['filename'],
                'prediction': r['prediction'],
                'primary_class': r['primary_class'],
                'confidence': round(float(r['confidence']), 1),
                'heart_rate': round(float(r['heart_rate']), 1),
                'status': r['status'],
                'is_abnormal': bool(r['is_abnormal']),
                'total_beats': r['total_beats']
            })
        conn.close()
        return records

    def delete_record(self, record_id):
        """Deletes a specific history record by ID."""
        if self.use_mysql:
            try:
                import mysql.connector
                conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM ecg_history WHERE id = %s", (record_id,))
                conn.commit()
                cursor.close()
                conn.close()
                return True
            except Exception:
                pass
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ecg_history WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True


# Global database instance
db = ECGDatabase()
