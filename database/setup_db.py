"""
setup_db.py
===========
Phase 9: Database Setup, Migration & Seeding Utility.

Usage:
  py -3.11 database/setup_db.py            # Checks connection & initializes DB
  py -3.11 database/setup_db.py --seed     # Initializes & seeds demo screening records
  py -3.11 database/setup_db.py --view     # Displays current screening history records
  py -3.11 database/setup_db.py --sqlite   # Force initialization of SQLite database
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime

# Add root to module search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import db, SQLITE_DB_PATH, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB


def seed_records():
    """Seeds sample demonstration records into the active database."""
    print("[*] Inserting sample screening records for demonstration...", end=' ', flush=True)
    
    # Record 1: Normal Sinus Rhythm
    r1 = {
        'prediction': 'Normal Sinus Rhythm',
        'primary_class': 'Normal Beat',
        'confidence': 99.2,
        'heart_rate_bpm': 75.3,
        'status': 'No abnormal pattern detected by the model.',
        'is_abnormal': False,
        'total_beats_analyzed': 12,
        'breakdown': {'Normal': {'count': 12, 'percentage': 100.0}, 'SVEB': {'count': 0, 'percentage': 0.0}, 'PVC': {'count': 0, 'percentage': 0.0}, 'Other': {'count': 0, 'percentage': 0.0}}
    }
    db.save_analysis("sample_normal_ecg.csv", r1)
    
    # Record 2: Premature Ventricular Contraction (PVC)
    r2 = {
        'prediction': 'Possible Arrhythmia (Premature Ventricular Contraction detected)',
        'primary_class': 'Premature Ventricular (PVC)',
        'confidence': 99.5,
        'heart_rate_bpm': 64.5,
        'status': 'Abnormal pattern detected — professional medical evaluation is recommended.',
        'is_abnormal': True,
        'total_beats_analyzed': 10,
        'breakdown': {'Normal': {'count': 8, 'percentage': 80.0}, 'SVEB': {'count': 0, 'percentage': 0.0}, 'PVC': {'count': 2, 'percentage': 20.0}, 'Other': {'count': 0, 'percentage': 0.0}}
    }
    db.save_analysis("sample_pvc_arrhythmia.csv", r2)
    
    # Record 3: Supraventricular Ectopic Beat (SVEB)
    r3 = {
        'prediction': 'Possible Arrhythmia (Supraventricular Ectopic Beat detected)',
        'primary_class': 'Supraventricular Ectopic (SVEB)',
        'confidence': 92.1,
        'heart_rate_bpm': 88.0,
        'status': 'Abnormal pattern detected — professional medical evaluation is recommended.',
        'is_abnormal': True,
        'total_beats_analyzed': 15,
        'breakdown': {'Normal': {'count': 13, 'percentage': 86.7}, 'SVEB': {'count': 2, 'percentage': 13.3}, 'PVC': {'count': 0, 'percentage': 0.0}, 'Other': {'count': 0, 'percentage': 0.0}}
    }
    db.save_analysis("sample_atrial_arrhythmia.csv", r3)
    
    print("[Done]")


def view_history():
    """Prints all historical screening records formatted as a CLI table."""
    records = db.get_all_history(limit=50)
    print("\n" + "=" * 90)
    print(f" DATABASE HISTORY RECORDS (Total: {len(records)}) | Backend: {'MySQL' if db.use_mysql else 'SQLite'} ")
    print("=" * 90)
    
    if len(records) == 0:
        print(" [No records found in database]")
        return
        
    print(f"{'ID':<5} | {'Date/Time':<19} | {'Heart Rate':<10} | {'Confidence':<10} | {'Abnormal':<8} | {'Prediction'}")
    print("-" * 90)
    for r in records:
        ab_str = "YES" if r['is_abnormal'] else "NO"
        print(f"#{r['id']:<4} | {r['date']:<19} | {r['heart_rate']:<5.1f} BPM  | {r['confidence']:<5.1f}%    | {ab_str:<8} | {r['prediction']}")
    print("-" * 90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECG AI Database Manager")
    parser.add_argument('--seed', action='store_true', help="Seed database with sample clinical records")
    parser.add_argument('--view', action='store_true', help="View all records in database")
    args = parser.parse_args()
    
    print("=" * 65)
    print(" PHASE 9: DATABASE SETUP & INITIALIZATION ")
    print("=" * 65)
    print(f" • MySQL Configuration  : {MYSQL_USER}@{MYSQL_HOST}/{MYSQL_DB}")
    print(f" • SQLite Fallback Path : {SQLITE_DB_PATH}")
    print(f" • Active Database Mode : {'MySQL' if db.use_mysql else 'SQLite (Automatic Local Fallback)'}")
    print("-" * 65)
    
    if args.seed:
        seed_records()
        
    view_history()
