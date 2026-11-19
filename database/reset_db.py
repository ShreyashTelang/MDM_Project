import sqlite3
import json

conn = sqlite3.connect('database/ecg_history.db')
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS ecg_history')
c.execute('''
CREATE TABLE ecg_history (
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
''')

seed_data = [
    ('2026-11-19 01:00:00', 'sample_normal_ecg.csv', 'Normal Sinus Rhythm', 'Normal Beat', 99.7, 74.9, 'No abnormal pattern detected by the model.', 0, 11, json.dumps({'Normal': {'count': 11, 'percentage': 100.0}})),
    ('2026-11-19 01:05:00', 'sample_pvc_arrhythmia.csv', 'Possible Arrhythmia (Premature Ventricular Contraction detected)', 'Premature Ventricular (PVC)', 99.5, 64.5, 'Abnormal pattern detected — professional medical evaluation is recommended.', 1, 10, json.dumps({'Normal': {'count': 8, 'percentage': 80.0}, 'PVC': {'count': 2, 'percentage': 20.0}})),
    ('2026-11-19 01:10:00', 'sample_bundle_branch_block.csv', 'Normal Sinus Rhythm (Conduction Morphology)', 'Normal Beat', 98.2, 71.3, 'No acute arrhythmia detected; conduction morphology present.', 0, 12, json.dumps({'Normal': {'count': 12, 'percentage': 100.0}})),
    ('2026-11-19 01:15:00', 'sample_atrial_arrhythmia.csv', 'Possible Arrhythmia (Supraventricular Ectopic Beat detected)', 'Supraventricular Ectopic (SVEB)', 92.1, 88.0, 'Abnormal pattern detected — professional medical evaluation is recommended.', 1, 15, json.dumps({'Normal': {'count': 13, 'percentage': 86.7}, 'SVEB': {'count': 2, 'percentage': 13.3}})),
    ('2026-11-19 01:20:00', 'sample_normal_ecg.csv', 'Normal Sinus Rhythm', 'Normal Beat', 99.4, 75.2, 'No abnormal pattern detected by the model.', 0, 12, json.dumps({'Normal': {'count': 12, 'percentage': 100.0}}))
]

for row in seed_data:
    c.execute('''
    INSERT INTO ecg_history 
    (analysis_date, filename, prediction, primary_class, confidence, heart_rate, status, is_abnormal, total_beats, breakdown_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', row)

conn.commit()
c.execute('SELECT id, analysis_date, filename, primary_class FROM ecg_history')
for r in c.fetchall():
    print(r)
conn.close()
