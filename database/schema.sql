-- =====================================================================
-- AI-Based ECG Abnormality Detection and Classification System
-- Database Schema DDL for MySQL (and SQLite Compatible)
-- =====================================================================

-- 1. Create Database (for MySQL environments)
CREATE DATABASE IF NOT EXISTS ecg_arrhythmia_db 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE ecg_arrhythmia_db;

-- 2. Drop table if recreating
-- DROP TABLE IF EXISTS ecg_history;

-- 3. Create ecg_history Table
CREATE TABLE IF NOT EXISTS ecg_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_date DATETIME NOT NULL,
    filename VARCHAR(255) NOT NULL,
    prediction VARCHAR(255) NOT NULL,
    primary_class VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    heart_rate FLOAT NOT NULL,
    status VARCHAR(255) NOT NULL,
    is_abnormal TINYINT NOT NULL DEFAULT 0,
    total_beats INT NOT NULL DEFAULT 0,
    breakdown_json TEXT,
    INDEX idx_date (analysis_date),
    INDEX idx_class (primary_class),
    INDEX idx_abnormal (is_abnormal)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Sample Demonstration Seed Records
INSERT INTO ecg_history 
(analysis_date, filename, prediction, primary_class, confidence, heart_rate, status, is_abnormal, total_beats, breakdown_json)
VALUES 
(
    NOW() - INTERVAL 2 HOUR, 
    'sample_normal_ecg.csv', 
    'Normal Sinus Rhythm', 
    'Normal Beat', 
    99.2, 
    75.3, 
    'No abnormal pattern detected by the model.', 
    0, 
    12, 
    '{"Normal": {"count": 12, "percentage": 100.0}, "SVEB": {"count": 0, "percentage": 0.0}, "PVC": {"count": 0, "percentage": 0.0}, "Other": {"count": 0, "percentage": 0.0}}'
),
(
    NOW() - INTERVAL 1 HOUR, 
    'sample_pvc_arrhythmia.csv', 
    'Possible Arrhythmia (Premature Ventricular Contraction detected)', 
    'Premature Ventricular (PVC)', 
    99.5, 
    64.5, 
    'Abnormal pattern detected — professional medical evaluation is recommended.', 
    1, 
    10, 
    '{"Normal": {"count": 8, "percentage": 80.0}, "SVEB": {"count": 0, "percentage": 0.0}, "PVC": {"count": 2, "percentage": 20.0}, "Other": {"count": 0, "percentage": 0.0}}'
);
