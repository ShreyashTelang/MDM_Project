"""
app.py
======
Flask Web Application for AI-Based ECG Abnormality Detection and Classification.

Endpoints:
- /                  : Home page with project overview & quick actions
- /analysis          : Interactive ECG upload & real-time waveform screening
- /history           : Historical patient screening database viewer
- /about             : Academic methodology, limitations, & disclaimers
- /api/analyze       : POST endpoint accepting ECG files for 1D CNN inference
- /api/sample/<type> : GET endpoint running instant analysis on pre-loaded samples
- /api/history       : GET endpoint returning past screening records
- /api/health        : Health check endpoint
"""

import os
import sys
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

# Ensure src/ is on Python module path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.predict import get_engine
from src.database import db

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ecg-academic-deeplearning-key-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB maximum file size
ALLOWED_EXTENSIONS = {'csv', 'txt', 'dat', 'npy'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------
# PAGE VIEW ROUTES
# -------------------------------------------------------------

@app.route('/')
def home():
    """Renders the Home Page."""
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    """Renders the ECG Analysis Dashboard."""
    return render_template('analysis.html')


@app.route('/history')
def history():
    """Renders the Analysis History Page."""
    records = db.get_all_history(limit=50)
    return render_template('history.html', records=records)


@app.route('/about')
def about():
    """Renders the Academic & Clinical About Page."""
    return render_template('about.html')


# -------------------------------------------------------------
# REST API ENDPOINTS
# -------------------------------------------------------------

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Main inference endpoint.
    Accepts file upload (multipart/form-data with key 'ecg_file').
    Validates, filters, detects R-peaks, segments beats, and returns
    full predictions + waveform coordinates.
    """
    if 'ecg_file' not in request.files:
        return jsonify({'success': False, 'error': 'No ECG file provided in request.'}), 400
        
    file = request.files['ecg_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
        
    try:
        # Secure unique filename
        orig_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{orig_filename}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(saved_path)
        
        # Run inference
        engine = get_engine()
        result = engine.predict_ecg(saved_path)
        
        # Persist to database
        record_id = db.save_analysis(orig_filename, result)
        result['analysis_id'] = record_id
        
        # Clean up uploaded file
        try:
            os.remove(saved_path)
        except Exception:
            pass
            
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f"Inference pipeline error: {str(e)}"}), 500


@app.route('/api/sample/<sample_type>', methods=['GET'])
def api_sample_analyze(sample_type):
    """
    Analyzes pre-loaded clinical sample ECG files for fast demonstration.
    Types: 'normal', 'pvc', 'bundle_branch'
    """
    sample_map = {
        'normal': 'sample_normal_ecg.csv',
        'pvc': 'sample_pvc_arrhythmia.csv',
        'bundle_branch': 'sample_bundle_branch_block.csv'
    }
    
    if sample_type not in sample_map:
        return jsonify({'success': False, 'error': f"Unknown sample type '{sample_type}'."}), 400
        
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_uploads', sample_map[sample_type])
    if not os.path.exists(sample_path):
        return jsonify({'success': False, 'error': f"Sample file '{sample_map[sample_type]}' not found on server."}), 404
        
    try:
        engine = get_engine()
        result = engine.predict_ecg(sample_path)
        record_id = db.save_analysis(f"Sample: {sample_map[sample_type]}", result)
        result['analysis_id'] = record_id
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def api_get_history():
    """Returns past analysis records as JSON."""
    records = db.get_all_history(limit=50)
    return jsonify({'success': True, 'records': records}), 200


@app.route('/api/history/delete/<int:record_id>', methods=['POST', 'DELETE'])
def api_delete_history(record_id):
    """Deletes a record from the database."""
    success = db.delete_record(record_id)
    return jsonify({'success': success, 'deleted_id': record_id}), 200


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint."""
    model_exists = os.path.exists("models/ecg_cnn_model.keras")
    return jsonify({
        'status': 'healthy' if model_exists else 'model_missing',
        'model_loaded': model_exists,
        'database_backend': 'MySQL' if db.use_mysql else 'SQLite (fallback)',
        'sampling_rate_hz': 360,
        'supported_classes': ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']
    }), 200


# -------------------------------------------------------------
# ERROR HANDLERS
# -------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': 'Uploaded file exceeds the maximum 16 MB limit.'}), 413

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500


if __name__ == '__main__':
    print("[*] Starting ECG AI Web Application on http://127.0.0.1:5000 ...", flush=True)
    app.run(host='127.0.0.1', port=5000, debug=False)
