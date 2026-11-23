"""
=============================================================================
PlantCare AI — Intelligent Plant Leaf Disease Detection and Classification System
Phase 1: Web Application Architecture, Image Validation Pipeline & User Interface
MCA Minor Project
=============================================================================
Author: vivek kumar (MCA University Student)
Technology Stack: Python, Flask, Jinja2, Pillow (PIL), Werkzeug, Bootstrap 5

Description:
This Flask application serves as the core foundation (Phase 1) for PlantCare AI.
It handles user navigation, drag-and-drop file uploads, strict image security
validation (MIME type verification, header validation, corrupt file rejection,
and file size enforcement), and renders a responsive agricultural UI.

NOTE FOR PHASE 1:
No AI model is loaded or simulated. Uploaded images are strictly validated,
and the user is explicitly informed that the trained MobileNetV2 CNN model will
be integrated in Phase 2. No fake or randomized predictions are generated.
=============================================================================
"""

import os
import io
import uuid
import base64
import secrets
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    session
)
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from ml.predictor import PlantDiseasePredictor
from database import (
    init_db,
    close_db,
    insert_prediction,
    get_all_predictions,
    get_prediction_by_id,
    delete_prediction,
    clear_all_predictions,
    get_dashboard_analytics,
    get_db
)

# ---------------------------------------------------------------------------
# Flask Application Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Base directory for application-root-relative path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'plantcare.keras')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.json')

# SQLite Database path
app.config['DATABASE'] = os.path.join(BASE_DIR, 'plantcare.db')

# Register database teardown
app.teardown_appcontext(close_db)

# Initialize database tables and indexes on startup
with app.app_context():
    init_db()

# Initialize MobileNetV2 predictor ONCE per server process
predictor = PlantDiseasePredictor(model_path=MODEL_PATH, class_indices_path=CLASS_INDICES_PATH)

# Secret key for encrypted session data and flash messages
app.config['SECRET_KEY'] = os.environ.get(
    'SECRET_KEY',
    'plantcare-ai-mca-minor-project-phase1-dev-key-2026'
)

# Maximum file upload size: 5 Megabytes (5 * 1024 * 1024 bytes)
# Any request body exceeding this size will trigger a 413 Request Entity Too Large error
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Set of allowed file extensions for leaf photo uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Set of valid image formats recognized by Pillow for plant leaf scans
VALID_PIL_FORMATS = {'JPEG', 'PNG'}

# ---------------------------------------------------------------------------
# Cross-Site Request Forgery (CSRF) Protection
# ---------------------------------------------------------------------------
def generate_csrf_token() -> str:
    """
    Generates or retrieves a cryptographically secure token tied to the user's session.
    """
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']

app.jinja_env.globals['csrf_token'] = generate_csrf_token

@app.before_request
def csrf_protect():
    """
    Validates CSRF tokens on all state-changing HTTP POST requests.
    Automatically bypassed in testing mode (app.config['TESTING'] or WTF_CSRF_ENABLED=False).
    """
    if request.method == "POST":
        if app.config.get('TESTING') or not app.config.get('WTF_CSRF_ENABLED', True):
            return
        token = session.get('_csrf_token', None)
        form_token = request.form.get('csrf_token') or request.headers.get('X-CSRFToken')
        if not token or token != form_token:
            flash("Security notice: Your form session has expired. Please submit again.", "warning")
            return redirect(request.referrer or url_for('home'))


# ---------------------------------------------------------------------------
# Helper / Utility Functions
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file's extension matches our allowed set.

    Args:
        filename (str): The original filename provided by the client.

    Returns:
        bool: True if file has an allowed extension (.jpg, .jpeg, .png), else False.
    """
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_image_stream(file_stream: io.BytesIO):
    """
    Perform deep validation of an image file stream using Pillow.
    This guarantees that the uploaded file is indeed a valid, uncorrupted
    image and not an arbitrary file disguised with an image extension.

    Args:
        file_stream (io.BytesIO): In-memory byte stream of the uploaded file.

    Returns:
        tuple: (is_valid: bool, image_info: dict, error_message: str)
               If valid, image_info contains format, dimensions, mode, and size_kb.
               If invalid, error_message explains the rejection reason.
    """
    try:
        # Seek to beginning of stream before reading
        file_stream.seek(0)
        
        # Verify file header and structural integrity
        with Image.open(file_stream) as img:
            # img.verify() checks image integrity without decoding full pixel data
            img.verify()
        
        # Re-open stream for metadata extraction (verify closes/invalidates the image object)
        file_stream.seek(0)
        with Image.open(file_stream) as img:
            img_format = img.format
            width, height = img.size
            img_mode = img.mode

        # Check if the internal format is strictly JPEG or PNG
        if img_format not in VALID_PIL_FORMATS:
            return False, {}, (
                f"Unsupported image encoding ({img_format}). "
                "Only standard JPEG and PNG formats are supported."
            )

        # Ensure image has non-zero dimensions
        if width <= 0 or height <= 0:
            return False, {}, "The uploaded image has invalid dimensions (0x0 pixels)."

        # Calculate file size in KB for user feedback
        file_stream.seek(0, io.SEEK_END)
        size_bytes = file_stream.tell()
        size_kb = round(size_bytes / 1024, 2)
        file_stream.seek(0)

        image_info = {
            'format': img_format,
            'width': width,
            'height': height,
            'dimensions': f"{width} × {height} px",
            'mode': img_mode,
            'size_kb': size_kb,
            'size_mb': round(size_kb / 1024, 2)
        }
        return True, image_info, ""

    except (UnidentifiedImageError, OSError, SyntaxError) as err:
        return False, {}, "Corrupted or unrecognized image file. Please upload an authentic photo."
    except Exception as err:
        return False, {}, f"Validation error occurred: {str(err)}"


# ---------------------------------------------------------------------------
# Application Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    """
    Renders the PlantCare AI Homepage with project hero, how it works,
    key features, and project scope summary.
    """
    return render_template('index.html', active_page='home')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles the leaf disease detection page.
    GET:  Renders the drag-and-drop upload interface.
    POST: Safely processes and validates the uploaded leaf photo.
          In Phase 1, it demonstrates full backend validation without
          generating fake or simulated AI predictions.
    """
    if request.method == 'GET':
        return render_template('predict.html', active_page='predict')

    # 1. Check if the POST request contains the file part
    if 'file' not in request.files:
        flash("No file part detected in the upload request. Please select a leaf photo.", "danger")
        return redirect(url_for('predict'))

    file = request.files['file']

    # 2. Check if a file was actually chosen by the user
    if file.filename == '' or file.filename is None:
        flash("No file selected. Please choose or drag a leaf photo before submitting.", "warning")
        return redirect(url_for('predict'))

    # 3. Sanitize filename using Werkzeug to prevent directory traversal attacks
    filename = secure_filename(file.filename)
    if not filename:
        filename = "leaf_upload.jpg"

    # 4. Check extension whitelist
    if not allowed_file(filename):
        flash(
            f"Invalid file extension for '{filename}'. Only JPG, JPEG, and PNG images are allowed.",
            "danger"
        )
        return redirect(url_for('predict'))

    # 5. Read file contents into an in-memory byte stream for validation & preview
    try:
        file_bytes = file.read()
        
        # Enforce empty file check
        if len(file_bytes) == 0:
            flash("The uploaded file is empty (0 bytes). Please upload a valid image file.", "danger")
            return redirect(url_for('predict'))

        # Check size explicitly before Pillow processing
        if len(file_bytes) > app.config['MAX_CONTENT_LENGTH']:
            flash("File exceeds the maximum upload limit of 5 MB. Please choose a smaller photo.", "danger")
            return redirect(url_for('predict'))

        stream = io.BytesIO(file_bytes)

        # 6. Deep image verification with Pillow
        is_valid, img_info, err_msg = validate_image_stream(stream)
        if not is_valid:
            flash(err_msg, "danger")
            return redirect(url_for('predict'))

        # 7. Convert image bytes to a safe Base64 Data URI for immediate client preview
        # This eliminates the need to store unverified files in the server file system
        mime_type = "image/jpeg" if img_info['format'] == 'JPEG' else "image/png"
        base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
        preview_data_uri = f"data:{mime_type};base64,{base64_encoded}"

        # 8. Real MobileNetV2 Inference
        stream.seek(0)
        prediction_result = predictor.predict(stream)

        if prediction_result.get('status') == 'success':
            # Phase 4: Privacy-Preserving Image Storage Consent
            # Store uploaded image file permanently only if the user explicitly opted in
            save_image_consent = request.form.get('save_image') in ('on', 'true', '1')
            saved_rel_path = None

            if save_image_consent:
                try:
                    upload_folder = os.path.join(BASE_DIR, 'static', 'uploads')
                    os.makedirs(upload_folder, exist_ok=True)
                    ext = 'jpg' if img_info['format'] == 'JPEG' else 'png'
                    unique_name = f"{uuid.uuid4().hex[:12]}_{filename}"
                    dest_path = os.path.join(upload_folder, unique_name)
                    with open(dest_path, 'wb') as f:
                        f.write(file_bytes)
                    saved_rel_path = f"uploads/{unique_name}"
                except Exception as save_err:
                    saved_rel_path = None

            # Phase 4: Save verified prediction record to SQLite database
            try:
                record_id = insert_prediction(
                    crop=prediction_result['crop'],
                    disease_name=prediction_result['disease_name'],
                    raw_class=prediction_result['raw_class'],
                    is_healthy=1 if prediction_result['is_healthy'] else 0,
                    confidence=prediction_result['confidence'],
                    confidence_percent=prediction_result['confidence_percent'],
                    image_path=saved_rel_path
                )
            except Exception as db_err:
                record_id = None

            validation_result = {
                'status': 'success',
                'filename': filename,
                'format': img_info['format'],
                'dimensions': img_info['dimensions'],
                'size_kb': img_info['size_kb'],
                'preview_uri': preview_data_uri,
                'raw_class': prediction_result['raw_class'],
                'display_name': prediction_result['display_name'],
                'crop': prediction_result['crop'],
                'disease_name': prediction_result['disease_name'],
                'is_healthy': prediction_result['is_healthy'],
                'confidence': prediction_result['confidence'],
                'confidence_percent': prediction_result['confidence_percent'],
                'is_uncertain': prediction_result.get('is_uncertain', False),
                'uncertainty_warning': prediction_result.get('uncertainty_warning', ''),
                'advisory_notice': prediction_result.get('advisory_notice', ''),
                'pathogen': prediction_result.get('pathogen', 'N/A'),
                'symptoms': prediction_result.get('symptoms', ''),
                'organic_remedies': prediction_result.get('organic_remedies', []),
                'chemical_treatments': prediction_result.get('chemical_treatments', []),
                'prevention': prediction_result.get('prevention', []),
                'distribution': prediction_result.get('distribution', {}),
                'record_id': record_id,
                'image_saved': bool(saved_rel_path)
            }
            history_msg = f" — Logged to History #{record_id}" if record_id else ""
            flash(
                f"Leaf image analyzed: {prediction_result['display_name']} ({prediction_result['confidence_percent']} confidence){history_msg}",
                "success"
            )
        elif prediction_result.get('status') == 'model_not_ready':
            validation_result = {
                'status': 'model_not_ready',
                'filename': filename,
                'format': img_info['format'],
                'dimensions': img_info['dimensions'],
                'size_kb': img_info['size_kb'],
                'preview_uri': preview_data_uri,
                'phase_message': (
                    "AI model not trained yet. Prediction will be available in Phase 2."
                ),
                'phase_details': (
                    "Your plant leaf image has passed all backend security validation checks "
                    "(MIME type verification, header validation, and dimensions check). "
                    "The MobileNetV2 model binary is currently offline or not initialized."
                )
            }
            flash("Leaf image validated. AI model is currently offline or uninitialized.", "info")
        else:
            flash(
                "An unexpected issue occurred while evaluating the leaf photo. Please ensure the leaf is clearly centered and try again.",
                "danger"
            )
            return redirect(url_for('predict'))

        return render_template(
            'predict.html',
            active_page='predict',
            result=validation_result
        )

    except Exception as e:
        flash(f"An unexpected error occurred while processing the upload: {str(e)}", "danger")
        return redirect(url_for('predict'))


@app.route('/history')
def history():
    """
    Renders the persistent prediction history page with filtering capabilities.
    """
    crop_filter = request.args.get('crop', 'all').strip()
    status_filter = request.args.get('status', 'all').strip()

    predictions = get_all_predictions(crop_filter=crop_filter, status_filter=status_filter)

    # Metrics for header KPI summary cards
    all_records = get_all_predictions()
    total_count = len(all_records)
    healthy_count = sum(1 for r in all_records if r.get('is_healthy') == 1)
    diseased_count = total_count - healthy_count

    return render_template(
        'history.html',
        active_page='history',
        predictions=predictions,
        current_crop=crop_filter,
        current_status=status_filter,
        total_count=total_count,
        healthy_count=healthy_count,
        diseased_count=diseased_count
    )


@app.route('/history/delete/<int:record_id>', methods=['POST'])
def delete_history_record(record_id: int):
    """
    Safely deletes a single prediction record by ID.
    If a disk photo was saved with user consent, purges it from static/uploads.
    Protected state-changing endpoint: requires HTTP POST.
    """
    success, image_rel_path = delete_prediction(record_id)
    if success:
        if image_rel_path:
            full_path = os.path.join(BASE_DIR, 'static', image_rel_path.replace('/', os.sep))
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
        flash(f"Prediction record #{record_id} successfully deleted.", "success")
    else:
        flash(f"Prediction record #{record_id} not found.", "warning")

    return redirect(url_for('history'))


@app.route('/history/clear', methods=['POST'])
def clear_history():
    """
    Purges all prediction records and cleans up stored image files.
    Protected state-changing endpoint: requires HTTP POST.
    """
    removed_images = clear_all_predictions()
    for rel_path in removed_images:
        if rel_path:
            full_path = os.path.join(BASE_DIR, 'static', rel_path.replace('/', os.sep))
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass

    flash("All prediction history and stored images have been cleared.", "success")
    return redirect(url_for('history'))


@app.route('/dashboard')
def dashboard():
    """
    Renders the Analytics Dashboard with real-time statistics and Chart.js charts.
    All data is computed strictly from genuine database records.
    """
    stats = get_dashboard_analytics()
    return render_template(
        'dashboard.html',
        active_page='dashboard',
        stats=stats
    )


@app.route('/api/dashboard-stats')
def api_dashboard_stats():
    """
    JSON API endpoint returning real-time analytics data.
    Useful for test assertions and programmatic inspection.
    """
    stats = get_dashboard_analytics()
    return jsonify(stats)


@app.route('/about')
def about():
    """
    Renders the About Project page with architectural details,
    simple CNN explanation, technology stack details, and future roadmap.
    """
    return render_template('about.html', active_page='about')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """
    Renders the Contact & Academic Information page.
    Demonstrates student credentials, university department details,
    and a preview contact form.
    """
    if request.method == 'POST':
        # Flash a friendly notice that form is for demonstration in Phase 1
        name = request.form.get('name', 'Visitor').strip()
        flash(
            f"Thank you, {name}! Your message has been received in Phase 1 preview mode. "
            "Full email notifications will be activated upon final deployment.",
            "info"
        )
        return redirect(url_for('contact'))

    return render_template('contact.html', active_page='contact')


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Gracefully handles uploads exceeding the 5 MB MAX_CONTENT_LENGTH limit.
    Renders the predict page with HTTP status 413 and a clear warning message.
    """
    flash(
        "Uploaded file is too large! The maximum allowed file size is 5 MB. "
        "Please compress or crop the leaf image and try again.",
        "danger"
    )
    return render_template('predict.html', active_page='predict'), 413


@app.errorhandler(404)
def page_not_found(error):
    """
    Renders a friendly 404 error response when an invalid URL is visited.
    """
    return render_template(
        'base.html',
        active_page='',
        custom_error={
            'code': 404,
            'title': 'Page Not Found',
            'message': 'The page or resource you requested could not be found on PlantCare AI.'
        }
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    """
    Renders a graceful 500 error response in case of unexpected backend exceptions.
    """
    return render_template(
        'base.html',
        active_page='',
        custom_error={
            'code': 500,
            'title': 'Internal Server Error',
            'message': 'An unexpected error occurred on the server. Please try again shortly.'
        }
    ), 500


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Run the Flask development server on localhost:5000 with debug enabled
    # Useful for university project viva demonstrations and local testing
    print("\n" + "=" * 65)
    print("  PlantCare AI — Intelligent Plant Leaf Disease Detection System")
    print("  Phase 1 Development Server Started Successfully!")
    print("  Open your browser and navigate to: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
