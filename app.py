"""
AlphabetAI Attendance — AI-Powered Smart Attendance System
A complete, responsive hackathon web application.
Stack: Python Flask, MongoDB Community (with seamless mongomock fallback), OpenCV DNN (YuNet + SFace), Vanilla HTML5/CSS3/JavaScript.
All main application code is contained in this single file.
"""

import os
import sys
import io
import re
import csv
import base64
import datetime
import urllib.request
from functools import wraps

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, request, jsonify, render_template_string, session,
    redirect, url_for, Response, make_response
)
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------------
# Flask App Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "alphabetai-secret-key-hackathon-2026-secure")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max payload

# ---------------------------------------------------------------------------
# MongoDB Configuration with Auto-Fallback
# ---------------------------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/alphabetai_db")
db = None
DB_MODE = "Unknown"

try:
    import pymongo
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    # Trigger connection probe
    client.server_info()
    db = client.get_database()
    DB_MODE = "MongoDB Community (Local 27017)"
    print(f"[Database] Connected to live MongoDB: {MONGO_URI}")
except Exception as mongo_err:
    print(f"[Database] MongoDB Community server unreachable ({mongo_err}). Using mongomock fallback.")
    import mongomock
    mock_client = mongomock.MongoClient()
    db = mock_client["alphabetai_db"]
    DB_MODE = "MongoDB In-Memory (Mock Mode)"

# Collections
teachers_col = db["teachers"]
students_col = db["students"]
attendance_col = db["attendance"]
audit_col = db["audit_logs"]

# Ensure default teacher exists
def seed_default_teacher():
    try:
        if teachers_col.count_documents({}) == 0:
            default_teacher = {
                "email": "teacher@alphabet.edu",
                "full_name": "Prof. Sarah Jenkins",
                "password_hash": generate_password_hash("Admin@123"),
                "department": "Computer Science & AI",
                "role": "Lead Faculty",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            teachers_col.insert_one(default_teacher)
            print("[Database] Seeded default teacher: teacher@alphabet.edu / Admin@123")
    except Exception as e:
        print(f"[Database] Teacher seed error: {e}")

seed_default_teacher()

def seed_sample_students():
    try:
        if students_col.count_documents({}) == 0:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            samples = [
                {
                    "full_name": "Alice Johnson",
                    "roll_no": "CS202601",
                    "email": "alice.j@alphabet.edu",
                    "department": "Computer Science & AI",
                    "biometric_enrolled": False,
                    "face_embedding": None,
                    "photo_thumbnail": "",
                    "consent_granted": True,
                    "consent_timestamp": now_iso,
                    "created_at": now_iso,
                    "updated_at": now_iso
                },
                {
                    "full_name": "Bob Martinez",
                    "roll_no": "CS202602",
                    "email": "bob.m@alphabet.edu",
                    "department": "Data Science & ML",
                    "biometric_enrolled": False,
                    "face_embedding": None,
                    "photo_thumbnail": "",
                    "consent_granted": True,
                    "consent_timestamp": now_iso,
                    "created_at": now_iso,
                    "updated_at": now_iso
                },
                {
                    "full_name": "Catherine Zhao",
                    "roll_no": "CS202603",
                    "email": "catherine.z@alphabet.edu",
                    "department": "Computer Science & AI",
                    "biometric_enrolled": False,
                    "face_embedding": None,
                    "photo_thumbnail": "",
                    "consent_granted": True,
                    "consent_timestamp": now_iso,
                    "created_at": now_iso,
                    "updated_at": now_iso
                },
                {
                    "full_name": "David Miller",
                    "roll_no": "CS202604",
                    "email": "david.m@alphabet.edu",
                    "department": "Information Technology",
                    "biometric_enrolled": False,
                    "face_embedding": None,
                    "photo_thumbnail": "",
                    "consent_granted": True,
                    "consent_timestamp": now_iso,
                    "created_at": now_iso,
                    "updated_at": now_iso
                },
                {
                    "full_name": "Emma Watson",
                    "roll_no": "CS202605",
                    "email": "emma.w@alphabet.edu",
                    "department": "Computer Science & AI",
                    "biometric_enrolled": False,
                    "face_embedding": None,
                    "photo_thumbnail": "",
                    "consent_granted": True,
                    "consent_timestamp": now_iso,
                    "created_at": now_iso,
                    "updated_at": now_iso
                }
            ]
            students_col.insert_many(samples)
            print("[Database] Seeded 5 initial sample students (A-Z) for immediate testing.")
    except Exception as e:
        print(f"[Database] Student seed error: {e}")

seed_sample_students()

# ---------------------------------------------------------------------------
# Local AI Face Engine (OpenCV YuNet + SFace)
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def ensure_model_files():
    for name, path, url in [("YuNet Detection", YUNET_PATH, YUNET_URL), ("SFace Recognition", SFACE_PATH, SFACE_URL)]:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"[AI Engine] Downloading {name} model from {url}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(path, "wb") as f:
                    f.write(resp.read())
                print(f"[AI Engine] Successfully downloaded {name} ({os.path.getsize(path)} bytes)")
            except Exception as e:
                print(f"[AI Engine] Error downloading {name}: {e}")

ensure_model_files()

face_detector = None
face_recognizer = None
AI_STATUS = "Offline"

try:
    if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
        face_detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
        face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        AI_STATUS = "OpenCV YuNet + SFace (Genuine Local AI)"
        print("[AI Engine] OpenCV YuNet & SFace neural network models loaded successfully!")
    else:
        AI_STATUS = "Model files missing"
except Exception as e:
    AI_STATUS = f"Initialization Error: {e}"
    print(f"[AI Engine] Model load error: {e}")

# Cosine similarity matching threshold for SFace (OpenCV Zoo recommends 0.363)
# We set standard threshold to 0.40 for robust high-accuracy matching
COSINE_SIMILARITY_THRESHOLD = 0.40

def decode_image_from_base64_or_bytes(data):
    """Safely decodes an image from base64 string or binary bytes to OpenCV BGR numpy array."""
    if isinstance(data, str):
        if "base64," in data:
            data = data.split("base64,")[1]
        raw_bytes = base64.b64decode(data)
    else:
        raw_bytes = data
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr

def extract_face_biometrics(img_bgr):
    """
    Genuine AI Face Feature Extraction.
    1. Uses OpenCV YuNet to detect face bounding box and 5-point landmarks.
    2. Uses OpenCV SFace to align and extract 128-dimensional L2-normalized embedding.
    3. Produces a safe thumbnail for UI verification.
    """
    if face_detector is None or face_recognizer is None:
        return None, None, "AI models are not loaded on server."

    h, w = img_bgr.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None, None, "No face detected in the image. Please face the camera directly with adequate lighting."

    # Select the most prominent face (highest confidence)
    best_face = faces[0]
    box = best_face[0:4].astype(int)
    confidence = float(best_face[-1])

    if confidence < 0.60:
        return None, None, f"Face detected but confidence is too low ({confidence*100:.1f}%). Please position face clearly."

    # Align face and extract 128-dimensional embedding vector
    aligned_face = face_recognizer.alignCrop(img_bgr, best_face)
    raw_feature = face_recognizer.feature(aligned_face) # shape: (1, 128)
    
    # L2 normalize vector
    norm = np.linalg.norm(raw_feature)
    if norm > 0:
        norm_feature = raw_feature / norm
    else:
        norm_feature = raw_feature
    embedding_list = norm_feature.flatten().tolist()

    # Generate small, safe thumbnail (80x80)
    x, y, bw, bh = box
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size > 0:
        thumb = cv2.resize(crop, (80, 80))
    else:
        thumb = cv2.resize(img_bgr, (80, 80))
    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    return embedding_list, thumb_b64, None

def verify_face_against_enrolled(img_bgr, candidate_student_id=None):
    """
    Genuine AI Face Verification against registered students.
    Uses Cosine Similarity between 128-d deep neural embeddings.
    Strictly verifies identity: no faking, no simple presence detection.
    """
    if face_detector is None or face_recognizer is None:
        return None, 0.0, "AI models are not loaded."

    h, w = img_bgr.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img_bgr)

    if faces is None or len(faces) == 0:
        return None, 0.0, "No face detected in camera frame."

    best_face = faces[0]
    det_conf = float(best_face[-1])
    if det_conf < 0.55:
        return None, 0.0, f"Face detection confidence too low ({det_conf*100:.1f}%)."

    aligned = face_recognizer.alignCrop(img_bgr, best_face)
    query_feature = face_recognizer.feature(aligned)
    q_norm = np.linalg.norm(query_feature)
    if q_norm > 0:
        query_feature = query_feature / q_norm

    # Fetch students with enrolled embeddings
    query_filter = {"biometric_enrolled": True, "face_embedding": {"$exists": True}}
    if candidate_student_id:
        query_filter["_id"] = candidate_student_id

    enrolled_students = list(students_col.find(query_filter))
    if not enrolled_students:
        return None, 0.0, "No enrolled students found with registered biometrics."

    best_student = None
    best_score = -1.0

    for s in enrolled_students:
        stored_emb = np.array(s["face_embedding"], dtype=np.float32).reshape(1, -1)
        s_norm = np.linalg.norm(stored_emb)
        if s_norm > 0:
            stored_emb = stored_emb / s_norm
        
        # Calculate Cosine Similarity: dot product of normalized vectors
        similarity = float(np.dot(query_feature.flatten(), stored_emb.flatten()))
        if similarity > best_score:
            best_score = similarity
            best_student = s

    # Compare with strict threshold
    if best_score >= COSINE_SIMILARITY_THRESHOLD and best_student:
        # Confidence scaled to 0-100%
        conf_pct = min(100.0, max(50.0, (best_score / 0.85) * 100.0))
        return best_student, round(conf_pct, 1), f"Cosine Similarity: {best_score:.4f}"
    else:
        return None, round(max(0.0, best_score * 100.0), 1), f"Similarity ({best_score:.4f}) below recognition threshold ({COSINE_SIMILARITY_THRESHOLD:.2f})."

# ---------------------------------------------------------------------------
# Helper Functions & Audit Logging
# ---------------------------------------------------------------------------
def log_audit(action, details, student_id=None, student_name=None, student_roll=None, confidence=None):
    """Records security and attendance audit events."""
    try:
        teacher_email = session.get("teacher_email", "System")
        ip = request.remote_addr or "127.0.0.1"
        audit_col.insert_one({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "teacher_email": teacher_email,
            "student_id": str(student_id) if student_id else None,
            "student_name": student_name,
            "student_roll": student_roll,
            "details": details,
            "confidence": confidence,
            "ip_address": ip
        })
    except Exception as e:
        print(f"[Audit Log Error] {e}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "teacher_email" not in session:
            if request.is_json:
                return jsonify({"error": "Unauthorized. Please log in."}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function

def get_today_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        teacher = teachers_col.find_one({"email": email})

        if teacher and check_password_hash(teacher["password_hash"], password):
            session["teacher_email"] = teacher["email"]
            session["teacher_name"] = teacher["full_name"]
            session["teacher_dept"] = teacher.get("department", "Faculty")
            log_audit("AUTH_LOGIN", f"Teacher {teacher['full_name']} logged in.")
            return redirect(url_for("dashboard_page"))
        else:
            return render_template_string(
                LOGIN_HTML,
                error="Invalid email or password. Please try again.",
                db_mode=DB_MODE
            )

    return render_template_string(LOGIN_HTML, db_mode=DB_MODE)

@app.route("/register-teacher", methods=["POST"])
def register_teacher():
    email = request.form.get("email", "").strip().lower()
    full_name = request.form.get("full_name", "").strip()
    password = request.form.get("password", "").strip()
    department = request.form.get("department", "Computer Science").strip()

    if not email or not full_name or not password:
        return render_template_string(LOGIN_HTML, error="All fields are required for teacher registration.", db_mode=DB_MODE)

    if teachers_col.find_one({"email": email}):
        return render_template_string(LOGIN_HTML, error=f"Teacher with email '{email}' is already registered.", db_mode=DB_MODE)

    teachers_col.insert_one({
        "email": email,
        "full_name": full_name,
        "password_hash": generate_password_hash(password),
        "department": department,
        "role": "Faculty",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })
    session["teacher_email"] = email
    session["teacher_name"] = full_name
    session["teacher_dept"] = department
    log_audit("AUTH_REGISTER", f"New teacher account registered: {full_name} ({email})")
    return redirect(url_for("dashboard_page"))

@app.route("/logout")
def logout():
    name = session.get("teacher_name", "Teacher")
    log_audit("AUTH_LOGOUT", f"{name} logged out.")
    session.clear()
    return redirect(url_for("login_page"))

# ---------------------------------------------------------------------------
# Dashboard & Main Interface
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if "teacher_email" in session:
        return redirect(url_for("dashboard_page"))
    return redirect(url_for("login_page"))

@app.route("/dashboard")
@login_required
def dashboard_page():
    return render_template_string(
        APP_HTML,
        teacher_name=session.get("teacher_name", "Teacher"),
        teacher_email=session.get("teacher_email", ""),
        teacher_dept=session.get("teacher_dept", ""),
        db_mode=DB_MODE,
        ai_status=AI_STATUS,
        today=get_today_str()
    )

# ---------------------------------------------------------------------------
# Student Management APIs (Alphabetical Sorting & Duplicate Prevention)
# ---------------------------------------------------------------------------
@app.route("/api/students", methods=["GET", "POST"])
@login_required
def api_students():
    if request.method == "GET":
        search = request.args.get("search", "").strip().lower()
        dept = request.args.get("dept", "").strip()
        letter = request.args.get("letter", "").strip().upper()

        query = {}
        if dept:
            query["department"] = dept

        # MongoDB query or filter
        students_cursor = list(students_col.find(query))

        # Always sort strictly alphabetically A–Z by full_name
        students_cursor.sort(key=lambda s: s.get("full_name", "").strip().lower())

        results = []
        for s in students_cursor:
            name = s.get("full_name", "")
            roll = s.get("roll_no", "")
            dep = s.get("department", "")

            # Alphabet filter
            if letter and not name.upper().startswith(letter):
                continue

            # Search filter
            if search and (search not in name.lower() and search not in roll.lower() and search not in dep.lower()):
                continue

            results.append({
                "id": str(s["_id"]),
                "full_name": name,
                "roll_no": roll,
                "email": s.get("email", ""),
                "department": dep,
                "biometric_enrolled": s.get("biometric_enrolled", False),
                "photo_thumbnail": s.get("photo_thumbnail", ""),
                "consent_granted": s.get("consent_granted", False),
                "consent_timestamp": s.get("consent_timestamp", ""),
                "created_at": s.get("created_at", "")
            })

        return jsonify({"students": results, "count": len(results)})

    elif request.method == "POST":
        # Register new student
        data = request.json or {}
        full_name = data.get("full_name", "").strip()
        roll_no = data.get("roll_no", "").strip().upper()
        email = data.get("email", "").strip().lower()
        department = data.get("department", "Computer Science").strip()
        photo_data = data.get("photo_data", "")
        consent = data.get("consent_granted", False)

        # Validation
        if not full_name or not roll_no or not email:
            return jsonify({"success": False, "error": "Name, Roll Number, and Email are required."}), 400

        # Duplicate checks
        if students_col.find_one({"roll_no": roll_no}):
            return jsonify({"success": False, "error": f"Student with Roll Number '{roll_no}' is already registered."}), 409

        if students_col.find_one({"email": email}):
            return jsonify({"success": False, "error": f"Student with Email '{email}' is already registered."}), 409

        # Consent check
        if not consent:
            return jsonify({"success": False, "error": "Biometric data protection requires explicit informed student consent."}), 400

        # Biometric AI processing
        embedding_list = None
        thumb_b64 = ""
        biometric_enrolled = False

        if photo_data:
            try:
                img_bgr = decode_image_from_base64_or_bytes(photo_data)
                if img_bgr is not None:
                    emb, thumb, err = extract_face_biometrics(img_bgr)
                    if err:
                        return jsonify({"success": False, "error": f"AI Face Enrollment Failed: {err}"}), 400
                    embedding_list = emb
                    thumb_b64 = thumb
                    biometric_enrolled = True
            except Exception as e:
                return jsonify({"success": False, "error": f"Image processing error: {e}"}), 400

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        student_doc = {
            "full_name": full_name,
            "roll_no": roll_no,
            "email": email,
            "department": department,
            "biometric_enrolled": biometric_enrolled,
            "face_embedding": embedding_list,
            "photo_thumbnail": thumb_b64,
            "consent_granted": True,
            "consent_timestamp": now_iso,
            "created_at": now_iso,
            "updated_at": now_iso
        }
        res = students_col.insert_one(student_doc)
        student_id = str(res.inserted_id)

        log_audit(
            "STUDENT_ENROLLED",
            f"Enrolled student {full_name} ({roll_no}) in {department}. Biometric: {'Yes' if biometric_enrolled else 'No'}.",
            student_id=student_id,
            student_name=full_name,
            student_roll=roll_no
        )

        return jsonify({
            "success": True,
            "message": f"Student '{full_name}' successfully enrolled!",
            "student_id": student_id
        })

@app.route("/api/students/<student_id>", methods=["PUT", "DELETE"])
@login_required
def api_student_detail(student_id):
    from bson.objectid import ObjectId
    try:
        oid = ObjectId(student_id)
    except Exception:
        oid = student_id

    student = students_col.find_one({"_id": oid})
    if not student:
        return jsonify({"success": False, "error": "Student not found."}), 404

    if request.method == "DELETE":
        students_col.delete_one({"_id": oid})
        # Clean attendance
        attendance_col.delete_many({"student_id": str(student_id)})
        log_audit(
            "STUDENT_DELETED",
            f"Removed student {student.get('full_name')} ({student.get('roll_no')}).",
            student_id=str(student_id),
            student_name=student.get("full_name")
        )
        return jsonify({"success": True, "message": "Student successfully removed."})

    elif request.method == "PUT":
        data = request.json or {}
        full_name = data.get("full_name", student.get("full_name")).strip()
        email = data.get("email", student.get("email")).strip().lower()
        department = data.get("department", student.get("department")).strip()
        photo_data = data.get("photo_data", "")

        # Check duplicate email if changed
        if email != student.get("email"):
            if students_col.find_one({"email": email, "_id": {"$ne": oid}}):
                return jsonify({"success": False, "error": f"Email '{email}' is used by another student."}), 409

        update_fields = {
            "full_name": full_name,
            "email": email,
            "department": department,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # If re-enrolling photo
        if photo_data:
            try:
                img_bgr = decode_image_from_base64_or_bytes(photo_data)
                if img_bgr is not None:
                    emb, thumb, err = extract_face_biometrics(img_bgr)
                    if err:
                        return jsonify({"success": False, "error": f"AI Face Update Failed: {err}"}), 400
                    update_fields["face_embedding"] = emb
                    update_fields["photo_thumbnail"] = thumb
                    update_fields["biometric_enrolled"] = True
            except Exception as e:
                return jsonify({"success": False, "error": f"Image processing error: {e}"}), 400

        students_col.update_one({"_id": oid}, {"$set": update_fields})
        log_audit(
            "STUDENT_UPDATED",
            f"Updated student profile for {full_name}.",
            student_id=str(student_id),
            student_name=full_name
        )
        return jsonify({"success": True, "message": "Student updated successfully."})

# ---------------------------------------------------------------------------
# Live Alphabetical Attendance Queue APIs
# ---------------------------------------------------------------------------
@app.route("/api/queue", methods=["GET"])
@login_required
def api_attendance_queue():
    date_str = request.args.get("date", get_today_str())
    dept = request.args.get("dept", "").strip()

    # Get all students
    query = {}
    if dept:
        query["department"] = dept

    all_students = list(students_col.find(query))
    # STRICT ALPHABETICAL SORTING (A-Z)
    all_students.sort(key=lambda s: s.get("full_name", "").strip().lower())

    # Get existing attendance for this date
    att_records = list(attendance_col.find({"date": date_str}))
    att_map = {str(a["student_id"]): a for a in att_records}

    queue_items = []
    present_cnt = 0
    absent_cnt = 0
    pending_cnt = 0

    for s in all_students:
        s_id = str(s["_id"])
        att = att_map.get(s_id)

        if att:
            status = att.get("status", "Pending")
            method = att.get("method", "MANUAL")
            verified_at = att.get("verified_at", "")
            confidence = att.get("confidence")
            remarks = att.get("remarks", "")
        else:
            status = "Pending"
            method = None
            verified_at = None
            confidence = None
            remarks = None

        if status == "Present":
            present_cnt += 1
        elif status == "Absent":
            absent_cnt += 1
        else:
            pending_cnt += 1

        queue_items.append({
            "student_id": s_id,
            "full_name": s.get("full_name", ""),
            "roll_no": s.get("roll_no", ""),
            "department": s.get("department", ""),
            "biometric_enrolled": s.get("biometric_enrolled", False),
            "photo_thumbnail": s.get("photo_thumbnail", ""),
            "status": status,
            "method": method,
            "verified_at": verified_at,
            "confidence": confidence,
            "remarks": remarks
        })

    total = len(queue_items)
    rate = round((present_cnt / total * 100), 1) if total > 0 else 0.0

    return jsonify({
        "date": date_str,
        "total_students": total,
        "present_count": present_cnt,
        "absent_count": absent_cnt,
        "pending_count": pending_cnt,
        "attendance_rate": rate,
        "queue": queue_items
    })

# ---------------------------------------------------------------------------
# Genuine AI Face Verification Endpoint
# ---------------------------------------------------------------------------
@app.route("/api/verify-face", methods=["POST"])
@login_required
def api_verify_face():
    data = request.json or {}
    image_data = data.get("image", "")
    target_student_id = data.get("target_student_id")
    date_str = data.get("date", get_today_str())

    if not image_data:
        return jsonify({"success": False, "error": "No image frame received."}), 400

    try:
        img_bgr = decode_image_from_base64_or_bytes(image_data)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Failed to decode camera image frame."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image format: {e}"}), 400

    # Execute genuine neural face recognition
    from bson.objectid import ObjectId
    target_oid = None
    if target_student_id:
        try:
            target_oid = ObjectId(target_student_id)
        except Exception:
            target_oid = target_student_id

    matched_student, confidence_pct, details = verify_face_against_enrolled(img_bgr, target_oid)

    if not matched_student:
        return jsonify({
            "success": False,
            "status": "UNRECOGNIZED",
            "message": details,
            "confidence": confidence_pct
        }), 200

    s_id = str(matched_student["_id"])
    s_name = matched_student["full_name"]
    s_roll = matched_student["roll_no"]
    s_dept = matched_student.get("department", "")

    # Duplicate Attendance Prevention: Check if already marked today
    existing_att = attendance_col.find_one({"student_id": s_id, "date": date_str})
    if existing_att and existing_att.get("status") == "Present":
        log_audit(
            "DUPLICATE_ATTENDANCE_BLOCKED",
            f"Prevented duplicate attendance for {s_name} ({s_roll}). Already marked at {existing_att.get('verified_at')}.",
            student_id=s_id,
            student_name=s_name,
            student_roll=s_roll
        )
        return jsonify({
            "success": False,
            "status": "DUPLICATE_PREVENTED",
            "message": f"Duplicate Prevented: {s_name} ({s_roll}) is already marked Present today at {existing_att.get('verified_at')}.",
            "student": {
                "id": s_id,
                "full_name": s_name,
                "roll_no": s_roll,
                "department": s_dept
            },
            "verified_at": existing_att.get("verified_at"),
            "confidence": existing_att.get("confidence")
        }), 200

    # Mark Present with AI verification details
    now_time = datetime.datetime.now().strftime("%I:%M:%S %p")
    att_doc = {
        "student_id": s_id,
        "student_roll": s_roll,
        "student_name": s_name,
        "student_dept": s_dept,
        "date": date_str,
        "status": "Present",
        "verified_at": now_time,
        "method": "AI_FACE_RECOGNITION",
        "confidence": confidence_pct,
        "verified_by_teacher": session.get("teacher_email"),
        "remarks": f"Genuine AI Match ({details})",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    attendance_col.update_one(
        {"student_id": s_id, "date": date_str},
        {"$set": att_doc},
        upsert=True
    )

    log_audit(
        "FACE_VERIFIED_ATTENDANCE",
        f"Marked Present via Genuine AI Face Recognition: {s_name} ({s_roll}) [Confidence: {confidence_pct}%, {details}].",
        student_id=s_id,
        student_name=s_name,
        student_roll=s_roll,
        confidence=confidence_pct
    )

    return jsonify({
        "success": True,
        "status": "MARKED_PRESENT",
        "message": f"Verified: {s_name} ({s_roll}) marked Present!",
        "student": {
            "id": s_id,
            "full_name": s_name,
            "roll_no": s_roll,
            "department": s_dept,
            "thumbnail": matched_student.get("photo_thumbnail", "")
        },
        "confidence": confidence_pct,
        "verified_at": now_time,
        "method": "AI_FACE_RECOGNITION",
        "ai_details": details
    })

# ---------------------------------------------------------------------------
# Manual Attendance & Correction API
# ---------------------------------------------------------------------------
@app.route("/api/mark-attendance", methods=["POST"])
@login_required
def api_mark_attendance():
    data = request.json or {}
    student_id = data.get("student_id")
    date_str = data.get("date", get_today_str())
    status = data.get("status", "Present") # Present, Absent, Pending
    reason = data.get("reason", "Manual Teacher Entry").strip()

    if not student_id:
        return jsonify({"success": False, "error": "Student ID is required."}), 400

    from bson.objectid import ObjectId
    try:
        oid = ObjectId(student_id)
    except Exception:
        oid = student_id

    student = students_col.find_one({"_id": oid})
    if not student:
        return jsonify({"success": False, "error": "Student not found."}), 404

    s_name = student.get("full_name")
    s_roll = student.get("roll_no")
    s_dept = student.get("department")

    # If resetting to Pending
    if status == "Pending":
        attendance_col.delete_one({"student_id": str(student_id), "date": date_str})
        log_audit(
            "ATTENDANCE_RESET",
            f"Reset attendance for {s_name} ({s_roll}) on {date_str}. Reason: {reason}",
            student_id=str(student_id),
            student_name=s_name,
            student_roll=s_roll
        )
        return jsonify({"success": True, "message": f"Attendance reset to Pending for {s_name}."})

    now_time = datetime.datetime.now().strftime("%I:%M:%S %p")
    att_doc = {
        "student_id": str(student_id),
        "student_roll": s_roll,
        "student_name": s_name,
        "student_dept": s_dept,
        "date": date_str,
        "status": status,
        "verified_at": now_time,
        "method": "MANUAL_TEACHER_OVERRIDE",
        "confidence": None,
        "verified_by_teacher": session.get("teacher_email"),
        "remarks": reason,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    attendance_col.update_one(
        {"student_id": str(student_id), "date": date_str},
        {"$set": att_doc},
        upsert=True
    )

    action_label = "MANUAL_CORRECTION" if status != "Present" else "MANUAL_ATTENDANCE_OVERRIDE"
    log_audit(
        action_label,
        f"Marked {status} manually for {s_name} ({s_roll}). Reason: {reason}",
        student_id=str(student_id),
        student_name=s_name,
        student_roll=s_roll
    )

    return jsonify({
        "success": True,
        "message": f"Student '{s_name}' marked as {status}.",
        "verified_at": now_time,
        "status": status
    })

# ---------------------------------------------------------------------------
# Statistics API
# ---------------------------------------------------------------------------
@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    date_str = request.args.get("date", get_today_str())
    
    total_students = students_col.count_documents({})
    enrolled_biometrics = students_col.count_documents({"biometric_enrolled": True})
    
    today_records = list(attendance_col.find({"date": date_str}))
    present_cnt = sum(1 for r in today_records if r.get("status") == "Present")
    absent_cnt = sum(1 for r in today_records if r.get("status") == "Absent")
    pending_cnt = max(0, total_students - (present_cnt + absent_cnt))

    ai_verified_cnt = sum(1 for r in today_records if r.get("method") == "AI_FACE_RECOGNITION" and r.get("status") == "Present")
    manual_verified_cnt = present_cnt - ai_verified_cnt

    att_rate = round((present_cnt / total_students * 100), 1) if total_students > 0 else 0.0
    ai_pct = round((ai_verified_cnt / present_cnt * 100), 1) if present_cnt > 0 else 0.0

    # Department breakdown
    dept_stats = {}
    for s in students_col.find():
        d = s.get("department", "General")
        dept_stats[d] = dept_stats.get(d, {"total": 0, "present": 0})
        dept_stats[d]["total"] += 1

    for r in today_records:
        if r.get("status") == "Present":
            d = r.get("student_dept", "General")
            if d in dept_stats:
                dept_stats[d]["present"] += 1

    audit_count = audit_col.count_documents({})

    return jsonify({
        "date": date_str,
        "total_students": total_students,
        "enrolled_biometrics": enrolled_biometrics,
        "present_count": present_cnt,
        "absent_count": absent_cnt,
        "pending_count": pending_cnt,
        "attendance_rate": att_rate,
        "ai_verified_count": ai_verified_cnt,
        "manual_verified_count": manual_verified_cnt,
        "ai_verified_pct": ai_pct,
        "audit_events_count": audit_count,
        "department_breakdown": dept_stats,
        "database_mode": DB_MODE,
        "ai_engine_status": AI_STATUS
    })

# ---------------------------------------------------------------------------
# Reports & Cumulative Percentages API
# ---------------------------------------------------------------------------
@app.route("/api/reports", methods=["GET"])
@login_required
def api_reports():
    date_str = request.args.get("date", get_today_str())
    dept = request.args.get("dept", "").strip()

    # Get distinct session dates in the database
    all_dates = sorted(attendance_col.distinct("date"))
    if not all_dates:
        all_dates = [date_str]

    # Fetch students
    query = {}
    if dept:
        query["department"] = dept
    students = list(students_col.find(query))
    students.sort(key=lambda s: s.get("full_name", "").strip().lower())

    # Calculate cumulative stats for each student
    reports = []
    total_sessions = len(all_dates)

    for s in students:
        s_id = str(s["_id"])
        records = list(attendance_col.find({"student_id": s_id}))
        present_days = sum(1 for r in records if r.get("status") == "Present")
        absent_days = sum(1 for r in records if r.get("status") == "Absent")
        
        # Today's status
        today_att = next((r for r in records if r.get("date") == date_str), None)
        today_status = today_att.get("status") if today_att else "Pending"

        pct = round((present_days / total_sessions * 100), 1) if total_sessions > 0 else 0.0

        reports.append({
            "student_id": s_id,
            "full_name": s.get("full_name", ""),
            "roll_no": s.get("roll_no", ""),
            "department": s.get("department", ""),
            "today_status": today_status,
            "present_days": present_days,
            "absent_days": absent_days,
            "total_sessions": total_sessions,
            "percentage": pct
        })

    return jsonify({
        "dates": all_dates,
        "total_sessions": total_sessions,
        "selected_date": date_str,
        "reports": reports
    })

# ---------------------------------------------------------------------------
# CSV Export Route
# ---------------------------------------------------------------------------
@app.route("/export-csv", methods=["GET"])
@login_required
def export_csv():
    date_str = request.args.get("date", get_today_str())
    report_type = request.args.get("type", "session") # 'session' or 'cumulative'
    dept = request.args.get("dept", "").strip()

    output = io.StringIO()
    writer = csv.writer(output)

    if report_type == "cumulative":
        # Cumulative Summary
        writer.writerow(["AlphabetAI Attendance — Cumulative Summary Report"])
        writer.writerow(["Generated At", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["Teacher In-Charge", session.get("teacher_name")])
        writer.writerow([])
        writer.writerow(["S.No", "Roll Number", "Student Name", "Department", "Sessions Attended", "Total Sessions", "Attendance %", "Eligibility Status"])

        all_dates = sorted(attendance_col.distinct("date"))
        total_sessions = max(1, len(all_dates))

        query = {}
        if dept:
            query["department"] = dept
        students = list(students_col.find(query))
        students.sort(key=lambda s: s.get("full_name", "").strip().lower())

        for idx, s in enumerate(students, 1):
            s_id = str(s["_id"])
            records = list(attendance_col.find({"student_id": s_id}))
            present_days = sum(1 for r in records if r.get("status") == "Present")
            pct = round((present_days / total_sessions * 100), 1)
            eligibility = "Eligible (>=75%)" if pct >= 75.0 else "Shortage Warning (<75%)"

            writer.writerow([
                idx,
                s.get("roll_no", ""),
                s.get("full_name", ""),
                s.get("department", ""),
                present_days,
                total_sessions,
                f"{pct}%",
                eligibility
            ])

        filename = f"AlphabetAI_Cumulative_Report_{date_str}.csv"

    else:
        # Daily Session Log
        writer.writerow(["AlphabetAI Attendance — Daily Session Report"])
        writer.writerow(["Date", date_str])
        writer.writerow(["Teacher In-Charge", session.get("teacher_name")])
        writer.writerow(["Generated At", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])
        writer.writerow(["S.No", "Roll Number", "Student Name", "Department", "Status", "Verification Method", "Confidence Score", "Verified At", "Remarks"])

        query = {}
        if dept:
            query["department"] = dept
        students = list(students_col.find(query))
        students.sort(key=lambda s: s.get("full_name", "").strip().lower())

        att_records = list(attendance_col.find({"date": date_str}))
        att_map = {str(a["student_id"]): a for a in att_records}

        for idx, s in enumerate(students, 1):
            s_id = str(s["_id"])
            att = att_map.get(s_id)
            if att:
                status = att.get("status", "Pending")
                method = att.get("method", "MANUAL")
                conf = f"{att.get('confidence')}%" if att.get("confidence") else "N/A"
                v_time = att.get("verified_at", "")
                remarks = att.get("remarks", "")
            else:
                status = "Pending"
                method = "Unrecorded"
                conf = "N/A"
                v_time = "N/A"
                remarks = "Not yet processed"

            writer.writerow([
                idx,
                s.get("roll_no", ""),
                s.get("full_name", ""),
                s.get("department", ""),
                status,
                method,
                conf,
                v_time,
                remarks
            ])

        filename = f"AlphabetAI_Attendance_{date_str}.csv"

    log_audit("CSV_EXPORT", f"Exported {report_type} CSV report for date {date_str}.")

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response

# ---------------------------------------------------------------------------
# Audit Logs API
# ---------------------------------------------------------------------------
@app.route("/api/audit-logs", methods=["GET"])
@login_required
def api_audit_logs():
    limit = int(request.args.get("limit", 50))
    action_filter = request.args.get("action", "").strip()

    query = {}
    if action_filter:
        query["action"] = action_filter

    logs = list(audit_col.find(query).sort("timestamp", -1).limit(limit))
    formatted = []
    for l in logs:
        formatted.append({
            "id": str(l["_id"]),
            "timestamp": l.get("timestamp", ""),
            "action": l.get("action", ""),
            "teacher_email": l.get("teacher_email", ""),
            "student_name": l.get("student_name", "N/A"),
            "student_roll": l.get("student_roll", "N/A"),
            "details": l.get("details", ""),
            "confidence": l.get("confidence"),
            "ip_address": l.get("ip_address", "")
        })

    return jsonify({"logs": formatted, "count": len(formatted)})

# ---------------------------------------------------------------------------
# UI Templates (HTML, CSS, Vanilla JavaScript inside Single File)
# ---------------------------------------------------------------------------

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Teacher Portal — AlphabetAI Attendance</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1d4ed8;
      --primary-hover: #1e40af;
      --primary-light: #eff6ff;
      --dark: #0f172a;
      --slate: #475569;
      --border: #e2e8f0;
      --bg: #f8fafc;
      --white: #ffffff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body {
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 50%, #f0fdf4 100%);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .card {
      background: var(--white);
      border-radius: 1.25rem;
      box-shadow: 0 20px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.05);
      width: 100%;
      max-width: 440px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .card-header {
      background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
      color: var(--white);
      padding: 2.25rem 2rem;
      text-align: center;
    }
    .brand-icon {
      width: 56px;
      height: 56px;
      background: rgba(255,255,255,0.2);
      border-radius: 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 1.75rem;
      margin-bottom: 0.75rem;
      backdrop-filter: blur(4px);
    }
    .card-header h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.025em; }
    .card-header p { font-size: 0.875rem; opacity: 0.9; margin-top: 0.35rem; }
    .card-body { padding: 2rem; }
    .form-group { margin-bottom: 1.25rem; }
    label { display: block; font-size: 0.875rem; font-weight: 600; color: var(--dark); margin-bottom: 0.4rem; }
    input, select {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 1.5px solid var(--border);
      border-radius: 0.65rem;
      font-size: 0.95rem;
      outline: none;
      transition: all 0.2s;
    }
    input:focus, select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37,99,235,0.15);
    }
    .btn {
      width: 100%;
      padding: 0.85rem;
      background: var(--primary);
      color: var(--white);
      border: none;
      border-radius: 0.65rem;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
      margin-top: 0.5rem;
    }
    .btn:hover { background: var(--primary-hover); }
    .alert {
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      margin-bottom: 1.25rem;
    }
    .alert-danger { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .toggle-link {
      text-align: center;
      margin-top: 1.25rem;
      font-size: 0.875rem;
      color: var(--slate);
    }
    .toggle-link a { color: var(--primary); font-weight: 600; text-decoration: none; cursor: pointer; }
    .db-pill {
      margin-top: 1rem;
      padding: 0.5rem 0.75rem;
      background: #f1f5f9;
      border-radius: 0.5rem;
      font-size: 0.75rem;
      color: var(--slate);
      text-align: center;
    }
    .demo-creds {
      background: #eff6ff;
      border: 1px dashed #93c5fd;
      border-radius: 0.5rem;
      padding: 0.75rem;
      margin-top: 1.25rem;
      font-size: 0.8rem;
      color: #1e40af;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="card-header">
      <div class="brand-icon">🎓</div>
      <h1>AlphabetAI Attendance</h1>
      <p>AI-Powered Smart Attendance System</p>
    </div>
    <div class="card-body">
      {% if error %}
      <div class="alert alert-danger">{{ error }}</div>
      {% endif %}

      <div id="loginFormContainer">
        <form method="POST" action="/login">
          <div class="form-group">
            <label for="email">Teacher Email Address</label>
            <input type="email" id="email" name="email" value="teacher@alphabet.edu" required autocomplete="username">
          </div>
          <div class="form-group">
            <label for="password">Password</label>
            <input type="password" id="password" name="password" value="Admin@123" required autocomplete="current-password">
          </div>
          <button type="submit" class="btn">Sign In to Attendance Portal</button>
        </form>

        <div class="demo-creds">
          <strong>Default Faculty Demo Access:</strong><br>
          Email: <code>teacher@alphabet.edu</code><br>
          Password: <code>Admin@123</code>
        </div>

        <div class="toggle-link">
          Need a faculty account? <a onclick="toggleForm('register')">Register Faculty</a>
        </div>
      </div>

      <div id="registerFormContainer" style="display: none;">
        <form method="POST" action="/register-teacher">
          <div class="form-group">
            <label for="reg_name">Full Name & Title</label>
            <input type="text" id="reg_name" name="full_name" placeholder="e.g. Prof. Alan Turing" required>
          </div>
          <div class="form-group">
            <label for="reg_email">Institutional Email</label>
            <input type="email" id="reg_email" name="email" placeholder="faculty@university.edu" required>
          </div>
          <div class="form-group">
            <label for="reg_dept">Department</label>
            <select id="reg_dept" name="department">
              <option value="Computer Science & AI">Computer Science & AI</option>
              <option value="Data Science & ML">Data Science & ML</option>
              <option value="Information Technology">Information Technology</option>
              <option value="Electrical Engineering">Electrical Engineering</option>
              <option value="Mechanical Engineering">Mechanical Engineering</option>
            </select>
          </div>
          <div class="form-group">
            <label for="reg_password">Create Password</label>
            <input type="password" id="reg_password" name="password" placeholder="At least 6 characters" required>
          </div>
          <button type="submit" class="btn">Create Faculty Account</button>
        </form>
        <div class="toggle-link">
          Already have an account? <a onclick="toggleForm('login')">Sign In</a>
        </div>
      </div>

      <div class="db-pill">
        Database Engine: <strong>{{ db_mode }}</strong>
      </div>
    </div>
  </div>

  <script>
    function toggleForm(mode) {
      if (mode === 'register') {
        document.getElementById('loginFormContainer').style.display = 'none';
        document.getElementById('registerFormContainer').style.display = 'block';
      } else {
        document.getElementById('registerFormContainer').style.display = 'none';
        document.getElementById('loginFormContainer').style.display = 'block';
      }
    }
  </script>
</body>
</html>
"""

APP_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AlphabetAI Attendance — AI-Powered Smart Attendance System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1e40af;
      --primary-hover: #1d4ed8;
      --primary-light: #eff6ff;
      --primary-border: #bfdbfe;
      --accent: #0284c7;
      --success: #059669;
      --success-light: #ecfdf5;
      --danger: #dc2626;
      --danger-light: #fef2f2;
      --warning: #d97706;
      --warning-light: #fffbeb;
      --dark: #0f172a;
      --slate: #334155;
      --muted: #64748b;
      --border: #e2e8f0;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --radius-sm: 8px;
      --radius: 12px;
      --radius-lg: 16px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
      --shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body { background-color: var(--bg); color: var(--dark); min-height: 100vh; display: flex; flex-direction: column; }

    /* Top Navigation */
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: var(--shadow-sm);
    }
    .navbar {
      max-width: 1400px;
      margin: 0 auto;
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: var(--dark);
    }
    .brand-logo {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, var(--primary) 0%, #3b82f6 100%);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 1.25rem;
      font-weight: 800;
    }
    .brand-text h1 { font-size: 1.15rem; font-weight: 800; color: var(--primary); letter-spacing: -0.02em; }
    .brand-text p { font-size: 0.75rem; color: var(--muted); font-weight: 500; }

    .nav-tabs {
      display: flex;
      gap: 0.5rem;
      background: #f1f5f9;
      padding: 0.35rem;
      border-radius: var(--radius);
      overflow-x: auto;
    }
    .nav-btn {
      border: none;
      background: transparent;
      padding: 0.5rem 1rem;
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--slate);
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .nav-btn.active {
      background: var(--card-bg);
      color: var(--primary);
      box-shadow: var(--shadow-sm);
    }

    .user-profile {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .user-badge {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }
    .user-badge .name { font-size: 0.875rem; font-weight: 700; color: var(--dark); }
    .user-badge .role { font-size: 0.75rem; color: var(--muted); }
    .btn-logout {
      background: var(--danger-light);
      color: var(--danger);
      border: 1px solid #fecaca;
      padding: 0.45rem 0.85rem;
      border-radius: var(--radius-sm);
      font-size: 0.8rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s;
    }
    .btn-logout:hover { background: #fee2e2; }

    /* Main Container */
    main {
      flex: 1;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      padding: 1.5rem;
    }

    /* System Status Bar */
    .status-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 0.75rem 1.25rem;
      border-radius: var(--radius);
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
    }
    .status-indicators {
      display: flex;
      align-items: center;
      gap: 1.25rem;
      flex-wrap: wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.75rem;
    }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-blue { background: var(--primary-light); color: var(--primary); }
    .pill-amber { background: var(--warning-light); color: #92400e; }

    /* KPI Summary Cards */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .kpi-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem;
      box-shadow: var(--shadow-sm);
      display: flex;
      align-items: center;
      gap: 1rem;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
    .kpi-icon {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      flex-shrink: 0;
    }
    .kpi-icon.blue { background: #eff6ff; color: #2563eb; }
    .kpi-icon.green { background: #ecfdf5; color: #059669; }
    .kpi-icon.red { background: #fef2f2; color: #dc2626; }
    .kpi-icon.amber { background: #fffbeb; color: #d97706; }
    .kpi-info .val { font-size: 1.65rem; font-weight: 800; color: var(--dark); line-height: 1.2; }
    .kpi-info .lbl { font-size: 0.8rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

    /* Layout Split */
    .split-layout {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 1024px) {
      .split-layout { grid-template-columns: 1fr; }
    }

    /* Cards */
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-sm);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .panel-header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }
    .panel-header h2 { font-size: 1.15rem; font-weight: 700; color: var(--dark); display: flex; align-items: center; gap: 0.5rem; }
    .panel-body { padding: 1.5rem; flex: 1; }

    /* Live Camera & Scanner Interface */
    .scanner-viewport {
      position: relative;
      width: 100%;
      height: 280px;
      background: #090d16;
      border-radius: var(--radius);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 2px solid #1e293b;
    }
    #cameraVideo {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1); /* mirror preview */
    }
    .reticle {
      position: absolute;
      width: 180px;
      height: 220px;
      border: 2px dashed rgba(59, 130, 246, 0.7);
      border-radius: 100px;
      pointer-events: none;
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.4);
      animation: pulseReticle 2.5s infinite ease-in-out;
    }
    @keyframes pulseReticle {
      0%, 100% { border-color: rgba(59, 130, 246, 0.5); }
      50% { border-color: rgba(34, 197, 94, 0.9); }
    }
    .scanner-overlay {
      position: absolute;
      bottom: 10px;
      left: 10px;
      right: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(6px);
      padding: 0.4rem 0.75rem;
      border-radius: var(--radius-sm);
      color: white;
      font-size: 0.75rem;
    }
    .scanner-controls {
      display: flex;
      gap: 0.5rem;
      margin-top: 1rem;
    }
    .btn-action {
      flex: 1;
      padding: 0.75rem;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: var(--radius);
      font-weight: 700;
      font-size: 0.9rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: background 0.2s;
    }
    .btn-action:hover { background: var(--primary-hover); }
    .btn-action:disabled { opacity: 0.6; cursor: not-allowed; }
    .btn-secondary {
      background: #f1f5f9;
      color: var(--slate);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover { background: #e2e8f0; }

    /* Verification Result Banner */
    .match-banner {
      margin-top: 1rem;
      padding: 1rem;
      border-radius: var(--radius);
      display: none;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
    .match-banner.success { background: #ecfdf5; border: 1.5px solid #a7f3d0; color: #065f46; }
    .match-banner.duplicate { background: #fffbeb; border: 1.5px solid #fde68a; color: #92400e; }
    .match-banner.error { background: #fef2f2; border: 1.5px solid #fecaca; color: #991b1b; }

    /* Alphabet Jump Bar */
    .alphabet-bar {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      padding: 0.5rem;
      background: #f8fafc;
      border-radius: var(--radius-sm);
      overflow-x: auto;
      margin-bottom: 1rem;
      border: 1px solid var(--border);
    }
    .alpha-chip {
      min-width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      border: 1px solid transparent;
      color: var(--slate);
      background: transparent;
    }
    .alpha-chip:hover { background: #e2e8f0; }
    .alpha-chip.active { background: var(--primary); color: white; }

    /* Search & Filter Bar */
    .filter-bar {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    .search-input {
      flex: 1;
      min-width: 200px;
      padding: 0.65rem 1rem;
      border: 1.5px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.875rem;
    }
    .select-input {
      padding: 0.65rem 1rem;
      border: 1.5px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.875rem;
      background: white;
    }

    /* Attendance Queue Table */
    .table-container {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.875rem;
    }
    th {
      background: #f1f5f9;
      padding: 0.85rem 1rem;
      font-weight: 700;
      color: var(--slate);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    td {
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background-color: #f8fafc; }

    /* Student Item Cell */
    .student-cell {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .student-avatar {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: #e2e8f0;
      object-fit: cover;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      color: var(--primary);
      flex-shrink: 0;
    }
    .student-meta .s-name { font-weight: 700; color: var(--dark); }
    .student-meta .s-roll { font-size: 0.75rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

    /* Badges */
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge-present { background: #dcfce7; color: #15803d; }
    .badge-absent { background: #fee2e2; color: #b91c1c; }
    .badge-pending { background: #fef3c7; color: #b45309; }

    /* Action Buttons in Queue */
    .action-group {
      display: flex;
      gap: 0.35rem;
    }
    .btn-sm {
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s;
    }
    .btn-present { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
    .btn-present:hover { background: #bbf7d0; }
    .btn-absent { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
    .btn-absent:hover { background: #fecaca; }
    .btn-correct { background: #f1f5f9; color: var(--slate); border-color: #cbd5e1; }
    .btn-correct:hover { background: #e2e8f0; }

    /* Modal */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1rem;
      z-index: 1000;
    }
    .modal-content {
      background: var(--card-bg);
      border-radius: var(--radius-lg);
      max-width: 540px;
      width: 100%;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border);
      overflow: hidden;
      animation: modalSlide 0.25s ease-out;
    }
    @keyframes modalSlide { from { transform: translateY(15px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .modal-header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .modal-header h3 { font-size: 1.15rem; font-weight: 700; color: var(--dark); }
    .btn-close {
      background: transparent;
      border: none;
      font-size: 1.25rem;
      color: var(--muted);
      cursor: pointer;
    }
    .modal-body { padding: 1.5rem; }
    .modal-footer {
      padding: 1rem 1.5rem;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
      background: #f8fafc;
    }

    /* Consent Checkbox Box */
    .consent-card {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: var(--radius-sm);
      padding: 0.85rem;
      margin-top: 1rem;
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
    }
    .consent-card input[type="checkbox"] { margin-top: 0.2rem; cursor: pointer; }
    .consent-card label { font-size: 0.8rem; color: #1e40af; line-height: 1.4; font-weight: 500; }

    /* Photo Snap for Registration */
    .enroll-photo-preview {
      width: 120px;
      height: 120px;
      border-radius: var(--radius);
      border: 2px dashed #cbd5e1;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 1rem;
      overflow: hidden;
      background: #f8fafc;
      position: relative;
    }
    .enroll-photo-preview img { width: 100%; height: 100%; object-fit: cover; }

    /* Tabs Content */
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header>
    <div class="navbar">
      <a href="/dashboard" class="brand">
        <div class="brand-logo">A</div>
        <div class="brand-text">
          <h1>AlphabetAI Attendance</h1>
          <p>AI-Powered Smart Attendance System</p>
        </div>
      </a>

      <!-- Navigation Tabs -->
      <nav class="nav-tabs">
        <button class="nav-btn active" onclick="switchTab('live-queue')">
          ⚡ Live Queue (A–Z)
        </button>
        <button class="nav-btn" onclick="switchTab('students-tab')">
          👥 Students Directory
        </button>
        <button class="nav-btn" onclick="switchTab('reports-tab')">
          📊 Reports & Export
        </button>
        <button class="nav-btn" onclick="switchTab('audit-tab')">
          🛡️ Audit Logs
        </button>
      </nav>

      <!-- User Profile & Logout -->
      <div class="user-profile">
        <div class="user-badge">
          <span class="name">{{ teacher_name }}</span>
          <span class="role">{{ teacher_dept }}</span>
        </div>
        <a href="/logout" class="btn-logout">Sign Out</a>
      </div>
    </div>
  </header>

  <!-- Main Body Content -->
  <main>

    <!-- System Status Bar -->
    <div class="status-bar">
      <div class="status-indicators">
        <span>Session Date: <strong><span id="displayDate">{{ today }}</span></strong></span>
        <span class="pill pill-green">● {{ db_mode }}</span>
        <span class="pill pill-blue">⚡ {{ ai_status }}</span>
        <span id="cameraStatusPill" class="pill pill-amber">📷 Camera Standby</span>
      </div>
      <div>
        <button class="btn-sm btn-correct" onclick="refreshDashboardData()">🔄 Sync Realtime</button>
      </div>
    </div>

    <!-- KPI Summary Row -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-icon blue">👥</div>
        <div class="kpi-info">
          <div class="val" id="kpiTotal">0</div>
          <div class="lbl">Total Registered (A–Z)</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon green">✅</div>
        <div class="kpi-info">
          <div class="val" id="kpiPresent">0</div>
          <div class="lbl">Present Today</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon red">❌</div>
        <div class="kpi-info">
          <div class="val" id="kpiAbsent">0</div>
          <div class="lbl">Absent Today</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon amber">⏳</div>
        <div class="kpi-info">
          <div class="val" id="kpiRate">0%</div>
          <div class="lbl">Today's Attendance Rate</div>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 1: LIVE ATTENDANCE QUEUE & CAMERA SCANNER (DEFAULT)                -->
    <!-- ===================================================================== -->
    <div id="tab-live-queue" class="tab-pane active">
      <div class="split-layout">

        <!-- Left Side: AI Face Scanner & Camera View -->
        <div class="panel">
          <div class="panel-header">
            <h2>📷 AI Face Verification</h2>
            <span class="badge badge-present" id="aiModelBadge">YuNet + SFace</span>
          </div>
          <div class="panel-body">
            <!-- Camera Viewport -->
            <div class="scanner-viewport">
              <video id="cameraVideo" autoplay playsinline muted></video>
              <canvas id="captureCanvas" style="display:none;"></canvas>
              <div class="reticle"></div>
              <div class="scanner-overlay">
                <span id="camResolution">Resolution: 640x480</span>
                <span>Genuine AI Verification</span>
              </div>
            </div>

            <!-- Scanner Buttons -->
            <div class="scanner-controls">
              <button id="btnStartCam" class="btn-action" onclick="initCamera()">
                Turn On Camera
              </button>
              <button id="btnScanFace" class="btn-action btn-secondary" onclick="scanFaceFromCamera()" disabled>
                ⚡ Verify Face
              </button>
            </div>

            <!-- Manual Photo Upload Fallback (Camera permission denied or external photo) -->
            <div style="margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 0.75rem;">
              <label style="font-size: 0.8rem; font-weight: 600; color: var(--muted); margin-bottom: 0.35rem; display: block;">
                Camera Permission Fallback (Upload Photo):
              </label>
              <input type="file" id="fallbackPhotoInput" accept="image/*" class="search-input" style="padding: 0.45rem;" onchange="handleFallbackPhotoUpload(event)">
            </div>

            <!-- Real-time Verification Result Banner -->
            <div id="matchBanner" class="match-banner">
              <strong id="matchBannerTitle">Verification Result</strong>
              <p id="matchBannerText" style="font-size: 0.85rem; margin-top: 0.25rem;"></p>
            </div>
          </div>
        </div>

        <!-- Right Side: Alphabetical Attendance Queue -->
        <div class="panel">
          <div class="panel-header">
            <h2>📋 Live Alphabetical Queue (A–Z)</h2>
            <div style="display: flex; gap: 0.5rem;">
              <button class="btn-sm btn-correct" onclick="openEnrollStudentModal()">+ Enroll Student</button>
            </div>
          </div>
          <div class="panel-body">

            <!-- Alphabet Jump Navigation Bar -->
            <div class="alphabet-bar">
              <button class="alpha-chip active" onclick="filterByLetter('')">All</button>
              <script>
                const alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
                alphabets.forEach(l => {
                  document.write(`<button class="alpha-chip" onclick="filterByLetter('${l}')">${l}</button>`);
                });
              </script>
            </div>

            <!-- Search & Filters -->
            <div class="filter-bar">
              <input type="text" id="queueSearchInput" class="search-input" placeholder="Search by name, roll number, or department..." oninput="filterQueueTable()">
              <select id="queueDeptFilter" class="select-input" onchange="loadQueueData()">
                <option value="">All Departments</option>
                <option value="Computer Science & AI">Computer Science & AI</option>
                <option value="Data Science & ML">Data Science & ML</option>
                <option value="Information Technology">Information Technology</option>
                <option value="Electrical Engineering">Electrical Engineering</option>
                <option value="Mechanical Engineering">Mechanical Engineering</option>
              </select>
            </div>

            <!-- Attendance Queue Table -->
            <div class="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Student Name (A–Z)</th>
                    <th>Department</th>
                    <th>Status</th>
                    <th>Verification Details</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody id="queueTableBody">
                  <tr>
                    <td colspan="5" style="text-align:center; padding: 2rem; color: var(--muted);">
                      Loading student attendance queue...
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

          </div>
        </div>

      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 2: STUDENTS DIRECTORY (REGISTRATION, EDIT, BIOMETRIC CONSENT)     -->
    <!-- ===================================================================== -->
    <div id="tab-students-tab" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <h2>👥 Registered Students Directory (Sorted Alphabetically A–Z)</h2>
          <button class="btn-action" style="padding: 0.5rem 1rem; width: auto;" onclick="openEnrollStudentModal()">
            + Register New Student
          </button>
        </div>
        <div class="panel-body">
          <div class="filter-bar">
            <input type="text" id="studentSearchInput" class="search-input" placeholder="Search student directory..." oninput="loadStudentsDirectory()">
            <select id="studentDeptFilter" class="select-input" onchange="loadStudentsDirectory()">
              <option value="">All Departments</option>
              <option value="Computer Science & AI">Computer Science & AI</option>
              <option value="Data Science & ML">Data Science & ML</option>
              <option value="Information Technology">Information Technology</option>
              <option value="Electrical Engineering">Electrical Engineering</option>
              <option value="Mechanical Engineering">Mechanical Engineering</option>
            </select>
          </div>

          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Student Name & Roll No</th>
                  <th>Institutional Email</th>
                  <th>Department</th>
                  <th>Biometric Status</th>
                  <th>Consent Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="studentsTableBody">
                <tr><td colspan="6" style="text-align:center; padding: 2rem;">Loading directory...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 3: REPORTS, PERCENTAGES & CSV EXPORT                             -->
    <!-- ===================================================================== -->
    <div id="tab-reports-tab" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <h2>📊 Attendance Analytics, History & Reports</h2>
          <div style="display:flex; gap: 0.5rem;">
            <button class="btn-action" style="width: auto; padding: 0.5rem 1rem;" onclick="exportCSV('session')">
              📥 Export Day CSV
            </button>
            <button class="btn-action btn-secondary" style="width: auto; padding: 0.5rem 1rem;" onclick="exportCSV('cumulative')">
              📥 Export Cumulative Summary CSV
            </button>
          </div>
        </div>
        <div class="panel-body">
          <div class="filter-bar">
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <label style="font-size:0.85rem; font-weight:600;">Select Date:</label>
              <input type="date" id="reportDateInput" class="select-input" value="{{ today }}" onchange="loadReportsData()">
            </div>
            <select id="reportDeptFilter" class="select-input" onchange="loadReportsData()">
              <option value="">All Departments</option>
              <option value="Computer Science & AI">Computer Science & AI</option>
              <option value="Data Science & ML">Data Science & ML</option>
              <option value="Information Technology">Information Technology</option>
              <option value="Electrical Engineering">Electrical Engineering</option>
              <option value="Mechanical Engineering">Mechanical Engineering</option>
            </select>
          </div>

          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Student Name (A–Z)</th>
                  <th>Roll No</th>
                  <th>Department</th>
                  <th>Selected Date Status</th>
                  <th>Total Present</th>
                  <th>Total Conducted</th>
                  <th>Cumulative %</th>
                  <th>Eligibility</th>
                </tr>
              </thead>
              <tbody id="reportsTableBody">
                <tr><td colspan="8" style="text-align:center; padding: 2rem;">Loading attendance reports...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 4: AUDIT TRAIL LOGS                                               -->
    <!-- ===================================================================== -->
    <div id="tab-audit-tab" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <h2>🛡️ Security & Attendance Audit Trail</h2>
          <button class="btn-sm btn-correct" onclick="loadAuditLogs()">🔄 Refresh Audit</button>
        </div>
        <div class="panel-body">
          <div class="filter-bar">
            <select id="auditActionFilter" class="select-input" onchange="loadAuditLogs()">
              <option value="">All Audit Actions</option>
              <option value="FACE_VERIFIED_ATTENDANCE">AI Face Verified</option>
              <option value="DUPLICATE_ATTENDANCE_BLOCKED">Duplicate Blocked</option>
              <option value="MANUAL_ATTENDANCE_OVERRIDE">Manual Override</option>
              <option value="ATTENDANCE_CORRECTION">Correction</option>
              <option value="STUDENT_ENROLLED">Student Enrolled</option>
              <option value="AUTH_LOGIN">Faculty Login</option>
            </select>
          </div>

          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Faculty</th>
                  <th>Student</th>
                  <th>Event Details</th>
                  <th>AI Confidence</th>
                </tr>
              </thead>
              <tbody id="auditTableBody">
                <tr><td colspan="6" style="text-align:center; padding: 2rem;">Loading audit trail...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- ===================================================================== -->
  <!-- MODAL: ENROLL STUDENT WITH BIOMETRIC CONSENT & FACE RECOGNITION       -->
  <!-- ===================================================================== -->
  <div id="enrollModal" class="modal-backdrop">
    <div class="modal-content">
      <div class="modal-header">
        <h3 id="enrollModalTitle">Enroll New Student</h3>
        <button class="btn-close" onclick="closeEnrollModal()">&times;</button>
      </div>
      <div class="modal-body">
        <input type="hidden" id="editStudentId">
        <div style="margin-bottom: 1rem;">
          <label style="font-size:0.85rem; font-weight:600; display:block; margin-bottom:0.35rem;">Full Name *</label>
          <input type="text" id="enrollName" class="search-input" placeholder="e.g. Alice Wonderland" style="width:100%;">
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.75rem; margin-bottom: 1rem;">
          <div>
            <label style="font-size:0.85rem; font-weight:600; display:block; margin-bottom:0.35rem;">Roll Number *</label>
            <input type="text" id="enrollRoll" class="search-input" placeholder="e.g. CS2026001" style="width:100%;">
          </div>
          <div>
            <label style="font-size:0.85rem; font-weight:600; display:block; margin-bottom:0.35rem;">Department *</label>
            <select id="enrollDept" class="select-input" style="width:100%;">
              <option value="Computer Science & AI">Computer Science & AI</option>
              <option value="Data Science & ML">Data Science & ML</option>
              <option value="Information Technology">Information Technology</option>
              <option value="Electrical Engineering">Electrical Engineering</option>
              <option value="Mechanical Engineering">Mechanical Engineering</option>
            </select>
          </div>
        </div>
        <div style="margin-bottom: 1rem;">
          <label style="font-size:0.85rem; font-weight:600; display:block; margin-bottom:0.35rem;">Email Address *</label>
          <input type="email" id="enrollEmail" class="search-input" placeholder="alice@university.edu" style="width:100%;">
        </div>

        <!-- Biometric Face Capture -->
        <div style="border-top:1px solid var(--border); padding-top: 1rem; margin-top: 1rem;">
          <label style="font-size:0.85rem; font-weight:700; color:var(--primary); display:block; margin-bottom:0.5rem;">
            📷 Biometric Face Enrollment (OpenCV SFace)
          </label>
          <div class="enroll-photo-preview" id="enrollPhotoBox">
            <span style="font-size:0.75rem; color:var(--muted); text-align:center; padding:0.5rem;">No Photo Selected</span>
          </div>
          <div style="display:flex; gap:0.5rem; justify-content:center;">
            <button class="btn-sm btn-correct" onclick="captureEnrollSnapshotFromCam()">Use Live Camera Frame</button>
            <label class="btn-sm btn-correct" style="cursor:pointer; display:inline-flex; align-items:center;">
              Upload Photo File
              <input type="file" id="enrollFileInput" accept="image/*" style="display:none;" onchange="handleEnrollFileSelected(event)">
            </label>
          </div>
        </div>

        <!-- Biometric Protection & Consent Notice -->
        <div class="consent-card">
          <input type="checkbox" id="enrollConsent" required checked>
          <label for="enrollConsent">
            <strong>Biometric Data Consent & Safeguard Agreement:</strong><br>
            I confirm that the student provides informed consent for extracting a secure, 128-dimensional facial embedding vector for the sole purpose of smart attendance verification. Raw biometrics are strictly protected under institutional data privacy policy.
          </label>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-sm btn-correct" onclick="closeEnrollModal()">Cancel</button>
        <button id="btnSaveStudent" class="btn-action" style="width:auto; padding:0.6rem 1.25rem;" onclick="saveStudent()">
          Save & Register Student
        </button>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- MODAL: MANUAL CORRECTION & AUDITED OVERRIDE                            -->
  <!-- ===================================================================== -->
  <div id="correctModal" class="modal-backdrop">
    <div class="modal-content">
      <div class="modal-header">
        <h3>Manual Attendance Correction</h3>
        <button class="btn-close" onclick="closeCorrectModal()">&times;</button>
      </div>
      <div class="modal-body">
        <input type="hidden" id="correctStudentId">
        <p style="font-size:0.9rem; margin-bottom:1rem;">
          Editing attendance record for: <strong id="correctStudentName">Student</strong> (<span id="correctStudentRoll">Roll</span>)
        </p>
        <div style="margin-bottom: 1rem;">
          <label style="font-size:0.85rem; font-weight:600; display:block; margin-bottom:0.35rem;">Attendance Status</label>
          <select id="correctStatusSelect" class="select-input" style="width:100%;">
            <option value="Present">Present</option>
            <option value="Absent">Absent</option>
            <option value="Pending">Reset to Pending (Unmarked)</option>
          </select>
        </div>
        <div style="margin-bottom: 1rem;">
          <label style="font-size:0.85rem; font-weight:600; display:block; margin-bottom:0.35rem;">Audit Reason / Correction Note *</label>
          <input type="text" id="correctReason" class="search-input" placeholder="e.g. Camera glare, medical leave note, late arrival" style="width:100%;" value="Manual faculty verification">
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-sm btn-correct" onclick="closeCorrectModal()">Cancel</button>
        <button class="btn-action" style="width:auto; padding:0.6rem 1.25rem;" onclick="submitManualCorrection()">
          Confirm & Log Correction
        </button>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- VANILLA JAVASCRIPT APPLICATION LOGIC                                  -->
  <!-- ===================================================================== -->
  <script>
    // State
    let currentTab = 'live-queue';
    let cameraStream = null;
    let currentFilterLetter = '';
    let enrolledPhotoBase64 = null;
    let fullQueueData = [];

    // --- Tab Switching ---
    function switchTab(tabId) {
      currentTab = tabId;
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

      const pane = document.getElementById('tab-' + tabId);
      if (pane) pane.classList.add('active');

      // Highlight active nav button
      event?.target?.classList.add('active');

      if (tabId === 'live-queue') loadQueueData();
      else if (tabId === 'students-tab') loadStudentsDirectory();
      else if (tabId === 'reports-tab') loadReportsData();
      else if (tabId === 'audit-tab') loadAuditLogs();
    }

    // --- Camera Handling ---
    async function initCamera() {
      const video = document.getElementById('cameraVideo');
      const pill = document.getElementById('cameraStatusPill');
      const btnStart = document.getElementById('btnStartCam');
      const btnScan = document.getElementById('btnScanFace');

      try {
        if (cameraStream) {
          cameraStream.getTracks().forEach(t => t.stop());
          cameraStream = null;
          video.srcObject = null;
          btnStart.innerText = 'Turn On Camera';
          btnScan.disabled = true;
          pill.className = 'pill pill-amber';
          pill.innerText = '📷 Camera Standby';
          return;
        }

        pill.innerText = '📷 Requesting Permission...';
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });

        cameraStream = stream;
        video.srcObject = stream;
        await video.play();

        btnStart.innerText = 'Turn Off Camera';
        btnScan.disabled = false;
        pill.className = 'pill pill-green';
        pill.innerText = '📷 Camera Active';
        document.getElementById('camResolution').innerText = `Active: ${video.videoWidth || 640}x${video.videoHeight || 480}`;

      } catch (err) {
        console.warn('Camera access denied or unavailable:', err);
        pill.className = 'pill pill-amber';
        pill.innerText = '📷 Camera Blocked / Unavailable';
        alert('Camera access could not be started (' + err.message + ').\\n\\nYou can use the "Camera Permission Fallback (Upload Photo)" option below to test AI Face Recognition directly with any photo file!');
      }
    }

    function grabCameraSnapshotBase64() {
      const video = document.getElementById('cameraVideo');
      if (!cameraStream || !video.videoWidth) return null;

      const canvas = document.getElementById('captureCanvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      // Mirror horizontal to match display
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.9);
    }

    // --- Face Verification from Live Camera ---
    async function scanFaceFromCamera() {
      const b64 = grabCameraSnapshotBase64();
      if (!b64) {
        showMatchBanner('error', 'Camera Frame Empty', 'Turn on the camera and position your face in the reticle.');
        return;
      }
      await performAIFaceVerification(b64);
    }

    // --- Fallback Photo Upload Verification ---
    function handleFallbackPhotoUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = async function(e) {
        const b64 = e.target.result;
        await performAIFaceVerification(b64);
      };
      reader.readAsDataURL(file);
    }

    async function performAIFaceVerification(imageBase64) {
      const banner = document.getElementById('matchBanner');
      showMatchBanner('duplicate', 'Scanning Face...', 'Executing genuine OpenCV YuNet detection & SFace embedding matching...');

      try {
        const resp = await fetch('/api/verify-face', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: imageBase64,
            date: document.getElementById('displayDate').innerText
          })
        });

        const data = await resp.json();

        if (data.status === 'MARKED_PRESENT') {
          showMatchBanner(
            'success',
            `Verified: ${data.student.full_name} (${data.student.roll_no})`,
            `Status: Marked Present! | AI Confidence: ${data.confidence}% | Time: ${data.verified_at}`
          );
          // Play subtle success sound
          playChime(true);
          refreshDashboardData();
        } else if (data.status === 'DUPLICATE_PREVENTED') {
          showMatchBanner(
            'duplicate',
            '⚠️ Duplicate Attendance Prevented',
            data.message
          );
          playChime(false);
        } else {
          showMatchBanner(
            'error',
            '❌ Face Not Recognized',
            data.message || 'Face did not match any enrolled student embeddings above the security threshold.'
          );
        }
      } catch (err) {
        showMatchBanner('error', 'Verification Error', err.message);
      }
    }

    function showMatchBanner(type, title, text) {
      const banner = document.getElementById('matchBanner');
      banner.className = 'match-banner ' + type;
      banner.style.display = 'block';
      document.getElementById('matchBannerTitle').innerText = title;
      document.getElementById('matchBannerText').innerText = text;
    }

    function playChime(success) {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = success ? 587.33 : 220; // D5 vs A3
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        osc.start();
        osc.stop(ctx.currentTime + (success ? 0.2 : 0.4));
      } catch (e) {}
    }

    // --- Alphabetical Queue Loading & Filtering ---
    async function loadQueueData() {
      const dept = document.getElementById('queueDeptFilter').value;
      const date = document.getElementById('displayDate').innerText;

      try {
        const res = await fetch(`/api/queue?date=${date}&dept=${encodeURIComponent(dept)}`);
        const data = await res.json();
        fullQueueData = data.queue || [];

        // Update KPIs
        document.getElementById('kpiTotal').innerText = data.total_students;
        document.getElementById('kpiPresent').innerText = data.present_count;
        document.getElementById('kpiAbsent').innerText = data.absent_count;
        document.getElementById('kpiRate').innerText = data.attendance_rate + '%';

        renderQueueTable();
      } catch (e) {
        console.error('Failed to load queue:', e);
      }
    }

    function filterByLetter(letter) {
      currentFilterLetter = letter;
      document.querySelectorAll('.alpha-chip').forEach(c => {
        if ((!letter && c.innerText === 'All') || c.innerText === letter) {
          c.classList.add('active');
        } else {
          c.classList.remove('active');
        }
      });
      renderQueueTable();
    }

    function filterQueueTable() {
      renderQueueTable();
    }

    function renderQueueTable() {
      const search = (document.getElementById('queueSearchInput')?.value || '').toLowerCase();
      const tbody = document.getElementById('queueTableBody');

      let filtered = fullQueueData.filter(s => {
        const name = s.full_name || '';
        const roll = s.roll_no || '';
        const dept = s.department || '';

        // Alphabet filter
        if (currentFilterLetter && !name.toUpperCase().startsWith(currentFilterLetter)) {
          return false;
        }
        // Search filter
        if (search && (!name.toLowerCase().includes(search) && !roll.toLowerCase().includes(search) && !dept.toLowerCase().includes(search))) {
          return false;
        }
        return true;
      });

      if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 2rem; color: var(--muted);">No students found matching current filter.</td></tr>`;
        return;
      }

      let html = '';
      filtered.forEach(s => {
        let statusBadge = `<span class="badge badge-pending">⏳ Pending</span>`;
        if (s.status === 'Present') statusBadge = `<span class="badge badge-present">✅ Present</span>`;
        if (s.status === 'Absent') statusBadge = `<span class="badge badge-absent">❌ Absent</span>`;

        let methodText = '<span style="color:var(--muted);">Not verified yet</span>';
        if (s.status === 'Present') {
          if (s.method === 'AI_FACE_RECOGNITION') {
            methodText = `<strong style="color:var(--primary);">⚡ AI Face Match</strong> (${s.confidence || 95}% conf) at ${s.verified_at}`;
          } else {
            methodText = `<span>Teacher Override at ${s.verified_at}</span>`;
          }
        } else if (s.status === 'Absent') {
          methodText = `<span style="color:var(--danger);">Marked Absent (${s.remarks || 'Faculty entry'})</span>`;
        }

        const avatar = s.photo_thumbnail 
          ? `<img src="${s.photo_thumbnail}" class="student-avatar" alt="Avatar">`
          : `<div class="student-avatar">${s.full_name.charAt(0)}</div>`;

        html += `
          <tr>
            <td>
              <div class="student-cell">
                ${avatar}
                <div class="student-meta">
                  <div class="s-name">${s.full_name}</div>
                  <div class="s-roll">${s.roll_no}</div>
                </div>
              </div>
            </td>
            <td>${s.department}</td>
            <td>${statusBadge}</td>
            <td style="font-size:0.8rem;">${methodText}</td>
            <td>
              <div class="action-group">
                <button class="btn-sm btn-present" title="Mark Present" onclick="quickMarkAttendance('${s.student_id}', 'Present')">✅ Present</button>
                <button class="btn-sm btn-absent" title="Mark Absent" onclick="quickMarkAttendance('${s.student_id}', 'Absent')">❌ Absent</button>
                <button class="btn-sm btn-correct" title="Manual Correction" onclick="openCorrectModal('${s.student_id}', '${escapeQuotes(s.full_name)}', '${s.roll_no}', '${s.status}')">✏️ Edit</button>
              </div>
            </td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
    }

    function escapeQuotes(str) {
      return (str || '').replace(/'/g, "\\\\'");
    }

    // --- Quick Manual Attendance Action ---
    async function quickMarkAttendance(studentId, status) {
      try {
        const resp = await fetch('/api/mark-attendance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            student_id: studentId,
            status: status,
            date: document.getElementById('displayDate').innerText,
            reason: 'Quick action by faculty'
          })
        });
        const data = await resp.json();
        if (data.success) {
          refreshDashboardData();
        } else {
          alert('Error: ' + data.error);
        }
      } catch (e) {
        alert('Network error marking attendance: ' + e.message);
      }
    }

    // --- Correction Modal ---
    function openCorrectModal(studentId, name, roll, currentStatus) {
      document.getElementById('correctStudentId').value = studentId;
      document.getElementById('correctStudentName').innerText = name;
      document.getElementById('correctStudentRoll').innerText = roll;
      document.getElementById('correctStatusSelect').value = currentStatus || 'Present';
      document.getElementById('correctReason').value = 'Manual faculty override';
      document.getElementById('correctModal').style.display = 'flex';
    }

    function closeCorrectModal() {
      document.getElementById('correctModal').style.display = 'none';
    }

    async function submitManualCorrection() {
      const studentId = document.getElementById('correctStudentId').value;
      const status = document.getElementById('correctStatusSelect').value;
      const reason = document.getElementById('correctReason').value;

      try {
        const resp = await fetch('/api/mark-attendance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            student_id: studentId,
            status: status,
            reason: reason,
            date: document.getElementById('displayDate').innerText
          })
        });
        const data = await resp.json();
        if (data.success) {
          closeCorrectModal();
          refreshDashboardData();
        } else {
          alert('Failed: ' + data.error);
        }
      } catch (e) {
        alert('Error: ' + e.message);
      }
    }

    // --- Student Directory Tab ---
    async function loadStudentsDirectory() {
      const search = document.getElementById('studentSearchInput')?.value || '';
      const dept = document.getElementById('studentDeptFilter')?.value || '';
      const tbody = document.getElementById('studentsTableBody');

      try {
        const res = await fetch(`/api/students?search=${encodeURIComponent(search)}&dept=${encodeURIComponent(dept)}`);
        const data = await res.json();
        const students = data.students || [];

        if (students.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:2rem; color:var(--muted);">No students registered yet. Click "+ Register New Student" to get started.</td></tr>`;
          return;
        }

        let html = '';
        students.forEach(s => {
          const avatar = s.photo_thumbnail 
            ? `<img src="${s.photo_thumbnail}" class="student-avatar" alt="Avatar">`
            : `<div class="student-avatar">${s.full_name.charAt(0)}</div>`;

          const bioBadge = s.biometric_enrolled
            ? `<span class="badge badge-present">✓ Enrolled (128-d)</span>`
            : `<span class="badge badge-absent">Not Enrolled</span>`;

          const consentBadge = s.consent_granted
            ? `<span class="badge badge-present">✓ Consented</span>`
            : `<span class="badge badge-pending">Pending</span>`;

          html += `
            <tr>
              <td>
                <div class="student-cell">
                  ${avatar}
                  <div class="student-meta">
                    <div class="s-name">${s.full_name}</div>
                    <div class="s-roll">${s.roll_no}</div>
                  </div>
                </div>
              </td>
              <td>${s.email}</td>
              <td>${s.department}</td>
              <td>${bioBadge}</td>
              <td>${consentBadge}</td>
              <td>
                <div class="action-group">
                  <button class="btn-sm btn-correct" onclick="deleteStudent('${s.id}', '${escapeQuotes(s.full_name)}')">🗑️ Remove</button>
                </div>
              </td>
            </tr>
          `;
        });
        tbody.innerHTML = html;
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:2rem; color:var(--danger);">Error loading directory: ${e.message}</td></tr>`;
      }
    }

    async function deleteStudent(id, name) {
      if (!confirm(`Are you sure you want to permanently remove student "${name}" and all attendance records?`)) return;

      try {
        const resp = await fetch(`/api/students/${id}`, { method: 'DELETE' });
        const data = await resp.json();
        if (data.success) {
          loadStudentsDirectory();
          loadQueueData();
        } else {
          alert('Delete error: ' + data.error);
        }
      } catch (e) {
        alert('Network error: ' + e.message);
      }
    }

    // --- Student Enrollment Modal ---
    function openEnrollStudentModal() {
      document.getElementById('enrollName').value = '';
      document.getElementById('enrollRoll').value = '';
      document.getElementById('enrollEmail').value = '';
      document.getElementById('enrollConsent').checked = true;
      enrolledPhotoBase64 = null;
      document.getElementById('enrollPhotoBox').innerHTML = `<span style="font-size:0.75rem; color:var(--muted); text-align:center; padding:0.5rem;">No Photo Selected</span>`;
      document.getElementById('enrollModal').style.display = 'flex';
    }

    function closeEnrollModal() {
      document.getElementById('enrollModal').style.display = 'none';
    }

    function captureEnrollSnapshotFromCam() {
      const b64 = grabCameraSnapshotBase64();
      if (!b64) {
        alert('Camera is not running. Please turn on the camera on the live queue panel first, or use "Upload Photo File" below.');
        return;
      }
      enrolledPhotoBase64 = b64;
      document.getElementById('enrollPhotoBox').innerHTML = `<img src="${b64}" alt="Face Snapshot">`;
    }

    function handleEnrollFileSelected(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(e) {
        enrolledPhotoBase64 = e.target.result;
        document.getElementById('enrollPhotoBox').innerHTML = `<img src="${enrolledPhotoBase64}" alt="Face File">`;
      };
      reader.readAsDataURL(file);
    }

    async function saveStudent() {
      const name = document.getElementById('enrollName').value.trim();
      const roll = document.getElementById('enrollRoll').value.trim();
      const dept = document.getElementById('enrollDept').value;
      const email = document.getElementById('enrollEmail').value.trim();
      const consent = document.getElementById('enrollConsent').checked;

      if (!name || !roll || !email) {
        alert('Please fill in Name, Roll Number, and Email.');
        return;
      }

      if (!consent) {
        alert('Biometric protection compliance requires explicit consent to be checked.');
        return;
      }

      const btnSave = document.getElementById('btnSaveStudent');
      btnSave.disabled = true;
      btnSave.innerText = 'Extracting AI Features...';

      try {
        const resp = await fetch('/api/students', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            full_name: name,
            roll_no: roll,
            department: dept,
            email: email,
            photo_data: enrolledPhotoBase64,
            consent_granted: consent
          })
        });

        const data = await resp.json();
        if (data.success) {
          alert('Success: ' + data.message);
          closeEnrollModal();
          loadStudentsDirectory();
          loadQueueData();
        } else {
          alert('Enrollment Error: ' + data.error);
        }
      } catch (e) {
        alert('Network error during enrollment: ' + e.message);
      } finally {
        btnSave.disabled = false;
        btnSave.innerText = 'Save & Register Student';
      }
    }

    // --- Reports & Export Tab ---
    async function loadReportsData() {
      const date = document.getElementById('reportDateInput').value;
      const dept = document.getElementById('reportDeptFilter').value;
      const tbody = document.getElementById('reportsTableBody');

      try {
        const resp = await fetch(`/api/reports?date=${date}&dept=${encodeURIComponent(dept)}`);
        const data = await resp.json();
        const reports = data.reports || [];

        if (reports.length === 0) {
          tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:2rem; color:var(--muted);">No records found.</td></tr>`;
          return;
        }

        let html = '';
        reports.forEach(r => {
          let statusBadge = `<span class="badge badge-pending">⏳ Pending</span>`;
          if (r.today_status === 'Present') statusBadge = `<span class="badge badge-present">✅ Present</span>`;
          if (r.today_status === 'Absent') statusBadge = `<span class="badge badge-absent">❌ Absent</span>`;

          const eligBadge = r.percentage >= 75.0
            ? `<span class="badge badge-present">Eligible</span>`
            : `<span class="badge badge-absent">Shortage</span>`;

          html += `
            <tr>
              <td><strong>${r.full_name}</strong></td>
              <td><code>${r.roll_no}</code></td>
              <td>${r.department}</td>
              <td>${statusBadge}</td>
              <td><strong>${r.present_days}</strong></td>
              <td>${r.total_sessions}</td>
              <td><strong>${r.percentage}%</strong></td>
              <td>${eligBadge}</td>
            </tr>
          `;
        });

        tbody.innerHTML = html;
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:2rem; color:var(--danger);">Error loading report: ${e.message}</td></tr>`;
      }
    }

    function exportCSV(type) {
      const date = document.getElementById('reportDateInput')?.value || document.getElementById('displayDate').innerText;
      const dept = document.getElementById('reportDeptFilter')?.value || '';
      window.location.href = `/export-csv?date=${date}&type=${type}&dept=${encodeURIComponent(dept)}`;
    }

    // --- Audit Logs Tab ---
    async function loadAuditLogs() {
      const action = document.getElementById('auditActionFilter')?.value || '';
      const tbody = document.getElementById('auditTableBody');

      try {
        const resp = await fetch(`/api/audit-logs?action=${encodeURIComponent(action)}`);
        const data = await resp.json();
        const logs = data.logs || [];

        if (logs.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:2rem; color:var(--muted);">No audit logs found.</td></tr>`;
          return;
        }

        let html = '';
        logs.forEach(l => {
          let actionBadge = `<span class="badge badge-pending">${l.action}</span>`;
          if (l.action.includes('FACE_VERIFIED')) actionBadge = `<span class="badge badge-present">⚡ AI Verified</span>`;
          if (l.action.includes('DUPLICATE')) actionBadge = `<span class="badge badge-absent">⚠️ Duplicate Blocked</span>`;
          if (l.action.includes('OVERRIDE') || l.action.includes('CORRECTION')) actionBadge = `<span class="badge badge-pending">✏️ Manual Action</span>`;

          const dt = new Date(l.timestamp).toLocaleString();

          html += `
            <tr>
              <td style="font-size:0.75rem; color:var(--muted);">${dt}</td>
              <td>${actionBadge}</td>
              <td style="font-size:0.8rem;">${l.teacher_email}</td>
              <td><strong>${l.student_name || '—'}</strong> <br><small style="color:var(--muted);">${l.student_roll || ''}</small></td>
              <td style="font-size:0.85rem;">${l.details}</td>
              <td style="font-size:0.8rem;">${l.confidence ? l.confidence + '%' : '—'}</td>
            </tr>
          `;
        });

        tbody.innerHTML = html;
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:2rem; color:var(--danger);">Error loading audit: ${e.message}</td></tr>`;
      }
    }

    // --- Master Realtime Dashboard Refresh ---
    function refreshDashboardData() {
      if (currentTab === 'live-queue') loadQueueData();
      else if (currentTab === 'students-tab') loadStudentsDirectory();
      else if (currentTab === 'reports-tab') loadReportsData();
      else if (currentTab === 'audit-tab') loadAuditLogs();
    }

    // Auto-load on page startup
    document.addEventListener('DOMContentLoaded', () => {
      loadQueueData();
    });
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("  AlphabetAI Attendance — AI-Powered Smart Attendance System")
    print(f"  Database Engine: {DB_MODE}")
    print(f"  AI Facial Model: {AI_STATUS}")
    print(f"  Access Portal:   http://127.0.0.1:{port}")
    print("  Faculty Access:  teacher@alphabet.edu / Admin@123")
    print("=" * 70)
    app.run(host="127.0.0.1", port=port, debug=False)
