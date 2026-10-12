import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template_string, Response, request, redirect, url_for, jsonify, session
import face_recognition
from threading import Thread, Lock
import logging
import atexit
import time
from collections import deque
from functools import wraps
import uuid

# ---------------- CONFIG ---------------- #
BASE_PATH = 'known_faces'
THRESHOLD = 0.45
FRAME_SKIP = 4
MAX_IMAGES_PER_STUDENT = 5
CAMERA_INDEX = 0
CAMERA_RETRY_ATTEMPTS = 3
LABEL_PERSIST_FRAMES = 5
FRAME_SIZE_SCALE = 0.5
USERS = {"admin": {"password": "admin123", "role": "admin"}}
ADMIN_NAME = "admin"  # Define admin identifier

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
os.makedirs(BASE_PATH, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- GLOBAL STATE ---------------- #
encodings_dict = {}
name_class_map = {}
running = False
current_class = None
attendance_file = None
cap = None
frame_count = 0
last_frame_bgr = None
last_frame_lock = Lock()
camera_error = None
last_detections = deque(maxlen=LABEL_PERSIST_FRAMES)
camera_thread = None
camera_running = False
cached_classes = []
cached_students = {}

# ---------------- CAMERA MANAGEMENT ---------------- #
def initialize_camera():
    global cap, camera_error, camera_running
    for attempt in range(CAMERA_RETRY_ATTEMPTS):
        try:
            if cap is not None and cap.isOpened():
                cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                camera_running = True
                logger.info("Camera initialized successfully")
                camera_error = None
                return True
            else:
                logger.warning(f"Camera initialization attempt {attempt + 1} failed")
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
        time.sleep(0.5)
    camera_error = "Failed to open webcam after multiple attempts"
    logger.error(camera_error)
    return False

def release_camera():
    global cap, camera_running
    camera_running = False
    if cap is not None and cap.isOpened():
        cap.release()
        logger.info("Camera released")
    cap = None

def camera_reader():
    global last_frame_bgr, camera_error
    while camera_running:
        try:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to capture frame")
                camera_error = "Failed to capture frame"
                continue
            with last_frame_lock:
                last_frame_bgr = frame
        except Exception as e:
            logger.error(f"Camera reader error: {str(e)}")
            camera_error = "Camera read error"
        time.sleep(0.01)

atexit.register(release_camera)

# ---------------- ENCODINGS ---------------- #
def load_known_faces(base_path=BASE_PATH):
    encodings = {}
    mapping = {}
    try:
        for class_folder in os.listdir(base_path):
            class_path = os.path.join(base_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            for file in os.listdir(class_path):
                if file.startswith(ADMIN_NAME + "_"):
                    logger.info(f"Skipping admin image: {file}")
                    continue
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_enc = face_recognition.face_encodings(img_rgb, model='hog')
                if face_enc:
                    full_name = os.path.splitext(file)[0]
                    base_name = full_name.split("_")[0]
                    if base_name not in encodings:
                        encodings[base_name] = []
                        mapping[base_name] = class_folder
                    encodings[base_name].append(face_enc[0])
        logger.info(f"Loaded {sum(len(v) for v in encodings.values())} encodings")
        return encodings, mapping
    except Exception as e:
        logger.error(f"Error loading encodings: {str(e)}")
        return {}, {}

# ---------------- ATTENDANCE FILE ---------------- #
def list_students_for_class(cls):
    global cached_students
    if cls not in cached_students:
        cached_students[cls] = sorted([n for n, c in name_class_map.items() if c == cls and n != ADMIN_NAME])
    return cached_students[cls]

def initialize_attendance_file(class_name, students):
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        fname = f"Attendance_{class_name}_{today}.xlsx"
        if not os.path.exists(fname):
            df = pd.DataFrame({"Name": students, "Status": ["U"] * len(students), "Timestamp": [""] * len(students)})
            df.to_excel(fname, index=False)
            logger.info(f"Created attendance file: {fname}")
        return fname
    except Exception as e:
        logger.error(f"Error initializing attendance file: {str(e)}")
        return None

def mark_attendance(name, class_name, filename):
    if name == ADMIN_NAME:
        logger.info(f"Skipping attendance marking for admin: {name}")
        return
    try:
        df = pd.read_excel(filename)
        if name in df['Name'].values:
            idx = df[df['Name'] == name].index[0]
            if df.at[idx, "Status"] != "A":
                df.at[idx, "Status"] = "A"
                df.at[idx, "Timestamp"] = datetime.now().strftime("%H:%M:%S")
                df.to_excel(filename, index=False)
                logger.info(f"Marked {name} present in {class_name}")
        else:
            logger.warning(f"{name} not in attendance list")
    except Exception as e:
        logger.error(f"Error marking attendance: {str(e)}")

# ---------------- CAMERA STREAM ---------------- #
def video_generator():
    global frame_count, last_frame_bgr, camera_error, last_detections
    if not camera_running:
        if not initialize_camera():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + create_error_image(camera_error).tobytes() + b'\r\n')
            return
        global camera_thread
        camera_thread = Thread(target=camera_reader)
        camera_thread.daemon = True
        camera_thread.start()

    while True:
        try:
            with last_frame_lock:
                if last_frame_bgr is None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + create_error_image("No frame available").tobytes() + b'\r\n')
                    continue
                processed_frame = last_frame_bgr.copy()

            current_detections = []
            if running and current_class is not None:
                frame_count += 1
                if frame_count % FRAME_SKIP == 0:
                    small_img = cv2.resize(processed_frame, (0, 0), fx=FRAME_SIZE_SCALE, fy=FRAME_SIZE_SCALE)
                    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                    faces_cur_frame = face_recognition.face_locations(rgb_small, model='hog')
                    encodes_cur_frame = face_recognition.face_encodings(rgb_small, faces_cur_frame, model='hog')

                    for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
                        name = "UNKNOWN"
                        min_distance = 1.0
                        for known_name, enc_list in encodings_dict.items():
                            if name_class_map.get(known_name) != current_class or known_name == ADMIN_NAME:
                                continue
                            distances = face_recognition.face_distance(enc_list, encodeFace)
                            best_distance = np.min(distances) if len(distances) > 0 else 1.0
                            if best_distance < min_distance and best_distance < THRESHOLD:
                                min_distance = best_distance
                                name = known_name
                        current_detections.append((name, faceLoc))
                        if name != "UNKNOWN" and name != ADMIN_NAME and attendance_file:
                            mark_attendance(name, current_class, attendance_file)

                last_detections.append(current_detections)
                detections_to_use = current_detections if current_detections else next(
                    (d for d in reversed(last_detections) if d), [])

                for name, faceLoc in detections_to_use:
                    y1, x2, y2, x1 = [int(v / FRAME_SIZE_SCALE) for v in faceLoc]
                    color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(processed_frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    cv2.putText(processed_frame, name, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ok, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Error in video generator: {str(e)}")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + create_error_image("Stream error").tobytes() + b'\r\n')

def create_error_image(message):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ok, buffer = cv2.imencode('.jpg', img)
    return buffer

# ---------------- HELPERS ---------------- #
def list_classes():
    global cached_classes
    if not cached_classes:
        try:
            cached_classes = sorted([f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))])
        except Exception as e:
            logger.error(f"Error listing classes: {str(e)}")
            cached_classes = []
    return cached_classes

def reload_encodings():
    global encodings_dict, name_class_map, cached_classes, cached_students
    encodings_dict, name_class_map = load_known_faces()
    cached_classes = []
    cached_students = {}

def clean_attendance_files(admin_name=ADMIN_NAME):
    try:
        for fname in os.listdir("."):
            if fname.startswith("Attendance_") and fname.endswith(".xlsx"):
                df = pd.read_excel(fname)
                if admin_name in df['Name'].values:
                    df = df[df['Name'] != admin_name]
                    df.to_excel(fname, index=False)
                    logger.info(f"Removed admin from attendance file: {fname}")
    except Exception as e:
        logger.error(f"Error cleaning attendance files: {str(e)}")

# ---------------- AUTH HELPERS ---------------- #
def login_required(roles=["admin"]):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if "username" not in session or session.get("role") not in roles:
                return redirect(url_for("admin_login"))
            return fn(*args, **kwargs)
        return wrapped
    return decorator

# ---------------- ROUTES: AUTH ---------------- #
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    msg = ""
    if request.method == 'POST':
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = USERS.get(username)
        if user and user["password"] == password and user["role"] == "admin":
            session["username"] = username
            session["role"] = user["role"]
            return redirect(url_for("admin_dashboard"))
        else:
            msg = "Invalid username or password."
    return render_template_string(ADMIN_LOGIN_HTML, msg=msg)

@app.route('/logout')
@login_required()
def logout():
    session.clear()
    return redirect(url_for('index'))

# ---------------- ROUTES: HOMEPAGE ---------------- #
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

# ---------------- ROUTES: SHARED ---------------- #
@app.route('/video_feed')
def video_feed():
    return Response(video_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/students/<cls>')
def students(cls):
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        fname = f"Attendance_{cls}_{today}.xlsx"
        students = list_students_for_class(cls)
        status = {}
        if os.path.exists(fname):
            df = pd.read_excel(fname)
            status = dict(zip(df['Name'].astype(str), df['Status'].astype(str)))
        return jsonify({"students": students, "status": status})
    except Exception as e:
        logger.error(f"Error fetching students: {str(e)}")
        return jsonify({"ok": False, "msg": "Failed to fetch students"}), 500

# ---------------- ROUTES: ADMIN ---------------- #
@app.route('/admin')
@login_required()
def admin_dashboard():
    classes = list_classes()
    return render_template_string(ADMIN_HTML, classes=classes, current_class=current_class, running=running,
                                 camera_error=camera_error)

@app.route('/start', methods=['POST'])
@login_required()
def start():
    global running, current_class, attendance_file
    try:
        cls = request.form.get('class_name', '').strip()
        if not cls:
            return jsonify({"ok": False, "msg": "Class name required"}), 400
        os.makedirs(os.path.join(BASE_PATH, cls), exist_ok=True)
        reload_encodings()
        current_class = cls
        students = list_students_for_class(current_class)
        attendance_file = initialize_attendance_file(current_class, students)
        running = True
        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        logger.error(f"Error starting attendance: {str(e)}")
        return jsonify({"ok": False, "msg": "Failed to start attendance"}), 500

@app.route('/stop', methods=['POST'])
@login_required()
def stop():
    global running
    running = False
    return redirect(url_for('admin_dashboard'))

@app.route('/create_class', methods=['POST'])
@login_required()
def create_class():
    try:
        cls = request.form.get('new_class', '').strip()
        if cls:
            os.makedirs(os.path.join(BASE_PATH, cls), exist_ok=True)
            logger.info(f"Created class: {cls}")
            cached_classes.append(cls)
            cached_classes.sort()
        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        logger.error(f"Error creating class: {str(e)}")
        return jsonify({"ok": False, "msg": "Failed to create class"}), 500

@app.route('/reload', methods=['POST'])
@login_required()
def reload_route():
    reload_encodings()
    return redirect(url_for('admin_dashboard'))

@app.route('/register')
@login_required()
def register_page():
    classes = list_classes()
    return render_template_string(REGISTER_HTML, classes=classes, max_images=MAX_IMAGES_PER_STUDENT,
                                 camera_error=camera_error)

@app.route('/capture', methods=['POST'])
@login_required()
def capture():
    try:
        cls = request.form.get('class_name', '').strip()
        user = request.form.get('user_name', '').strip()
        if user.lower() == ADMIN_NAME.lower():
            return jsonify({"ok": False, "msg": "Cannot register admin as a student"}), 400
        if not cls or not user:
            return jsonify({"ok": False, "msg": "Missing class_name or user_name"}), 400
        save_dir = os.path.join(BASE_PATH, cls)
        os.makedirs(save_dir, exist_ok=True)
        existing = [f for f in os.listdir(save_dir) if f.startswith(user + "_") and f.lower().endswith(".jpg")]
        if len(existing) >= MAX_IMAGES_PER_STUDENT:
            return jsonify({"ok": False, "msg": f"Maximum {MAX_IMAGES_PER_STUDENT} images reached for {user}"}), 400
        next_idx = len(existing) + 1
        filename = os.path.join(save_dir, f"{user}_{next_idx}.jpg")
        with last_frame_lock:
            if last_frame_bgr is None:
                return jsonify({"ok": False, "msg": "No camera frame available"}), 400
            ok = cv2.imwrite(filename, last_frame_bgr)
        if not ok:
            return jsonify({"ok": False, "msg": "Failed to save image"}), 500
        logger.info(f"Saved image: {filename}")
        return jsonify({"ok": True, "msg": f"Saved {os.path.basename(filename)}", "count": next_idx})
    except Exception as e:
        logger.error(f"Error capturing image: {str(e)}")
        return jsonify({"ok": False, "msg": "Internal server error"}), 500

# ---------------- ROUTES: USER ---------------- #
@app.route('/user')
def user():
    classes = list_classes()
    return render_template_string(USER_HTML, classes=classes, current_class=current_class, running=running,
                                 camera_error=camera_error)

@app.route('/user/start', methods=['POST'])
def user_start():
    global running, current_class, attendance_file
    try:
        cls = request.form.get('class_name', '').strip()
        if not cls:
            return jsonify({"ok": False, "msg": "Class name required"}), 400
        os.makedirs(os.path.join(BASE_PATH, cls), exist_ok=True)
        reload_encodings()
        current_class = cls
        students = list_students_for_class(current_class)
        attendance_file = initialize_attendance_file(current_class, students)
        running = True
        return redirect(url_for('user'))
    except Exception as e:
        logger.error(f"Error starting attendance: {str(e)}")
        return jsonify({"ok": False, "msg": "Failed to start attendance"}), 500

@app.route('/user/stop', methods=['POST'])
def user_stop():
    global running
    running = False
    return redirect(url_for('user'))

# ---------------- HTML TEMPLATES ---------------- #
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Attendance System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] },
                    colors: {
                        primary: {'50':'#f0f9ff','100':'#e0f2fe','200':'#bae6fd','300':'#7dd3fc','400':'#38bdf8','500':'#0ea5e9','600':'#0284c7','700':'#0369a1','800':'#075985','900':'#0c4a6e'},
                        accent: {'500':'#10b981','600':'#059669'},
                    }
                }
            }
        }
    </script>
    <style>
        body {
            position: relative;
            background-image: url('/static/bg.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
        }
        body::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3); /* Semi-transparent overlay for readability */
            z-index: -1;
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center bg-gray-100 text-gray-900">
    <div class="bg-white p-8 rounded-2xl shadow-2xl max-w-md w-full text-center">
        <div class="mb-6 flex justify-center">
            <svg class="w-12 h-12 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.121 17.804A8.947 8.947 0 0112 15a8.947 8.947 0 016.879 2.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
        </div>
        <h1 class="text-3xl font-bold mb-6">Face Attendance</h1>
        <a href="/user" class="block w-full bg-primary-600 text-white py-3 rounded-lg hover:bg-primary-700 transition mb-4">View Attendance</a>
        <a href="/admin_login" class="block w-full bg-accent-500 text-white py-3 rounded-lg hover:bg-accent-600 transition">Admin Login</a>
    </div>
</body>
</html>
"""

ADMIN_LOGIN_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Login - Face Attendance</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: { sans: ['Inter', 'sans-serif'] },
          colors: {
            primary: {'50':'#f0f9ff','100':'#e0f2fe','200':'#bae6fd','300':'#7dd3fc','400':'#38bdf8','500':'#0ea5e9','600':'#0284c7','700':'#0369a1','800':'#075985','900':'#0c4a6e'},
            accent: {'500':'#10b981','600':'#059669'},
          }
        }
      }
    }
  </script>
</head>
<body class="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 relative">

  <!-- Background image -->
  <div class="absolute inset-0 bg-cover bg-center" 
       style="background-image: url('/static/bg.jpg');">
  </div>

  <!-- Overlay to make form readable -->
  <div class="absolute inset-0 bg-black/40 dark:bg-black/60"></div>

  <!-- Login card -->
  <div class="relative z-10 bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-2xl max-w-md w-full">
      <div class="flex justify-center mb-6">
          <svg class="w-12 h-12 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 11c0 1.104-.896 2-2 2s-2-.896-2-2 2-4 2-4 2 .896 2 2zM12 11c0 1.104.896 2 2 2s2-.896 2-2-2-4-2-4-2 .896-2 2zM8 11h8m-8 0H5m11 0h3m-7-7v14" />
          </svg>
      </div>
      <h1 class="text-3xl font-bold text-center mb-6">Admin Login</h1>
      {% if msg %}
          <div class="mb-6 bg-red-50 dark:bg-red-900/50 text-red-600 dark:text-red-400 p-4 rounded-lg flex items-center">
              <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {{ msg }}
          </div>
      {% endif %}
      <form method="post" class="space-y-5">
          <div>
              <label for="username" class="block text-sm font-medium mb-2">Username</label>
              <input id="username" name="username" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition" required>
          </div>
          <div>
              <label for="password" class="block text-sm font-medium mb-2">Password</label>
              <input id="password" type="password" name="password" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition" required>
          </div>
          <button type="submit" class="w-full bg-primary-600 text-white py-3 rounded-lg hover:bg-primary-700 focus:ring-4 focus:ring-primary-300 dark:focus:ring-primary-800 transition">Sign In</button>
      </form>
      <a href="/" class="block mt-4 text-sm text-primary-600 dark:text-primary-400 text-center hover:underline">Back to Home</a>
  </div>

  <script>
    if (localStorage.getItem('theme') === 'dark') document.documentElement.classList.add('dark');
  </script>
</body>
</html>

"""

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Face Attendance</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] },
                    colors: {
                        primary: {'50':'#f0f9ff','100':'#e0f2fe','200':'#bae6fd','300':'#7dd3fc','400':'#38bdf8','500':'#0ea5e9','600':'#0284c7','700':'#0369a1','800':'#075985','900':'#0c4a6e'},
                        accent: {'500':'#10b981','600':'#059669'},
                    }
                }
            }
        }
    </script>
    <style>
        .status-running { @apply bg-accent-500/20 text-accent-600 dark:bg-accent-600/20 dark:text-accent-400; }
        .status-stopped { @apply bg-red-500/20 text-red-600 dark:bg-red-600/20 dark:text-red-400; }
        .snapshot-present { @apply text-accent-600 dark:text-accent-400 bg-accent-50 dark:bg-accent-900/50; }
        .snapshot-absent { @apply text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50; }
    </style>
</head>
<body class="bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <header class="bg-white dark:bg-gray-800 shadow-lg">
        <div class="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
            <h1 class="text-2xl font-bold flex items-center">
                <svg class="w-8 h-8 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                Admin Dashboard
            </h1>
            <div class="flex items-center space-x-4">
                <a href="/" class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Home</a>
                <a href="/logout" class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Logout</a>
                <button id="theme-toggle" class="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition">
                    <svg class="w-5 h-5 text-gray-800 dark:text-gray-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                    </svg>
                </button>
            </div>
        </div>
    </header>
    <main class="max-w-7xl mx-auto px-4 py-6">
        {% if camera_error %}
            <div class="mb-6 bg-red-50 dark:bg-red-900/50 text-red-600 dark:text-red-400 p-4 rounded-lg flex items-center">
                <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                    <p class="font-bold">Camera Error</p>
                    <p>{{ camera_error }}</p>
                </div>
            </div>
        {% endif %}
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
                    </svg>
                    Manage Classes
                </h2>
                <form method="post" action="/start" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium mb-2">Select Class</label>
                        <select name="class_name" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500">
                            {% for c in classes %}
                                <option value="{{c}}" {% if c==current_class %}selected{% endif %}>{{c}}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <button type="submit" class="w-full bg-primary-600 text-white py-3 rounded-lg hover:bg-primary-700 transition">Start Attendance</button>
                </form>
                <form method="post" action="/create_class" class="mt-4 space-y-4">
                    <input name="new_class" placeholder="New class name" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500">
                    <button type="submit" class="w-full bg-gray-600 text-white py-3 rounded-lg hover:bg-gray-700 transition">Create Class</button>
                </form>
                <div class="mt-6 flex items-center space-x-3">
                    <span class="text-sm font-medium">Status:</span>
                    <span class="{% if running %}status-running{% else %}status-stopped{% endif %} px-4 py-1 rounded-full text-sm font-semibold">
                        {% if running %}Running{% else %}Stopped{% endif %}
                    </span>
                    {% if current_class %}
                        <span class="bg-primary-50 dark:bg-primary-900/50 text-primary-600 dark:text-primary-400 px-4 py-1 rounded-full text-sm font-semibold">{{ current_class }}</span>
                    {% endif %}
                </div>
                {% if running %}
                    <form method="post" action="/stop" class="mt-4">
                        <button type="submit" class="w-full bg-red-600 text-white py-3 rounded-lg hover:bg-red-700 transition">Stop Attendance</button>
                    </form>
                {% endif %}
            </div>
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    Today's Attendance
                </h2>
                <div class="flex space-x-4 mb-4">
                    <select id="snapClass" class="flex-1 p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500">
                        {% for c in classes %}
                            <option value="{{c}}" {% if c==current_class %}selected{% endif %}>{{c}}</option>
                        {% endfor %}
                    </select>
                    <button onclick="loadSnapshot()" class="bg-primary-600 text-white px-6 py-3 rounded-lg hover:bg-primary-700 transition">Refresh</button>
                </div>
                <ul id="snapshot" class="mt-4 space-y-3 max-h-96 overflow-y-auto"></ul>
            </div>
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    Register Student
                </h2>
                <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">Capture photos for face recognition.</p>
                <a href="/register" class="block w-full bg-accent-500 text-white py-3 rounded-lg hover:bg-accent-600 transition">Register New Student</a>
            </div>
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10h3.5M15 14h3.5m-7-7v14m-4-14v14" />
                    </svg>
                    Live Camera Feed
                </h2>
                <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">Real-time face recognition for attendance.</p>
                <div class="relative">
                    <img id="cam" src="/video_feed" class="w-full rounded-xl border border-gray-200 dark:border-gray-600"/>
                    <div class="absolute top-2 right-2 bg-white dark:bg-gray-800 px-3 py-1 rounded-full text-sm font-medium">
                        {% if running %}Active{% else %}Idle{% endif %}
                    </div>
                </div>
            </div>
        </div>
    </main>
    <script>
        const html = document.documentElement;
        document.getElementById('theme-toggle').addEventListener('click', () => {
            html.classList.toggle('dark');
            localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
        });
        if (localStorage.getItem('theme') === 'dark') html.classList.add('dark');
        async function loadSnapshot() {
            const sel = document.getElementById('snapClass');
            const res = await fetch('/students/' + encodeURIComponent(sel.value));
            const data = await res.json();
            const ul = document.getElementById('snapshot');
            ul.innerHTML = '';
            const status = data.status || {};
            data.students.forEach(n => {
                const li = document.createElement('li');
                const s = status[n] || 'U';
                li.textContent = `${n} — ${s === 'A' ? 'Present' : 'Absent'}`;
                li.className = `py-2 px-4 rounded-lg ${s === 'A' ? 'snapshot-present' : 'snapshot-absent'}`;
                ul.appendChild(li);
            });
        }
        window.onload = loadSnapshot;
    </script>
</body>
</html>
"""

REGISTER_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register Student - Face Attendance</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] },
                    colors: {
                        primary: {'50':'#f0f9ff','100':'#e0f2fe','200':'#bae6fd','300':'#7dd3fc','400':'#38bdf8','500':'#0ea5e9','600':'#0284c7','700':'#0369a1','800':'#075985','900':'#0c4a6e'},
                        accent: {'500':'#10b981','600':'#059669'},
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <header class="bg-white dark:bg-gray-800 shadow-lg">
        <div class="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
            <h1 class="text-2xl font-bold flex items-center">
                <svg class="w-8 h-8 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Register Student
            </h1>
            <div class="flex items-center space-x-4">
                <a href="/admin" class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Back</a>
                <a href="/logout" class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Logout</a>
                <button id="theme-toggle" class="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition">
                    <svg class="w-5 h-5 text-gray-800 dark:text-gray-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                    </svg>
                </button>
            </div>
        </div>
    </header>
    <main class="max-w-7xl mx-auto px-4 py-6">
        {% if camera_error %}
            <div class="mb-6 bg-red-50 dark:bg-red-900/50 text-red-600 dark:text-red-400 p-4 rounded-lg flex items-center">
                <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                    <p class="font-bold">Camera Error</p>
                    <p>{{ camera_error }}</p>
                </div>
            </div>
        {% endif %}
        <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
                <h3 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10h3.5M15 14h3.5m-7-7v14m-4-14v14" />
                    </svg>
                    Camera Preview
                </h3>
                <img src="/video_feed" class="w-full rounded-xl border border-gray-200 dark:border-gray-600"/>
                <p class="text-sm text-gray-500 dark:text-gray-400 mt-3">Ensure face is centered with good lighting.</p>
            </div>
            <div>
                <h3 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    Capture Photos
                </h3>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium mb-2">Select Class</label>
                        <select id="classSel" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500">
                            {% for c in classes %}
                                <option value="{{c}}">{{c}}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-2">Student Name</label>
                        <input id="userName" placeholder="e.g. John Doe" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500"/>
                    </div>
                    <button onclick="capture()" class="w-full bg-primary-600 text-white py-3 rounded-lg hover:bg-primary-700 transition">Capture Photo</button>
                    <div class="flex items-center justify-between">
                        <p>Captured: <span id="count" class="font-bold text-primary-600 dark:text-primary-400">0</span> / {{ max_images }}</p>
                        <div class="w-1/2 bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                            <div id="progress" class="bg-primary-600 h-2.5 rounded-full" style="width:0%"></div>
                        </div>
                    </div>
                    <button onclick="reloadEncodings()" class="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition">Reload Encodings</button>
                    <p id="msg" class="text-sm text-gray-500 dark:text-gray-400 text-center"></p>
                </div>
            </div>
        </div>
    </main>
    <script>
        const html = document.documentElement;
        document.getElementById('theme-toggle').addEventListener('click', () => {
            html.classList.toggle('dark');
            localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
        });
        if (localStorage.getItem('theme') === 'dark') html.classList.add('dark');
        let captured = 0;
        const maxCaptures = {{ max_images }};
        const progress = document.getElementById('progress');
        async function capture() {
            const cls = document.getElementById('classSel').value.trim();
            const user = document.getElementById('userName').value.trim();
            const msg = document.getElementById('msg');
            if (!cls || !user) {
                msg.textContent = 'Please select class and enter student name.';
                msg.className = 'text-red-600 dark:text-red-400';
                return;
            }
            const form = new FormData();
            form.append('class_name', cls);
            form.append('user_name', user);
            const res = await fetch('/capture', { method: 'POST', body: form });
            const data = await res.json();
            if (data.ok) {
                captured = data.count;
                document.getElementById('count').textContent = captured;
                progress.style.width = `${(captured / maxCaptures) * 100}%`;
                msg.textContent = data.msg;
                msg.className = 'text-accent-600 dark:text-accent-400';
            } else {
                msg.textContent = data.msg || 'Capture failed';
                msg.className = 'text-red-600 dark:text-red-400';
            }
        }
        async function reloadEncodings() {
            await fetch('/reload', { method: 'POST' });
            document.getElementById('msg').textContent = 'Encodings reloaded successfully.';
            document.getElementById('msg').className = 'text-blue-600 dark:text-blue-400';
        }
    </script>
</body>
</html>
"""

USER_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attendance Dashboard - Face Attendance</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] },
                    colors: {
                        primary: {'50':'#f0f9ff','100':'#e0f2fe','200':'#bae6fd','300':'#7dd3fc','400':'#38bdf8','500':'#0ea5e9','600':'#0284c7','700':'#0369a1','800':'#075985','900':'#0c4a6e'},
                        accent: {'500':'#10b981','600':'#059669'},
                    }
                }
            }
        }
    </script>
    <style>
        .status-running { @apply bg-accent-500/20 text-accent-600 dark:bg-accent-600/20 dark:text-accent-400; }
        .status-stopped { @apply bg-red-500/20 text-red-600 dark:bg-red-600/20 dark:text-red-400; }
        .snapshot-present { @apply text-accent-600 dark:text-accent-400 bg-accent-50 dark:bg-accent-900/50; }
        .snapshot-absent { @apply text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50; }
    </style>
</head>
<body class="bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <header class="bg-white dark:bg-gray-800 shadow-lg">
        <div class="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
            <h1 class="text-2xl font-bold flex items-center">
                <svg class="w-8 h-8 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.121 17.804A8.947 8.947 0 0112 15a8.947 8.947 0 016.879 2.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Attendance Dashboard
            </h1>
            <div class="flex items-center space-x-4">
                <a href="/" class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Home</a>
                <button id="theme-toggle" class="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition">
                    <svg class="w-5 h-5 text-gray-800 dark:text-gray-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                    </svg>
                </button>
            </div>
        </div>
    </header>
    <main class="max-w-7xl mx-auto px-4 py-6">
        {% if camera_error %}
            <div class="mb-6 bg-red-50 dark:bg-red-900/50 text-red-600 dark:text-red-400 p-4 rounded-lg flex items-center">
                <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                    <p class="font-bold">Camera Error</p>
                    <p>{{ camera_error }}</p>
                </div>
            </div>
        {% endif %}
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
                    </svg>
                    Manage Classes
                </h2>
                <form method="post" action="/user/start" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium mb-2">Select Class</label>
                        <select name="class_name" class="w-full p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500">
                            {% for c in classes %}
                                <option value="{{c}}" {% if c==current_class %}selected{% endif %}>{{c}}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <button type="submit" class="w-full bg-primary-600 text-white py-3 rounded-lg hover:bg-primary-700 transition">Start Attendance</button>
                </form>
                <div class="mt-6 flex items-center space-x-3">
                    <span class="text-sm font-medium">Status:</span>
                    <span class="{% if running %}status-running{% else %}status-stopped{% endif %} px-4 py-1 rounded-full text-sm font-semibold">
                        {% if running %}Running{% else %}Stopped{% endif %}
                    </span>
                    {% if current_class %}
                        <span class="bg-primary-50 dark:bg-primary-900/50 text-primary-600 dark:text-primary-400 px-4 py-1 rounded-full text-sm font-semibold">{{ current_class }}</span>
                    {% endif %}
                </div>
                {% if running %}
                    <form method="post" action="/user/stop" class="mt-4">
                        <button type="submit" class="w-full bg-red-600 text-white py-3 rounded-lg hover:bg-red-700 transition">Stop Attendance</button>
                    </form>
                {% endif %}
            </div>
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10h3.5M15 14h3.5m-7-7v14m-4-14v14" />
                    </svg>
                    Live Camera Feed
                </h2>
                <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">Real-time face recognition for attendance.</p>
                <div class="relative">
                    <img id="cam" src="/video_feed" class="w-full rounded-xl border border-gray-200 dark:border-gray-600"/>
                    <div class="absolute top-2 right-2 bg-white dark:bg-gray-800 px-3 py-1 rounded-full text-sm font-medium">
                        {% if running %}Active{% else %}Idle{% endif %}
                    </div>
                </div>
            </div>
            <div class="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-lg">
                <h2 class="text-xl font-semibold mb-4 flex items-center">
                    <svg class="w-6 h-6 mr-2 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    Today's Attendance
                </h2>
                <div class="flex space-x-4 mb-4">
                    <select id="snapClass" class="flex-1 p-3 rounded-lg bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-primary-500">
                        {% for c in classes %}
                            <option value="{{c}}" {% if c==current_class %}selected{% endif %}>{{c}}</option>
                        {% endfor %}
                    </select>
                    <button onclick="loadSnapshot()" class="bg-primary-600 text-white px-6 py-3 rounded-lg hover:bg-primary-700 transition">Refresh</button>
                </div>
                <ul id="snapshot" class="mt-4 space-y-3 max-h-96 overflow-y-auto"></ul>
            </div>
        </div>
    </main>
    <script>
        const html = document.documentElement;
        document.getElementById('theme-toggle').addEventListener('click', () => {
            html.classList.toggle('dark');
            localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
        });
        if (localStorage.getItem('theme') === 'dark') html.classList.add('dark');
        async function loadSnapshot() {
            const sel = document.getElementById('snapClass');
            const res = await fetch('/students/' + encodeURIComponent(sel.value));
            const data = await res.json();
            const ul = document.getElementById('snapshot');
            ul.innerHTML = '';
            const status = data.status || {};
            data.students.forEach(n => {
                const li = document.createElement('li');
                const s = status[n] || 'U';
                li.textContent = `${n} — ${s === 'A' ? 'Present' : 'Absent'}`;
                li.className = `py-2 px-4 rounded-lg ${s === 'A' ? 'snapshot-present' : 'snapshot-absent'}`;
                ul.appendChild(li);
            });
        }
        window.onload = loadSnapshot;
    </script>
</body>
</html>
"""

# ---------------- MAIN ---------------- #
if __name__ == '__main__':
    clean_attendance_files()
    reload_encodings()
    logger.info(f"Encodings loaded: {sum(len(v) for v in encodings_dict.values())} images")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)