import os

# Configuration
BASE_PATH = 'known_faces'
THRESHOLD = 0.45
FRAME_SKIP = 3  # Increased for smoother video; reduce to 1 for faster detection
MAX_IMAGES_PER_STUDENT = 3
CAMERA_INDEX = 0
CAMERA_RETRY_ATTEMPTS = 3
LABEL_PERSIST_FRAMES = 5
ENCODINGS_CACHE = os.path.join(BASE_PATH, "encodings_cache.pkl")
CAPTURE_RESOLUTION = (640, 480)
DETECTION_LOG_SIZE = 10

# Demo admin user
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
}