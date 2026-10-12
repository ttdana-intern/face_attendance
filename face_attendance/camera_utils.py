import cv2
import numpy as np
import logging
import time
from threading import Lock
from collections import deque
from datetime import datetime
from config import CAMERA_INDEX, CAMERA_RETRY_ATTEMPTS, FRAME_SKIP, THRESHOLD, LABEL_PERSIST_FRAMES, CAPTURE_RESOLUTION, DETECTION_LOG_SIZE
import face_recognition

# Set up logging (set level to DEBUG for more details)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
cap = None
frame_count = 0
last_frame_bgr = None
last_frame_lock = Lock()
camera_error = None
last_detections = deque(maxlen=LABEL_PERSIST_FRAMES)
detection_log = deque(maxlen=DETECTION_LOG_SIZE)

def initialize_camera():
    global cap, camera_error
    for attempt in range(CAMERA_RETRY_ATTEMPTS):
        try:
            if cap is not None and cap.isOpened():
                cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_RESOLUTION[1])
            if cap.isOpened():
                logger.info("Camera initialized successfully")
                camera_error = None
                return True
            else:
                logger.warning(f"Camera initialization attempt {attempt + 1} failed")
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
        time.sleep(1)
    camera_error = "Failed to open webcam after multiple attempts"
    logger.error(camera_error)
    return False

def release_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        logger.info("Camera released")
    cap = None

def create_error_image(message):
    img = np.zeros((CAPTURE_RESOLUTION[1], CAPTURE_RESOLUTION[0], 3), dtype=np.uint8)
    cv2.putText(img, message, (50, CAPTURE_RESOLUTION[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ok, buffer = cv2.imencode('.jpg', img)
    return buffer

def video_generator():
    global frame_count, last_frame_bgr, camera_error, last_detections, detection_log
    from app import running, current_class, attendance_file, encodings_dict, name_class_map, mark_attendance
    logger.debug(f"Starting video_generator. Running: {running}, Current class: {current_class}, Encodings count: {len(encodings_dict) if encodings_dict else 0}")
    if not initialize_camera():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + create_error_image(camera_error).tobytes() + b'\r\n')
        return

    while True:
        try:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to capture frame")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + create_error_image(
                    "Failed to capture frame").tobytes() + b'\r\n')
                initialize_camera()
                continue

            with last_frame_lock:
                last_frame_bgr = frame.copy()

            processed_frame = frame.copy()
            current_detections = []

            if running and current_class is not None:
                cv2.putText(processed_frame, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                logger.debug(f"Processing frame {frame_count}. Current class: {current_class}")

                frame_count += 1
                if frame_count % FRAME_SKIP == 0:
                    start_time = time.time()
                    small_img = cv2.resize(processed_frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

                    faces_cur_frame = face_recognition.face_locations(rgb_small)
                    encodes_cur_frame = face_recognition.face_encodings(rgb_small, faces_cur_frame)

                    logger.debug(f"Faces detected: {len(faces_cur_frame)}, Encodings generated: {len(encodes_cur_frame)}")

                    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                        name = "UNKNOWN"
                        min_distance = 1.0

                        for known_name, enc_list in encodings_dict.items():
                            if name_class_map.get(known_name) != current_class:
                                continue
                            distances = face_recognition.face_distance(enc_list, encode_face)
                            best_distance = np.min(distances) if distances.size > 0 else 1.0
                            logger.debug(f"Comparing to {known_name}: best distance {best_distance:.4f}")
                            if best_distance < min_distance and best_distance < THRESHOLD:
                                min_distance = best_distance
                                name = known_name

                        current_detections.append((name, face_loc))
                        logger.info(f"Face recognized: {name} (distance: {min_distance:.4f})")

                        if name != "UNKNOWN" and attendance_file:
                            mark_attendance(name, current_class, attendance_file)
                            current_time = datetime.now().strftime("%H:%M:%S")
                            detection_msg = f"Who is this? {name} detected at {current_time}"
                            detection_log.append({
                                'message': detection_msg,
                                'timestamp': current_time
                            })
                            logger.info(detection_msg)

                    logger.debug(f"Frame processing took {time.time() - start_time:.2f} seconds")

                last_detections.append(current_detections)
                detections_to_use = current_detections if current_detections else next(
                    (d for d in reversed(last_detections) if d), [])

                for name, face_loc in detections_to_use:
                    y1, x2, y2, x1 = [v * 4 for v in face_loc]  # Scale back to original frame size
                    color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(processed_frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    display_text = f"{name} at {current_time}" if name != "UNKNOWN" else "Detecting..."
                    cv2.putText(processed_frame, display_text, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    logger.debug(f"Drawing text: {display_text} at ({x1}, {y2 - 6})")

            else:
                cv2.putText(processed_frame, "Attendance Stopped", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ok, buffer = cv2.imencode('.jpg', processed_frame)
            if not ok:
                logger.error("Failed to encode frame")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Error in video generator: {str(e)}")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + create_error_image("Stream error").tobytes() + b'\r\n')
            initialize_camera()