import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import face_recognition
import pickle
import logging
from config import BASE_PATH, ENCODINGS_CACHE, CAPTURE_RESOLUTION

# Set up logging
logger = logging.getLogger(__name__)

def load_known_faces(base_path=BASE_PATH, class_name=None):
    encodings = {}
    mapping = {}
    try:
        folders = [class_name] if class_name else os.listdir(base_path)
        for class_folder in folders:
            class_path = os.path.join(base_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                # Resize image to reduce encoding time
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_enc = face_recognition.face_encodings(img_rgb)
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

def save_encodings_cache(encodings, mapping):
    try:
        with open(ENCODINGS_CACHE, 'wb') as f:
            pickle.dump({'encodings': encodings, 'mapping': mapping}, f)
        logger.info(f"Saved encodings to cache: {ENCODINGS_CACHE}")
    except Exception as e:
        logger.error(f"Error saving encodings cache: {str(e)}")

def load_encodings_cache():
    try:
        if os.path.exists(ENCODINGS_CACHE):
            with open(ENCODINGS_CACHE, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded encodings from cache: {ENCODINGS_CACHE}")
            return data['encodings'], data['mapping']
        return {}, {}
    except Exception as e:
        logger.error(f"Error loading encodings cache: {str(e)}")
        return {}, {}

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

def list_classes():
    try:
        return sorted([f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))])
    except Exception as e:
        logger.error(f"Error listing classes: {str(e)}")
        return []

def list_students_for_class(cls):
    from app import name_class_map
    return sorted([n for n, c in name_class_map.items() if c == cls])

def reload_encodings(class_name=None):
    from app import encodings_dict, name_class_map
    if class_name:
        # Update encodings for a specific class
        new_encodings, new_mapping = load_known_faces(class_name=class_name)
        encodings_dict.update(new_encodings)
        name_class_map.update(new_mapping)
        # Update cache with merged encodings
        save_encodings_cache(encodings_dict, name_class_map)
    else:
        # Full reload from cache or images
        encodings_dict.clear()
        name_class_map.clear()
        encodings, mapping = load_encodings_cache()
        if encodings and mapping:
            encodings_dict.update(encodings)
            name_class_map.update(mapping)
            logger.info("Using cached encodings")
        else:
            encodings, mapping = load_known_faces()
            encodings_dict.update(encodings)
            name_class_map.update(mapping)
            save_encodings_cache(encodings, mapping)