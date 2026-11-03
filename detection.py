"""Detect potholes in video and camera streams with YOLO."""

from datetime import datetime
import math
from pathlib import Path
import time

import cv2

from capture_location import DeviceLocation, photo_filename

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pothole.pt"
# Use a video path, camera number (0), or stream URL.
VIDEO_PATH = str(ROOT / "pothole1.mp4")
CONFIDENCE = 0.35
IMAGE_SIZE = 800
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SAVE_PHOTOS = True
PHOTO_CONFIDENCE = 0.5
PHOTO_INTERVAL = 1.0  # Seconds between photos; this does not track individual potholes.
PHOTO_DIR = ROOT / "pothole_photos"
WINDOW_NAME = "Pothole Detection"


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model missing: {MODEL_PATH}. Install the README checkpoint or run python trainer.py first.")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    if model.task not in ("detect", "segment") or {k: str(v).casefold() for k, v in model.names.items()} != {0: "pothole"}:
        raise ValueError("Expected a single-class pothole detection or segmentation model.")
    return model


def detect_potholes(img, model):
    result = model.predict(img, conf=CONFIDENCE, imgsz=IMAGE_SIZE, classes=[0], quantize="fp16", verbose=False)[0]
    annotated = result.plot()
    h, w = annotated.shape[:2]
    scale = min(1, PROCESS_WIDTH / w, PROCESS_HEIGHT / h)
    if scale < 1:
        annotated = cv2.resize(annotated, (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)))
    boxes = result.boxes
    found = bool(boxes) and any(float(c) > PHOTO_CONFIDENCE for c in getattr(boxes, "conf", [1.0]))
    return annotated, found


def run_video(model):
    live = isinstance(VIDEO_PATH, int) or str(VIDEO_PATH).startswith(("rtsp://", "http://", "https://"))
    cap = cv2.VideoCapture(VIDEO_PATH)
    location = DeviceLocation()
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")
        if live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30
        frame_number = 0
        last_photo_time = -float("inf")
        if SAVE_PHOTOS:
            location.start()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            t0 = time.perf_counter()
            if not live and frame_number > 0:
                while frame_number / fps < time.perf_counter() - start_time and cap.grab():
                    frame_number += 1
            elif frame_number == 0:
                start_time = t0
            ok, img = cap.read()
            if not ok:
                break
            captured_at = datetime.now().astimezone()
            location_label = location.label()
            output, found = detect_potholes(img, model)
            frame_time = time.monotonic() if live else frame_number / fps
            frame_number += 1
            if SAVE_PHOTOS and found and frame_time - last_photo_time >= PHOTO_INTERVAL:
                PHOTO_DIR.mkdir(parents=True, exist_ok=True)
                photo_path = PHOTO_DIR / photo_filename(location_label, captured_at)
                if cv2.imwrite(str(photo_path), output):
                    last_photo_time = frame_time
                    print(f"Saved pothole photo: {photo_path}")
            cv2.imshow(WINDOW_NAME, output)
            delay = 1 if live else max(1, round(1000 / fps - (time.perf_counter() - t0) * 1000))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        location.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video(load_model())
