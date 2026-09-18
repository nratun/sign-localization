import multiprocessing as mp
import queue
import time

import cv2
from paddleocr import PaddleOCR

from room_lookup import find_room

# Issues arise when using PaddleOCR in the same environment as YOLO
# Use different environments for the two or use cpu rather than gpu

def preprocess_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Denoise
    denoised = cv2.medianBlur(gray, ksize=3)  
    
    # Binarize: Black text on white background
    # TODO sign is split into two regions (black text/light bg, light text/black bg)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

    return thresh

def ocr_text(ocr, image) -> list[dict]:
    # Preprocess image (not using for now)
    # processed = preprocess_ocr(image)
    # Run OCR
    result = ocr.predict(image)

    if not result:
        return []

    result = result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    coords = result.get("rec_polys", [])
    detections = []

    # Example format:
    # { "text": "397", "conf": 0.97, "box": np.array([[12, 8], [30, 8], [30, 20], [12, 20]]) }
    for text, score, points in zip(texts, scores, coords):
        detections.append({
            "text": text,
            "conf": float(score),
            "points": points
        })
    return detections

def _ocr_worker_loop(jobs, results, stop_event, rooms):
    print("[OCR] Worker started")

    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        enable_mkldnn=False, # Need this otherwise issues with YOLO?
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    )

    while not stop_event.is_set():
        try:
            sign_id, crop = jobs.get(timeout=0.1)

        except queue.Empty:
            continue

        try:
            start_time = time.perf_counter()
            detections = ocr_text(ocr, crop)
            room = find_room(detections, rooms)
            ocr_time = time.perf_counter() - start_time

            results.put({
                "id": sign_id,
                "detections": detections,
                "room": room,
                "time": ocr_time
            })

        except Exception as error:
            results.put({
                "id": sign_id,
                "detections": [],
                "room": None,
                "error": str(error)
            })
    print("[OCR] Worker stopped")

class OCRWorker:
    def __init__(self, rooms: set[str], max_queue_size: int = 1):
        self.rooms = rooms
        self.jobs = mp.Queue(maxsize=max_queue_size)
        self.results = mp.Queue()
        self.stop_event = mp.Event()

        # Start OCR in separate process
        ctx = mp.get_context("spawn")

        self.process = ctx.Process(
            target=_ocr_worker_loop,
            args=(
                self.jobs,
                self.results,
                self.stop_event,
                self.rooms
            ),
            daemon=True
        )

        self.process.start()

    def submit(self, sign_id: int, crop) -> bool:
        if crop is None or crop.size == 0:
            return False

        try:
            # Copy crop because og frame continues changing
            self.jobs.put_nowait((sign_id, crop.copy()))
            return True

        except queue.Full:
            return False

    def poll(self) -> list[dict]:
        results = []

        while True:
            try:
                results.append(self.results.get_nowait())

            except queue.Empty:
                break
        return results

    def stop(self):
        self.stop_event.set()

        # Give worker chance to finish
        self.process.join(timeout=2.0)

        # stop if PaddleOCR still running
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()