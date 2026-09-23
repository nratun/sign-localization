#!/usr/bin/env python

"""
ocr.py: FIX FIX.

FIX FIX.
"""
import multiprocessing as mp
import queue
import time

from paddleocr import PaddleOCR

from room_lookup import find_room

# Issues arise when using PaddleOCR in the same environment as YOLO
# Use different environments for the two or use cpu rather than gpu

def ocr_text(ocr, image) -> list[dict]:
    # Run OCR
    result = ocr.predict(image)

    if not result:
        return []

    result = result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])

    detections = []

    for text, score in zip(texts, scores):
        detections.append({
            "text": text,
            "conf": float(score)
        })

    return detections

def _ocr_worker_loop(jobs, results, stop_event, ready_event, startup_errors, rooms):
    print("[OCR] Loading Worker...")

    try:
        ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False, # Need this otherwise issues with YOLO?
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,

            text_detection_model_name="PP-OCRv5_mobile_det", # Around same performance as v6_small
            text_recognition_model_name="PP-OCRv6_small_rec",
        )

    except Exception as error:
        startup_errors.put(str(error))
        ready_event.set()
        print(f"[OCR ERROR] Failed to initialize PaddleOCR: {error}")
        return

    # PaddleOCR fully initialized
    ready_event.set()
    print("[OCR] Worker ready")

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
    def __init__(
        self,
        rooms: set[str],
        max_queue_size: int = 1
    ):
        ctx = mp.get_context("spawn")
        self.rooms = rooms
        self.jobs = ctx.Queue(maxsize=max_queue_size)
        self.results = ctx.Queue()
        self.stop_event = ctx.Event()
        self.ready_event = ctx.Event()
        self.startup_errors = ctx.Queue()

        # Start OCR in separate process
        self.process = ctx.Process(
            target=_ocr_worker_loop,
            args=(
                self.jobs,
                self.results,
                self.stop_event,
                self.ready_event,
                self.startup_errors,
                self.rooms
            ),
            daemon=True
        )

        self.process.start()

    def wait_till_ready(self, timeout: float | None = None) -> bool:
        ready = self.ready_event.wait(timeout)

        if not ready:
            return False

        if not self.startup_errors.empty():
            error = self.startup_errors.get_nowait()
            print(f"[OCR ERROR] {error}")
            return False

        return self.process.is_alive()

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
        self.process.join(timeout=2.0) # Give worker chance to finish

        # stop if PaddleOCR still running
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()