#!/usr/bin/env python

"""
ocr.py: Handles all OCR logic using PaddleOCR.

Multiprocessing is used to prevent delays in the video feed while running OCR.
In the project, issues occur when attempting to use YOLO and Paddle together on GPU.
For this reason, the CPU is leveraged (the GPU is used only when dealing with the two separately).
"""
import multiprocessing as mp
import queue
import time

import numpy as np
from paddleocr import PaddleOCR

from room_lookup import find_room


def ocr_text(ocr: PaddleOCR, image: np.ndarray) -> list[dict]:
    '''
    Takes in an OCR model and cropped photo of a building sign and runs OCR on it.
    The extracted texts and confidence levels are returned as a list.

    Params:
        ocr (PaddleOCR): The PaddleOCR model that will be used to detect and recognize text
        image (np.ndarray): Cropped building sign photo

    Returns:
        detections (list[dict]): List containing extracted texts & confidence scores 
    '''
    # Run OCR
    result = ocr.predict(image)

    if not result:
        return []

    result = result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])

    detections = []

    # Iterate through texts and confidences in parallel
    for text, score in zip(texts, scores):
        detections.append({
            "text": text,
            "conf": float(score)
        })

    return detections

def _ocr_worker_loop(jobs, results, stop_event, ready_event, startup_errors, rooms: dict[str, int]):
    '''
    Runs OCR worker process:
    1. Initializes PaddleOCR then waits for sign crop submissions from job queue
    2. Processes each crop with OCR, then identifies valid room from OCR
    3. Places result into results queue

    Params:
        jobs: Queue containing sign IDs & image crops
        results: Queue used to return OCR results
        stop_event: Event indicating that the worker should shut down
        ready_event: Event indicating that PaddleOCR is ready
        startup_errors: Queue used to communicate initialization errors
        rooms (dict[str, int]): The valid rooms to compare the extracted text against
    '''
    # ----------------------------Initialize OCR----------------------------
    try:
        ocr = PaddleOCR(
            device="cpu",
            enable_mkldnn=False, # If this isn't used, DNN runtime error occurs
            
            # Unnecessary for our use case, only increases processing time
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,

            text_detection_model_name="PP-OCRv5_mobile_det", # Around same performance as v6_small
            text_recognition_model_name="PP-OCRv6_small_rec",
        )

    except Exception as error:
        startup_errors.put(str(error))
        ready_event.set()
        print(f"[ERROR] Failed to initialize PaddleOCR: {error}")
        return

    # PaddleOCR fully initialized, ready_event toggled
    ready_event.set()
    # ------------------------------Job Loop------------------------------
    # While stop hasn't been established
    while not stop_event.is_set():
        try:
            sign_id, crop = jobs.get(timeout=0.1) # Wait up to 0.1s for job
        except queue.Empty: # Nothing in queue -> Loop and check again
            continue

        try:
            start_time = time.perf_counter()
            detections = ocr_text(ocr, crop)
            room = find_room(detections, rooms)
            ocr_time = time.perf_counter() - start_time

            # Put answer into results queue
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
        rooms: dict[str, int],
        max_queue_size: int = 1 # Only one queued OCR job at a time
    ):
        '''
        Initialize the OCR worker process with specified maximum queue size.
        The queue is intentionally small to prevent stale processes during video stream

        Params:
            rooms (dict[str, int]): The valid rooms to compare the extracted text against
            max_queue_size (int): Maximum number of OCR jobs that can wait in queue
        '''
        ctx = mp.get_context("spawn") # Spawn in a worker
        self.rooms = rooms

        # Communication channels
        self.jobs = ctx.Queue(maxsize=max_queue_size)
        self.results = ctx.Queue()

        # Signals for when to start/stop
        self.stop_event = ctx.Event()
        self.ready_event = ctx.Event()
        self.startup_errors = ctx.Queue() # Any issues during startup

        # When process starts, runs ocr_worker_loop
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

        self.process.start() # Actually starts the process

    def wait_till_ready(self, timeout: float | None = None) -> bool:
        '''
        Waits for the OCR worker to finish initializing

        Params:
            timeout (float | None): Maximum number of seconds to wait (Forever if None)

        Returns:
            bool: Whether OCR has successfully initialized or not
        '''
        ready = self.ready_event.wait(timeout)

        # Couldn't ready in time
        if not ready:
            return False

        try:
            error = self.startup_errors.get_nowait()
        except queue.Empty:
            error = None
            
        # Error with OCR startup
        if error is not None:
            print(f"[ERROR] OCR: {error}")
            return False

        return self.process.is_alive()

    def submit(self, sign_id: int, crop: np.ndarray) -> bool:
        '''
        Submits cropped sign image for OCR processing (rejected if queue is full)

        Params:
            sign_id (int): ID assigned by sign_tracker
            crop (np.ndarray): The cropped building sign image

        Returns:
            bool: Whether the OCR request was added to the queue or not
        '''
        if crop is None or crop.size == 0:
            return False

        try:
            # Push job, if queue full don't
            # Copy crop because OG frame keeps changing as video continues
            self.jobs.put_nowait((sign_id, crop.copy()))
            return True
        
        except queue.Full: # Queue backed up, can't push job 
            return False

    def poll(self) -> list[dict]:
        '''
        Retrieves all currently available OCR results (doesn't wait for OCR to finish).
        If no OCR requests have completed since the last poll, the returned results list is empty

        Returns:
            results (list[dict]): The resulting OCR extractions (Empty if no requests finished since last poll)
        '''
        # Checks whether OCR has finished
        results = []

        while True:
            try:
                results.append(self.results.get_nowait()) # Get result if available, otherwise don't wait

            except queue.Empty: # Nothing in queue
                break
        return results

    def stop(self):
        '''
        Shuts down OCR worker process (Given 2 seconds grace period before ending process)
        '''
        self.stop_event.set()
        self.process.join(timeout=2.0) # Give worker chance to finish

        # Stop if PaddleOCR still running
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()