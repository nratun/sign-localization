#!/usr/bin/env python

"""
stream.py: FIX

FIX FIX.

Usage:
    python stream.py <video_path>
Example:
    python stream.py demo/v1.mp4
"""
import argparse
import time
from pathlib import Path

import cv2

from floor_data import build_rooms, load_floor
from live_crop import simple_crop
from map import Map, build_graph
from ocr import OCRWorker
from sign_tracker import SignTracker

# Number frames to wait before retrying OCR after unsuccessful attempt
OCR_RETRY = 5
# TODO fix map size

def stream_video(video: Path):
    '''
    Takes in an MP4 video and displays it as a continuous video stream. For each frame:
    1. Frame is tracked for any possible building signs
    2. Detected building signs are processed with OCR to extract room numbers
    3. Room numbers extracted successfully are used to update the floor plan map with current position

    Params:
        video (Path): The path to the video that will be processed

    Returns:
        None
    '''
    # ----------------------------Initialization----------------------------
    tracker = SignTracker(
        model="runs/train/r/weights/best.pt",
        conf=0.95
    )

    floor_data = load_floor()
    rooms = build_rooms(floor_data)
    vertices, edges = build_graph(floor_data)

    floor_map = Map(floor_data, vertices, edges, rooms)

    known_signs = set() # Signs that already have valid rooms found
    ocr_jobs = set() # Signs that have OCR job being processed    
    ocr_counters = {} # Number frames since last ocr for each sign
    # -----------------------------Intake Video-----------------------------
    vid = cv2.VideoCapture(str(video))

    # Can't open video
    if not vid.isOpened():
        print(f"[ERROR] Could not open {video.name}")
        return

    fps = vid.get(cv2.CAP_PROP_FPS)

    # Invalid frame rate
    if fps <= 0:
        print(f"[ERROR] Invalid FPS for {video.name}")
        vid.release()
        return

    frame_interval = 1.0 / fps
    frame_number = 0
    # -------------------------------Load OCR-------------------------------
    print("[INFO] Loading OCR model...")
    ocr_worker = OCRWorker(rooms)

    # Models take awhile to load, which affects sign detections in the beginning, add buffer
    if not ocr_worker.wait_till_ready(timeout=30):
        print("[ERROR] OCR worker failed to initialize")
        ocr_worker.stop()
        vid.release()
        return

    print("[INFO] OCR ready, starting video")
    start_time = time.perf_counter()
    # ------------------------------Video Loop------------------------------
    while True:
        success, frame = vid.read()

        # No more frames to process
        if not success:
            break
        # ------------------------------OCR Asynch------------------------------
        for result in ocr_worker.poll():
            id = result["id"]
            ocr_jobs.discard(id) # Don't OCR the sign anymore
            room = result["room"]

            # OCR failed
            if result.get("error") is not None:
                print(f"[ERROR] Sign ID {id}: " f"{result['error']}")
                continue

            # Valid room
            if room is not None:
                print(f"[FOUND] ID {id} | ROOM: {room} | TIME: {result['time']:.3f}s")
                moved = floor_map.move(room)

                # Position updated, don't OCR the sign anymore
                if moved:
                    known_signs.add(id)
                    ocr_counters.pop(id, None)

        # -----------------------------Track & Crop-----------------------------
        signs = tracker.track_signs(frame) # Pass frame into sign detector/tracker

        for sign in signs:
            id = sign["id"]

            # Don't OCR already known signs or signs that already have a job submitted
            if id in known_signs or id in ocr_jobs:
                continue
            
            # First time seeing sign = OCR immediately
            if id not in ocr_counters:
                ocr_counters[id] = 0
                do_ocr = True
            # Already seen sign before, check if needs OCR attempt again
            else:
                ocr_counters[id] += 1
                do_ocr = ocr_counters[id] >= OCR_RETRY

            # No need to OCR this sign
            if not do_ocr:
                continue

            crop = simple_crop(frame, sign["points"])

            if crop is None: # ndarray check None
                continue

            # Send crop to OCR worker
            submitted = ocr_worker.submit(id, crop)

            if submitted:
                ocr_jobs.add(id) # Add to job queue
                ocr_counters[id] = 0

        # Remove OCR state for signs YOLO isn't tracking
        for id in list(ocr_counters):
            if tracker.get_sign_info(id) is None:
                ocr_counters.pop(id, None)
                ocr_jobs.discard(id)

        # ----------------------------Update Displays----------------------------
        display = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Video feed too large for screen, reduce 50%
        cv2.imshow("Video Feed", display)

        floor_map.display()
        frame_number += 1

        # Playback close to OG video
        target_time = start_time + frame_number * frame_interval
        remaining = target_time - time.perf_counter()

        if remaining > 0:
            key = cv2.waitKey(max(1, int(remaining * 1000))) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
    ocr_worker.stop()
    return

def main():
    parser = argparse.ArgumentParser(description="Display MP4 video as continuous stream")
    parser.add_argument(
        "video",
        type=Path,
        help="Path to MP4 video",
    )
    args = parser.parse_args()
    video = args.video.resolve()

    # Can not find video from given path
    if not video.exists():
        raise FileNotFoundError(video)

    stream_video(video)
    print("\nFinished")

if __name__ == "__main__":
    main()