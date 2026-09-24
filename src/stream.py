#!/usr/bin/env python

"""
stream.py: This file is responsible for video playback and connecting the entire sign mapping system together.

The system is designed to simulate a live camera feed from an autonomous vehicle; ffor demonstration purposes,
it takes a video recording instead. The vehicle uses detected building signs to estimate its location on a floor
plan, acting as a lightweight system to complement more computationally intensive localization & mapping methods.

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

OCR_RETRY = 5 # Number frames to wait before retrying OCR

def stream_video(video: Path):
    '''
    Takes in an MP4 video and displays it as a continuous video stream. For each frame:
    1. Frame is tracked for any possible building signs
    2. Detected building signs are processed with OCR to extract room numbers
    3. Room numbers extracted successfully are used to update the floor plan map with current position

    Params:
        video (Path): The path to the video that will be processed
    '''
    # ----------------------------Initialization----------------------------
    floor_data = load_floor()
    rooms = build_rooms(floor_data)
    vertices, edges = build_graph(floor_data)

    floor_map = Map(floor_data, vertices, edges, rooms)
    tracker = SignTracker(model="models/sign-detector.pt", conf=0.95)

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

    # Wait for OCR to finish initializing before starting video playback
    # Main process paused until ready_event == True (OCR loaded), or 30s pass (failed)
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
        # ----------------------------Pull OCR Results----------------------------
        # Go through obtained results
        for result in ocr_worker.poll():
            id = result["id"]
            ocr_jobs.discard(id) # Don't OCR the sign anymore
            room = result["room"]

            # OCR failed for the sign
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

        # ------------------------------Track & Crop------------------------------
        signs = tracker.track_signs(frame) # Pass frame into sign detector/tracker

        for sign in signs:
            id = sign["id"]

            # Don't OCR already known signs or signs that already have a job submitted
            if id in known_signs or id in ocr_jobs:
                continue
            
            # First time seeing sign -> OCR immediately
            if id not in ocr_counters:
                ocr_counters[id] = 0
                do_ocr = True
            # Already seen sign before -> Check if needs OCR attempt again
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

            # Queue had room, job was submitted/added to queue
            if submitted:
                ocr_jobs.add(id) # Sign has OCR request in progress
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

        # Wait if the feed is ahead
        if remaining > 0:
            # int(x) = 0 if number is too small, use max to ensure value is 1 at minimum
            key = cv2.waitKey(max(1, int(remaining * 1000))) & 0xFF # Keeps last 8 bits of key
        # Feed behind, wait as little as possible
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