import argparse
import time
from pathlib import Path

import cv2

from live_crop_signs import simple_crop
from map import Map, build_graph, get_rooms
from ocr import OCRWorker
from room_lookup import build_rooms, load_yaml
from sign_tracker import SignTracker

# Number frames to wait before retrying OCR after unsuccessful attempt
OCR_RETRY = 5
PLAYBACK_SPEED = 1

'''
Takes in an MP4 video and displays it as a continuous video stream.
Each frame is passed to the sign tracker, where a YOLO model detect & track room signs.

Tracked signs are later used to infer the location during the video and updated accordingly.

Params:
    video (Path): The path to the video that will be processed

Returns:
    None
'''
def stream_video(video: Path):
    # -------------------------Initialize everything-------------------------
    tracker = SignTracker(
        model="runs/train/r/weights/best.pt",
        conf=0.95
    )

    floor_data = load_yaml()
    rooms = build_rooms(floor_data)
    vertices, edges = build_graph(floor_data)
    landmarks = get_rooms(vertices)

    floor_map = Map(floor_data, vertices, edges, landmarks)

    known_signs = set() # Signs that already have valid rooms found
    ocr_jobs = set() # Signs that have OCR job being processed    
    ocr_counters = {} # Number frames since last ocr for each sign
    # ----------------------------INtake video----------------------------
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

    ocr_worker = OCRWorker(rooms)

    # Models take awhile to load, which affects sign detections in the beginning, add buffer
    print("[INFO] Loading OCR model...")
    ocr_worker = OCRWorker(rooms)

    # Wait for worker to  finish initialization
    if not ocr_worker.wait_until_ready(timeout=30):
        print("[ERROR] OCR worker failed to initialize")
        ocr_worker.stop()
        vid.release()
        return

    print("[INFO] OCR ready")
    print("[INFO] Starting video")
    start_time = time.perf_counter()
    # ----------------------------Process video----------------------------
    while True:
        success, frame = vid.read()

        # No more frames to process
        if not success:
            break
        # ----------------------------OCR Asynch----------------------------
        for result in ocr_worker.poll():
            id = result["id"]
            ocr_jobs.discard(id) # No more OCR job running for sign
            detections = result["detections"]
            room = result["room"]

            print(f"Sign ID {id} | OCR: {detections}")

            # OCR failed
            if result.get("error") is not None:
                print(f"[OCR ERROR] Sign ID {id}: " f"{result['error']}")
                continue

            # Valid room
            if room is not None:
                print(f"FOUND ID {id} | ROOM: {room} | TIME: {result['time']:.3f}s")
                moved = floor_map.move_to_room(room)

                if moved:
                    known_signs.add(id)
                    ocr_counters.pop(id, None)

        # ----------------------------Track & Crop----------------------------
        # Pass frame into sign detector/tracker
        signs = tracker.track_signs(frame)

        for sign in signs:
            id = sign["id"]

            # Don't ocr already known signs
            if id in known_signs:
                continue

            # Don't submit OCR while one being processed for the sign
            if id in ocr_jobs:
                continue
            
            # First time seeing sign = OCR immediately
            if id not in ocr_counters:
                ocr_counters[id] = 0
                do_ocr = True
            # Already seen sign before, check if needs ocr attempt again
            else:
                ocr_counters[id] += 1
                do_ocr = ocr_counters[id] >= OCR_RETRY

            if not do_ocr:
                continue

            crop = simple_crop(frame, sign["points"])

            if crop is None: # ndarray check None
                continue

            # Send crop to OCR worker
            submitted = ocr_worker.submit(id, crop)

            if submitted:
                ocr_jobs.add(id)
                ocr_counters[id] = 0

        # Remove OCR state for signs yolo isn't tracking
        for id in list(ocr_counters):
            if tracker.get_sign_info(id) is None:
                ocr_counters.pop(id, None)
                ocr_jobs.discard(id)

        # ----------------------------Update Displays----------------------------
        display = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Video feed was too large for screen, reduce 50%
        cv2.imshow("Video Feed", display)

        floor_map.display()
        frame_number += 1

        # Playback close to og video
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