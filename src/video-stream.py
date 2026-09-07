from pathlib import Path
from sign_tracker import SignTracker
import argparse
import cv2

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
    tracker = SignTracker(
        model="runs/train/r/weights/best.pt",
        conf=0.95
    )

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

    delay = int(1000 / fps) # In ms, may not keep up with video feed if processing takes too long

    while True:
        success, frame = vid.read()

        # No more frames to process
        if not success:
            break

        # Pass frame into sign detector/tracker
        signs = tracker.track_signs(frame)

        for sign in signs:
            print(
                f"Sign ID: {sign['id']} | "
                f"Confidence: {sign['conf']:.3f}"
            )

        # next steps:
        # choose specific frame to crop (maybe based on confidence?)
        # crop specific tracked sign with live_crop_signs
        # Run OCR on crop
        # Show changed floor plan/location

        display = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Video feed was too large for screen, reduce 50%
        cv2.imshow("Video Feed", display)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
    return

def main():
    parser = argparse.ArgumentParser(
        description="Display MP4 video as continuous stream"
    )
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
    return

if __name__ == "__main__":
    main()