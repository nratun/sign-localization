from pathlib import Path
import argparse
import cv2

'''
Takes in an MP4 video and displays it as a continuous video stream.
Each frame is passed onto the cropping phase, where a YOLO model will check for room signs.
Detected signs are used to infer the location during the video and updated accordingly.

Params:
    video (Path): The path to the video that will be processed

Returns:
    None
'''
def input_video(video: Path):
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

    delay = int(1000 / fps) # In ms

    while True:
        success, frame = vid.read()

        # No more frames to process
        if not success:
            break

        # Pass frame into crop
        # Show changed floor plan here?

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Video feed was too large for screen, reduce 50%
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
    return

def main():
    parser = argparse.ArgumentParser(
        description="DIsplay MP4 video as continuous stream"
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

    input_video(video)
    print("\nFinished")
    return

if __name__ == "__main__":
    main()