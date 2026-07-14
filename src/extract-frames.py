from pathlib import Path
import argparse
import cv2

'''
Takes in an MP4 video and extracts a specific amount of frames from it.
The amount of frames extracted is dependent on the interval selected (smaller = more).
The extracted frames are stored in a separate directory.

Params:
    video_path (Path): The path to the video that will be processed
    output_dir (Path): The path to the directory where the processed frames will be stored
    interval (float): How often we want frames to be extracted (in seconds), 0.5 by default

Returns:
    None
'''
def extract_frames(video_path: Path, output_dir: Path, interval: float):
    vid = cv2.VideoCapture(str(video_path))

    # Can't open video
    if not vid.isOpened():
        print(f"[ERROR] Could not open {video_path.name}")
        return

    fps = vid.get(cv2.CAP_PROP_FPS)

    # Invalid frame rate
    if fps <= 0:
        print(f"[ERROR] Invalid FPS for {video_path.name}")
        vid.release()
        return

    # How much frames we want to skip by 
    # (Ex. 30 fps, every 0.5s = Skip every 15 frames)
    frame_interval = max(1, int(round(fps * interval)))
    output_dir.mkdir(parents=True, exist_ok=True)
    curr_frame = 0
    curr_saved = 0

    while True:
        success, frame = vid.read()

        # No more frames to process
        if not success:
            break

        if curr_frame % frame_interval == 0:
            filename = output_dir / f"{video_path.stem}_{curr_saved:03d}.jpg"

            # Save frame as photo (95 quality vs 100 to save some space)
            cv2.imwrite(
                str(filename),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            curr_saved += 1
        curr_frame += 1
    vid.release()
    print(f"[DONE] {video_path.name}: {curr_saved} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from every MP4 in a directory"
    )
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing MP4 videos",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between extracted frames (default = 0.5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame folder",
    )
    args = parser.parse_args()
    video_dir = args.video_dir.resolve()

    # Can not find directory from given path
    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    frames_root = video_dir.parent / "frames"
    frames_root.mkdir(exist_ok=True)
    videos = sorted(video_dir.glob("*.mp4"))

    # Can't find any MP4 videos in directory
    if not videos:
        print("No MP4 files found")
        return
    
    print(f"Found {len(videos)} video(s)\n")

    # Go through each video and extract its frames
    for video in videos:
        output_dir = frames_root / video.stem

        # Make separate directory for each processed video
        # If overwrite == False, skip existing directory instead
        if output_dir.exists():
            if not args.overwrite:
                print(f"[SKIP] {video.name}")
                continue

            # Delete pre-existing files in directory
            for file in output_dir.glob("*"):
                file.unlink()

        extract_frames(video, output_dir, args.interval)
    print("\nFinished")


if __name__ == "__main__":
    main()