from pathlib import Path
import argparse
import cv2

'''
Takes in an MP4 video and extracts a specific amount of frames from it.
The amount of frames extracted is dependent on the interval selected (smaller = more).
The extracted frames are stored in a separate directory.

Params:
    video (Path): The path to the video that will be processed
    out_dir (Path): The path to the directory where the processed frames will be stored
    interval (float): How often we want frames to be extracted (in seconds), 0.5 by default

Returns:
    None
'''
def extract_frames(video: Path, out_dir: Path, interval: float):
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

    # How much frames we want to skip by 
    # (Ex. 30 fps, every 0.5s = Skip every 15 frames)
    frame_interval = max(1, int(round(fps * interval)))
    out_dir.mkdir(parents=True, exist_ok=True)
    curr_frame = 0
    curr_saved = 0

    while True:
        success, frame = vid.read()

        # No more frames to process
        if not success:
            break

        if curr_frame % frame_interval == 0:
            filename = out_dir / f"{video.stem}_{curr_saved:03d}.jpg"

            # Save frame as photo (95 quality vs 100 to save some space)
            cv2.imwrite(
                str(filename),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            curr_saved += 1
        curr_frame += 1
    vid.release()
    print(f"[DONE] {video.name}: {curr_saved} frames")
    return


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from every MP4 in a directory"
    )
    parser.add_argument(
        "vid_dir",
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
    vid_dir = args.vid_dir.resolve()

    # Can not find directory from given path
    if not vid_dir.exists():
        raise FileNotFoundError(vid_dir)

    frames_root = vid_dir.parent / "frames"
    frames_root.mkdir(exist_ok=True)
    videos = sorted(vid_dir.glob("*.mp4"))

    # Can't find any MP4 videos in directory
    if not videos:
        print("No MP4 files found")
        return
    
    print(f"Found {len(videos)} video(s)\n")

    # Go through each video and extract its frames
    for video in videos:
        out_dir = frames_root / video.stem

        # Delete anything that is already in the directory
        # If overwrite == False, skip existing directory instead
        if out_dir.exists():
            if not args.overwrite:
                print(f"[SKIP] {video.name}")
                continue

            # Delete pre-existing files in directory
            for file in out_dir.glob("*"):
                file.unlink()

        extract_frames(video, out_dir, args.interval)
    print("\nFinished")
    return

if __name__ == "__main__":
    main()