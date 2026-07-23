from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
import cv2

# Takes OBB points (top  left, right, bottom right, left)
def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]      # top left
    rect[2] = points[np.argmax(s)]      # bottom right

    rect[1] = points[np.argmin(diff)]   # top right
    rect[3] = points[np.argmax(diff)]   # bottom left

    return rect

# Make warped sign into straight rectangle
def perspective_crop(image, points):
    rect = order_points(points)

    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))
    if width <= 0 or height <= 0:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # TODO probably make it padded so it isn't cut too severely

    return warped

def crop_signs(model, img_dir: Path, output_dir: Path, confidence: float):
    img = cv2.imread(str(img_dir))

    # Can't open photo
    if img is None:
        print(f"[ERROR] {img_dir.name} not found or unable to read")
        return

    results = model(
        img,
        conf=confidence
    )

    num_signs = 0

    for result in results:
        if result.obb is None:
            continue

        boxes = result.obb.xyxyxyxy.cpu().numpy()

        for box in boxes:
            crop = perspective_crop(img, box)

            if crop is None:
                continue

            output_name = (f"{img_dir.stem}_sign_{num_signs}.jpg")
            output_path = output_dir / output_name

            cv2.imwrite(str(output_path), crop)

            num_signs += 1

    print(
        f"[DONE] {img_dir.name}: "
        f"{num_signs} signs cropped"
    )
    
    return

def main():
    parser = argparse.ArgumentParser(
        description="Crop detected signs in a frame"
    )
    parser.add_argument(
        "img_dir",
        type=Path,
        help="Directory containing photos",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="runs/train/r/weights/best.pt",
        help="YOLO OBB model path"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing crops folder"
    )
    args = parser.parse_args()
    photos_dir = args.img_dir.resolve()

    # Can not find directory from given path
    if not photos_dir.exists():
        raise FileNotFoundError(photos_dir)

    photos_root = photos_dir.parent / "crops"
    photos_root.mkdir(exist_ok=True)

    if args.overwrite:
        for file in photos_root.glob("*"):
            file.unlink()

    photos = sorted(photos_dir.glob("*.jpeg"))

    # Can't find any photos in directory
    if not photos:
        print("No photos found")
        return
    
    print(f"Found {len(photos)} photo(s)\n")

    model = YOLO(args.model)

    # Go through each photo and crop the signs
    for photo in photos:
        crop_signs(model, photo, photos_root, args.confidence)

    print("\nFinished")
    return

if __name__ == "__main__":
    main()