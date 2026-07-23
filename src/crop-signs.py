from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
import cv2

'''
Takes in 4 (x,y) points representing a bounding box and places them in order.
The order of the points is determined by their values when added/subtracted.

Params:
    points (np.ndarray): An unordered array of 4 (x,y) points representing a bounding box

Returns:
    rect (np.ndarray): An ordered array of 4 (x,y) points representing a bounding box
        Order: top left, top right, bottom right, bottom left
'''
def order_points(points: np.ndarray) -> np.ndarray:
    # An array of 4 (x, y) points representing a rectangle
    rect = np.zeros((4, 2), dtype="float32")

    # axis = 1 to +/- across each row
    s = points.sum(axis=1)              # s = [x1 + y1, x2 + y2, ...]
    diff = np.diff(points, axis=1)      # diff = [y1 - x1, y2 - x2, ...]

    rect[0] = points[np.argmin(s)]      # top left      (smallest sum of x+y)
    rect[2] = points[np.argmax(s)]      # bottom right  (largest sum of x+y)

    rect[1] = points[np.argmin(diff)]   # top right     (larger x -> y-x = larger negative # = min value)
    rect[3] = points[np.argmax(diff)]   # bottom left   (smaller x -> y-x = smaller negative # = max value)

    return rect

'''
Warps the perspective of an image while cropping it to fit only the desired object
Takes in an image and 4 (x,y) points representing a rectangular bounding box.
The output image provides a straight, cropped rectangular image of a building sign.
The output image is intended to be passed through an OCR to extract text off the sign.

Params:
    image (np.ndarray): The original image to be transformed

Returns:
    warped (np.ndarray): The modified image that has been transformed
'''
def perspective_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    # Norm finds Euclidean distance between points
    width_a = np.linalg.norm(br - bl)  # Bottom edge length
    width_b = np.linalg.norm(tr - tl)  # Top edge length

    height_a = np.linalg.norm(tr - br) # Right edge height
    height_b = np.linalg.norm(tl - bl) # Left edge height

    # To make a straight rectangle, find longest width/height of the sides
    # Thus,  if the OG  points make a trapezoidal shape, the result will be a rectangle
    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))
    if width <= 0 or height <= 0:
        return None

    # Make straight rectangle (Coordinates are 0-based)
    # Ex. w=100, h=50 -> tl=(0,0), tr=(99,0), br=(99,49), bl=(0,49)
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    # Transform box to ideal rectangle
    matrix = cv2.getPerspectiveTransform(rect, destination)         # Transformation matrix
    warped = cv2.warpPerspective(image, matrix, (width, height))    # Apply transformation matrix
    return warped

'''
Takes in a photo & runs a model to detect regions of interest (building signs).
The photo is then cropped and transformed to provide a straight rectangular view of the ROI.
If the model is not confident enough in its inference, the photo is ignored

Params:
    model (YOLO): The YOLO model that detects ROIs
    img_path (Path): The path to the image that will be processed
    output_dir (Path): The path to the directory where the processed images will be stored
    confidence (float): The minimum confidence required for the ROI to be considered

Returns:
    None
'''
def crop_signs(model: YOLO, img_path: Path, output_dir: Path, confidence: float):
    img = cv2.imread(str(img_path))

    # Can't open photo
    if img is None:
        print(f"[ERROR] {img_path.name} not found or unable to read")
        return

    results = model(
        img,
        conf=confidence
    )

    num_signs = 0

    # TODO Don't need this for loop right now because only processing one photo at a time
    for result in results:
        if result.obb is None:
            continue

        # xyxyxyxy = OBB polygon format with 4-corner points
        # need to ensure process is specifically on CPU before numpy
        boxes = result.obb.xyxyxyxy.cpu().numpy() # Convets tensor to numpy array

        for box in boxes:
            crop = perspective_crop(img, box)

            if crop is None:
                continue

            output_name = (f"{img_path.stem}_sign_{num_signs}.jpg")
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), crop)

            num_signs += 1
    print(
        f"[DONE] {img_path.name}: "
        f"{num_signs} signs cropped"
    )
    return

def main():
    parser = argparse.ArgumentParser(
        description="Crop detected signs in a frame"
    )
    parser.add_argument(
        "img_path",
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
    photos_dir = args.img_path.resolve()

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