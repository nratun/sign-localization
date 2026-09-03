#!/usr/bin/env python

"""
live_crop_signs.py: Crops photos to contain only building signs detected by a YOLO model.

This file is different from crop_signs.py in that it is intended to work in tandem with live footage.
The detect_signs function is called in video_stream.py for each frame to detect signs in real-time.
"""
import cv2
import numpy as np
from ultralytics import YOLO

def order_points(points: np.ndarray) -> np.ndarray:
    '''
    Takes in 4 (x,y) points representing a bounding box and places them in order.
    The order of the points is determined by their values when added/subtracted.

    Params:
        points (np.ndarray): An unordered array of 4 (x,y) points representing a bounding box

    Returns:
        rect (np.ndarray): An ordered array of 4 (x,y) points representing a bounding box
            Order: top left, top right, bottom right, bottom left
    '''
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

def perspective_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    '''
    Takes in an image and 4 (x,y) points representing a rectangular bounding box.
    The output image provides a cropped rectangular image of a building sign.
    The output is intended to be passed through an OCR to extract text off the sign.

    Params:
        image (np.ndarray): The original image to be transformed

    Returns:
        warped (np.ndarray): The modified image that has been transformed
    '''
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

def detect_signs(model: YOLO, frame: np.ndarray, conf: float) -> list[dict]:
    '''
    Takes in a photo & runs a YOLO model to detect building signs.
    The photo is then cropped to only contain the building sign.
    If the model is not confident enough in its inference, the photo is ignored.

    Params:
        model (YOLO): The YOLO model that detects building signs
        frame (np.ndarray): The frame to be transformed
        conf (float): The minimum confidence required for the building sign to be considered

    Returns:
        signs (list[dict]): The detected signs & their attributes
    '''
    results = model(frame, conf=conf)
    signs = []

    # Note: Don't need this for loop right now because only processing one photo at a time
    for result in results:
        if result.obb is None:
            continue

        # xyxyxyxy = OBB polygon format with 4-corner points
        # need to ensure process is specifically on CPU before numpy
        boxes = result.obb.xyxyxyxy.cpu().numpy() # Convets tensor to numpy array
        confs = result.obb.conf.cpu().numpy() # Used later down the line

        # Iterate through both lists in parallel
        for box, conf in zip(boxes, confs):
            crop = perspective_crop(frame, box)

            if crop is None:
                continue

            signs.append({
                "box": box,
                "crop": crop,
                "conf": float(conf)
            })

    return signs