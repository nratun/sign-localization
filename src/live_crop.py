#!/usr/bin/env python

"""
live_crop.py: Crops photos to contain only building signs detected by a YOLO model.

This file is different from crop.py in that it is intended to work in tandem with live footage.
Uses a more simple cropping method that doesn't involve warping the photo.
"""
import numpy as np


def simple_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    '''
    Takes in an image and 4 (x,y) points representing a bounding box.
    Returns a cropped rectangular image of a building sign.
    The output is intended to be passed through an OCR to extract text off the sign.

    Unlike perspective_crop(), this function does not transform the image.
    It simply takes the rectangular region surrounding the detected sign.

    Params:
        image (np.ndarray): The original image to be cropped
        points (np.ndarray): The four corner points of the bounding box

    Returns:
        crop (np.ndarray | None): Rectangular region containing the sign (None if dimensions are invalid)
    '''
    # (x1, y1) make top left corner of box
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))

    # (x2, y2) make bottom right corner of box
    x2 = min(image.shape[1], int(np.ceil(points[:, 0].max())))
    y2 = min(image.shape[0], int(np.ceil(points[:, 1].max())))

    # Bad coords
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop