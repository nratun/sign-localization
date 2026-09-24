# Indoor Sign Localization

A lightweight indoor localization system that uses building room signs to estimate an autonomous vehicle's location on a floor plan.

The goal of this project is to explore whether ordinary building signs can serve as a practical source of localization for an autonomous vehicle, much like the behavior of a human navigating the inside of a building for the first time. Rather than performaing computationally intensive mapping, the system uses recognizable environmental landmarks to establish where the vehicle is within a building.

The system is designed to simulate a live camera feed from an autonomous vehicle. For demonstration purposes, it processes a prerecorded video instead. Detected room signs are recognized using computer vision and optical character recognition (OCR), then matched to known locations on a floor plan. The movement is visualized on the provided floor plan.

The project is intended to complement more precise localization and mapping methods such as LiDAR or odometry.

```text
PUT DEMO GIF HERE
```

## Table of Contents

- [How It Works](#how-it-works)
    - [Sign Detection & Tracking](#1-sign-detection-and-tracking)
    - [Sign Cropping](#2-sign-cropping)
    - [OCR](#3-ocr)
    - [Room Matching](#4-room-matching)
    - [Graph Movement](#5-graph-movement)
- [How to Use](#how-to-use)
    - [Running the Project](#running-the-project)
    - [Requirements](#requirements)
    - [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Design Considerations](#design-considerations)
    - [Using CPU](#using-cpu)
    - [Asynchronous OCR](#asynchronous-ocr)
    - [Floor-Plan Data](#floor-plan-data)
    - [Sign-Based Localization](#sign-based-localization)
- [Challenges & Limitations](#challenges-and-limitations)
- [Future Work](#future-work)

## How It Works

The system processes each video frame through the following pipeline:

```text
PUT DIAGRAM HERE
```
### 1. Sign Detection and Tracking

A custom YOLO oriented bounding box (OBB) model detects building signs in each frame.

- YOLO's built-in ByteTrack maintains a tracking ID for each detected sign across frames.
    - This allows the system to recognize the same sign across frames without treating every frame as a new detection.

### 2. Sign Cropping

The detected sign's OBB coordinates are converted into a simple rectangular region containing the sign.

- The system uses a lightweight axis-aligned crop to reduce processing, while the crop used during testing is a more complex perspective warp.
    - The simple crop generally produced better room number recognition, however, the warp crop is kept in `crop.py` for reference.

### 3. OCR

PaddleOCR extracts text from the cropped sign.

- OCR runs in a separate multiprocessing worker so that OCR inference does not block the main video processing loop.
    - The worker returns recognized text and confidence values to the main process through a queue.
    - If multiprocessing is not implemented, video playback is paused every time OCR is run.

- OCR is attempted periodically for unresolved signs rather than on every frame.

### 4. Room Matching

OCR output is compared against valid room names defined in `floor-plan.yaml`.

Variations in OCR output are accounted for:

- Complete room numbers such as `316A`
- Three-digit room numbers such as `316`
- Separate room suffix detections such as `316` and `A`

### 5. Graph Movement

The floor plan contains a predefined graph consisting of vertices and connecting edges (established in `floor-plan.yaml`).

- Each recognized room corresponds to a graph vertex.
- After the initial room is recognized, the system uses breadth-first search (BFS) to find a route between the current location and the newly recognized room.
- The robot is then animated to move along the graph using elapsed time and constant movement speed.

## How to Use

### Requirements

- Python 3.12
- NumPy
- OpenCV
- PyYAML
- Ultralytics
- PaddlePaddle
- PaddleOCR
- PyTorch

Exact tested package versions are listed in `requirements.txt`.
- The default configuration uses the CPU for both YOLO and PaddleOCR. Compatibility for GPU acceleration depends on the hardware and environment.

**Note:** PaddleOCR may have compatibility issues with YOLO when using the GPU.
- For parts of the project that only require one or the other, the GPU works without issue.

**Note:** PaddleOCR may have compatibility issues with newer verions of Python.

### Installation

Once the repository has been cloned, download the necessary libraries by running this command from the repository root:

```bash
python -m pip install -r requirements.txt
```

### Running the Project

The main application is located in `src/stream.py`.

The program can be run with the following command from the repository root:

```bash
# Syntax:
python src/stream.py video-path.mp4

# Example:
python src/stream.py dataset/demo/v1.mp4
```

The program opens two windows:

- **Video Feed** — The input video being processed
- **Floor Map** — The floor plan with the robot location

Pressing `q` stops playback.

## Repository Structure

```bash
└── sign-localization/
    ├── yolo/
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   ├── labels/
    │   │   ├── train/
    │   │   └── val/
    │   └── data.yaml
    ├── models/
    │   └── sign-detector.pt
    ├── dataset/
    │   ├── demo/
    │   └── floors/
    │       ├── floor-3.png
    │       ├── floor-4.png
    │       ├── floor-5.png
    │       └── floor-plan.yaml
    ├── src/
    │   ├── testing/
    │   │   ├── crop.py
    │   │   ├── extract-frames.py
    │   │   └── model.py
    │   ├── floor_data.py
    │   ├── live_crop.py
    │   ├── map.py
    │   ├── ocr.py
    │   ├── room_lookup.py
    │   ├── sign_tracker.py
    │   └── stream.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

**Note:**
- The `yolo/` directory shows the dataset structure expected by YOLO to train the model. The training images and labels themselves are not included.
- The `src/testing/` scripts were used during model development and OCR testing, and are not required for normal system execution.

## Design Considerations

### Using CPU

- The provided configuration uses CPU inference for both YOLO and PaddleOCR. During development, using the GPU for either framework in the same environment caused runtime conflicts.

- Separate environments can be used for GPU acceleration for YOLO or PaddleOCR.

### Asynchronous OCR

- OCR can take significantly longer than a single video frame. Running PaddleOCR in a separate process allows the main video loop to continue processing frames while OCR work is performed asynchronously.

- A small OCR queue is intentionally used so that older sign crops do not accumulate during live processing.

### Floor Plan Data

- The provided demo focuses on **Floor 3** of a campus building. Other floors are provided for reference, but are not used.

- `dataset/floors/` contains all floor plan content:
    - `floor-plan.yaml` provides the vertices, edges, room labels, and image information used by the graph system.
    - `floor-3.png`, `floor-4.png`, and `floor-5.png` provide photos of the each floor layout.

### Sign-Based Localization

- The system does not attempt to identify exact camera position or reconstruct the building.
- Instead, recognizable room signs act as landmarks that position the vehicle at a known location on the floor-plan graph.
- This makes the approach lightweight while still providing useful global location information.

## Challenges and Limitations

The current implementation is intentionally lightweight, but has several limitations:

- Localization depends on visible and readable building signs
- The system uses a predefined graph rather than constructing a map
- The YOLO model is limited to detecting signs specific to the particular building it was trained on
- Room recognition can be affected by OCR errors, especially missing or misread room suffixes
- The animated map represents approximate graph movement rather than precise physical robot positions
- The robot's movement on the map is determined by the shortest path rather than the actual path taken
- The application is currently demonstrated using prerecorded video rather than a physical vehicle camera

## Future Work

Possible future improvements include:

- Improving text recognition speeds for faster performance
- Adding logic to prevent unwanted/unexpected movement
- Incorporating camera position or heading estimates
- Integrating the sign-based localization with LiDAR or wheel odometry
- Optimizing the pipeline to work on a portable robotics computer (Ex. NVIDIA Jetson Orin Nano)
