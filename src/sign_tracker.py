import cv2
import numpy as np
from ultralytics import YOLO

class SignTracker:
    def __init__(
        self,
        model: str = "runs/train/r/weights/best.pt",
        conf: float = 0.95
    ):
        '''
        Initialize the YOLO sign tracker with specified confidence.

        Params:
            model (str): Path to the trained YOLO OBB model
            conf (float): The minimum confidence required for the building sign to be considered
        '''
        self.model = YOLO(model)
        self.conf = conf

    def track_signs(self, frame: np.ndarray) -> list[dict]:
        '''
        Takes in a photo & runs a YOLO model to detect and track building signs.
        If the model is not confident enough in its inference, the photo is ignored.

        Tracking persists across photos so the same physical sign has the same ID across frames.

        Params:
            frame (np.ndarray): The frame to be processed

        Returns:
            signs (list[dict]): The tracked signs & their attributes
        '''
        results = self.model.track(frame, conf=self.conf, persist=True)
        signs = []

        # Note: Don't need this for loop right now because only processing one photo at a time
        for result in results:
            if result.obb is None:
                continue

            track_ids = result.obb.id
            if track_ids is None:
                continue

            # xyxyxyxy = OBB polygon format with 4-corner points:
            # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            # need to ensure process is specifically on CPU before numpy
            boxes = result.obb.xyxyxyxy.cpu().numpy() # Convets tensor to numpy array
            confs = result.obb.conf.cpu().numpy() # Used later down the line
            ids = track_ids.cpu().numpy().astype(int)

            # Iterate through both lists in parallel
            for points, conf, id in zip(boxes, confs, ids):
                signs.append({
                    "id": int(id),
                    "points": points,
                    "conf": float(conf)
                })

        return signs