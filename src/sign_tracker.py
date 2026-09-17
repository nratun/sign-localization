from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class SignInfo:
    missed: int = 0
    frames_seen: int = 0

class SignTracker:
    def __init__(
        self,
        model: str = "runs/train/r/weights/best.pt",
        conf: float = 0.95,
        max_missed: int = 30
    ):
        '''
        Initialize the YOLO sign tracker with specified confidence.

        Params:
            model (str): Path to the trained YOLO OBB model
            conf (float): The minimum confidence required for the building sign to be considered
            max_missed (int): The number of frames a sign disappears before tracking is discarded
        '''
        self.model = YOLO(model)
        self.conf = conf
        self.max_missed = max_missed
        self.signs: dict[int, SignInfo] = {}

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
        results = self.model.track(frame, conf=self.conf, persist=True, tracker="bytetrack.yaml", verbose=False)
        signs = []
        curr_ids = set()

        for result in results:
            if result.obb is None:
                continue

            track_ids = result.obb.id
            if track_ids is None:
                continue

            # xyxyxyxy = OBB polygon format with 4-corner points:
            # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            # need to ensure process is specifically on CPU before numpy
            boxes = result.obb.xyxyxyxy.cpu().numpy() # Sign corner points
            confs = result.obb.conf.cpu().numpy() # Confidence
            ids = track_ids.cpu().numpy().astype(int) # Assigned ID

            # Iterate through lists in parallel
            for points, conf, id in zip(boxes, confs, ids):
                id = int(id)
                conf = float(conf)

                curr_ids.add(id)

                if id not in self.signs:
                    self.signs[id] = SignInfo()

                sign = self.signs[id]
                sign.missed = 0
                sign.frames_seen += 1

                signs.append({
                    "id": id,
                    "points": points,
                    "conf": conf,
                    "frames_seen": sign.frames_seen
                })

        # Signs not detected in current frame
        for id in list(self.signs):
            if id not in curr_ids:
                self.signs[id].missed += 1

                if self.signs[id].missed > self.max_missed:
                    del self.signs[id]

        return signs

    # Basically a getter function for other files like live_crop to get the necessary info
    def get_sign_info(self, id: int) -> SignInfo | None:
        return self.signs.get(id)