import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class SignInfo:
    missed: int = 0
    best_quality: float = -1.0
    best_conf: float = 0.0
    best_area: int = 0
    best_frame: np.ndarray | None = None
    points: np.ndarray | None = None

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
        self.signs = dict[int, SignInfo] = {}

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
        results = self.model.track(frame, conf=self.conf, persist=True, tracker="bytetrack.yaml")
        signs = []
        curr_ids = set()

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

                self.signs[id].missed = 0
                self.set_best_info(id, frame, points, conf)

                signs.append({
                    "id": id,
                    "points": points,
                    "conf": conf
                })

        for id in list(self.signs):
            if id not in curr_ids:
                self.signs[id].missed += 1

                if self.signs[id].missed > self.max_missed:
                    del self.signs[id]

        return signs

    # Calcs img quality to be used to calc best frame
    def img_quality(self, crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(sharpness)

    # Gets area of the sign to be later used to calc best frame
    def sign_area(self, points: np.ndarray) -> int:
        width = points[:, 0].max() - points[:, 0].min()
        height = points[:, 1].max() - points[:, 1].min()
        return int(width * height)

    # Setter function to set best frame for sign
    def set_best_info(self, id: int, frame: np.ndarray, points: np.ndarray, conf: float):
        # Taking idea of perspective warp and trying to only box the building sign
        # (x1, y1)----(x2, y1)
        #    |           |
        # (x1, y2)----(x2, y2)

        # (x1, y1) make top left corner of box
        x1 = max(0, int(np.floor(points[:, 0].min())))
        y1 = max(0, int(np.floor(points[:, 1].min())))

        # (x2, y2) make bottom right corner of region of interest
        x2 = min(frame.shape[1], int(np.ceil(points[:, 0].max())))
        y2 = min(frame.shape[0],int(np.ceil(points[:, 1].max())))

        # Bad coords
        if x2 <= x1 or y2 <= y1:
            return

        box = frame[y1:y2, x1:x2]
        if box.size == 0:
            return

        quality = self.img_quality(box)
        area = self.sign_area(points)
        sign = self.signs[id]

        # Use measure of sharpness to determine if it's best captured frame of a sign
        # TODO incorporate area & conf, potentially even more complex method to determine best frame
        if quality > sign.best_quality:
            sign.best_quality = quality
            sign.best_conf = conf
            sign.best_area = area

            # Store entire frame because crops need og frame coordinates
            sign.best_frame = frame.copy()
            sign.points = points.copy()

        return

    # Basically a getter function for other files like live_crop to get the necessary info
    def get_best_info(self, id: int) -> SignInfo | None:
        return self.signs.get(id)