import time
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO

@dataclass
class DetectorConfig:
    model_path: str = "models/yolov8m.pt"
    person_id: int = 0
    conf_thres: float = 0.25


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    draw_img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
    return draw_img


class Detector:
    def __init__(self, config: DetectorConfig = DetectorConfig()):
        model_path = config.model_path
        if not model_path.endswith(".pt"):
            model_path = model_path.split(".")[0] + ".pt"
        self.model = YOLO(model_path)
        self.person_id = config.person_id
        self.conf_thres = config.conf_thres

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        results = self.model(img, conf=self.conf_thres)
        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

        # Filter out only person
        person_detections = detections[detections[:, -1] == self.person_id]
        boxes = person_detections[:, :-2].astype(int)

        print(f"Detection inference took: {time.perf_counter() - start:.4f} seconds")
        return boxes


if __name__ == "__main__":
    import cv2

    detector = Detector()
    img = cv2.imread("../ComfyUI_00074_.png")
    boxes = detector.detect(img)
    draw_img = draw_boxes(img, boxes)
    cv2.imshow("img", draw_img)
    cv2.waitKey(0)
