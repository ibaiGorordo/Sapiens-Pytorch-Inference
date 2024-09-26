import time
from enum import Enum
from typing import List
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import pose_estimation_preprocessor, TaskType, download_hf_model
from .detector import Detector, DetectorConfig

from .pose_classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)


class SapiensPoseEstimationType(Enum):
    POSE_ESTIMATION_03B = "sapiens_0.3b_goliath_best_goliath_AP_575_torchscript.pt2"
    POSE_ESTIMATION_06B = "sapiens_0.6b_goliath_best_goliath_AP_600_torchscript.pt2"
    POSE_ESTIMATION_1B = "sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2"
    


class SapiensPoseEstimation:
    def __init__(self,
                 type: SapiensPoseEstimationType = SapiensPoseEstimationType.POSE_ESTIMATION_03B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        # Load the model
        self.device = device
        self.dtype = dtype
        path = download_hf_model(type.value, TaskType.POSE)
        self.model = torch.jit.load(path).eval().to(device).to(dtype)
        self.preprocessor = pose_estimation_preprocessor(input_size=(1024, 768))

        # Initialize the YOLO-based detector
        self.detector = Detector()



    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Detect persons in the image
        bboxes = self.detector.detect(img)

        # Process the image and estimate the pose
        pose_result_image, keypoints = self.estimate_pose(img, bboxes)

        print(f"Pose estimation inference took: {time.perf_counter() - start:.4f} seconds")
        return pose_result_image, keypoints


    @torch.inference_mode()
    def estimate_pose(self, img: np.ndarray, bboxes: List[List[float]]) -> (np.ndarray, List[dict]):
        all_keypoints = []
        result_img = img.copy()

        for bbox in bboxes:
            cropped_img = self.crop_image(img, bbox)
            tensor = self.preprocessor(cropped_img).unsqueeze(0).to(self.device).to(self.dtype)

            heatmaps = self.model(tensor)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
            all_keypoints.append(keypoints)

            # Draw the keypoints on the original image
            result_img = self.draw_keypoints(result_img, keypoints, bbox)

        return result_img, all_keypoints

    def crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        return img[y1:y2, x1:x2]


    def heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> dict:
        keypoints = {}
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < heatmaps.shape[0]:
                y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
                conf = heatmaps[i, y, x]
                keypoints[name] = (float(x), float(y), float(conf))
        return keypoints



    def draw_keypoints(self, img: np.ndarray, keypoints: dict, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        bbox_width, bbox_height = x2 - x1, y2 - y1
        img_copy = img.copy()

        # Draw keypoints on t1Bhe image
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > 0.3:  # Only draw confident keypoints
                x_coord = int(x * bbox_width / 192) + x1
                y_coord = int(y * bbox_height / 256) + y1
                cv2.circle(img_copy, (x_coord, y_coord), 3, GOLIATH_KPTS_COLORS[i], -1)

        # Optionally draw skeleton
        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    x1_coord = int(pt1[0] * bbox_width / 192) + x1
                    y1_coord = int(pt1[1] * bbox_height / 256) + y1
                    x2_coord = int(pt2[0] * bbox_width / 192) + x1
                    y2_coord = int(pt2[1] * bbox_height / 256) + y1
                    cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), GOLIATH_KPTS_COLORS[i], 2)

        return img_copy




if __name__ == "__main__":
    type = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "test.jpg"
    img = cv2.imread(img_path)

    model_type = SapiensPoseEstimationType.POSE_ESTIMATION_03B
    estimator = SapiensPoseEstimation(model_type)

    start = time.perf_counter()
    result_img, keypoints = estimator(img)
    
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

    
    cv2.imshow("pose_estimation", result_img)
    cv2.waitKey(0)
