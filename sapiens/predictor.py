from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .depth import SapiensDepth, SapiensDepthType, draw_depth_map
from .segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map
from .normal import SapiensNormal, SapiensNormalType, draw_normal_map
from .detector import Detector, DetectorConfig


@dataclass
class SapiensConfig:
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal_type: SapiensNormalType = SapiensNormalType.NORMAL_03B
    segmentation_type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B
    depth_type: SapiensDepthType = SapiensDepthType.OFF
    detector_config: DetectorConfig = DetectorConfig()
    minimum_person_height: int = 0.1  # 10% of the image height

    def __str__(self):
        return f"SapiensConfig(dtype={self.dtype}\n" \
               f"device={self.device}\n" \
               f"normal_type={self.normal_type}\n" \
               f"segmentation_type={self.segmentation_type}\n" \
               f"depth_type={self.depth_type}\n" \
               f"detector_config={self.detector_config}\n" \
               f"minimum_person_height={self.minimum_person_height * 100}% of the image height"


def filter_small_boxes(boxes: np.ndarray, img_height: int, height_thres: float = 0.1) -> np.ndarray:
    person_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        person_height = y2 - y1
        if person_height < height_thres * img_height:
            continue
        person_boxes.append(box)
    return np.array(person_boxes)


class SapiensPredictor:
    def __init__(self, config: SapiensConfig):
        self.has_normal = config.normal_type != SapiensNormalType.OFF
        self.has_depth = config.depth_type != SapiensDepthType.OFF
        self.minimum_person_height = config.minimum_person_height

        self.normal_predictor = SapiensNormal(config.normal_type, config.device,
                                              config.dtype) if self.has_normal else None
        self.segmentation_predictor = SapiensSegmentation(config.segmentation_type, config.device, config.dtype)
        self.depth_predictor = SapiensDepth(config.depth_type, config.device, config.dtype) if self.has_depth else None
        self.detector = Detector(config.detector_config)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.predict(img)

    def predict(self, img: np.ndarray) -> np.ndarray:
        img_shape = img.shape

        print("Detecting people...")
        person_boxes = self.detector.detect(img)
        person_boxes = filter_small_boxes(person_boxes, img_shape[0], self.minimum_person_height)

        if len(person_boxes) == 0:
            return img

        person_boxes = np.array(person_boxes)

        normal_maps = []
        segmentation_maps = []
        depth_maps = []
        print(f"{len(person_boxes)} people detected, predicting maps...")
        for box in person_boxes:
            crop = img[box[1]:box[3], box[0]:box[2]]

            segmentation_maps.append(self.segmentation_predictor(crop))
            if self.has_normal:
                normal_maps.append(self.normal_predictor(crop))
            if self.has_depth:
                depth_maps.append(self.depth_predictor(crop))

        return self.draw_maps(img, person_boxes, normal_maps, segmentation_maps, depth_maps)

    # TODO: Clean this up
    def draw_maps(self, img, person_boxes, normal_maps, segmentation_maps, depth_maps):
        draw_img = []
        segmentation_img = img.copy()
        for segmentation_map, box in zip(segmentation_maps, person_boxes):
            mask = segmentation_map > 0
            crop = img[box[1]:box[3], box[0]:box[2]]
            segmentation_draw = draw_segmentation_map(segmentation_map)
            crop_draw = cv2.addWeighted(crop, 0.5, segmentation_draw, 0.7, 0)
            segmentation_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
        draw_img.append(segmentation_img)

        if self.has_normal:
            normal_img = img.copy()
            for i, (normal_map, box) in enumerate(zip(normal_maps, person_boxes)):
                mask = segmentation_maps[i] > 0
                crop = img[box[1]:box[3], box[0]:box[2]]
                normal_draw = draw_normal_map(normal_map)
                crop_draw = cv2.addWeighted(crop, 0.5, normal_draw, 0.7, 0)
                normal_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
            draw_img.append(normal_img)

        if self.has_depth:
            depth_img = img.copy()
            for i, (depth_map, box) in enumerate(zip(depth_maps, person_boxes)):
                mask = segmentation_maps[i] > 0
                crop = img[box[1]:box[3], box[0]:box[2]]
                depth_map[~mask] = 0
                depth_draw = draw_depth_map(depth_map)
                crop_draw = cv2.addWeighted(crop, 0.5, depth_draw, 0.7, 0)
                depth_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
            draw_img.append(depth_img)

        return np.hstack(draw_img)
