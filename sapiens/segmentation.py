import time
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor, TaskType, download_hf_model


class SapiensSegmentationType(Enum):
    SEGMENTATION_03B = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
    SEGMENTATION_06B = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
    SEGMENTATION_1B = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"


random = np.random.RandomState(11)
classes = ["Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand", "Left Lower Arm", "Left Lower Leg",
           "Left Shoe", "Left Sock", "Left Upper Arm", "Left Upper Leg", "Lower Clothing", "Right Foot", "Right Hand",
           "Right Lower Arm", "Right Lower Leg", "Right Shoe", "Right Sock", "Right Upper Arm", "Right Upper Leg",
           "Torso", "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth", "Tongue"]

colors = random.randint(0, 255, (len(classes) - 1, 3))
colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
colors = colors[:, ::-1]


def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
    h, w = segmentation_map.shape
    segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        segmentation_img[segmentation_map == i] = color

    return segmentation_img


def postprocess_segmentation(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Perform argmax to get the segmentation map
    segmentation_map = logits.argmax(dim=0, keepdim=True)

    # Covert to numpy array
    segmentation_map = segmentation_map.float().numpy().squeeze()

    return segmentation_map


class SapiensSegmentation():
    def __init__(self,
                 type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        path = download_hf_model(type.value, TaskType.SEG)
        model = torch.jit.load(path)
        model = model.eval()
        self.model = model.to(device).to(dtype)
        self.device = device
        self.dtype = dtype
        self.preprocessor = create_preprocessor(input_size=(1024, 768))  # Only these values seem to work well

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)
        segmentation_map = postprocess_segmentation(results, img.shape[:2])

        print(f"Segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return segmentation_map


if __name__ == "__main__":
    type = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "test.jpg"
    img = cv2.imread(img_path)

    model_type = SapiensSegmentationType.SEGMENTATION_1B
    estimator = SapiensSegmentation(model_type)

    start = time.perf_counter()
    segmentations = estimator(img)
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

    segmentation_img = draw_segmentation_map(segmentations)
    combined = cv2.addWeighted(img, 0.5, segmentation_img, 0.5, 0)

    cv2.imshow("segmentation_map", combined)
    cv2.waitKey(0)
