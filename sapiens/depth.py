from enum import Enum
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor, download_hf_model, TaskType


class SapiensDepthType(Enum):
    OFF = "off"
    DEPTH_03B = "sapiens_0.3b_render_people_epoch_100_torchscript.pt2"
    DEPTH_06B = "sapiens_0.6b_render_people_epoch_70_torchscript.pt2"
    DEPTH_1B = "sapiens_1b_render_people_epoch_88_torchscript.pt2"
    DEPTH_2B = "sapiens_2b_render_people_epoch_25_torchscript.pt2"


def draw_depth_map(depth_map: np.ndarray) -> np.ndarray:
    min_depth, max_depth = np.min(depth_map), np.max(depth_map)

    norm_depth_map = 1 - (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth_map = (norm_depth_map * 255).astype(np.uint8)

    # Normalize and color the image
    color_depth = cv2.applyColorMap(norm_depth_map, cv2.COLORMAP_INFERNO)
    color_depth[depth_map == 0] = 128
    return color_depth


def postprocess_depth(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Covert to numpy array
    depth_map = logits.float().numpy().squeeze()
    return depth_map


class SapiensDepth():
    def __init__(self,
                 type: SapiensDepthType = SapiensDepthType.DEPTH_03B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        path = download_hf_model(type.value, TaskType.DEPTH)
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

        depth_map = postprocess_depth(results, img.shape[:2])
        print(f"Depth inference took: {time.perf_counter() - start:.4f} seconds")
        return depth_map


if __name__ == "__main__":
    type = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "test.jpg"
    img = cv2.imread(img_path)

    model_type = SapiensDepthType.DEPTH_03B
    estimator = SapiensDepth(model_type)

    start = time.perf_counter()
    depth_map = estimator(img)
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

    depth_img = draw_depth_map(depth_map)

    cv2.imshow("depth_map", depth_img)
    cv2.waitKey(0)
