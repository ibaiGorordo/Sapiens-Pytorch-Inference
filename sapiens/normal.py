import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor


def draw_normal_map(normal_map: np.ndarray) -> np.ndarray:
    # Normalize the normal map
    normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small epsilon to avoid division by zero
    normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)

    # Convert to BGR
    return cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)


def postprocess_normal(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].detach().cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Covert to numpy array
    normal_map = logits.float().numpy().transpose(1, 2, 0)

    return normal_map


class SapiensNormal():
    def __init__(self,
                 path: str,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):

        model = torch.jit.load(path)
        model = model.eval()
        self.model = model.to(device).to(dtype)
        self.device = device
        self.dtype = dtype
        self.preprocessor = create_preprocessor(input_size=(1024, 768)) # Only these values seem to work well

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():

            results = self.model(tensor)




        normals = postprocess_normal(results, img.shape[:2])
        print(f"fps: {1 / (time.perf_counter() - start)}")
        return normals


if __name__ == "__main__":
    type = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "test3.jpg"
    img = cv2.imread(img_path)

    model_path = "../models/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2"
    estimator = SapiensNormal(model_path, device=device, dtype=type)

    start = time.perf_counter()
    normals = estimator(img)
    print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

    normal_img = draw_normal_map(normals)

    cv2.imshow("normal_map", normal_img)
    cv2.waitKey(0)
