import torch
import cv2
from imread_from_url import imread_from_url

from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensDepthType, SapiensNormalType

config = SapiensConfig()
# config.dtype = torch.float16
config.depth_type = SapiensDepthType.DEPTH_03B  # Disabled by default
config.normal_type = SapiensNormalType.NORMAL_1B  # Disabled by default
predictor = SapiensPredictor(config)

# Load the image
img = imread_from_url("https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference/blob/assets/test2.png?raw=true")

# Estimate the maps
result = predictor(img)

cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
cv2.imshow("Combined", result)
cv2.waitKey(0)
