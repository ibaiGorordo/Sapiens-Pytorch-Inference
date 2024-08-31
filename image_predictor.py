import torch
import cv2
from imread_from_url import imread_from_url

from sapiens import SapiensPredictor, SapiensConfig, SapiensDepthType, SapiensNormalType

config = SapiensConfig()
config.dtype = torch.float16
config.depth_type = SapiensDepthType.DEPTH_03B  # Disabled by default
config.normal_type = SapiensNormalType.NORMAL_1B  # Disabled by default
predictor = SapiensPredictor(config)

# Load the image
# img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg")
img = cv2.imread("ComfyUI_00074_.png")

# Estimate the maps
result = predictor(img)

cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
cv2.imshow("Combined", result)
cv2.waitKey(0)
