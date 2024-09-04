import torch
import cv2
from imread_from_url import imread_from_url

from sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map

# dtype = torch.float16
estimator = SapiensNormal(SapiensNormalType.NORMAL_1B, dtype=dtype)

img = imread_from_url("https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference/blob/assets/test1.png?raw=true")

normal_map = estimator(img)

normal_map = draw_normal_map(normal_map)
combined = cv2.addWeighted(img, 0.3, normal_map, 0.8, 0)

cv2.namedWindow("Normal Map", cv2.WINDOW_NORMAL)
cv2.imshow("Normal Map", combined)
cv2.waitKey(0)
