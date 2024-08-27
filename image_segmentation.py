import torch
import cv2
from imread_from_url import imread_from_url

from sapiens.segmentation import SapiensSegmentation, draw_segmentation_map

model_path = "models/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
dtype = torch.float16
estimator = SapiensSegmentation(model_path, dtype=dtype)

img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg")

segmentation_map = estimator(img)

segmentation_image = draw_segmentation_map(segmentation_map)
combined = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)

cv2.imshow("Normal Map", combined)
cv2.waitKey(0)