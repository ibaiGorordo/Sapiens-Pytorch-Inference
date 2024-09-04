import torch
import cv2
from imread_from_url import imread_from_url

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map

# dtype = torch.float16
estimator = SapiensSegmentation(SapiensSegmentationType.SEGMENTATION_1B, dtype=dtype)

img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg")

segmentation_map = estimator(img)

segmentation_image = draw_segmentation_map(segmentation_map)
combined = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)

cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
cv2.imshow("Segmentation Map", combined)
cv2.waitKey(0)