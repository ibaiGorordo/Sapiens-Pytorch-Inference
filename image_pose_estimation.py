import torch
import cv2
from imread_from_url import imread_from_url

from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType

dtype = torch.float16
estimator = SapiensPoseEstimation(SapiensPoseEstimationType.POSE_ESTIMATION_03B, dtype=dtype)

img = imread_from_url("https://learnopencv.com/wp-content/uploads/2024/09/football-soccer-scaled.jpg")

segmentation_map = estimator(img)

result_img, keypoints = estimator(img)

cv2.imshow("pose_estimation", result_img)
cv2.waitKey(0)
