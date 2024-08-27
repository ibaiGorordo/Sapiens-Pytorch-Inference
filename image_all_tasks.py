import torch
import cv2
from imread_from_url import imread_from_url

from sapiens.segmentation import SapiensSegmentation, draw_segmentation_map
from sapiens.normal import SapiensNormal, draw_normal_map
from sapiens.depth import SapiensDepth, draw_depth_map

normal_path = "models/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
segmentation_path = "models/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
depth_path = "models/sapiens_0.3b_render_people_epoch_100_torchscript.pt2"

# Load the models
dtype = torch.float16
normal_estimator = SapiensNormal(normal_path, dtype=dtype)
segmentation_estimator = SapiensSegmentation(segmentation_path, dtype=dtype)
depth_estimator = SapiensDepth(depth_path, dtype=dtype)

# Load the image
# img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg")
img = cv2.imread("ComfyUI_00074_.png")

# Estimate the maps
segmentation_map = segmentation_estimator(img)
normal_map = normal_estimator(img)
depth_map = depth_estimator(img)

# Mask the image with the segmentation map
mask = segmentation_map > 0
normal_map[~mask] = 0
depth_map[~mask] = 0

# Draw the images
segmentation_image = draw_segmentation_map(segmentation_map)
normal_image = draw_normal_map(normal_map)
depth_image = draw_depth_map(depth_map)

# Combine the images
segmentation_image = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)
normal_image = cv2.addWeighted(img, 0.5, normal_image, 0.7, 0)
depth_image = cv2.addWeighted(img, 0.5, depth_image, 0.7, 0)
combined_image = cv2.hconcat([segmentation_image, normal_image, depth_image])

cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
cv2.imshow("Combined", combined_image)
cv2.waitKey(0)
