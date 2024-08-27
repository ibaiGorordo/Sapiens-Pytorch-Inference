import torch
import cv2
from imread_from_url import imread_from_url

from sapiens.normal import SapiensNormal, draw_normal_map


model_path = "models/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
dtype = torch.float16
estimator = SapiensNormal(model_path, dtype=dtype)

# img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Han_Solo_in_Carbonite_%2831649435213%29.jpg/768px-Han_Solo_in_Carbonite_%2831649435213%29.jpg")
img = cv2.imread("ComfyUI_00070_.png")

normal_map = estimator(img)

normal_map = draw_normal_map(normal_map)
combined = cv2.addWeighted(img, 0.3, normal_map, 0.8, 0)

cv2.namedWindow("Normal Map", cv2.WINDOW_NORMAL)
cv2.imshow("Normal Map", combined)
cv2.waitKey(0)
