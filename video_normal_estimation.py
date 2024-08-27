from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube
from sapiens.normal import SapiensNormal, draw_normal_map

videoUrl = 'https://youtu.be/c1pKUR6bvhA?si=4-hhLNgbdHUKh-N7'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=10))

model_path = "models/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
dtype = torch.float16
estimator = SapiensNormal(model_path, dtype=dtype)

cv2.namedWindow("Normal Map", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    normal_map = estimator(frame)

    normal_image = draw_normal_map(normal_map)
    combined = cv2.addWeighted(frame, 0.3, normal_image, 0.8, 0)

    cv2.imshow("Normal Map", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
