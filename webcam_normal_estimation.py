import torch
import cv2

from sapiens.normal import SapiensNormal, draw_normal_map

cap = cv2.VideoCapture(0)

model_path = "models/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
dtype = torch.float16
estimator = SapiensNormal(model_path, dtype=dtype)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    normal_map = estimator(frame)

    normal_map = draw_normal_map(normal_map)
    combined = cv2.addWeighted(frame, 0.2, normal_map, 0.8, 0)

    cv2.imshow("Normal Map", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


