import torch
import cv2

from sapiens.segmentation import SapiensSegmentation, draw_segmentation_map

cap = cv2.VideoCapture(0)

model_path = "models/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
dtype = torch.float16
estimator = SapiensSegmentation(model_path, dtype=dtype)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    normal_map = estimator(frame)

    normal_map = draw_segmentation_map(normal_map)
    combined = cv2.addWeighted(frame, 0.5, normal_map, 0.7, 0)

    cv2.imshow("Normal Map", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
