from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube
from sapiens.segmentation import SapiensSegmentation, draw_segmentation_map

videoUrl = 'https://youtu.be/c1pKUR6bvhA?si=4-hhLNgbdHUKh-N7'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=10))

model_path = "models/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
dtype = torch.float16
estimator = SapiensSegmentation(model_path, dtype=dtype)

cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    segmentation_map = estimator(frame)

    segmentation_image = draw_segmentation_map(segmentation_map)
    combined = cv2.addWeighted(frame, 0.5, segmentation_image, 0.7, 0)

    cv2.imshow("Segmentation Map", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
