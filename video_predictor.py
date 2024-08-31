from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube
from sapiens import SapiensPredictor, SapiensConfig, SapiensNormalType, SapiensDepthType

videoUrl = 'https://youtube.com/shorts/DCpxd2ii_sM?si=oWbLTd1RveKThxYL'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=0))

config = SapiensConfig()
config.dtype = torch.float16
config.normal_type = SapiensNormalType.NORMAL_1B # Disabled by default
# config.depth_type = SapiensDepthType.DEPTH_03B # Disabled by default
predictor = SapiensPredictor(config)

cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = predictor(frame)

    cv2.imshow("Segmentation Map", results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
