from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube

from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensNormalType, SapiensDepthType

config = SapiensConfig()
# config.dtype = torch.float16
config.normal_type = SapiensNormalType.NORMAL_1B # Disabled by default
# config.depth_type = SapiensDepthType.DEPTH_03B # Disabled by default
predictor = SapiensPredictor(config)

videoUrl = 'https://youtube.com/shorts/lXfX9qw0yAo?si=SrMq4-PGhBEau91l'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=0))


cv2.namedWindow("Predicted Map", cv2.WINDOW_NORMAL)
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = predictor(frame)

    cv2.imshow("Predicted Map", results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
