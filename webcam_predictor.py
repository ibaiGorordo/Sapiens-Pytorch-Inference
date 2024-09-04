import torch
import cv2

from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensSegmentationType, SapiensNormalType, SapiensDepthType

cap = cv2.VideoCapture(0)

config = SapiensConfig()
# config.depth_type = SapiensDepthType.DEPTH_03B # Disabled by default
# config.normal_type = SapiensNormalType.NORMAL_1B # Disabled by default
# config.dtype = torch.float16
predictor = SapiensPredictor(config)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = predictor(frame)

    cv2.imshow("Normal Map", results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
