from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube

from sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map

videoUrl = 'https://youtu.be/comTX7mxSzU?si=LL2ilfJ6tDXeFTkQ'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=2, seconds=54))

# dtype = torch.float16
estimator = SapiensNormal(SapiensNormalType.NORMAL_03B, dtype=dtype)

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
