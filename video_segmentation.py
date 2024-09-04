from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube

from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map

videoUrl = 'https://youtube.com/shorts/DCpxd2ii_sM?si=oWbLTd1RveKThxYL'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=0))

# dtype = torch.float16
estimator = SapiensSegmentation(SapiensSegmentationType.SEGMENTATION_1B, dtype=dtype)

cv2.namedWindow("Segmentation Map", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    segmentation_map = estimator(frame)

    segmentation_map = draw_segmentation_map(segmentation_map)
    combined = cv2.addWeighted(frame, 0.5, segmentation_map, 0.7, 0)

    cv2.imshow("Segmentation Map", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
