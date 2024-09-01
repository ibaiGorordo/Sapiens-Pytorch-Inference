from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube
from cv_videowriter import VideoWriter

from sapiens import SapiensPredictor, SapiensConfig, SapiensNormalType, SapiensDepthType

config = SapiensConfig()
config.dtype = torch.float16
config.normal_type = SapiensNormalType.NORMAL_1B # Disabled by default
# config.depth_type = SapiensDepthType.DEPTH_03B # Disabled by default
predictor = SapiensPredictor(config)

# videoUrl = 'https://youtube.com/shorts/DCpxd2ii_sM?si=oWbLTd1RveKThxYL'
# videoUrl = 'https://youtube.com/shorts/fxRZAp70s2M?si=bY8zTm-0aHeCzvDw'
# videoUrl = 'https://youtube.com/shorts/2D5SxTIRGBg?si=r0Zk2KJrrKmMsW5T'
# videoUrl = 'https://youtube.com/shorts/7WywUt8SchA?si=lzIonGGkXgDNyoK6'
# videoUrl = 'https://youtube.com/shorts/YNjCnEiKnyM?si=C-FnOVH3c7zAG46M'
videoUrl = 'https://youtube.com/shorts/lXfX9qw0yAo?si=SrMq4-PGhBEau91l'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=0))

frames = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frames.append(frame)

writer = VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20)

cv2.namedWindow("Predicted Map", cv2.WINDOW_NORMAL)
for frame in frames:

    results = predictor(frame)

    writer.write(results)

    cv2.imshow("Predicted Map", results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
