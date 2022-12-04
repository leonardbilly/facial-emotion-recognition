import cv2
import numpy as np


video_footage = cv2.VideoCapture(0)

while True:
    ret, frame = video_footage.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    cv2.imshow('FACIAL EMOTION RECOGNITION', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_footage.release()
cv2.destroyAllWindows()
