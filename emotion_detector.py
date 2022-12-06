import cv2
import numpy as np


video_footage = cv2.VideoCapture(0)

while True:
    ret, frame = video_footage.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    # Facial identification and processing
    face_identifier = cv2.CascadeClassifier(
        'facial_haarcascade/haarcascade_frontalface_default.xml'
    )
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_identifier.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_image = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1
        ), 0)

    cv2.imshow('FACIAL EMOTION RECOGNITION', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_footage.release()
cv2.destroyAllWindows()
