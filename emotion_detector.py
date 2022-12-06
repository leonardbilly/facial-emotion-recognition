import cv2
import numpy as np
from keras.models import model_from_json


emotions = {0: "angry", 1: "disgust", 2: "fear",
            3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

try:
    json_file = open('models/facial_emotion_recognition_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fer_model = model_from_json(loaded_model_json)

    fer_model.load_weights("models/facial_emotion_recognition_model.h5")
    print('Successfully loaded model.')
except:
    print('Failed to load saved model.')

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

        emotion_prediction = fer_model.predict(cropped_image)
        emotion_label = int(np.argmax(emotion_prediction))
        emotion_label_position = (x+5, y-15)
        cv2.putText(frame, emotions[emotion_label], emotion_label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('FACIAL EMOTION RECOGNITION', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_footage.release()
cv2.destroyAllWindows()
