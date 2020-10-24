import dlib
import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

detector = dlib.get_frontal_face_detector()
emotions = load_model("models/Emotions.h5")
#print(emotions.summary())
cap = cv2.VideoCapture(0)

classes = ["Angry","Happy","Neutral","Sad","Surprise"]

face_emotions = {}

while cap.isOpened():
    flag, frame = cap.read()

    faces = detector(frame, 0)
    font = cv2.FONT_ITALIC

    kk = cv2.waitKey(1)

    if kk == ord("q"):
        break
    else:
        if len(faces) != 0:
            for face in faces:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255), 2)
                
                face_area = frame[face.top():face.bottom(), face.left():face.right()]
                face_area_g = cv2.cvtColor(face_area, cv2.COLOR_RGB2GRAY)
                roi_gray = cv2.resize(face_area_g, (48, 48), interpolation = cv2.INTER_AREA)
                roi = roi_gray.astype("float")/255.
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis = 0)
                pred = emotions.predict(roi)[0]
                index = pred.argmax()

                cv2.putText(frame, classes[index], (int((face.left()+face.right())/2.3), face.bottom()+30), font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            
    cv2.imshow("image",frame)
cap.release()
cv2.destroyAllWindows()