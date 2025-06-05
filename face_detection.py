import cv2
import numpy as np
from collections import deque


face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt','age_net.caffemodel')


gender_list = ['Female', 'Male']
age_list =  ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(21-25)', '(27-32)', '(38-43)', '(48-53)', '(60-100)']


age_buffer = deque(maxlen=15)
gender_buffer = deque(maxlen=15)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for detection in detections[0, 0]:
        confidence = float(detection[2])
        if confidence > 0.5:
            box = detection[3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

        
            margin = 20
            x1_m = max(0, x1 - margin)
            y1_m = max(0, y1 - margin)
            x2_m = min(w, x2 + margin)
            y2_m = min(h, y2 + margin)

            face = frame[y1_m:y2_m, x1_m:x2_m]
            if face.shape[0] < 100 or face.shape[1] < 100:
                continue  

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=True)

            # Predict gender
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender_index = gender_preds[0].argmax()
            gender_buffer.append(gender_index)

            # Predict age
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age_index = age_preds[0].argmax()
            age_buffer.append(age_index)

    
            stable_gender = gender_list[max(set(gender_buffer), key=gender_buffer.count)]
            stable_age = age_list[max(set(age_buffer), key=age_buffer.count)]

         
            label = f"{stable_gender}, age: {stable_age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Age & Gender Detection', frame)
    if cv2.waitKey(1) & 0xFF == 13:  
        break

cap.release()
cv2.destroyAllWindows()
