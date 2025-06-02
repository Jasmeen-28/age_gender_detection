import cv2
import numpy as np

load_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt','age_net.caffemodel')

gender_list = ['Female', 'Male']
age_list =  ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

cap = cv2.VideoCapture(0)
# img = cv2.imread('images/img_3.png')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104.0, 177.0, 123.0), scalefactor=1.0)
    load_model.setInput(blob)
    detections = load_model.forward()

    for detection in detections[0, 0]:
        confidence = float(detection[2])
        if confidence > 0.3:
            box = detection[3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            face = frame[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face_blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227),
                                                  mean=(78.4263377603, 87.7689143744, 114.895847746),
                                                  swapRB=True, crop=False)

                gender_model.setInput(face_blob)
                gender_preds = gender_model.forward()
                gender = gender_list[gender_preds[0].argmax()]

                age_model.setInput(face_blob)
                age_pred = age_model.forward()
                age = age_list[age_pred[0].argmax()]

                label = f"{gender}, age:{age}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow('Original Image', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


