# age_gender_detection
This project performs real-time gender and age prediction on faces detected from a webcam video stream. 

## Features

- Detects faces in real-time from webcam input
- Predicts gender (Male/Female) 
- Predicts approximate age range for each detected face
- Displays bounding box and label (gender + age) on detected faces in video feed

  ## Errors
  - The gender and age predictions can fluctuate noticeably while the webcam is running.
  - The models provide approximate estimations and may not always correctly classify gender(in most cases shows male only) or predict age ranges. 

## Pre-trained Models

## Face detection model:

- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel

## Gender detection model:

- gender_deploy.prototxt
- gender_net.caffemodel

## Age detection model:

- age_deploy.prototxt
- age_net.caffemodel
