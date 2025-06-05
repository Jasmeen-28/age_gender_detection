# age_gender_detection
This project implements age and gender prediction using deep learning techniques. It is divided into two parts:

- Training a CNN model using the UTKFace dataset.
- Real-time prediction using webcam input with pre-trained OpenCV models.

## Dataset
The project uses the UTKFace dataset located in:
-C:\Users\Jasmine kaur\Downloads\archive\crop_part1

- Image format: {age}_{gender}_{...}.jpg
- age: 0–100
- gender: 0 = male, 1 = female

## Requirements 
- tensorflow==2.19.0
- opencv-python==4.11.0.86
- numpy==2.1.3
- matplotlib==3.10.3
- scikit-learn==1.6.1
- pandas==2.2.3
- seaborn==0.13.2
- 
## 🧠 Model Training
- 👨‍💻 Requirements
- Install the required libraries:
- pip install numpy opencv-python tensorflow matplotlib scikit-learn
  
## 🏗️ Model Architecture
- CNN with Batch Normalization, MaxPooling
- Two outputs:
 1. age_output: Regression (scaled 0–1, Mean Squared Error loss)
 2. gender_output: Binary classification (sigmoid, Binary Crossentropy loss)


## 📊 Training Configuration
- Input Size: 64x64
- Augmentation: Flip, brightness, contrast
- Optimizer: Adam
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Loss weights: Age (2.0), Gender (1.0)

## 🧪 Evaluation & Visualization
- Evaluates model performance on test set
- Displays prediction vs. ground truth for 10 random samples
- Plots loss trends for both age and gender outputs

## 🎥 Real-Time Detection
- 🛠️ Dependencies
-> Download the following models from OpenCV's GitHub model zoo:
  
## 1. Face detection model:

- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel
  
## 2. Gender detection model:

- gender_deploy.prototxt
- gender_net.caffemodel

## 3. Age detection model:

- age_deploy.prototxt
- age_net.caffemodel

## 🚀 Running the Webcam Demo
- The webcam captures faces and uses OpenCV DNN modules to predict:
- Gender (from gender_net.caffemodel)
- Age group (e.g., 0–2, 4–6, etc.)
- To run : python your_script_name.py

 ##  📌 Notes
- Age prediction is normalized between 0–1 during training and rescaled to 0–100 during inference.

- Real-time detection uses category-based age prediction (e.g., 15–20, 21–25) unlike the continuous regression used in the CNN model.



## 📷 Example Output
- During training prediction:  True Age: 25, Pred: 23, Gender: Female
  
- Real-time webcam output:  [ ] Female, age: (21-25)

## 📞 Contact

Created by: Jasmeen Kaur
- 📧 jasmeeenkaur463@gmail.com
- 🔗 LinkedIn: kaurjasmeen00




