Umpire Pose Detection using Computer Vision

This project detects cricket umpire signals using computer vision and machine learning techniques. 
The system captures video from a webcam, detects human body landmarks, and classifies the pose into different umpire signals.

 Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Joblib

Project Workflow
1. Capture video frames from webcam using OpenCV
2. Detect human body landmarks using MediaPipe Pose
3. Extract and normalize pose keypoints
4. Train machine learning models on the extracted features
5. Predict umpire signals in real time

 Machine Learning Models
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Ensemble Majority Voting for final prediction

Dataset
The dataset contains images of different cricket umpire signals. 
MediaPipe Pose is used to extract body landmarks from each image which are used as features for training the models.

Files

main.py  
Runs real-time umpire pose detection using a webcam.

train_model_complete.py  
Trains the machine learning models and saves them for prediction.

 How to Run

1. Install required libraries

pip install opencv-python mediapipe scikit-learn numpy joblib

2. Train the model

python train_model_complete.py

3. Run real-time detection

python main.py

