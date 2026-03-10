import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load Models
svm = joblib.load("svm_model.pkl")
knn = joblib.load("knn_model.pkl")
rf = joblib.load("rf_model.pkl")
label_names = joblib.load("label_names.pkl")

# Load scaler + PCA (IMPORTANT FOR KNN)
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Normalization Function (MUST MATCH TRAINING)
def normalize_keypoints(kp):
    kp = np.array(kp).reshape(-1, 2)
    left_shoulder = kp[11]
    right_shoulder = kp[12]

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width == 0:
        shoulder_width = 1

    kp = kp / shoulder_width
    return kp.flatten()


def extract_features(results):
    if results.pose_landmarks:
        kp = []
        for lm in results.pose_landmarks.landmark:
            kp.append([lm.x, lm.y])

        kp_norm = normalize_keypoints(kp)  # SAME AS TRAINING
        return np.array(kp_norm).reshape(1, -1)

    return None


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    features = extract_features(results)

    if features is not None:

        # KNN requires PCA + scaling
        features_scaled = scaler.transform(features)
        knn_ready = pca.transform(features_scaled)

        # Predictions
        svm_pred = svm.predict(features)[0]         # raw normalized features
        rf_pred = rf.predict(features)[0]           # raw normalized features
        knn_pred = knn.predict(knn_ready)[0]        # PCA + scaled features

        svm_label = label_names[svm_pred]
        knn_label = label_names[knn_pred]
        rf_label = label_names[rf_pred]

        # Majority vote
        preds = [svm_label, knn_label, rf_label]
        final = max(set(preds), key=preds.count)

        # Confidence
        if final == svm_label:
            conf = svm.predict_proba(features)[0][svm_pred]
        elif final == knn_label:
            conf = knn.predict_proba(knn_ready)[0][knn_pred]
        else:
            conf = rf.predict_proba(features)[0][rf_pred]

        conf_percent = conf * 100

        cv2.putText(frame, f"SVM: {svm_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"KNN: {knn_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"RF: {rf_label}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"FINAL: {final} ({conf_percent:.1f}%)", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    else:
        cv2.putText(frame, "Pose Not Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Umpire Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
