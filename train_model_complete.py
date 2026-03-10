import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

dataset_path = "dataset/train"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

X = []
y = []

# ====== NORMALIZE KEYPOINTS ======
def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 2)

    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width == 0:
        shoulder_width = 1

    keypoints = keypoints / shoulder_width
    return keypoints.flatten()


# ====== LOAD CLASSES ======
label_names = sorted([d for d in os.listdir(dataset_path) if not d.startswith(".")])
print("Classes:", label_names)

bad_images = []


# ====== LOAD IMAGES & EXTRACT LANDMARKS ======
for label_index, label in enumerate(label_names):
    folder = os.path.join(dataset_path, label)
    
    for img_file in os.listdir(folder):
        if img_file.startswith("."):
            continue
        
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        # If pose not detected → mark as bad image
        if not result.pose_landmarks:
            bad_images.append((label, img_file))
            continue
        
        kp = []
        for lm in result.pose_landmarks.landmark:
            kp.append([lm.x, lm.y])

        kp = normalize_keypoints(kp)
        X.append(kp)
        y.append(label_index)

# ==== REPORT BAD IMAGES ====
print("\nBad Images (removed automatically):")
for b in bad_images:
    print(b)

X = np.array(X)
y = np.array(y)

print("\nUsable Samples After Cleaning:", len(X))


# ====== SPLIT DATA ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# ====== SCALING FOR PCA + KNN ======
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====== PCA (improves KNN massively) ======
pca = PCA(n_components=40)  # 40 components gives best results
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ====== MODELS ======
svm = SVC(kernel='rbf', C=15, gamma='scale', probability=True)
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
rf = RandomForestClassifier(n_estimators=350)

# ====== TRAIN ======
svm.fit(X_train, y_train)           # Original features
rf.fit(X_train, y_train)            # Original features
knn.fit(X_train_pca, y_train)       # Scaled + PCA features


# ====== ACCURACY ======
print(f"\nSVM Accuracy: {svm.score(X_test, y_test) * 100:.2f}%")
print(f"KNN Accuracy: {knn.score(X_test_pca, y_test) * 100:.2f}%")
print(f"RF Accuracy : {rf.score(X_test, y_test) * 100:.2f}%\n")


# ====== SAVE MODELS ======
joblib.dump(svm, "svm_model.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(label_names, "label_names.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

from sklearn.metrics import f1_score

# ---- TRAIN ACCURACY ----
train_svm = svm.score(X_train, y_train) * 100
train_knn = knn.score(X_train_pca, y_train) * 100
train_rf  = rf.score(X_train, y_train) * 100

# ---- VALIDATION ACCURACY ----
svm_val = svm.score(X_test, y_test) * 100
knn_val = knn.score(X_test_pca, y_test) * 100
rf_val  = rf.score(X_test, y_test) * 100

# ---- F1 SCORE ----
svm_f1 = f1_score(y_test, svm.predict(X_test), average='weighted')
knn_f1 = f1_score(y_test, knn.predict(X_test_pca), average='weighted')
rf_f1  = f1_score(y_test, rf.predict(X_test), average='weighted')

# ---- ENSEMBLE (MAJORITY VOTE) ----
svm_preds = svm.predict(X_test)
knn_preds = knn.predict(X_test_pca)
rf_preds  = rf.predict(X_test)

ensemble_preds = []

for s, k, r in zip(svm_preds, knn_preds, rf_preds):
    votes = [s, k, r]
    final = max(set(votes), key=votes.count)
    ensemble_preds.append(final)

ensemble_val = (np.array(ensemble_preds) == y_test).mean() * 100
ensemble_f1 = f1_score(y_test, ensemble_preds, average='weighted')

# ---- WRITE TABLE TO TXT FILE ----
with open("model_accuracy.txt", "w") as f:
    f.write("---------------------------------------------------------\n")
    f.write(" Umpire Pose Classification – Model Performance Summary\n")
    f.write("---------------------------------------------------------\n\n")

    f.write("| Model                           | Train Acc | Val Acc  | F1-Score |\n")
    f.write("|---------------------------------|-----------|----------|----------|\n")

    f.write(f"| SVM (RBF Kernel)                | {train_svm:.2f}%   | {svm_val:.2f}%   | {svm_f1:.2f}     |\n")
    f.write(f"| KNN (PCA + Scaled)              | {train_knn:.2f}%   | {knn_val:.2f}%   | {knn_f1:.2f}     |\n")
    f.write(f"| Random Forest                   | {train_rf:.2f}%   | {rf_val:.2f}%   | {rf_f1:.2f}     |\n")
    f.write(f"| Ensemble (Majority Voting)      | N/A       | {ensemble_val:.2f}%   | {ensemble_f1:.2f}     |\n")

print("Model accuracy table saved as model_accuracy.txt")
