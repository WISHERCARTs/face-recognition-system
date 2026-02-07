# Face Recognition System using PCA + SVM
# Dataset: LFW (Labeled Faces in the Wild)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. โหลดข้อมูล LFW (Labeled Faces in the Wild)
# ============================================================
print("Loading LFW dataset...")
faces = fetch_lfw_people(min_faces_per_person=60)

# ดูข้อมูลเบื้องต้น
n_samples, h, w = faces.images.shape
X = faces.data  # ข้อมูล pixel (flatten แล้ว)
y = faces.target  # label (ชื่อบุคคล)
target_names = faces.target_names
n_classes = len(target_names)
n_features = X.shape[1]

print(f"Dataset loaded!")
print(f"  - Total samples: {n_samples}")
print(f"  - Image size: {h} x {w} pixels")
print(f"  - Number of features (pixels): {n_features}")
print(f"  - Number of classes (people): {n_classes}")
print(f"  - People: {target_names}")

# ============================================================
# 2. แบ่งข้อมูล Train/Test (75/25)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nData split:")
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Testing samples: {X_test.shape[0]}")

# ============================================================
# 3. สร้าง PCA และ SVM แล้วรวมเป็น Pipeline
# ============================================================
# PCA: ลดมิติจากพันกว่า features เหลือ 150 components
# whiten=True: ปรับ variance ของแต่ละ component ให้เท่ากัน
pca = PCA(n_components=150, whiten=True, random_state=42)

# SVM: ใช้ kernel RBF สำหรับข้อมูลที่ซับซ้อน
# class_weight='balanced': ให้ความสำคัญทุก class เท่ากัน
svc = SVC(kernel='rbf', class_weight='balanced')

# รวมเป็น Pipeline เดียว
model = make_pipeline(pca, svc)
print("\nPipeline created: PCA(150 components) -> SVM(RBF kernel)")

# ============================================================
# 4. ปรับจูน Hyperparameters ด้วย GridSearchCV
# ============================================================
print("\nStarting GridSearchCV (this may take a few minutes)...")
param_grid = {
    'svc__C': [1, 5, 10, 50],
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]
}

grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"\nBest parameters found:")
print(f"  - C: {grid.best_params_['svc__C']}")
print(f"  - gamma: {grid.best_params_['svc__gamma']}")
print(f"  - Best cross-validation score: {grid.best_score_:.4f}")

# ============================================================
# 5. ทำนายและวัดผล
# ============================================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================================
# 6. แสดง Confusion Matrix เป็น Heatmap
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Face Recognition')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# 7. แสดงตัวอย่างการทำนาย
# ============================================================
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    if i < len(X_test):
        ax.imshow(X_test[i].reshape(h, w), cmap='gray')
        pred_name = target_names[y_pred[i]]
        true_name = target_names[y_test[i]]
        color = 'green' if y_pred[i] == y_test[i] else 'red'
        ax.set_title(f"Pred: {pred_name.split()[-1]}\nTrue: {true_name.split()[-1]}", 
                     color=color, fontsize=10)
        ax.axis('off')

plt.suptitle('Face Recognition Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('prediction_samples.png', dpi=150)
plt.show()

print("\n Done! Images saved:")
print("  - confusion_matrix.png")
print("  - prediction_samples.png")
