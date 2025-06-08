import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import joblib

# === 1. Load and prepare data ===
df = pd.read_csv(r"D:/BA/PID-Controller-optimization-with-machine-learning/pid_dataset_control.csv")
df = df[df['Label'].notna()]

label_encoder = LabelEncoder()
df['Label_encoded'] = label_encoder.fit_transform(df['Label'])

core_features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Overshoot"]
type_features = [col for col in df.columns if col.startswith("type_")]
features = core_features + type_features
X = df[features].fillna(0)
y = df['Label_encoded']

# === 2. Output directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"D:/BA/PID-Controller-optimization-with-machine-learning/models/classifier_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 4. Train classifier ===
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# === 5. Evaluation ===
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# === 6. ROC Curves ===
classes = label_encoder.classes_
y_test_bin = label_binarize(y_test, classes=range(len(classes)))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curves.png"))
plt.close()

# === 7. Save model and encoder ===
joblib.dump((clf, label_encoder), os.path.join(output_dir, "label_classifier.pkl"))
print(f"✅ Classifier saved to {output_dir}")
