"""
=====================================================================
 GESTURE MODEL EVALUATION (RUN AFTER TRAINING, IN GOOGLE COLAB)
=====================================================================
 Generates:
   - Accuracy, Precision, Recall, F1 Score
   - Full classification report
   - Confusion matrix plot
=====================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================================
# CONFIG - must match training config
# =====================================================================
DATASET_DIR = "/content/dataset"
VALID_DIR = os.path.join(DATASET_DIR, "valid")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["fist", "one", "two", "three", "four", "five"]

# =====================================================================
# LOAD MODEL
# =====================================================================
model = tf.keras.models.load_model("gesture_model.h5")

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASS_NAMES,
    shuffle=False
)

# =====================================================================
# PREDICT ON VALIDATION SET
# =====================================================================
valid_generator.reset()
predictions = model.predict(valid_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_generator.classes

# =====================================================================
# METRICS
# =====================================================================
acc = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average="weighted")
recall = recall_score(true_classes, predicted_classes, average="weighted")
f1 = f1_score(true_classes, predicted_classes, average="weighted")

print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy  : {acc * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print("=" * 60)

print("\nFull Classification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

# =====================================================================
# CONFUSION MATRIX
# =====================================================================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title("Confusion Matrix - Gesture Recognition Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nSaved confusion_matrix.png")
print("Evaluation complete. Proceed to 03_export_model.py for export.")
