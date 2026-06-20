"""
=====================================================================
 AI AIR DRAWING SYSTEM - GESTURE MODEL TRAINING (GOOGLE COLAB)
=====================================================================
 Run this entire file as cells in Google Colab.
 Model: MobileNetV2 (Transfer Learning)
 Classes: fist, one, two, three, four, five

 HOW TO USE IN COLAB:
 1. Upload your dataset as a zip to Colab, or mount Google Drive.
 2. Update DATASET_DIR below to point to your dataset folder.
 3. Run all cells top to bottom.
 4. Download gesture_model.h5 and gesture_model.tflite at the end.
=====================================================================
"""

# =====================================================================
# CELL 1: Install / Import Libraries
# =====================================================================
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# =====================================================================
# CELL 2: Mount Google Drive (optional - if dataset is in Drive)
# =====================================================================
# from google.colab import drive
# drive.mount('/content/drive')


# =====================================================================
# CELL 3: Dataset Configuration
# =====================================================================
# Expected folder structure:
#
# dataset/
# │
# ├── train/
# │   ├── fist/
# │   ├── one/
# │   ├── two/
# │   ├── three/
# │   ├── four/
# │   └── five/
# │
# └── valid/
#     ├── fist/
#     ├── one/
#     ├── two/
#     ├── three/
#     ├── four/
#     └── five/

DATASET_DIR = "/content/dataset"          # change this to your dataset path
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")

IMG_SIZE = (224, 224)      # MobileNetV2 default input size
BATCH_SIZE = 32
CLASS_NAMES = ["fist", "one", "two", "three", "four", "five"]
NUM_CLASSES = len(CLASS_NAMES)


# =====================================================================
# CELL 4: Data Augmentation + Loading
# =====================================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,   # IMPORTANT: keep False, gestures are direction-specific
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASS_NAMES,
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASS_NAMES,
    shuffle=False
)

print("Class indices:", train_generator.class_indices)


# =====================================================================
# CELL 5: Build Model (MobileNetV2 Transfer Learning)
# =====================================================================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially (we fine-tune later in Cell 7)
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =====================================================================
# CELL 6: Callbacks
# =====================================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model_stage1.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# =====================================================================
# CELL 7: Stage 1 Training (Frozen Base Model)
# =====================================================================
EPOCHS_STAGE1 = 15

history_stage1 = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_STAGE1,
    callbacks=[early_stop, reduce_lr, checkpoint]
)


# =====================================================================
# CELL 8: Stage 2 - Fine Tuning (Unfreeze top layers of MobileNetV2)
# =====================================================================
base_model.trainable = True

# Freeze all layers except the last 30 (fine-tune only top layers)
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),   # much lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_finetune = ModelCheckpoint(
    "gesture_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

EPOCHS_STAGE2 = 10

history_stage2 = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_STAGE2,
    callbacks=[early_stop, reduce_lr, checkpoint_finetune]
)


# =====================================================================
# CELL 9: Save Final Model
# =====================================================================
model.save("gesture_model.h5")
print("Model saved as gesture_model.h5")


# =====================================================================
# CELL 10: Plot Accuracy / Loss Graphs
# =====================================================================
def plot_history(h1, h2, key, title):
    plt.figure(figsize=(8, 5))
    plt.plot(h1.history[key] + h2.history[key], label=f"Train {key}")
    plt.plot(h1.history[f"val_{key}"] + h2.history[f"val_{key}"], label=f"Val {key}")
    plt.axvline(x=len(h1.history[key]), color="gray", linestyle="--", label="Fine-tune start")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(key.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{key}_graph.png")
    plt.show()

plot_history(history_stage1, history_stage2, "accuracy", "Model Accuracy Over Training")
plot_history(history_stage1, history_stage2, "loss", "Model Loss Over Training")

print("\nTraining complete. Proceed to 02_evaluate_model.py for evaluation.")
