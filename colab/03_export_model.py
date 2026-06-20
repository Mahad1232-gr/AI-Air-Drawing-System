"""
=====================================================================
 EXPORT MODEL: .h5 -> .tflite (RUN IN GOOGLE COLAB)
=====================================================================
 Produces:
   - gesture_model.h5      (already saved during training)
   - gesture_model.tflite  (lightweight version for faster inference)

 WHY BOTH FORMATS:
   .h5      -> Standard Keras format. Easiest to use in VS Code with
               plain TensorFlow. Slightly larger, slightly slower.
   .tflite  -> Optimized, quantized, much smaller and faster.
               Best choice if real-time FPS is a priority on a
               weaker laptop CPU. Slightly more setup to load.
=====================================================================
"""

import tensorflow as tf
import os

MODEL_PATH = "gesture_model.h5"
TFLITE_PATH = "gesture_model.tflite"

# =====================================================================
# LOAD TRAINED MODEL
# =====================================================================
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")
print(f"Original .h5 size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

# =====================================================================
# CONVERT TO TFLITE (with dynamic range quantization)
# =====================================================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantization for speed + size

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved {TFLITE_PATH}")
print(f"TFLite size: {os.path.getsize(TFLITE_PATH) / (1024*1024):.2f} MB")

# =====================================================================
# QUICK SANITY CHECK - LOAD TFLITE AND VERIFY
# =====================================================================
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite Model Input Details:", input_details[0]["shape"], input_details[0]["dtype"])
print("TFLite Model Output Details:", output_details[0]["shape"], output_details[0]["dtype"])

print("\nExport complete.")
print("Download both gesture_model.h5 and gesture_model.tflite")
print("Place gesture_model.h5 inside: AI_Air_Drawing/model/")
