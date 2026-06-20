import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter


class GestureDetector:
    CLASS_NAMES = ["fist", "one", "two", "three", "four", "five"]

    def __init__(self, model_path="model/gesture_model.keras", buffer_size=8, confidence_threshold=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.img_size = (224, 224)
        self.buffer_size = buffer_size
        self.gesture_buffer = deque(maxlen=buffer_size)
        self.confidence_threshold = confidence_threshold
        self.last_stable_gesture = "none"
        self.current_confidence = 0.0

        # Load model - try multiple formats
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        import os
        # Try .keras first, then .h5, then SavedModel
        paths_to_try = [
            model_path,
            model_path.replace('.h5', '.keras').replace('.keras', '.keras'),
            "model/gesture_model.keras",
            "model/gesture_model.h5",
            "model/gesture_model_final.h5",
        ]
        # Remove duplicates
        seen = set()
        unique_paths = []
        for p in paths_to_try:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        for path in unique_paths:
            if not os.path.exists(path):
                continue
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(path)
                print(f"[GestureDetector] Model loaded from: {path}")
                return
            except Exception as e:
                print(f"[GestureDetector] Could not load {path}: {e}")

        # Last resort - use finger counting only (no CNN)
        print("[GestureDetector] WARNING: No model loaded - using finger counting only")

    def _get_hand_bbox(self, frame, landmarks):
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        pad = 40
        x_min = max(int(min(xs)) - pad, 0)
        x_max = min(int(max(xs)) + pad, w)
        y_min = max(int(min(ys)) - pad, 0)
        y_max = min(int(max(ys)) + pad, h)
        return x_min, y_min, x_max, y_max

    def _preprocess_crop(self, hand_crop):
        if hand_crop.size == 0:
            return None
        img = cv2.resize(hand_crop, self.img_size)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def _count_raised_fingers(self, landmarks):
        lm = landmarks.landmark
        tips_ids = [4, 8, 12, 16, 20]
        fingers_up = 0
        if lm[4].x < lm[3].x:
            fingers_up += 1
        for tip_id in tips_ids[1:]:
            if lm[tip_id].y < lm[tip_id - 2].y:
                fingers_up += 1
        return fingers_up

    def _finger_count_to_gesture(self, count):
        mapping = {0: "fist", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
        return mapping.get(count, "none")

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_raw = "none"
        confidence = 0.0
        landmarks = None
        index_tip_xy = None

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            index_tip_xy = (int(landmarks.landmark[8].x * w), int(landmarks.landmark[8].y * h))

            finger_count = self._count_raised_fingers(landmarks)
            gesture_raw = self._finger_count_to_gesture(finger_count)
            confidence = 0.85  # fixed confidence for finger counting

            # If CNN model available, use it
            if self.model is not None:
                x_min, y_min, x_max, y_max = self._get_hand_bbox(frame, landmarks)
                hand_crop = frame[y_min:y_max, x_min:x_max]
                processed = self._preprocess_crop(hand_crop)
                if processed is not None:
                    try:
                        preds = self.model.predict(processed, verbose=0)[0]
                        class_idx = int(np.argmax(preds))
                        cnn_confidence = float(preds[class_idx])
                        cnn_gesture = self.CLASS_NAMES[class_idx]
                        if cnn_confidence > 0.7:
                            gesture_raw = cnn_gesture
                            confidence = cnn_confidence
                    except:
                        pass  # fallback to finger counting

        if confidence >= self.confidence_threshold:
            self.gesture_buffer.append(gesture_raw)
        else:
            self.gesture_buffer.append("none")

        if len(self.gesture_buffer) == self.buffer_size:
            most_common, count = Counter(self.gesture_buffer).most_common(1)[0]
            if count >= int(self.buffer_size * 0.6):
                self.last_stable_gesture = most_common

        self.current_confidence = confidence
        return self.last_stable_gesture, confidence, landmarks, index_tip_xy

    def draw_landmarks(self, frame, landmarks):
        if landmarks:
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame