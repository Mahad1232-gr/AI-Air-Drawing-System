"""
=====================================================================
 ai_snap.py
=====================================================================
 Purpose:
   Takes the user's rough hand-drawn stroke, classifies it using
   ShapeClassifier, and draws a clean, perfect geometric version of
   that shape onto the canvas in the same position/size.
=====================================================================
"""

import cv2
import numpy as np


class AISnap:
    def __init__(self, shape_classifier, min_confidence=0.55):
        self.classifier = shape_classifier
        self.min_confidence = min_confidence

    def snap(self, canvas, points, color, thickness):
        """
        Classify the drawn points and replace them with a clean shape.
        Returns: (success: bool, shape_name: str, confidence: float)
        """
        if len(points) < 5:
            return False, None, 0.0

        shape_name, confidence = self.classifier.predict(points)

        if shape_name is None or confidence < self.min_confidence:
            return False, shape_name, confidence

        pts_arr = np.array(points)
        x_min, y_min = pts_arr.min(axis=0)
        x_max, y_max = pts_arr.max(axis=0)
        cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        w, h = int(x_max - x_min), int(y_max - y_min)

        if shape_name == "circle":
            radius = max(int(max(w, h) / 2), 5)
            cv2.circle(canvas, (cx, cy), radius, color, thickness)

        elif shape_name == "square":
            half = max(int(max(w, h) / 2), 5)
            cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), color, thickness)

        elif shape_name == "triangle":
            half_w = max(int(w / 2), 5)
            half_h = max(int(h / 2), 5)
            p1 = (cx, cy - half_h)
            p2 = (cx - half_w, cy + half_h)
            p3 = (cx + half_w, cy + half_h)
            pts = np.array([p1, p2, p3], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        elif shape_name == "line":
            p1 = tuple(pts_arr[0].astype(int))
            p2 = tuple(pts_arr[-1].astype(int))
            cv2.line(canvas, p1, p2, color, thickness)

        else:
            return False, shape_name, confidence

        return True, shape_name, confidence
