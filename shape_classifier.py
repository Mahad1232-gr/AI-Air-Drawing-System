"""
=====================================================================
 shape_classifier.py
=====================================================================
 Purpose:
   Classifies a hand-drawn rough shape (circle, square, triangle,
   line) so the AI Snap feature can replace it with a clean version.

 Why KNN + Random Forest here instead of a CNN:
   This is a small, 4-class, geometry-based problem with hand-crafted
   features (compactness, convexity, aspect ratio, etc.) that already
   describe shape very well mathematically. A CNN would require a
   large image dataset and add latency for a problem that classical
   ML already solves at very high accuracy with near-zero latency.
   This keeps AI Snap instant, explainable in a viva, and accurate.

 Models:
   - KNeighborsClassifier
   - RandomForestClassifier
   - Final prediction = average of both models' probabilities
     (ensemble learning)
=====================================================================
"""

import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ShapeClassifier:
    SHAPES = ["circle", "square", "triangle", "line"]

    def __init__(self, n_samples_per_shape=300, points_per_shape=64):
        self.points_per_shape = points_per_shape
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
        self._train(n_samples_per_shape)

    # -----------------------------------------------------------
    # SYNTHETIC DATA GENERATION
    # -----------------------------------------------------------
    def _make_circle(self, noise=0.05):
        t = np.linspace(0, 2 * np.pi, self.points_per_shape)
        r = 1 + np.random.normal(0, noise, self.points_per_shape)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y])

    def _make_square(self, noise=0.05):
        n = self.points_per_shape // 4
        pts = []
        for i in range(4):
            t = np.linspace(0, 1, n)
            if i == 0:
                x, y = t * 2 - 1, np.full(n, -1.0)
            elif i == 1:
                x, y = np.full(n, 1.0), t * 2 - 1
            elif i == 2:
                x, y = 1 - t * 2, np.full(n, 1.0)
            else:
                x, y = np.full(n, -1.0), 1 - t * 2
            pts.append(np.column_stack([x, y]))
        pts = np.vstack(pts)
        pts += np.random.normal(0, noise, pts.shape)
        return pts

    def _make_triangle(self, noise=0.05):
        n = self.points_per_shape // 3
        verts = np.array([[0, 1], [-1, -1], [1, -1]])
        pts = []
        for i in range(3):
            t = np.linspace(0, 1, n)[:, None]
            seg = verts[i] * (1 - t) + verts[(i + 1) % 3] * t
            pts.append(seg)
        pts = np.vstack(pts)
        pts += np.random.normal(0, noise, pts.shape)
        return pts

    def _make_line(self, noise=0.05):
        t = np.linspace(0, 1, self.points_per_shape)
        x = t * 2 - 1
        y = np.random.normal(0, noise, self.points_per_shape)
        return np.column_stack([x, y])

    # -----------------------------------------------------------
    # FEATURE EXTRACTION
    # -----------------------------------------------------------
    def resample_points(self, points, n=64):
        """$1-Recognizer style resampling: normalize point count for
        scale- and speed-invariant comparison."""
        points = np.array(points, dtype=np.float32)
        if len(points) < 2:
            return np.tile(points, (n, 1))

        distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])
        total_length = distances[-1]
        if total_length == 0:
            return np.tile(points[0], (n, 1))

        target_distances = np.linspace(0, total_length, n)
        resampled = np.zeros((n, 2))
        for i, d in enumerate(target_distances):
            idx = np.searchsorted(distances, d)
            idx = min(idx, len(points) - 1)
            resampled[i] = points[idx]
        return resampled

    def extract_features(self, points):
        """Extract geometric features used for classification."""
        points = np.array(points, dtype=np.float32)
        if len(points) < 3:
            return None

        resampled = self.resample_points(points, self.points_per_shape)

        # Bounding box / aspect ratio
        x_min, y_min = resampled.min(axis=0)
        x_max, y_max = resampled.max(axis=0)
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)
        aspect_ratio = width / height

        # Normalize to unit box for scale invariance
        norm = resampled.copy()
        norm[:, 0] = (norm[:, 0] - x_min) / width
        norm[:, 1] = (norm[:, 1] - y_min) / height

        # Convex hull based features
        hull_input = norm.astype(np.float32)
        hull = cv2.convexHull(hull_input)
        hull_area = cv2.contourArea(hull)
        shape_area = cv2.contourArea(norm.astype(np.float32))
        convexity = shape_area / hull_area if hull_area > 0 else 0

        perimeter = np.sum(np.linalg.norm(np.diff(norm, axis=0), axis=1))
        compactness = (4 * np.pi * shape_area) / (perimeter ** 2) if perimeter > 0 else 0
        hull_perimeter = cv2.arcLength(hull, True)
        hull_circularity = (4 * np.pi * hull_area) / (hull_perimeter ** 2) if hull_perimeter > 0 else 0

        # Angle statistics
        vecs = np.diff(norm, axis=0)
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        angle_diffs = np.diff(angles)
        angle_diffs = np.mod(angle_diffs + np.pi, 2 * np.pi) - np.pi
        mean_angle = np.mean(np.abs(angle_diffs))
        angle_variance = np.var(angle_diffs)

        # Distance statistics from centroid
        centroid = norm.mean(axis=0)
        dists = np.linalg.norm(norm - centroid, axis=1)
        dist_mean = np.mean(dists)
        dist_std = np.std(dists)

        # End to end distance (high for "line", low for closed shapes)
        end_to_end = np.linalg.norm(norm[0] - norm[-1])

        features = [
            aspect_ratio, compactness, convexity, hull_circularity,
            mean_angle, angle_variance, dist_mean, dist_std, end_to_end
        ]
        return np.array(features, dtype=np.float32)

    # -----------------------------------------------------------
    # TRAINING (on synthetic generated dataset)
    # -----------------------------------------------------------
    def _train(self, n_samples_per_shape):
        X, y = [], []
        generators = {
            "circle": self._make_circle,
            "square": self._make_square,
            "triangle": self._make_triangle,
            "line": self._make_line,
        }

        for shape_name, gen_fn in generators.items():
            for _ in range(n_samples_per_shape):
                pts = gen_fn(noise=np.random.uniform(0.02, 0.12))
                feats = self.extract_features(pts)
                if feats is not None:
                    X.append(feats)
                    y.append(shape_name)

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled, y)
        self.rf.fit(X_scaled, y)

        print(f"[ShapeClassifier] Trained on {len(X)} synthetic samples "
              f"({n_samples_per_shape} per shape).")

    # -----------------------------------------------------------
    # PREDICTION (ensemble: average KNN + RF probabilities)
    # -----------------------------------------------------------
    def predict(self, points):
        feats = self.extract_features(points)
        if feats is None:
            return None, 0.0

        feats_scaled = self.scaler.transform([feats])

        knn_proba = self.knn.predict_proba(feats_scaled)[0]
        rf_proba = self.rf.predict_proba(feats_scaled)[0]

        # Ensure label order matches across both classifiers
        classes = self.knn.classes_
        avg_proba = (knn_proba + rf_proba) / 2.0

        best_idx = int(np.argmax(avg_proba))
        predicted_shape = classes[best_idx]
        confidence = float(avg_proba[best_idx])

        return predicted_shape, confidence
