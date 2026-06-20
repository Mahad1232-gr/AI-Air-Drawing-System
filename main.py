"""
=====================================================================
 main.py
=====================================================================
 AI BASED AIR DRAWING SYSTEM
 Using Hand Gesture Recognition (Deep Learning) + Machine Learning
 Shape Classification

 GESTURES:
   Fist        -> Clear Canvas
   1 Finger    -> Draw
   2 Fingers   -> AI Snap (clean up rough shape)
   3 Fingers   -> Change Color
   4 Fingers   -> Save Drawing
   5 Fingers   -> Stop / Unlock

 KEYBOARD (ONLY for lock control + brush size, NOT replacing gestures):
   L -> Lock current gesture mode
   U -> Unlock
   B -> Increase brush size
   N -> Decrease brush size
   Q -> Quit

 LOCK SYSTEM (resolves open questions from project design phase):
   - Hold any action gesture (1/2/3/4) for ~1 second -> that mode
     LOCKS. While locked, only that action's gesture is recognized;
     all other hand shapes are ignored. This is what stops accidental
     color changes / canvas clears / failed saves.
   - Unlock = hold 5 fingers (open palm) for ~1 second, OR press U.
   - Save and Clear both require a short hold (not instant single
     frame) to avoid accidental triggers from a hand passing through
     that pose for one frame.
=====================================================================
"""

import cv2
import time

from gesture_detector import GestureDetector
from shape_classifier import ShapeClassifier
from ai_snap import AISnap
from utils.smoothing import HoltSmoother
from utils.drawing_utils import (
    COLORS, COLOR_NAMES, create_blank_canvas, is_canvas_empty,
    save_canvas, overlay_canvas_on_frame, draw_gesture_guide,
    draw_status_bar, draw_lock_progress, draw_cursor,
    draw_brush_size_indicator, draw_ai_snap_result
)

# =====================================================================
# CONFIGURATION
# =====================================================================
GESTURE_MODEL_PATH = "model/gesture_model.h5"
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720

HOLD_TO_LOCK_SECONDS = 1.0       # how long to hold a gesture to lock it
HOLD_TO_ACTION_SECONDS = 0.6     # hold time required for Save / Clear
MIN_BRUSH, MAX_BRUSH, BRUSH_STEP = 2, 30, 2

GESTURE_TO_MODE = {
    "one": "draw",
    "two": "snap",
    "three": "color",
    "four": "save",
}


class AirDrawingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        self.gesture_detector = GestureDetector(model_path=GESTURE_MODEL_PATH)
        self.shape_classifier = ShapeClassifier()
        self.ai_snap = AISnap(self.shape_classifier)
        self.smoother = HoltSmoother(alpha=0.5, beta=0.3)

        self.canvas = create_blank_canvas(FRAME_H, FRAME_W)

        self.mode = "draw"           # current active mode
        self.locked = False
        self.locked_mode = None

        self.color_idx = 0
        self.brush_size = 5

        self.current_stroke = []     # points of the in-progress stroke (for AI Snap)
        self.prev_point = None

        self.status_message = ""
        self.status_message_until = 0

        # Hold-tracking state for gesture-based locking and hold-actions
        self.pending_gesture = None
        self.pending_gesture_start = None

    # -----------------------------------------------------------
    def set_status(self, msg, duration=1.5):
        self.status_message = msg
        self.status_message_until = time.time() + duration

    # -----------------------------------------------------------
    def get_active_color(self):
        name = COLOR_NAMES[self.color_idx]
        return COLORS[name], name

    # -----------------------------------------------------------
    def handle_hold_logic(self, gesture):
        """
        Tracks how long a gesture has been continuously held, to
        support: lock-in (1/2/3/4), unlock (five), and hold-to-confirm
        for save/clear.
        Returns progress in [0, 1] for UI feedback, and fires the
        corresponding action exactly once when the hold completes.
        """
        now = time.time()

        if gesture != self.pending_gesture:
            self.pending_gesture = gesture
            self.pending_gesture_start = now
            return 0.0

        held_for = now - self.pending_gesture_start

        # ---- LOCK: holding one/two/three/four while unlocked ----
        if not self.locked and gesture in GESTURE_TO_MODE:
            progress = min(held_for / HOLD_TO_LOCK_SECONDS, 1.0)

            if gesture == "four":
                # four = save acts immediately (hold-to-confirm), doesn't lock
                if held_for >= HOLD_TO_ACTION_SECONDS:
                    self.do_save()
                    self.pending_gesture_start = now + 999  # prevent re-trigger until gesture changes
                return min(held_for / HOLD_TO_ACTION_SECONDS, 1.0)

            if held_for >= HOLD_TO_LOCK_SECONDS:
                self.lock_mode(GESTURE_TO_MODE[gesture])
            return progress

        # ---- UNLOCK: holding five (open palm) while locked ----
        if self.locked and gesture == "five":
            progress = min(held_for / HOLD_TO_LOCK_SECONDS, 1.0)
            if held_for >= HOLD_TO_LOCK_SECONDS:
                self.unlock_mode()
            return progress

        # ---- CLEAR: holding fist (works locked or unlocked) ----
        if gesture == "fist":
            progress = min(held_for / HOLD_TO_ACTION_SECONDS, 1.0)
            if held_for >= HOLD_TO_ACTION_SECONDS:
                self.do_clear()
                self.pending_gesture_start = now + 999
            return progress

        return 0.0

    # -----------------------------------------------------------
    def lock_mode(self, mode_name):
        self.locked = True
        self.locked_mode = mode_name
        self.mode = mode_name
        self.set_status(f"{mode_name.upper()} MODE LOCKED")

        if mode_name == "color":
            self.color_idx = (self.color_idx + 1) % len(COLOR_NAMES)
            _, name = self.get_active_color()
            self.set_status(f"Color -> {name} (LOCKED)")

    def unlock_mode(self):
        self.locked = False
        self.locked_mode = None
        self.mode = "draw"
        self.current_stroke = []
        self.prev_point = None
        self.set_status("UNLOCKED")

    # -----------------------------------------------------------
    def do_save(self):
        if is_canvas_empty(self.canvas):
            self.set_status("Nothing to save - canvas empty")
            return
        path = save_canvas(self.canvas)
        self.set_status(f"Saved: {path}")

    def do_clear(self):
        self.canvas = create_blank_canvas(FRAME_H, FRAME_W)
        self.current_stroke = []
        self.prev_point = None
        self.set_status("Canvas Cleared")

    def do_ai_snap(self):
        if len(self.current_stroke) < 5:
            self.set_status("Draw something first")
            return
        color, _ = self.get_active_color()
        success, shape_name, confidence = self.ai_snap.snap(
            self.canvas, self.current_stroke, color, self.brush_size
        )
        if success:
            self.set_status(f"AI Snap: {shape_name} ({confidence*100:.0f}%)")
        else:
            self.set_status("Shape not recognized confidently")
        self.current_stroke = []
        self.prev_point = None

    # -----------------------------------------------------------
    def handle_drawing(self, gesture, index_tip):
        """Drawing only happens in 'draw' mode (locked or in the
        unlocked default state when gesture == 'one')."""
        active_draw = (self.locked and self.mode == "draw") or \
                      (not self.locked and gesture == "one")

        if active_draw and index_tip is not None:
            smoothed = self.smoother.smooth(index_tip)
            color, _ = self.get_active_color()

            if self.prev_point is not None:
                cv2.line(self.canvas, self.prev_point, smoothed, color, self.brush_size)

            self.prev_point = smoothed
            self.current_stroke.append(smoothed)
        else:
            self.prev_point = None
            self.smoother.reset()

    # -----------------------------------------------------------
    def handle_snap_mode(self, gesture):
        """In snap mode (locked or gesture == 'two'), trigger AI snap once."""
        active_snap = (self.locked and self.mode == "snap") or \
                      (not self.locked and gesture == "two")
        if active_snap and self.current_stroke:
            self.do_ai_snap()

    # -----------------------------------------------------------
    def run(self):
        print("AI Air Drawing System started. Press Q to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame = cv2.flip(frame, 1)

            gesture, confidence, landmarks, index_tip = self.gesture_detector.detect(frame)

            # If locked, restrict effective gesture to only what's relevant,
            # ignoring any other stray detected gesture entirely.
            effective_gesture = gesture
            if self.locked:
                # while locked, only "five" (unlock) and "fist" (clear) matter
                if gesture not in ("five", "fist"):
                    effective_gesture = self.locked_mode  # keep current action going

            progress = self.handle_hold_logic(gesture)

            self.handle_drawing(effective_gesture, index_tip)
            self.handle_snap_mode(effective_gesture)

            # Frame composition
            display_frame = overlay_canvas_on_frame(frame, self.canvas)
            display_frame = self.gesture_detector.draw_landmarks(display_frame, landmarks)
            display_frame = draw_cursor(display_frame, index_tip, self.get_active_color()[0], self.brush_size)
            display_frame = draw_gesture_guide(display_frame)
            display_frame = draw_brush_size_indicator(display_frame, self.brush_size, self.get_active_color()[0], None)

            msg = self.status_message if time.time() < self.status_message_until else ""
            display_frame = draw_status_bar(
                display_frame, self.mode, self.locked,
                self.get_active_color()[1], self.brush_size, msg, confidence
            )

            if index_tip is not None and progress > 0:
                display_frame = draw_lock_progress(display_frame, progress, index_tip[0], index_tip[1])

            cv2.imshow("AI Air Drawing System", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                if not self.locked and self.mode in GESTURE_TO_MODE.values():
                    self.lock_mode(self.mode)
            elif key == ord('u'):
                if self.locked:
                    self.unlock_mode()
            elif key == ord('b'):
                self.brush_size = min(self.brush_size + BRUSH_STEP, MAX_BRUSH)
                self.set_status(f"Brush size: {self.brush_size}px")
            elif key == ord('n'):
                self.brush_size = max(self.brush_size - BRUSH_STEP, MIN_BRUSH)
                self.set_status(f"Brush size: {self.brush_size}px")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirDrawingApp()
    app.run()
