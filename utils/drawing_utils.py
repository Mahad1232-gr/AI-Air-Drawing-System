import cv2
import numpy as np
import os
from datetime import datetime

COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Purple": (255, 0, 255),
    "Cyan": (255, 255, 0),
}
COLOR_NAMES = list(COLORS.keys())


def create_blank_canvas(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)


def is_canvas_empty(canvas, threshold=50):
    return np.count_nonzero(canvas) < threshold


def save_canvas(canvas, save_dir="saved_drawings"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"drawing_{timestamp}.png")
    cv2.imwrite(filename, canvas)
    return filename


def overlay_canvas_on_frame(frame, canvas):
    canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
    canvas_gray = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    mask = mask.astype('uint8')
    result = frame.copy()
    for c in range(3):
        result[:, :, c] = np.where(mask > 0, canvas_resized[:, :, c], frame[:, :, c])
    return result


def draw_gesture_guide(frame):
    """Right side pe vertical gesture guide panel"""
    h, w = frame.shape[:2]
    panel_w = 210
    x_start = w - panel_w - 10

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 8, 60), (w - 5, 60 + 185), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Title
    cv2.putText(frame, "GESTURE GUIDE", (x_start, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Divider line
    cv2.line(frame, (x_start - 5, 88), (w - 8, 88), (0, 255, 255), 1)

    guide = [
        ("FIST", "Clear Canvas", (0, 100, 255)),
        ("1 Finger", "Draw", (0, 255, 0)),
        ("2 Fingers", "AI Snap", (255, 200, 0)),
        ("3 Fingers", "Color Change", (255, 0, 200)),
        ("4 Fingers", "Save", (0, 200, 255)),
        ("5 Fingers", "Stop/Unlock", (200, 200, 200)),
    ]

    for i, (gesture, action, color) in enumerate(guide):
        y = 108 + i * 22
        cv2.putText(frame, f"{gesture}", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"= {action}", (x_start + 75, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Keyboard shortcuts
    cv2.line(frame, (x_start - 5, 245), (w - 8, 245), (100, 100, 100), 1)
    cv2.putText(frame, "B/N = Brush +/-  L=Lock  U=Unlock", (x_start - 5, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def draw_status_bar(frame, mode, locked, color_name, brush_size, message="", confidence=None):
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # Colored left accent bar
    mode_colors = {
        "draw": (0, 255, 0),
        "snap": (255, 200, 0),
        "color": (255, 0, 200),
        "save": (0, 200, 255),
        "stop": (200, 200, 200),
    }
    accent = mode_colors.get(mode, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (5, 55), accent, -1)

    # MODE
    cv2.putText(frame, "MODE:", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    cv2.putText(frame, mode.upper(), (70, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, accent, 2, cv2.LINE_AA)

    # LOCK STATUS
    lock_text = "🔒 LOCKED" if locked else "🔓 UNLOCKED"
    lock_color = (0, 0, 255) if locked else (0, 255, 100)
    cv2.putText(frame, "LOCKED" if locked else "UNLOCKED", (200, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 2, cv2.LINE_AA)

    # COLOR dot + name
    dot_color = COLORS.get(color_name, (255, 255, 255))
    cv2.circle(frame, (370, 16), 10, dot_color, -1)
    cv2.circle(frame, (370, 16), 10, (255, 255, 255), 1)
    cv2.putText(frame, color_name, (388, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot_color, 1, cv2.LINE_AA)

    # BRUSH SIZE with visual bar
    cv2.putText(frame, f"Brush:", (510, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, str(brush_size), (572, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    # brush size bar
    bar_max = 30
    bar_w = int((brush_size / bar_max) * 80)
    cv2.rectangle(frame, (600, 10), (680, 22), (50, 50, 50), -1)
    cv2.rectangle(frame, (600, 10), (600 + bar_w, 22), accent, -1)

    # CONFIDENCE
    if confidence is not None:
        conf_pct = int(confidence * 100)
        conf_color = (0, 255, 0) if conf_pct > 70 else (0, 200, 255) if conf_pct > 40 else (0, 0, 255)
        cv2.putText(frame, f"Conf: {conf_pct}%", (710, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1, cv2.LINE_AA)

    # Bottom message bar
    if message:
        cv2.rectangle(frame, (0, 55), (w, 82), (20, 20, 40), -1)
        cv2.putText(frame, message, (15, 74), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_brush_size_indicator(frame, brush_size, color, position):
    """Right side pe brush size scale"""
    h, w = frame.shape[:2]
    x = w - 225
    y_start = 290
    
    cv2.putText(frame, "BRUSH", (x, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Scale markings
    sizes = [2, 5, 10, 15, 20, 25, 30]
    bar_h = 120
    for i, s in enumerate(sizes):
        y = y_start + 15 + int((s / 30) * bar_h)
        line_w = 8 + int((s / 30) * 20)
        thickness = max(1, int(s / 6))
        line_color = color if s == brush_size else (80, 80, 80)
        cv2.line(frame, (x, y), (x + line_w, y), line_color, thickness)
        if s == brush_size:
            cv2.putText(frame, str(s), (x + line_w + 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # B/N hint
    cv2.putText(frame, "B+ N-", (x, y_start + bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    return frame


def draw_lock_progress(frame, progress, x, y, radius=25):
    if progress <= 0:
        return frame
    angle = int(360 * min(progress, 1.0))
    cv2.ellipse(frame, (x, y), (radius, radius), -90, 0, angle, (0, 255, 255), 4)
    pct = int(progress * 100)
    cv2.putText(frame, f"{pct}%", (x - 15, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return frame


def draw_cursor(frame, position, color, brush_size):
    if position is not None:
        radius = max(brush_size // 2 + 2, 6)
        cv2.circle(frame, position, radius, color, 2)
        cv2.circle(frame, position, 2, (255, 255, 255), -1)
    return frame


def draw_ai_snap_result(frame, shape_name, confidence):
    """AI snap result popup"""
    h, w = frame.shape[:2]
    msg = f"AI SNAP: {shape_name.upper()} ({int(confidence*100)}%)"
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 180, h//2 - 30), (w//2 + 180, h//2 + 30), (0, 30, 60), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    cv2.putText(frame, msg, (w//2 - 170, h//2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
