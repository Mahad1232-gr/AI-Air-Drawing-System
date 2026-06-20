"""
=====================================================================
 utils/smoothing.py
=====================================================================
 Purpose:
   Smooths fingertip movement using Holt's Double Exponential
   Smoothing so the drawn line doesn't jitter/shake from natural
   hand tremor or detection noise.
=====================================================================
"""


class HoltSmoother:
    def __init__(self, alpha=0.5, beta=0.3):
        """
        alpha: smoothing factor for the position (higher = less smoothing, more responsive)
        beta : smoothing factor for the trend/velocity
        """
        self.alpha = alpha
        self.beta = beta
        self.level = None      # smoothed position estimate
        self.trend = None      # smoothed velocity estimate

    def reset(self):
        self.level = None
        self.trend = None

    def smooth(self, point):
        """
        point: (x, y) tuple - the raw, noisy fingertip position
        returns: (x, y) tuple - the smoothed position
        """
        x, y = point

        if self.level is None:
            self.level = (x, y)
            self.trend = (0.0, 0.0)
            return self.level

        prev_level = self.level
        prev_trend = self.trend

        new_level_x = self.alpha * x + (1 - self.alpha) * (prev_level[0] + prev_trend[0])
        new_level_y = self.alpha * y + (1 - self.alpha) * (prev_level[1] + prev_trend[1])

        new_trend_x = self.beta * (new_level_x - prev_level[0]) + (1 - self.beta) * prev_trend[0]
        new_trend_y = self.beta * (new_level_y - prev_level[1]) + (1 - self.beta) * prev_trend[1]

        self.level = (new_level_x, new_level_y)
        self.trend = (new_trend_x, new_trend_y)

        return (int(self.level[0]), int(self.level[1]))
