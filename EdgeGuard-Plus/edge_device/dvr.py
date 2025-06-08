# Placeholder for dvr.py
import cv2
import os
from collections import deque
import time

class DVRBuffer:
    def __init__(self, fps=30, buffer_seconds=5):
        self.buffer = deque(maxlen=fps * buffer_seconds)
        self.fps = fps
        self.output_dir = "data/clips"
        os.makedirs(self.output_dir, exist_ok=True)

    def add_frame(self, frame):
        self.buffer.append(frame.copy())

    def save_clip(self, post_frames=150):  # 5s after = 150 frames at 30fps
        filename = os.path.join(self.output_dir, f"clip_{int(time.time())}.avi")
        if not self.buffer:
            return None

        height, width, _ = self.buffer[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))

        # Write pre-anomaly frames
        for frame in self.buffer:
            out.write(frame)

        # Record post-anomaly frames live
        cap = cv2.VideoCapture(0)
        for _ in range(post_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        return filename
