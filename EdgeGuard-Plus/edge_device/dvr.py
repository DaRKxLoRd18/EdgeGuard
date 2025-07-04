# dvr.py
import os
import cv2
from collections import deque
import time

class DVRBuffer:
    def __init__(self, fps=30, buffer_seconds=5):
        self.fps = fps
        self.buffer_size = fps * buffer_seconds
        self.buffer = deque(maxlen=self.buffer_size)

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save_clip_start(self):
        timestamp = int(time.time())
        filename = f"data/clips/clip_{timestamp}.avi"
        if len(self.buffer) == 0:
            return None, None

        height, width = self.buffer[0].shape[:2]
        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.fps,
            (width, height)
        )

        # ✅ Safe copy to avoid RuntimeError from threading
        safe_copy = list(self.buffer)
        for frame in safe_copy:
            writer.write(frame)

        return writer, filename
