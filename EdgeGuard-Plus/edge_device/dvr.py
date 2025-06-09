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

    def save_clip_start(self):
        filename = os.path.join(self.output_dir, f"clip_{int(time.time())}.avi")
        if not self.buffer:
            return None, None

        height, width, _ = self.buffer[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))

        for frame in self.buffer:
            out.write(frame)

        return out, filename  # Return writer so more frames can be added
