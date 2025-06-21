import os
import time
import imageio
import cv2

def save_gif_from_frames(frames, output_dir="data/previews", fps=10):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    gif_path = os.path.join(output_dir, f"clip_{timestamp}.gif")

    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(gif_path, rgb_frames, fps=fps)

    return gif_path
