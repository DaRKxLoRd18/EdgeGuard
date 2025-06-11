import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def get_clips(path, prefix):
    return sorted([clip for clip in os.listdir(path) if clip.startswith(prefix) and os.path.isdir(os.path.join(path, clip))])

def extract_sequences(root_path, split='Train', seq_len=12, resize=64, out_dir='data/processed'):
    input_dir = os.path.join(root_path, split)
    output_dir = os.path.join(out_dir, split.lower())
    os.makedirs(output_dir, exist_ok=True)

    prefix = 'Train' if split == 'Train' else 'Test'
    clip_dirs = get_clips(input_dir, prefix)

    for clip in tqdm(clip_dirs, desc=f'Processing {split} clips'):
        clip_path = os.path.join(input_dir, clip)
        frame_paths = sorted(glob(os.path.join(clip_path, '*.tif')))
        frames = []

        for frame_file in frame_paths:
            frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, (resize, resize))
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=-1)
            frames.append(frame)

        sequences = []
        for i in range(len(frames) - seq_len + 1):
            sequences.append(np.array(frames[i:i+seq_len]))

        sequences = np.array(sequences)
        np.save(os.path.join(output_dir, f"{clip}.npy"), sequences)
        print(f"✅ Saved {clip}: {sequences.shape} to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
                        help="Path to UCSDped2 folder (e.g., UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2)")
    parser.add_argument('--split', type=str, default='Train', choices=['Train', 'Test'])
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--resize', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='data/processed')

    args = parser.parse_args()

    extract_sequences(
        root_path=args.root_path,
        split=args.split,
        seq_len=args.seq_len,
        resize=args.resize,
        out_dir=args.out_dir
    )
