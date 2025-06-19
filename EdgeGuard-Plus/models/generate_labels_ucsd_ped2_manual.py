# models/generate_labels_ucsd_ped2_manual.py

import os
import numpy as np

# Ground truth frame ranges (1-based, inclusive)
GT_FRAMES = {
    "Test001": list(range(61, 181)),
    "Test002": list(range(95, 181)),
    "Test003": list(range(1, 147)),
    "Test004": list(range(31, 181)),
    "Test005": list(range(1, 130)),
    "Test006": list(range(1, 160)),
    "Test007": list(range(46, 181)),
    "Test008": list(range(1, 181)),
    "Test009": list(range(1, 121)),
    "Test010": list(range(1, 151)),
    "Test011": list(range(1, 181)),
    "Test012": list(range(88, 181)),
}

def generate_labels(seq_file, anomaly_frames, seq_len=12):
    data = np.load(seq_file)
    num_seqs = data.shape[0]

    labels = []
    for i in range(num_seqs):
        frame_range = range(i + 1, i + seq_len + 1)  # 1-indexed
        label = int(any(f in anomaly_frames for f in frame_range))
        labels.append(label)

    return np.array(labels)

def process_all(test_dir):
    for clip_id, anomaly_frames in GT_FRAMES.items():
        npy_path = os.path.join(test_dir, f"{clip_id}.npy")
        label_path = os.path.join(test_dir, f"{clip_id}_labels.npy")

        labels = generate_labels(npy_path, anomaly_frames)
        np.save(label_path, labels)

        print(f"✅ {clip_id}: {labels.sum()} anomalies out of {len(labels)} sequences")

if __name__ == '__main__':
    test_dir = "data/processed/test"
    process_all(test_dir)
