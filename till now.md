# 🚨 EdgeGuard-Plus: Anomaly Detection Model Summary

## 🎯 Objective
Develop a robust **hybrid anomaly detection pipeline** for edge surveillance systems that:
- Detects unusual activities in video feeds using temporal reconstruction error.
- Integrates object detection for context-aware reasoning.
- Supports encrypted alerting and DVR-like anomaly capture.

---

## 🧠 Core Components

### 1. ConvLSTM Autoencoder (ONNX-based)
- **Purpose**: Learn normal temporal motion patterns via sequence reconstruction.
- **Input shape**: `(1, 12, 64, 64, 1)`
- **Output**: Reconstructed sequence, MSE used as anomaly score.
- **Exported to ONNX**: Converted from `.h5` to `.onnx` for efficient inference with `onnxruntime`.

### 2. YOLOv8 Integration (Ultralytics)
- **Purpose**: Detect known suspicious object classes (vehicles/persons).
- **Model**: `yolov8n.pt` (lightweight, real-time).
- **Classes of interest**:
  - Person (contextual filtering)
  - Car, Motorcycle, Bus, Truck (hardcoded object anomalies)
- **Used for**:
  - Cross-checking anomalies
  - Enabling context filtering (e.g., people in restricted hours)

---

## 🧪 Anomaly Detection Logic

### Hybrid Triggering Mechanism:
- Anomaly is flagged if:
  - 🔍 ConvLSTM Score > Threshold (`score > threshold * margin`)
  - OR
  - 🕵️ YOLO detects suspicious object (vehicles in restricted zones, or persons during restricted hours)

### ConvLSTM Threshold:
- F1-optimized threshold calculated over test data.
- Example global threshold: `0.00004017`
- Optional margin multiplier (`1.2`) added to adapt to runtime noise.

---

## 🖼️ Visual Output + Logging

- Live feed displayed in window with:
  - Overlayed anomaly score
  - Contextual time window label (e.g., "Context Allowed: ✅")
- Annotated video saved to:
  - `data/output_annotated.avi`

---

## 🎞️ DVR Buffer + Clip Saving

- Rolling buffer stores last 5 seconds (150 frames at 30 FPS).
- When anomaly is triggered:
  - Buffer is flushed and 5 seconds of future frames are saved.
  - Total ~10s clip saved as `clip_TIMESTAMP.avi` in `data/clips/`

---

## 🔐 Encrypted Metadata + Alerts

- Metadata (clip path, timestamp, zone, etc.) is:
  - AES-encrypted (256-bit CBC mode)
  - IV and ciphertext base64-encoded
- Sent to backend endpoint (e.g., `http://localhost:5000/api/alerts`) as JSON.
- Example metadata:
```json
{
  "timestamp": "2025-06-21 18:00:00",
  "clip_path": "data/clips/clip_1718992800.avi",
  "type": "reconstruction_anomaly",
  "location": "Zone A",
  "iv": "...",
  "ciphertext": "..."
}
