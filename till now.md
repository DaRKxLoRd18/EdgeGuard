- Created a mock model to detect anomaly.
- Capture.py : captures live feed from webcam, detects anomaly saves +=5 sec of clip around anomaly. Encrypts data using AES-256 and sends to backend to save data in mongoDB(local)
- created demo backend using nodejs(Just POST command) using chatgpt.
- Saved complete metadata in MongoDB.

### Model work 
- Downloaded and preprocessed raw data for model development and saved processed npy file in data folder. Check models/readme file for complete model development process.


# Model :
# 📊 EdgeGuard-Plus: Anomaly Detection Progress Log

## ✅ Week 1: Foundation + Model Development

### 🔹 Day 1: Dataset Setup + Preprocessing
- Downloaded and explored **UCSD Ped2** dataset.
- Used `preprocess.py` to generate `.npy` grayscale video tensors.
- Frame shape standardized to `(12, 64, 64, 1)` using sliding windows.
- Saved data and binary labels in: `data/processed/test/`.

### 🔹 Day 2: Contextual Anomaly Understanding
- Studied Section 2.2 of Chandola et al.
- Designed labeling logic with collective + contextual anomaly support.
- Decided binary score > threshold = anomaly.

### 🔹 Day 3: ConvLSTM Model Prototyping
- Built `ConvLSTM2D` autoencoder.
- Input: `(batch, 12, 64, 64, 1)`, Output: reconstructed sequence.
- Output is not classification, but anomaly **reconstruction error score**.

### 🔹 Day 4: Training Framework
- `train_model.py` uses:
  - `Adam(1e-4)`, `EarlyStopping`, `ModelCheckpoint`.
  - Binary F1, accuracy, AUC.
- Data augmentation: horizontal flip, blur, etc.

### 🔹 Day 5: Evaluation & Threshold Optimization
- Created `get_threshold_anomaly_detection.py`.
- Found best thresholds using:
  - `f1_score`, `95th percentile`, and `roc_auc_score`.
- Global threshold chosen: **~0.00004017** (Best F1 from all 12 videos).

### 🔹 Day 6: Export to ONNX
- Exported trained `.h5` model to `final_conv_lstm_ae.onnx`.
- Validated ONNX model using `onnxruntime`.

### 🔹 Day 7: DVR Pipeline Integration
- Built `capture.py`:
  - Reads from webcam or video.
  - Uses `DVRBuffer` to save clips with 5s pre-buffer + post frames.
  - Encrypts metadata with AES.
  - Sends encrypted alert via `sender.py`.

---

## ✅ Week 2: Advanced Features + YOLO Integration

### 🔹 Day 8: Real-time Deployment
- Successfully ran live anomaly detection on UCSD test videos.
- ONNX model achieved high F1 and AUC on evaluation.

### 🔹 YOLOv8 Integration (Replacement for YOLOv5)
- Switched to **Ultralytics YOLOv8** for lightweight object detection.
- Automatically downloads `yolov8n.pt`.
- Detects suspicious object classes: car, motorcycle, bus, truck.
- Uses confidence threshold: `0.4`.

### 🔹 Combined Detection Logic
- `detect.py` now includes both:
  - ConvLSTM score-based detection.
  - YOLOv8 object detection logic.
- If **either** detects anomaly → triggers DVR clip + alert.
- All detections logged with:
  - Class ID
  - Confidence
  - Bounding Box
- Annotated output video saved as `data/output_annotated.avi`.

### 🔹 YOLO Error Handling
- Fails gracefully if Ultralytics not installed.
- Keeps using ConvLSTM-only if YOLO fails or unavailable.

---

## 🔄 In Progress / Upcoming (Week 2)

- [ ] Day 9: Context engine (e.g., ignore known persons at night).
- [ ] Day 10: Multi-class labeling (optional).
- [ ] Day 11: Overlay scores on live stream (visual explainability).
- [ ] Day 12: Quantize + prune to `.tflite`.
- [ ] Day 13: Deploy on edge device (Jetson / Pi).
- [ ] Day 14: Final demo video + README polish.



## 📦 Current Folder Structure

```text
EdgeGuard-Plus/
├── edge_device/
│ ├── capture.py
│ ├── detect.py
│ ├── dvr.py
│ ├── encrypt.py
│ ├── sender.py
│ ├── yolov5/ (old, unused)
│ └── yolov8 (via ultralytics)
├── saved_model/
│ └── final_conv_lstm_ae.onnx
├── data/
│ ├── clips/
│ └── processed/
└── till_now.md

```

## 📌 Notes
- 🔐 Alerts use AES-encrypted metadata.
- 📡 Sending fails if backend API (localhost:5000) not running.
- 🎯 Anomalies = high reconstruction error **or** YOLO-detected suspicious object.

