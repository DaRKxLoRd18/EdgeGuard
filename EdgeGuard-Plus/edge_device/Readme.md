# 🚀 EdgeGuard++: Edge Device Anomaly Detection Pipeline

This README describes the **edge-only component** of the EdgeGuard++ system — a hybrid anomaly detection pipeline that runs on an edge device to detect suspicious events in real-time.

---

## 🧠 Features

- ✅ **ConvLSTM Autoencoder (ONNX)**: Detects motion anomalies using reconstruction error.
- ✅ **YOLOv8 Integration (optional)**: Flags vehicles/persons in restricted times or zones.
- ✅ **DVR Buffer**: Captures full 10-second anomaly clips (5s before and after).
- ✅ **AES Encryption**: Metadata is encrypted using 256-bit AES before being sent.
- ✅ **Preview GIFs**: Automatically generates animated `.gif` of anomaly moment.
- ✅ **Context-Aware Classification**: Tags alerts as `"vehicle_or_person"` or `"motion_anomaly"`.

---

## 🗂️ Folder Structure

```bash
edge_device/
├── capture.py              # Main loop (uses webcam or video)
├── detect.py               # ONNX + YOLO hybrid detection
├── dvr.py                  # Rolling frame buffer for DVR
├── encrypt.py              # AES encryption module
├── sender.py               # POST metadata to backend server
├── gif_generator.py        # Saves animated GIFs of anomalies
saved_model/
└── final_conv_lstm_ae.onnx # Trained ConvLSTM model
data/
├── clips/                  # Saved anomaly clips
└── previews/               # Saved preview GIFs
```

---

## 🧪 How It Works

1. Captures frames via webcam or video.
2. Buffers frames into a 5-second rolling window.
3. Runs **YOLOv8** and **ConvLSTM-AE**:
   - If YOLO finds a vehicle/person during restricted times → anomaly.
   - If ConvLSTM MSE > threshold → anomaly.
4. If anomaly is detected:
   - Saves full 10-second `.avi` clip.
   - Generates a `.gif` preview.
   - Encrypts metadata.
   - Sends POST to backend API.

---

## 🔐 Metadata Sent

```json
{
  "timestamp": "2025-06-21 18:00:00",
  "clip_path": "data/clips/clip_1718992800.avi",
  "gif_path": "data/previews/clip_1718992800.gif",
  "type": "motion_anomaly",
  "location": "Zone A",
  "iv": "…",
  "ciphertext": "…"
}
```

---

## ▶️ Running the Edge Pipeline

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run capture (use webcam or local video)
python edge_device/capture.py
```

> ℹ️ Modify the source video or model threshold in `capture.py` as needed.

---

## 📦 Model & Asset Expectations

- `saved_model/final_conv_lstm_ae.onnx` – Required ONNX model
- `yolov8n.pt` – Automatically downloaded via Ultralytics if YOLO is enabled

---

## ⚠️ Notes

- If no YOLO: ConvLSTM still works independently.
- Tested on: Python 3.8+, OpenCV 4.7+, ONNXRuntime 1.15+

---

<!-- ## 🙌 Author

Built by a CS undergrad as part of a job-ready smart surveillance project. -->
