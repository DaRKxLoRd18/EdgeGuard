
# Two-Week Beast Mode Roadmap

## 🗓️ Week 1: Foundation + Model Development

### ✅ Day 1: Dataset Setup + Preprocessing
- Download & explore UCSD Ped1/Ped2 or ShanghaiTech Campus dataset
- Use `preprocess.py` to convert videos to grayscale sequence tensors
  - Frame shape: `(12, 64, 64, 1)` for temporal window
- Save numpy sequences with normal/anomaly labels

### ✅ Day 2: Literature & Contextual Design
- Read Section 2.2 of Chandola et al. paper: focus on contextual + collective anomalies
- Design anomaly labeling logic to support:
  - Temporal continuity
  - Location-based or object-centric outliers
- Define output format: binary or score > 0.5 = anomaly

### ✅ Day 3: Model Prototyping
- Build baseline `ConvLSTM2D` model (see README)
- Explore alternatives:
  - `TimeDistributed(CNN) + LSTM`
  - Lightweight backbones: `MobileNetV2`, `EfficientNet-B0`
- Input shape: `(batch, 12, 64, 64, 1)`
- Output layer: `Dense(1, activation='sigmoid')`

### ✅ Day 4: Training Framework
- Expand `train_model.py`:
  - Use `EarlyStopping`, `ModelCheckpoint`
  - Optimizer: `Adam`, Learning Rate: `1e-4`
  - Metrics: binary accuracy, F1, AUC
- Validate on held-out sequences
- Add data augmentation: flip, rotate, blur

### ✅ Day 5: Model Tuning & Evaluation
- Experiment with temporal depth: 8, 12, 16 frames
- Add visualizations:
  - Loss curves
  - Anomaly score timeline
- Save best `.h5` model

### ✅ Day 6: Export & Runtime Optimization
- `model_export.py`:
  - Convert to `.onnx` using `tf2onnx` or `keras2onnx`
  - Validate with `onnxruntime`
  - Optionally export `.tflite`

### ✅ Day 7: Dry Run on DVR Pipeline
- Integrate ONNX model in `detect.py`
- Feed sequences from `DVRBuffer`
- Trigger encrypted alert if score > 0.5
- Save `.avi` of anomaly clip (5s before & after)

---

## 🗓️ Week 2: Advanced Features + Context Engine

### ✅ Day 8: Hard Negative Mining
- Use near-anomaly normal clips (false positives) to retrain model
- Boost precision while maintaining recall

### ✅ Day 9: Auto-Label Context Data
- Create `context_filter.py`: time-of-day, location ID, object type
- Filter out contextually normal anomalies
  - E.g., authorized person in office at night = normal

### ✅ Day 10: Multi-class Extension (Optional)
- Label anomalies by type: vehicle, intrusion, abandoned object
- Modify output to `Dense(n, activation='softmax')`

### ✅ Day 11: Visualization + Explainability
- Overlay anomaly score on live video frames
- If using attention layers, save anomaly heatmap

### ✅ Day 12: Model Slimming + TFLite
- Apply:
  - Quantization (float16, int8)
  - Pruning (TF Model Optimization Toolkit)
- Export smallest `.tflite` model

### ✅ Day 13: Stress Test on Edge
- Deploy on Raspberry Pi or Jetson Nano
- Benchmark:
  - FPS
  - Memory usage
  - Latency
- Monitor with `psutil` + logs

### ✅ Day 14: Final Integration + GitHub Polish
- Organize:
  - `models/`, `scripts/`, `dvr/`, `server/`
- Finalize:
  - `README.md`
  - `requirements.txt`
  - Deployment guide
  - Short demo video of full pipeline

---

## 📚 Bonus from Chandola Survey
- Use contextual anomaly formulation: combine spatial + temporal metadata
- Consider semi-supervised learning with only normal labels
- Use sliding window for collective anomalies (e.g., repeated frames)
- Evaluate with Precision-Recall AUC, not just accuracy

---

## 🚀 Stretch Goals (Post 2 Weeks)
- Train autoencoder-based anomaly model as baseline
- Add visual explainability: Grad-CAM, Integrated Gradients
- Deploy drone alerts for anomaly detection in smart security zones
