# 🧠 EdgeGuard++ Model Training: Real-Time Context-Aware Anomaly Detector

This module focuses on building the **Tier-1 anomaly detection model** for EdgeGuard++ — a lightweight, high-accuracy, real-time model based on CNN + LSTM (or ConvLSTM2D), exportable to ONNX or TFLite and deployable on edge devices.

---

## 📌 Goal

- Train a generic anomaly detection model that detects *anything unusual* in frame sequences.
- Integrate with a lightweight **context-aware reasoning engine** (Tier-2).
- Run in real-time on webcam video feed using only CPU/GPU.

---

## 🧱 Model Architecture

- **Input:** Sequence of 12–16 grayscale frames (64×64 or 96×96)
- **Backbone:** CNN (e.g., MobileNetV2 or Conv2D)
- **Temporal Modeling:** LSTM or ConvLSTM2D
- **Output:** Binary (normal vs anomaly)

---

## 📦 Folder Structure

```
models/
├── preprocess.py          # Extract sequences from raw videos
├── cnn_lstm_model.py      # Model builder
├── train_model.py         # Training + validation
├── model_export.py        # Convert to ONNX/TFLite
├── saved_model/
│   └── generic_anomaly.onnx  # Final exported model
```

---

## 📚 Dataset Options

- UCSD Ped1/Ped2: [http://www.svcl.ucsd.edu/projects/anomaly/](http://www.svcl.ucsd.edu/projects/anomaly/)
- ShanghaiTech Campus
- S-UCF-Crime (cropped videos)

All videos are preprocessed into sequence tensors using `preprocess.py`.

---

## 🚀 Training Instructions

1. Install dependencies:
```bash
pip install tensorflow numpy opencv-python scikit-learn
```

2. Run preprocessing:
```bash
python models/preprocess.py --input_dir data/raw --output_dir data/processed
```

3. Train the model:
```bash
python models/train_model.py
```

4. Export to ONNX:
```bash
python models/model_export.py
```

---

## 🧪 Sample Model (ConvLSTM2D-based)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout

def build_model(input_shape=(12, 64, 64, 1)):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu',
                         return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model
```

---

## 🧠 Output Format (Prediction)

The model outputs a float score `0.0–1.0`:
- `> 0.5` → likely anomaly
- `≤ 0.5` → normal

---

## 🛠 Integration in Edge Pipeline

Used in `detect.py` on edge device as Tier-1 model.  
Sequence of frames is maintained and passed through the ONNX model to return anomaly score.

---

## 📁 Model Exports

- ✅ `.h5` → for local testing
- ✅ `.onnx` → for edge inference using `onnxruntime`
- ✅ `.tflite` (optional) → for mobile devices or TFLite micro

---

## 📌 Next Steps

- [ ] Fine-tune on context-specific videos
- [ ] Plug into `context_filter.py` for Tier-2 scoring
- [ ] Improve recall by augmenting normal sequences

---

## 🙌 Author

This module is part of the [EdgeGuard++] project



































<!-- [Edge Device Pipeline]
┌────────────────────────────────────────────────────────────────┐
│ 1. Video Capture & Preprocessing                               │
│    - Continuous capture via OpenCV, maintain circular buffer   │
│    - Preprocess frames: resizing, normalization                 │
│    - Scene calibration: load zone maps, camera parameters      │
│                                                                │
│ 2. Context Extraction                                          │
│    - Time-based context: current time-of-day, day-of-week      │
│    - Scene-based context: zones definitions (e.g., “entrance”,│
│      “aisle”, “restricted area”)                               │
│    - User-specified anomaly interest parsed into structured    │
│      config: target classes, zone/time constraints             │
│    - Additional sensors (optional): door sensors, IoT signals  │
│                                                                │
│ 3. Feature & Event Detection                                   │
│    - Object Detection module: e.g., lightweight YOLO/TinyYOLO  │
│      or MobileNet-SSD for on-device detection of classes       │
│    - Semantic Segmentation (optional): lightweight models to   │
│      delineate zones                                       │
│    - Motion analysis: optical flow or frame differencing for   │
│      movement patterns                                        │
│    - Temporal/event segmentation: group consecutive frames into│
│      candidate events based on motion or object presence      │
│                                                                │
│ 4. Context-Conditioned Anomaly Scoring                         │
│    - For each event segment, compute spatiotemporal embeddings │
│      via CNN+LSTM or 3D-CNN or lightweight video transformer,  │
│      trained to model “normal” patterns for given context      │
│      (e.g., learned via contrastive learning aligning context/ │
│      appearance/motion) :contentReference[oaicite:5]{index=5}                          │
│    - Alternatively, apply rule-based checks based on parsed    │
│      user config: e.g., if “person in restricted zone after    │
│      hours” → immediate alert                                     │
│    - For more nuanced anomalies (e.g., “vehicle behaving      │
│      unusually”), compare features against context-conditioned │
│      normal distribution; high deviation → anomaly            │
│    - Use thresholding calibrated per context cluster           │
│                                                                │
│ 5. DVR Buffering & Clip Extraction                             │
│    - On anomaly detection: save pre-buffer (e.g., 5s before),  │
│      during, and post-buffer (e.g., 5s after) as a clip (.avi) │
│    - Store locally or upload encrypted clip (depending on      │
│      privacy settings)                                         │
│                                                                │
│ 6. Metadata Encryption & Transmission                          │
│    - Prepare JSON metadata: timestamp, context tags, object    │
│      attributes, anomaly score, event ID                       │
│    - AES-256 encrypt metadata payload via PyCryptodome         │
│    - Send encrypted metadata (and optionally small clip or clip│
│      reference) via HTTPS/WebSocket to Node.js backend         │
│                                                                │
│ 7. Local Feedback Loop & Adaptation                            │
│    - Provide user option (via dashboard) to label false alarms │
│      or confirm true alerts                                    │
│    - On-device fine-tuning: accumulate labeled samples of      │
│      “normal”/“anomaly” for context; periodically update local │
│      model (small-scale) or send anonymized gradients for      │
│      federated aggregation                                    │
└────────────────────────────────────────────────────────────────┘

[Backend & Dashboard]
- Node.js + Express: receive encrypted metadata, store in MongoDB
- WebSocket: push alerts to React dashboard in real time
- Dashboard UI: decode metadata client-side if appropriate, display alert list, replay clips, allow user feedback (e.g., mark false positives)
- Analytics: aggregate anomaly statistics per zone/time/context for model improvement guidance

[Federated Learning Server]
- Collect encrypted model updates (gradients) from edge devices
- Aggregate updates to improve global context-conditioned model
- Distribute updated model weights to devices periodically

[Model Training & Export]
- Offline training pipeline (in models/): 
  - Pretrain base spatiotemporal feature extractor on large-scale video datasets
  - Use contrastive learning (e.g., Trinity-like) to align context vectors (time, zone label) with appearance/motion embeddings :contentReference[oaicite:6]{index=6}
  - For known anomaly classes (if labeled data exists), fine-tune detectors or one-class classifiers per class
  - Export lightweight versions (e.g., TensorFlow Lite, ONNX) for edge inference
- Provide mechanism for incremental fine-tuning based on on-device feedback

---

## Detailed Steps for Context-Aware Model Development

### 1. Define and Structure Context

- **Identify context dimensions** relevant to target deployments:  
  - **Temporal context**: time-of-day slots (e.g., business hours vs. after hours), day-of-week (weekday/weekend), seasonal events.  
  - **Spatial context**: zone maps defined by the user (e.g., entrance area, cashier zone, prohibited zone). Represent via coordinates/polygons in config.json.  
  - **Operational context**: schedule (e.g., opening/closing times), special events (sales, holidays).  
  - **User-specific interest**: natural-language descriptions parsed into structured rules or parameters (object classes of interest, behavior patterns).  

- **Encoding context**:  
  - Convert temporal context into embeddings or discrete tokens (e.g., one-hot for time slots or sinusoidal time encodings).  
  - Represent zone IDs or labels as embeddings.  
  - Combine into a context vector appended as input to the anomaly detection model or used in contrastive learning.  

### 2. Data Collection and Normal Pattern Learning

- **Collect long-term “normal” videos per deployment**:  
  - Record for weeks/months to capture variations (day/night, weekdays/weekends).  
  - Annotate context: tag segments by time slot or special events.  
- **Unsupervised/weakly-supervised learning**:  
  - Train a context-conditioned autoencoder or predictive model: input is video segment + context vector; model learns to reconstruct or predict next frames under that context.  
  - Use contrastive learning: treat segments from the same context as positive pairs; segments from different contexts as negatives; learn embeddings where alignment between context and features is maximized for normal behavior :contentReference[oaicite:7]{index=7}.  
- **Clustering normal patterns**:  
  - In embedding space, cluster patterns per context; record cluster centers and distribution (e.g., via Gaussian mixture) to compute distance-based anomaly scores.  

### 3. Anomaly Scoring and Detection

- **Reconstruction/Predictive error**: High error under current context indicates potential anomaly.  
- **Embedding distance**: Distance from context-conditioned normal clusters signals anomaly.  
- **Object- and event-based checks**:  
  - Use object detector to identify entity classes. For user-specified classes, check presence/absence or counts in zones. Example: “vehicle in pedestrian zone” triggers anomaly if a vehicle bounding box is wholly or partly within the polygon of a pedestrian-only zone.  
  - Behavior-based checks: speed, trajectory deviation (e.g., a person loitering too long in restricted zone), group patterns (e.g., crowd formation at odd times).  
- **Hybrid scoring**: Combine ML-based anomaly score (normalized) with rule-based binary flags; weight according to configuration.

### 4. User-Specified Anomaly Interests

- **Natural Language Interface**:  
  - Provide a UI where user enters description, e.g., “Alert me if a person enters after 10 PM,” or “Detect vehicles moving against traffic flow in delivery bay.”  
  - **Parse** via simple NLP pipeline:  
    - Extract object classes (“person,” “vehicle”), conditions (“after 10 PM,” “in delivery bay zone”), behaviors (“moving against flow,” “loitering > 30s”).  
    - Map extracted terms to internal parameters: class IDs from object detection model, zone IDs from config, time thresholds, behavior templates.  
- **Configuration Generation**:  
  - Create or update a JSON rule set:  
    ```json
    {
      "anomaly_rules": [
        {
          "object": "person",
          "zone": "entrance",
          "time_after": "22:00",
          "action": "alert"
        },
        {
          "object": "vehicle",
          "zone": "delivery_bay",
          "behavior": "against_flow",
          "action": "alert"
        }
      ]
    }
    ```
  - For complex behaviors not easily rule-defined, trigger adaptation: e.g., collect a few labeled video snippets illustrating the behavior; use few-shot fine-tuning module.  

### 5. Model Architecture & Training

- **Base spatiotemporal backbone**:  
  - Options:  
    - **CNN+LSTM**: frame-level CNN to extract spatial features, followed by LSTM for temporal patterns.  
    - **3D-CNN**: captures spatiotemporal features directly (e.g., lightweight I3D-like).  
    - **Video Transformer**: ViT-based temporal modeling (if compute permits).  
  - **Lightweight variants**: MobileNet3D or Tiny video models for edge inference.  

- **Context conditioning**:  
  - **Concatenate context vector** (time embedding, zone embedding) with features at a chosen layer; or  
  - **Adaptive normalization**: e.g., FiLM layers where context vector modulates feature maps via scaling and shifting.  
  - **Contrastive pretraining**: as in Trinity, train embeddings such that appearance+motion features align with context embeddings for normal segments :contentReference[oaicite:8]{index=8}.  

- **Anomaly head**:  
  - For unsupervised: reconstruction/predictive subnetwork or one-class classification head (e.g., distance to cluster means).  
  - For supervised (if labeled anomalies available): binary/multiclass classification head, possibly lightweight MLP on top of features.  
  - For hybrid: separate modules for rule-based detection that override or boost anomaly score when triggered.  

- **Training pipeline**:  
  - **Offline phase**: Pretrain on large public video datasets to learn general spatiotemporal features.  
  - **Context fine-tuning**: Use deployment-specific normal data with annotated contexts to train context-conditioned normalcy model.  
  - **Few-shot anomaly adaptation**: If user provides sample anomalous events, fine-tune or calibrate thresholds.  
  - **Federated aggregation**: When multiple edge devices share models, aggregate updates without raw data transfer.  

### 6. Edge Inference and Efficiency

- **Model optimization**: Convert to TensorFlow Lite or ONNX with quantization (e.g., 8-bit) to reduce footprint.  
- **Frame sampling**: Process frames at moderate rate (e.g., 5-10 FPS) to balance detection fidelity vs. compute.  
- **Event-triggered processing**: Use lightweight motion detection to trigger heavier analysis only when there is activity.  
- **Memory buffering**: Maintain circular buffer of frames (pre-event) to enable DVR saving.  
- **Asynchronous pipelines**: Separate threads/processes for capture, inference, buffering, encryption/transmission to utilize multi-core and avoid blocking capture.  

### 7. Feedback Loop and Continuous Learning

- **User feedback UI**: Dashboard allows marking false positives/negatives.  
- **Local adaptation**: On-device keep a small buffer of labeled segments; periodically fine-tune anomaly thresholds or small network layers to adjust to new patterns.  
- **Federated updates**: Send encrypted gradient updates (or model diffs) to central server for aggregation, then distribute improved global model back to devices. This helps generalize context-aware patterns across deployments while preserving privacy.  

### 8. Evaluation Metrics

- **Standard VAD metrics**: Frame-level and event-level AUC, precision-recall curves on labeled test data.  
- **Context-aware metrics**: Evaluate separately across different contexts (e.g., business hours vs. after hours) to ensure model adapts correctly.  
- **User-centric metrics**: False positive rate in production scenarios (e.g., how many benign events flagged), time-to-alert latency, user satisfaction feedback.  
- **Resource metrics**: On-device CPU/GPU/memory usage, inference latency, battery usage (for mobile/embedded devices).  

---

## Implementation Roadmap

1. **Prototype Phase**  
   - Build a minimal edge pipeline: capture + motion-triggered object detection + simple rule-based alerts (e.g., person after hours).  
   - Develop UI to define zones and simple rules.  
   - Validate DVR buffering and metadata encryption/transmission.  
   - Integrate with backend & dashboard to view alerts and feedback.

2. **Context-Conditioned Model Development**  
   - Collect normal data for a pilot environment (e.g., an office corridor).  
   - Implement context embedding (time, zone) and train a small-scale context-conditioned autoencoder or contrastive model using local compute or cloud.  
   - Deploy lightweight inference on edge and evaluate anomaly detection on held-out events.

3. **User-Specified Interest Parsing**  
   - Design and implement NLP parsing module (could start with rule-based keyword extraction; later integrate a small LLM locally or via backend for richer parsing).  
   - Map parsed inputs to configuration schema and integrate with detection pipeline to adjust rules or model parameters.

4. **Advanced Spatiotemporal Learning**  
   - Experiment with different backbone choices (CNN+LSTM vs. 3D-CNN vs. transformer) prioritized by edge constraints.  
   - Implement contrastive pretraining aligning features with context. Consider using off-the-shelf code from related research (Trinity-like).  
   - Evaluate performance improvements in reducing false alarms across varying contexts.

5. **Feedback Loop Integration**  
   - Create dashboard features for marking alerts as false/true.  
   - Implement local data collection of feedback-labeled segments; design small-scale retraining scripts to update thresholds or fine-tune layers.  
   - Set up federated learning server and integrate secure update flows.

6. **Scalability and Robustness Testing**  
   - Stress-test for multiple anomaly types, variable lighting, camera shakes, occlusion.  
   - Evaluate under different contexts: weekends, holidays, special events.  
   - Monitor resource usage; profile and optimize model size, sampling rates, threading.

7. **Documentation and Packaging**  
   - Document configuration formats, user guide for specifying anomaly interests, deployment instructions for edge device.  
   - Containerize components: Docker for backend, edge Dockerfile for inference environment.  
   - CI/CD: automate tests for model inference, data pipelines, encryption workflows.

8. **Pilot Deployment and Iteration**  
   - Deploy in sample real-world settings (e.g., small retail store after hours).  
   - Gather real usage feedback, refine context definitions, improve parsing accuracy.  
   - Iterate model improvements based on diverse environments (e.g., homes, warehouses, hospitals, vehicles).

---

## Example: Context-Conditioned Contrastive Pretraining (Sketch)

1. **Data Preparation**: Organize normal video segments with associated context labels (e.g., segment A: “weekday morning,” “zone: lobby”; segment B: “weekday evening,” “zone: lobby”).
2. **Context Embeddings**: Encode each context attribute:  
   - Time-of-day: embed as continuous periodic features or discrete one-hot bins.  
   - Zone ID: embed via learnable embedding vector.  
   - Combine into a context vector `c`.
3. **Feature Extractor**: Use a lightweight 3D-CNN to extract spatiotemporal features `f` from each segment.
4. **Contrastive Loss**:  
   - For segments `(f_i, c_i)` and `(f_j, c_j)`: if contexts match (same time-zone combination), treat as positive pair; if contexts differ, negative pair.  
   - Use a contrastive loss (e.g., InfoNCE) to pull together embeddings of matched pairs and push apart mismatched pairs.  
   - This ensures `f` captures patterns aligned with context `c`.  
5. **Anomaly Scoring**: At inference, given new segment with context `c_new`, compute `f_new` and measure its alignment score with learned normal clusters for that context. Low alignment → anomaly.

This procedure follows ideas from Trinity-style algorithms tailored to long-term context-aware VAD :contentReference[oaicite:9]{index=9}.

---

## Handling User-Defined Anomaly Types

- **Simple Cases (Rule-Based)**: For “object presence/absence” or basic conditions (time, zone), directly use object detection outputs + geometric checks.  
- **Behavioral Patterns**: For more complex behaviors (e.g., “vehicle driving against flow,” “loitering,” “fall detection”), implement specialized modules:  
  - **Trajectory analysis**: Track object centroids over frames; detect direction deviations relative to permitted flows (requires mapping permitted paths in zone).  
  - **Temporal analysis**: Measure dwell time in zone; flag if exceeds threshold.  
  - **Pose estimation**: For fall detection in hospitals, use pose estimation lightweight model to detect abnormal human poses.  
- **Few-Shot Adaptation**: If user provides example clips illustrating the anomaly:  
  - Use a small fine-tuning routine to adapt the anomaly head or thresholds: e.g., train a one-class SVM on embeddings augmented with these examples, or fine-tune a lightweight classification layer.  
  - This may require a small annotation UI in the dashboard for uploading/labeling clips.  
- **NLP Parsing**: Design a mapping table of keywords to parameters (e.g., “after closing” → time_after config). For more flexibility, consider integrating a small LLM service in backend to parse freeform user descriptions into structured config.

---

## Privacy, Security, and Edge Considerations

- **On-Device Inference**: Raw video never leaves device; only encrypted metadata or anonymized features are transmitted.  
- **Encryption**: Use AES-256 to encrypt metadata payloads. If transmitting clips, either avoid sending raw video or encrypt video segments before upload.  
- **Federated Learning**: Utilize frameworks (Flower, PySyft) to aggregate model updates without sharing raw data, preserving user privacy across deployments.  
- **Resource Constraints**: Continuously profile CPU/GPU/memory; use model pruning, quantization; dynamically adjust frame rates or processing schedules based on load.  
- **Fail-Safe Mechanisms**: In case of connectivity loss, store encrypted metadata locally and sync when connection restored; ensure buffer management avoids overflow.  
- **Security**: Secure the backend API (e.g., JWT-based authentication), secure WebSocket channels with TLS; ensure proper environment variable management for keys in .env; enforce least privilege for database access.

---

## Evaluation and Iteration

- **Simulated Scenarios**: Create synthetic test cases for different anomaly definitions (e.g., simulate person entering after hours) to validate rule-based logic.  
- **Pilot Data Collection**: Deploy at a few sites, collect anonymous metadata and user feedback to measure detection accuracy, false alarm rates in different contexts.  
- **A/B Testing of Models**: Compare baseline (no context) vs. context-conditioned model vs. hybrid rule+ML approach to quantify gains in precision/recall across contexts.  
- **Continuous Monitoring**: Use dashboard analytics to surface patterns of false positives tied to specific contexts; refine model or rules accordingly.  
- **Model Retraining Schedule**: Periodically retrain global model with aggregated insights; distribute updated models to edge. For highly dynamic environments, consider more frequent updates.

---

## Practical Tips

- **Start Small**: Begin with a clear use-case (e.g., after-hours intrusion detection) to validate the pipeline before adding more complex context dimensions and anomaly types.  
- **Modular Design**: Build detection modules (object detection, trajectory analysis, pose estimation) as separate components that can be composed per user config.  
- **Lightweight Models**: Use proven lightweight architectures (e.g., MobileNet variants, TinyYOLO) for on-device inference; reserve heavier processing for selective segments or backend processing if needed.  
- **Automate Context Labeling**: For initial training, automate tagging of normal video segments by time slots and zones.  
- **Logging and Observability**: Log inference latency, resource usage, anomaly scores distribution to detect drift or performance degradation.  
- **User Experience**: In the dashboard, provide intuitive interfaces for zone drawing (e.g., interactive map overlay), time schedule inputs (calendar picker), and anomaly rule editing. Clear visualization of past events helps users trust and refine the system.

---

## Roadmap Example (2-Week Sprints)

1. **Sprint 1: Baseline Pipeline**  
   - Implement basic capture + object detection + simple rule (e.g., person after hours) + DVR saving + metadata encryption/transmission.  
   - Build minimal dashboard to view alerts and clips.

2. **Sprint 2: Context Embedding & Normal Modeling**  
   - Collect normal data, implement context embedding (time, zone).  
   - Train simple autoencoder or distance-based model on-device or cloud.  
   - Deploy context-conditioned anomaly scoring and compare with baseline; adjust thresholds.

3. **Sprint 3: User Interest Parsing & Configuration UI**  
   - Develop UI for user to specify anomalies; implement parsing to structured config.  
   - Integrate config into detection pipeline; test multiple rule scenarios.

4. **Sprint 4: Advanced Feature Modules**  
   - Add trajectory-based behavior detection (e.g., loitering, wrong direction) and pose estimation if needed.  
   - Integrate these modules conditionally based on user config.

5. **Sprint 5: Feedback Loop & Adaptation**  
   - Develop dashboard feedback marking; collect labeled events; implement local threshold adaptation or lightweight fine-tuning.  
   - Prototype federated update flow (simulated multi-device).

6. **Sprint 6: Optimization & Hardening**  
   - Profile and optimize inference speed, memory; quantize models.  
   - Harden security for metadata transmission and backend integration.  
   - Extensive testing under varied lighting, contexts, and anomaly scenarios.

7. **Sprint 7+: Scaling & Research Integration**  
   - Explore integrating more advanced contrastive learning (Trinity-like) with larger datasets.  
   - Experiment with event-aware modules (EventVAD insights) for training-free quick adaptation.  
   - Iterate based on user feedback from pilot deployments; refine UI/UX and model performance.

---

## Conclusion

Building a “hell of a model” that is context-aware, user-configurable, and production-ready requires blending state-of-the-art research (e.g., contrastive context-conditioned learning, event-aware reasoning), modular rule-based logic for immediate adaptability, robust edge deployment practices (lightweight inference, privacy-preserving design), and a strong feedback and continuous learning pipeline. By following the structured roadmap above—starting with simple rule-based prototypes, layering context-conditioned modeling, enabling user-driven anomaly definitions via NLP parsing, and integrating feedback loops/federated learning—you can iteratively develop and refine an anomaly detection system that not only impresses recruiters but also delivers tangible value to real-world users across various scenarios.  

Feel free to dive deeper into any component (e.g., specific model architectures, contrastive pretraining code examples, NLP parsing techniques) as you progress; the modular breakdown facilitates focused development and experimentation. Good luck building EdgeGuard++’s advanced anomaly detection engine!
::contentReference[oaicite:10]{index=10} -->
