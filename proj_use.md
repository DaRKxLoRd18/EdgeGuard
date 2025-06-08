# 🛡️ EdgeGuard: Privacy-Preserving Real-Time Anomaly Detection for Edge Devices

## 🔍 Overview

**EdgeGuard** is a lightweight, privacy-first anomaly detection system designed to run on **resource-constrained edge devices** like Raspberry Pi, smartphones, or even laptops. It detects anomalies (e.g., intrusions, accidents, unusual behavior) in **real-time** and only sends **encrypted metadata or alerts** to the cloud — never raw video or personal data.

> 🧠 Think: *AI-powered security camera that respects your privacy.*

---

## 🎯 Use Case

Traditional IoT surveillance devices stream **raw video to cloud servers** for analysis. This approach:
- Compromises privacy (anyone who hacks the cloud sees everything),
- Consumes high bandwidth,
- Increases latency,
- Is expensive to scale.

**EdgeGuard solves this by**:
- Performing detection **on-device**,
- Encrypting and sending only **metadata** (e.g., “anomaly at 15:30, door zone”),
- Optionally participating in **federated learning** to improve the global model without leaking local data.

---

## 👥 Target Users

| User Type           | How They Benefit                              |
|---------------------|-----------------------------------------------|
| 🏠 Home Users        | Local privacy-preserving security monitoring |
| 🏢 Small Businesses  | Edge-based surveillance on budget hardware   |
| 🏥 Hospitals         | Detect equipment/room anomalies without leaking sensitive footage |
| 🚗 Smart Vehicles    | On-device anomaly/event detection             |
| 🧠 CS Students       | Build a full-stack AI + security + systems portfolio project |

---

## 🚀 Key Features

- 📸 **Live Video Processing** — capture from webcam or video files
- 🔍 **Anomaly Detection** — lightweight CNN+LSTM-based event classification
- 🔐 **Encryption Engine** — AES-256 for secure metadata transmission
- ☁️ **Cloudless or Cloud-lite** — works offline, integrates with cloud optionally
- 💻 **React Dashboard** — see alerts in real-time with location/timestamp
- 🔁 **Federated Learning Support** — update models without sharing raw data
- 🐳 **Docker + CI/CD Ready** — production-grade deployment

---

## 🧠 Tech Stack

| Component           | Tech Used                          |
|---------------------|------------------------------------|
| Edge AI             | Python, OpenCV, TensorFlow Lite    |
| Anomaly Detection   | CNN + LSTM                         |
| Security Layer      | PyCryptodome / cryptography (AES)  |
| Backend             | Flask / FastAPI + SQLite/MQTT      |
| Frontend            | React + Mapbox + WebSocket         |
| DevOps              | Docker, GitHub Actions             |

---

## 📈 Real-World Impact

- ✅ Reduce cloud storage costs by 90%+
- ✅ Enable compliance with **GDPR**, **CCPA**, and data privacy laws
- ✅ Make AI accessible on **low-cost hardware**
- ✅ Build **trustable tech**: security without surveillance

---

## 🧪 Demo Scenario

1. Webcam captures a person entering an unauthorized zone
2. Edge AI detects anomaly with CNN+LSTM
3. Metadata `{ time: '15:30', zone: 'A1', anomaly: 'unauthorized entry' }` is encrypted and sent
4. Cloud backend stores and forwards to dashboard
5. Dashboard shows alert in real-time

---

## 🗓️ Development Roadmap (Laptop-Based)

| Week | Task                                |
|------|-------------------------------------|
| 1    | Webcam/video feed + edge simulation |
| 2    | Train CNN+LSTM model                |
| 3    | Integrate encryption & backend      |
| 4    | React dashboard                     |
| 5    | Add federated learning mock setup   |
| 6    | Final integration + polish & deploy |

---

## 💼 Why Employers Will Love It

- ✅ Demonstrates **AI in production**, not just notebooks
- ✅ Combines **ML, security, systems**, and **frontend**
- ✅ Shows initiative with real-world relevance
- ✅ Privacy-by-design mindset (a rare, valuable skill!)
- ✅ Solves a hot problem: edge privacy & anomaly detection

---

## 🙌 Credits

- Project by: [Your Name]
- Inspired by: OpenMMLab, UCSD Anomaly Dataset, Flower Federated Learning

