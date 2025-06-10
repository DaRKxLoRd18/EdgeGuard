# 🛡️ EdgeGuard++: Smart, Privacy-First, Full-Stack Anomaly Detection Platform for Edge Devices

> A complete AI-powered, privacy-preserving anomaly detection system — featuring real-time video analysis, DVR-style event capture, AES-encrypted metadata, a Node.js + MongoDB backend, and a React-based dashboard.

---

## 🚨 Problem Statement

Most security and monitoring systems **stream raw video to the cloud**, leading to:
- ⚠️ **Privacy violations**
- 🛑 High **bandwidth and cloud cost**
- 🐢 Increased **latency**

**EdgeGuard++ solves this by:**
- Performing AI inference on-device
- Saving only anomalies (DVR-style)
- Encrypting & sending metadata only
- Storing events in a Node.js + MongoDB backend
- Displaying via a React dashboard

---

## 🎯 Key Features

| Category            | Feature                                                                 |
|---------------------|-------------------------------------------------------------------------|
| 🎥 Edge Capture     | Webcam or video file input with DVR buffering                           |
| 🧠 Anomaly Detection| CNN + LSTM model for spatial-temporal pattern detection (real model)    |
| 🔐 Privacy Engine   | AES-256 encryption for metadata                                          |
| 🎞️ DVR System       | Saves 5s before and after each anomaly                                  |
| 📡 Cloud Sync       | Sends encrypted alerts to Express.js API                                |
| 🧮 Data Storage      | MongoDB for event logging                                               |
| 📊 Dashboard (Coming)| Real-time alert list + playback + map zones (React)                    |
| 🔁 Real-time Ready   | WebSocket support for live feed push (planned)                         |

---


## 👥 Target Users

| User               | Use Case                                        |
|--------------------|--------------------------------------------------|
| 🏠 Homeowners       | Affordable & private surveillance at home       |
| 🏢 Business Owners  | Smart zone alerting in shops/warehouses         |
| 🏥 Hospitals        | Non-invasive, real-time monitoring of patients  |
| 🚗 Smart Vehicles   | On-device anomaly/event detection (black box)   |
| 👩‍💻 CS Students     | Build production-grade ML + systems projects    |

---

## 🧰 Tech Stack

| Layer           | Stack/Tools Used                            |
|------------------|---------------------------------------------|
| 🧠 Edge Inference | Python, OpenCV, TensorFlow (CNN+LSTM), NumPy|
| 📦 Metadata Sec. | PyCryptodome (AES-256)                      |
| 🎞️ Video Buffer  | OpenCV + Deques (DVR logic)                 |
| ☁️ Backend API   | Node.js, Express.js, dotenv                 |
| 🗃️ Database       | MongoDB (local or Atlas)                   |
| 📊 Frontend      | React.js (dashboard UI), Axios, Mapbox (TBD)|
| 🧪 Model Training| TensorFlow, Keras                           |
| 🛠 DevOps         | Nodemon, .env configs                      |

---

## 🔗 Architecture Overview

```text


                [Edge Device]
                ┌───────────────────────┐
                │ - Webcam / Video Feed│
                │ - CNN+LSTM Detector  │
                │ - AES Encryptor      │
                │ - DVR Saver (.avi)   │
                └───────┬──────────────┘
                        ▼
                POST Encrypted Metadata (JSON)
                        ▼
                [Node.js + Express Backend]
                ┌──────────────────────┐
                │ - Save to MongoDB    │
                │ - GET /api/alerts    │
                │ - WebSocket Support  │
                └───────┬──────────────┘
                        ▼
                [React.js Dashboard UI]
                ┌──────────────────────┐
                │ - Live alert feed    │
                │ - Replay .avi clips  │
                │ - Analytics & Zones  │
                └──────────────────────┘
