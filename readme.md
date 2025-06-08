# 🛡️ EdgeGuard++: Smart, Privacy-First, Full-Stack Anomaly Detection Platform for Edge Devices

> Real-time, privacy-aware anomaly detection on edge devices with encrypted cloud sync, heatmaps, DVR-like event rewind, and a full-stack dashboard experience. Built collaboratively by a 3-member team.

---

## 🚨 Problem Statement

Most security and monitoring systems **stream raw video to the cloud**, leading to:
- ⚠️ **Privacy violations**
- 🛑 High **bandwidth and cloud cost**
- 🐢 Increased **latency**

**EdgeGuard++** brings intelligence to the **edge**, detecting anomalies **locally**, encrypting sensitive metadata, and offering a rich **dashboard and analytics layer** — all without sending raw data to the cloud.

---

## 🎯 Key Features

| Category              | Feature                                                   |
|-----------------------|-----------------------------------------------------------|
| 🧠 AI on Edge         | Real-time anomaly detection with lightweight CNN+LSTM     |
| 🔐 Privacy Engine     | AES-256 encryption, local processing, cloud opt-in        |
| 📦 Mini-DVR           | Save only 5s before & after an anomaly (privacy-respecting) |
| 🌍 Zone Detection     | Define zones + real-time heatmaps on the dashboard        |
| 📊 Analytics          | Anomaly trends, daily stats, most-affected zones          |
| 🛠️ Configurable       | Privacy control center (retention, sync toggle, opt-in FL) |
| 🔄 Feedback Loop      | Users can confirm/correct detection for model retraining  |
| 🌐 Multi-Device       | Manage multiple edge devices from a central dashboard     |
| 📱 Mobile Companion   | (Optional) Push notifications for anomaly alerts          |

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

## 🧑‍💻 Tech Stack

| Layer              | Tech Used                                |
|--------------------|-------------------------------------------|
| Edge Device        | Python, OpenCV, TensorFlow Lite, AES      |
| ML Model           | CNN + LSTM, Federated Learning (Flower)   |
| Backend API        | FastAPI / Flask, SQLite / MongoDB         |
| Dashboard Frontend | React.js, Mapbox, WebSocket, Chart.js     |
| DevOps             | Docker, GitHub Actions, CI/CD, MQTT       |

---

## 🔗 Architecture Overview

```text
          [Edge Device A]         [Edge Device B]
         ┌───────────────┐       ┌───────────────┐
         │ Webcam/Input  │       │ Video File    │
         │ Anomaly Model │       │ CNN+LSTM      │
         │ DVR Buffer    │       │ AES Encrypt   │
         └─────┬─────────┘       └─────┬─────────┘
               │                          │
               ▼                          ▼
       [Encrypted Metadata & Clip]    [Encrypted Metadata]
               │                          │
               └───────────┬──────────────┘
                           ▼
                   🌩️ Cloud Backend API
               (FastAPI + DB + MQTT Broker)
                           │
                           ▼
                     📊 React Dashboard
                   - Live map & feed
                   - Zone editor
                   - Analytics & replay
                   - Privacy settings
