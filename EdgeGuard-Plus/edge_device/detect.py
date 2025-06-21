import cv2
import numpy as np
import os
from datetime import datetime

class ONNXAnomalyDetector:
    def __init__(self, model_path, threshold=0.02, confidence_margin=1.2):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold
        self.margin = confidence_margin
        self.frame_buffer = []

        # For output video saving
        self.video_writer = None
        self.output_video_path = "data/output_annotated.avi"
        os.makedirs("data", exist_ok=True)

        # Setup YOLOv8
        self.yolo_available = False
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            self.detect_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            self.conf_thres = 0.4
            self.yolo_available = True
            print("✅ Ultralytics YOLO detection enabled")
        except ImportError:
            print("⚠️ Install YOLOv8: pip install ultralytics")
            print("📝 Using ConvLSTM detection only")
        except Exception as e:
            print(f"⚠️ YOLO setup error: {e}")
            print("📝 Using ConvLSTM detection only")

    def is_within_allowed_time(self):
        now = datetime.now()
        return 6 <= now.hour < 12  # 6:00 AM to 11:59 AM

    def detect_objects(self, frame):
        if not self.yolo_available:
            return False, frame

        try:
            results = self.yolo_model(frame, conf=self.conf_thres, verbose=False)
            suspicious = False
            annotated_frame = frame.copy()

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                        if class_id in self.detect_classes:
                            suspicious = True

                    annotated_frame = result.plot()

            return suspicious, annotated_frame

        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            return False, frame

    def is_anomaly(self, frame):
        # Step 1: YOLO detection
        suspicious, annotated_frame = self.detect_objects(frame)

        # Initialize video writer once
        if self.video_writer is None:
            height, width = annotated_frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                self.output_video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                30,
                (width, height)
            )

        # If YOLO detected object of interest
        if suspicious:
            cv2.putText(
                annotated_frame,
                "YOLO: Suspicious Object Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            self.video_writer.write(annotated_frame)
            print("🚨 YOLO detected suspicious object!")
            return True

        # Step 2: ConvLSTM-based detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64)).astype('float32') / 255.0
            gray = np.expand_dims(gray, axis=-1)  # (64, 64, 1)
            self.frame_buffer.append(gray)

            if len(self.frame_buffer) < 12:
                self.video_writer.write(annotated_frame)
                return False
            if len(self.frame_buffer) > 12:
                self.frame_buffer.pop(0)

            sequence = np.stack(self.frame_buffer, axis=0)     # (12, 64, 64, 1)
            sequence = np.expand_dims(sequence, axis=0)        # (1, 12, 64, 64, 1)
            pred = self.session.run(None, {self.input_name: sequence})[0]
            score = np.mean((sequence - pred) ** 2)

            # Overlay score on frame
            cv2.putText(
                annotated_frame,
                f"ConvLSTM Score: {score:.6f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if score <= self.threshold * self.margin else (0, 0, 255),
                2
            )

            self.video_writer.write(annotated_frame)
            print(f"🔍 ConvLSTM Score: {score:.8f}")

            return score > self.threshold * self.margin

        except Exception as e:
            print(f"❌ ConvLSTM error: {e}")
            self.video_writer.write(annotated_frame)
            return False

