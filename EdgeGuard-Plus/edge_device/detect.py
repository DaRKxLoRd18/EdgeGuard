class ONNXAnomalyDetector:
    def __init__(self, model_path, threshold=0.01, confidence_margin=1.2):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold
        self.margin = confidence_margin  # Multiplier

        self.frame_buffer = []  # For accumulating 12 frames

    def is_anomaly(self, frame):
        import cv2
        import numpy as np

        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gray = gray.astype('float32') / 255.0
        gray = np.expand_dims(gray, axis=-1)  # (64, 64, 1)

        self.frame_buffer.append(gray)

        if len(self.frame_buffer) < 12:
            return False
        if len(self.frame_buffer) > 12:
            self.frame_buffer.pop(0)

        # Correct shape: (1, 12, 64, 64, 1)
        sequence = np.stack(self.frame_buffer, axis=0)
        sequence = np.expand_dims(sequence, axis=0)

        pred = self.session.run(None, {self.input_name: sequence})[0]
        score = np.mean((sequence - pred) ** 2)

        print(f"🔍 Anomaly score: {score:.8f}")

        return score > (self.threshold * self.margin)

