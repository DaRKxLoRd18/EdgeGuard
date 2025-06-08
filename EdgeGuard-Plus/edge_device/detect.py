# Placeholder for detect.py
import random
import time

class MockAnomalyDetector:
    def __init__(self, trigger_prob=0.01):
        self.trigger_prob = trigger_prob
        random.seed(time.time())

    def is_anomaly(self, frame):
        # Simulate anomaly randomly
        return random.random() < self.trigger_prob
