import cv2
import time
import os
from detect import ONNXAnomalyDetector  # ✅ Real detector
from dvr import DVRBuffer
from encrypt import MetadataEncryptor
from sender import send_encrypted_alert
from gif_generator import save_gif_from_frames  # ✅ New

def run_capture():
    cap = cv2.VideoCapture(r"test-videos\test.avi")
    # cap = cv2.VideoCapture(0)

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/clips", exist_ok=True)
    os.makedirs("data/previews", exist_ok=True)  # For GIFs

    if not cap.isOpened():
        print("❌ Error: Cannot open video source.")
        return

    window_name = 'Edge Device Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = ONNXAnomalyDetector(
        model_path=os.path.join("saved_model", "final_conv_lstm_ae.onnx"),
        threshold=0.016,
        confidence_margin=1.2
    )
    
    buffer = DVRBuffer(fps=30, buffer_seconds=5)
    encryptor = MetadataEncryptor()

    print("🎥 Streaming started. Press 'q' or close window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame. Exiting.")
            break

        cv2.imshow(window_name, frame)
        buffer.add_frame(frame)

        # ✅ Get score, type, and flag
        triggered, score, anomaly_type = detector.is_anomaly(frame)

        if triggered:
            print(f"🚨 {anomaly_type.upper()} detected! Score = {score:.8f}")
            writer, clip_path = buffer.save_clip_start()

            if writer and clip_path:
                post_frames = 150
                gif_frames = []

                for _ in range(post_frames):
                    ret_post, post_frame = cap.read()
                    if not ret_post:
                        break
                    writer.write(post_frame)
                    gif_frames.append(post_frame)

                writer.release()

                # ✅ Save GIF preview
                gif_path = save_gif_from_frames(gif_frames)

                metadata = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "clip_path": clip_path,
                    "gif_path": gif_path,
                    "type": anomaly_type,
                    "location": "Zone A"
                }

                encrypted = encryptor.encrypt(metadata)

                # ✅ Send to backend
                send_encrypted_alert(
                    api_url="http://localhost:5000/api/alerts",
                    encrypted_data={
                        "timestamp": metadata["timestamp"],
                        "clip_path": metadata["clip_path"],
                        "gif_path": metadata["gif_path"],
                        "type": metadata["type"],
                        "location": metadata["location"],
                        "iv": encrypted["iv"],
                        "ciphertext": encrypted["ciphertext"]
                    }
                )
            else:
                print("❌ Clip save failed.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Stopping stream via 'q'.")
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 Stopping stream via window close.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_capture()
