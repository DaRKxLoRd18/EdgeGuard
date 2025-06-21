import cv2
import time
from detect import ONNXAnomalyDetector  # ✅ Replaced Mock with ONNX
from dvr import DVRBuffer
from encrypt import MetadataEncryptor
from sender import send_encrypted_alert

def run_capture():
    cap = cv2.VideoCapture(r"C:\proj\test-videos\test.avi")
    # cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path



    if not cap.isOpened():
        print("❌ Error: Cannot open video source.")
        return

    window_name = 'Edge Device Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ✅ Real anomaly detection using ONNX model
    detector = ONNXAnomalyDetector(
        model_path = r"C:\proj\saved_model\final_conv_lstm_ae.onnx",
        threshold=0.01, # Adjust threshold as needed
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

        # Display frame
        cv2.imshow(window_name, frame)

        # Add frame to DVR buffer
        buffer.add_frame(frame)

        # Check for anomaly using ONNX model
        if detector.is_anomaly(frame):
            print("🚨 Anomaly Detected! Saving clip...")

            writer, clip_path = buffer.save_clip_start()
            if writer and clip_path:
                # Write 5 seconds of future frames (post-anomaly)
                post_frames = 150
                for _ in range(post_frames):
                    ret_post, post_frame = cap.read()
                    if not ret_post:
                        break
                    writer.write(post_frame)

                writer.release()

                metadata = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "clip_path": clip_path,
                    "type": "reconstruction_anomaly",
                    "location": "Zone A"
                }

                encrypted = encryptor.encrypt(metadata)

                # Send to backend
                send_encrypted_alert(
                    api_url="http://localhost:5000/api/alerts",
                    encrypted_data={
                        "timestamp": metadata["timestamp"],
                        "clip_path": metadata["clip_path"],
                        "type": metadata["type"],
                        "location": metadata["location"],
                        "iv": encrypted["iv"],
                        "ciphertext": encrypted["ciphertext"]
                    }
                )

            else:
                print("❌ Clip save failed.")

        # Exit conditions
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
