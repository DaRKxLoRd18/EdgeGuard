import cv2
import time
from detect import MockAnomalyDetector
from dvr import DVRBuffer
from encrypt import MetadataEncryptor

def run_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open video source.")
        return

    window_name = 'Edge Device Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = MockAnomalyDetector(trigger_prob=0.01)
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

        # Check for anomaly
        if detector.is_anomaly(frame):
            print("🚨 Anomaly Detected! Saving clip...")

            writer, clip_path = buffer.save_clip_start()
            if writer and clip_path:
                # Write 5 seconds of future frames
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
                    "type": "mock_anomaly",
                    "location": "Zone A"
                }

                encrypted = encryptor.encrypt(metadata)
                print("🔐 Encrypted Metadata:")
                print(encrypted)
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
