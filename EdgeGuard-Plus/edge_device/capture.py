import cv2
import time
import os
import requests
from detect import ONNXAnomalyDetector
from dvr import DVRBuffer
from encrypt import MetadataEncryptor
from sender import send_encrypted_alert
from gif_generator import save_gif_from_frames


def fetch_user_id(email):
    try:
        response = requests.get("http://localhost:5000/api/users/by-email", params={"email": email})
        if response.status_code == 200:
            user = response.json()
            print(f"✅ User found: {user['_id']}")
            return user["_id"]
        else:
            print("❌ User fetch failed:", response.json().get("message"))
            return None
    except Exception as e:
        print(f"❌ Exception while fetching user ID: {e}")
        return None


def run_capture(user_email, video_path=None):
    # Step 1: Get user ID
    user_id = fetch_user_id(user_email)
    if not user_id:
        print("Exiting: Could not find user by email:", user_email)
        return

    # Step 2: Prepare folders
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/clips", exist_ok=True)
    os.makedirs("data/previews", exist_ok=True)

    # Step 3: Start capturing
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("❌ Error: Cannot open video source.")
        return

    window_name = 'Edge Device Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Step 4: Setup detector, buffer and encryption
    detector = ONNXAnomalyDetector(
        model_path=os.path.join("saved_model", "final_conv_lstm_ae.onnx"),
        threshold=0.012,
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

        triggered, score, anomaly_type = detector.is_anomaly(frame)

        if triggered:
            print(f"🚨 {anomaly_type.upper()} Detected! Score = {score:.8f}")

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

                gif_path = save_gif_from_frames(gif_frames)

                metadata = {
                    "userId": user_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "clip_path": clip_path,
                    "gif_path": gif_path,
                    "type": anomaly_type,
                    "location": "Zone A"
                }

                encrypted = encryptor.encrypt(metadata)

                send_encrypted_alert(
                    api_url="http://localhost:5000/api/alerts",
                    encrypted_data={
                        "userId": metadata["userId"],
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
    user_email = "gautammayank292@gmail.com"
    video_path = r"D:\Security App\proj\test-videos\test.avi"  # or None for webcam
    run_capture(user_email=user_email, video_path=video_path)
