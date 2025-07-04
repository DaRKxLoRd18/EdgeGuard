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

def handle_anomaly(user_id, buffer, cap, encryptor, anomaly_type, score):
    print(f"🚨 {anomaly_type.upper()} Detected! Score = {score:.8f}")

    writer, clip_path = buffer.save_clip_start()
    if not writer or not clip_path:
        print("❌ Clip save failed.")
        return

    gif_frames = []
    fps = 60
    post_frames = fps * 2  # 2 seconds of future frames

    for _ in range(post_frames):
        ret_post, post_frame = cap.read()
        if ret_post and post_frame is not None:
            writer.write(post_frame)
            gif_frames.append(post_frame)
        else:
            print("⚠️ Skipped bad frame during post-capture.")

    writer.release()
    time.sleep(0.1)  # Ensure FFmpeg flushes internal buffers

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

def run_capture(user_email, video_path=None):
    user_id = fetch_user_id(user_email)
    if not user_id:
        print("Exiting: Could not find user by email:", user_email)
        return

    os.makedirs("data/clips", exist_ok=True)
    os.makedirs("data/previews", exist_ok=True)

    cap = cv2.VideoCapture(video_path if video_path else 0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("❌ Error: Cannot open video source.")
        return

    window_name = 'Edge Device Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = ONNXAnomalyDetector(
        model_path=os.path.join("saved_model", "final_conv_lstm_ae.onnx"),
        threshold=0.012,
        confidence_margin=1.2
    )
    buffer = DVRBuffer(fps=60, buffer_seconds=2)
    encryptor = MetadataEncryptor()

    print("🎥 Streaming started. Press 'q' or close window to stop.")

    cooldown_frames = 0  # Number of frames to skip detection after anomaly

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame. Exiting.")
            break

        cv2.imshow(window_name, frame)
        buffer.add_frame(frame)

        if cooldown_frames > 0:
            cooldown_frames -= 1
            continue

        triggered, score, anomaly_type = detector.is_anomaly(frame)
        if triggered:
            cooldown_frames = 120  # Skip detection for 2 seconds at 60 FPS
            handle_anomaly(user_id, buffer, cap, encryptor, anomaly_type, score)

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
    video_path = "test-videos/test.avi"  # Set to None to use webcam
    run_capture(user_email=user_email, video_path=video_path)
