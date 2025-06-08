# edge_device/capture.py
import cv2

def run_capture():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam. Replace with video file path if needed

    if not cap.isOpened():
        print("❌ Error: Cannot open video source.")
        return

    window_name = 'Edge Device Feed'
    print("✅ Starting video stream... Press 'q' or close window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame. Exiting.")
            break

        cv2.imshow(window_name, frame)

        # Detect if user pressed 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Stopping stream via key.")
            break

        # Detect if the window is closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 Stopping stream via window close.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_capture()
