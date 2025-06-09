# edge_device/sender.py
import requests

def send_encrypted_alert(api_url, encrypted_data):
    try:
        response = requests.post(api_url, json=encrypted_data)
        if response.status_code == 201:
            print("✅ Alert successfully sent to backend.")
            print("🔁 Status:", response.status_code)
            print("📝 Response:", response.text)
        else:
            print(f"⚠️ Backend responded with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Error sending alert: {e}")
