- Created a mock model to detect anomaly.
- Capture.py : captures live feed from webcam, detects anomaly saves +=5 sec of clip around anomaly. Encrypts data using AES-256 and sends to backend to save data in mongoDB(local)
- created demo backend using nodejs(Just POST command) using chatgpt.
- Saved complete metadata in MongoDB.

### Model work 
- Downloaded and preprocessed raw data for model development and saved processed npy file in data folder. Check models/readme file for complete model development process.