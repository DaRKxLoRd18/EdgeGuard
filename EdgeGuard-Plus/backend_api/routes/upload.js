const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// 🔧 Ensure uploads directory exists
const uploadPath = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadPath)) {
  fs.mkdirSync(uploadPath);
}

// 📦 Configure Multer for file upload
const storage = multer.diskStorage({
  destination: uploadPath,
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});
const upload = multer({ storage });

// 📤 POST route for file upload and trigger anomaly detection
router.post('/', upload.single('video'), (req, res) => {
  const { email } = req.body;
  const videoPath = req.file?.path;

  if (!email || !videoPath) {
    return res.status(400).json({ message: 'Missing email or video file' });
  }

  // ✅ Resolve absolute path for capture.py (adjust if necessary)
  const capturePath = path.resolve(__dirname, '../../edge_device/capture.py');

  // 🧠 Spawn Python process for anomaly detection
  const pythonProcess = spawn('python', [
    capturePath,
    '--email', email,
    '--stream', videoPath,
    '--headless' // Optional: prevents window popups (OpenCV)
  ]);

  // 🔍 Listen for stdout
  pythonProcess.stdout.on('data', data => {
    console.log(`[📤 Python stdout]: ${data}`);
  });

  // ⚠️ Listen for stderr
  pythonProcess.stderr.on('data', data => {
    console.error(`[❌ Python stderr]: ${data}`);
  });

  // ✅ When Python process ends, send response
  pythonProcess.on('close', code => {
    console.log(`🎬 capture.py exited with code ${code}`);

    if (code === 0) {
      return res.status(200).json({
        message: '✅ Video processed successfully.',
        file: path.basename(videoPath),
        status: 'success'
      });
    } else {
      return res.status(500).json({
        message: '❌ Video processing failed.',
        file: path.basename(videoPath),
        status: 'error',
        code
      });
    }
  });
});

module.exports = router;
