const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// Ensure uploads directory exists
const uploadPath = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadPath)) fs.mkdirSync(uploadPath);

// Setup Multer for file uploads
const storage = multer.diskStorage({
  destination: uploadPath,
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});
const upload = multer({ storage });

// POST /api/upload
router.post('/', upload.single('video'), (req, res) => {
  const { email } = req.body;
  const videoPath = req.file?.path;

  if (!email || !videoPath) {
    return res.status(400).json({ message: 'Missing email or video file' });
  }

  // ✅ FIX: Use path.resolve with proper escaping for Windows paths
  const capturePath = path.resolve(__dirname, '../../edge_device/capture.py'); // Adjust if needed

  // ✅ OPTIONAL: Add "--headless" flag if running on server
  const pythonProcess = spawn('python', [
    capturePath,
    '--email', email,
    '--stream', videoPath,
    '--headless'
  ]);

  // Log output from Python process
  pythonProcess.stdout.on('data', data => console.log(`[📤 Python stdout]: ${data}`));
  pythonProcess.stderr.on('data', data => console.error(`[❌ Python stderr]: ${data}`));
  pythonProcess.on('close', code => console.log(`🎬 capture.py exited with code ${code}`));

  // Respond immediately
  res.status(200).json({
    message: '📦 Video uploaded and processing started.',
    file: path.basename(videoPath)
  });
});

module.exports = router;
