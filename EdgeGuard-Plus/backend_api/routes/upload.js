const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// Local uploads folder
const uploadPath = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadPath)) fs.mkdirSync(uploadPath);

// Configure Multer
const storage = multer.diskStorage({
  destination: uploadPath,
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});
const upload = multer({ storage });

router.post('/', upload.single('video'), (req, res) => {
  const { email } = req.body;
  const videoPath = req.file.path;

  if (!email || !req.file) {
    return res.status(400).json({ message: 'Missing email or file' });
  }

  const process = spawn('python', [
    './edge_device/capture.py',
    '--email', email,
    '--stream', videoPath
  ]);

  process.stdout.on('data', data => console.log(`[STDOUT]: ${data}`));
  process.stderr.on('data', data => console.error(`[STDERR]: ${data}`));
  process.on('close', code => {
    console.log(`Process exited with code ${code}`);
  });

  res.json({ message: '🎥 Video uploaded & processing started.' });
});

module.exports = router;
