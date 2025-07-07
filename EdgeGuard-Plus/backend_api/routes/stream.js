const express = require("express");
const { spawn } = require("child_process");
const router = express.Router();

router.post("/start", (req, res) => {
  const { email, rtsp } = req.body;

  if (!email || !rtsp) {
    return res.status(400).json({ message: "Missing RTSP or email" });
  }

  const proc = spawn("python", [
    "./edge_device/capture.py",
    "--email",
    email,
    "--stream",
    rtsp,
  ]);

  proc.stdout.on("data", (data) => console.log(`[PYTHON STDOUT]: ${data}`));
  proc.stderr.on("data", (data) => console.error(`[PYTHON STDERR]: ${data}`));
  proc.on("close", (code) => console.log(`🔚 Python process exited with code ${code}`));

  res.json({ message: "📡 Stream started successfully" });
});

module.exports = router;