const express = require('express');
const router = express.Router();
const { saveAlert, getAllAlerts, getAlertsByUser } = require('../controllers/alertController');
const { authMiddleware } = require('../middlewares/authMiddleware');

router.post('/', saveAlert);
router.get('/', getAllAlerts);
router.get('/user/:userId', authMiddleware, getAlertsByUser);

// ✅ Debug route to test token auth
router.get('/ping', authMiddleware, (req, res) => {
  res.status(200).json({ message: "✅ Authenticated", user: req.user });
});

module.exports = router;
