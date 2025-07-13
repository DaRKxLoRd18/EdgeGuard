const express = require('express');
const router = express.Router();

const {
  saveAlert,
  getAllAlerts,
  getAlertsByUser,
  getAlertStatsByUser, 
} = require('../controllers/alertController');

const { authMiddleware } = require('../middlewares/authMiddleware');
router.post('/', saveAlert);
router.get('/', getAllAlerts);
router.get('/user/:userId', authMiddleware, getAlertsByUser);
router.get('/user/:userId/stats', authMiddleware, getAlertStatsByUser);

module.exports = router;
