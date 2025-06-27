const express = require('express');
const router = express.Router();
const { saveAlert, getAllAlerts, getAlertsByUser } = require('../controllers/alertController');

router.post('/', saveAlert);
router.get('/', getAllAlerts);
router.get('/user/:userId', getAlertsByUser);

module.exports = router;
