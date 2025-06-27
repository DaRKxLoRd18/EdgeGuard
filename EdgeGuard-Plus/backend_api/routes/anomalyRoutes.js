const express = require('express');
const router = express.Router();
const { logAnomaly, getAnomaliesByUser, getAllAnomalies } = require('../controllers/anomalyController');

router.post('/', logAnomaly);
router.get('/', getAllAnomalies);
router.get('/user/:userId', getAnomaliesByUser);

module.exports = router;
