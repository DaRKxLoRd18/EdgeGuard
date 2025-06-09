const express = require('express');
const router = express.Router();
const { postAlert } = require('../controllers/alertController');

router.post('/', postAlert);

module.exports = router;
