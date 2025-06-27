const express = require('express');
const router = express.Router();
const { registerUser, getAllUsers, getUserById, getUserWithHistory, getUserByEmail  } = require('../controllers/userController');

router.get('/by-email', getUserByEmail);

router.post('/register', registerUser);
router.get('/', getAllUsers);
router.get('/:id', getUserById);
router.get('/:id/full', getUserWithHistory);


module.exports = router;
