const express = require('express');
const router = express.Router();
const {
  registerUser,
  loginUser,
  getAllUsers,
  getUserById,
  getUserWithHistory,
  getUserByEmail
} = require('../controllers/userController');

const { authMiddleware } = require('../middlewares/authMiddleware');

router.post('/register', registerUser);
router.post('/login', loginUser);
router.get('/by-email', getUserByEmail);
router.get('/', getAllUsers);
router.get('/:id', getUserById);
router.get('/:id/full', authMiddleware, getUserWithHistory); // ✅ protected

module.exports = router;
