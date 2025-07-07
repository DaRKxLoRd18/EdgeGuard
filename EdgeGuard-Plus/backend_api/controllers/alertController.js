const Alert = require('../models/Alert');
const User = require('../models/User');

exports.saveAlert = async (req, res) => {
  try {
    const alert = await Alert.create(req.body);
    await User.findByIdAndUpdate(req.body.userId, {
      $push: { alerts: alert._id }
    });
    res.status(201).json(alert);
  } catch (error) {
    res.status(400).json({ message: 'Error saving alert', error });
  }
};

exports.getAllAlerts = async (req, res) => {
  const alerts = await Alert.find().populate('userId');
  res.json(alerts);
};

exports.getAlertsByUser = async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;

    if (!req.user || req.user.userId !== req.params.userId) {
      return res.status(403).json({ message: "Access denied" });
    }

    const alerts = await Alert.find({ userId: req.params.userId })
      .sort({ timestamp: -1 })
      .skip((page - 1) * limit)
      .limit(parseInt(limit));

    res.status(200).json(alerts);
  } catch (err) {
    console.error("❌ Error fetching user alerts:", err.message);
    res.status(500).json({ message: "Failed to fetch alerts" });
  }
};