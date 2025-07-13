const Alert = require('../models/Alert');
const User = require('../models/User');

// Save a new alert and associate it with the user
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

// Get all alerts (admin/global use)
exports.getAllAlerts = async (req, res) => {
  const alerts = await Alert.find().populate('userId');
  res.json(alerts);
};

// Get paginated alerts for a specific user
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

// Get full alert stats for dashboard counters (independent of pagination)
exports.getAlertStatsByUser = async (req, res) => {
  try {
    const userId = req.params.userId;

    if (!req.user || req.user.userId !== userId) {
      return res.status(403).json({ message: "Access denied" });
    }

    const alerts = await Alert.find({ userId });

    const stats = {
      total: alerts.length,
      motion_anomaly: alerts.filter(a => a.type === "motion_anomaly").length,
      vehicle_or_person: alerts.filter(a => a.type === "vehicle_or_person").length,
      today: alerts.filter(
        a => new Date(a.timestamp).toDateString() === new Date().toDateString()
      ).length,
    };

    res.status(200).json(stats);
  } catch (err) {
    console.error("❌ Error in getAlertStatsByUser:", err.message);
    res.status(500).json({ message: "Server error" });
  }
};
