// const Alert = require('../models/Alert');

// const postAlert = async (req, res) => {
//   try {
//     const alert = new Alert(req.body);
//     await alert.save();
//     res.status(201).json({ message: 'Alert stored successfully!' });
//   } catch (error) {
//     res.status(500).json({ error: 'Error saving alert.' });
//   }
// };

// module.exports = { postAlert };


const Alert = require('../models/Alert');
const User = require('../models/User');

exports.saveAlert = async (req, res) => {
  try {
    const alert = await Alert.create(req.body);
    
    // Update user's alert history
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
    console.log("🔐 Token decoded userId:", req.user?.userId);
    console.log("📥 Request param userId:", req.params.userId);

    if (!req.user || req.user.userId !== req.params.userId) {
      return res.status(403).json({ message: "Access denied" });
    }

    const alerts = await Alert.find({ userId: req.params.userId });
    console.log("✅ Alerts fetched:", alerts.length);
    res.status(200).json(alerts);
  } catch (err) {
    console.error("❌ Error fetching user alerts:", err.message);
    res.status(500).json({ message: "Failed to fetch alerts" });
  }
};


