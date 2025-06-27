const Anomaly = require('../models/Anomaly');
const User = require('../models/User');

exports.logAnomaly = async (req, res) => {
  try {
    const anomaly = await Anomaly.create(req.body);

    // Update user's anomaly history
    await User.findByIdAndUpdate(req.body.userId, {
      $push: { anomalies: anomaly._id }
    });

    res.status(201).json(anomaly);
  } catch (error) {
    res.status(400).json({ message: 'Error logging anomaly', error });
  }
};

exports.getAnomaliesByUser = async (req, res) => {
  const anomalies = await Anomaly.find({ userId: req.params.userId }).populate('alertId');
  res.json(anomalies);
};

exports.getAllAnomalies = async (req, res) => {
  const anomalies = await Anomaly.find().populate('userId').populate('alertId');
  res.json(anomalies);
};
