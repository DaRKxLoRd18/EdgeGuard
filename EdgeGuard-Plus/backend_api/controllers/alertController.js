const Alert = require('../models/Alert');

const postAlert = async (req, res) => {
  try {
    const alert = new Alert(req.body);
    await alert.save();
    res.status(201).json({ message: 'Alert stored successfully!' });
  } catch (error) {
    res.status(500).json({ error: 'Error saving alert.' });
  }
};

module.exports = { postAlert };
