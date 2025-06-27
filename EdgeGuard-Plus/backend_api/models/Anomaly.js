const mongoose = require('mongoose');

const anomalySchema = new mongoose.Schema({
  userId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User' },
    alertId: { 
        type: mongoose.Schema.Types.ObjectId, 
        ref: 'Alert' 
    },
    modelUsed: String,
    confidence: Number,
    zone: String,
    createdAt: { 
        type: Date, 
        default: Date.now 
    }
});

module.exports = mongoose.model('Anomaly', anomalySchema);
