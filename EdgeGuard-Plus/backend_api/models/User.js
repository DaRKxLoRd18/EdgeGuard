const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  name: String,
  email: {
    type: String,
    unique: true,
},
  location: String,
  registeredAt: { 
    type: Date, 
    default: Date.now 
    },
  alerts: [{ 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "Alert" 
    }],
  anomalies: [{ 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "Anomaly" 
    }],
});

module.exports = mongoose.model("User", userSchema);
