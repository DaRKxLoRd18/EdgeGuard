// const mongoose = require('mongoose');

// const alertSchema = new mongoose.Schema({
//   timestamp: String,
//   clip_path: String,
//   type: String,
//   location: String,
//   iv: String,
//   ciphertext: String
// }, { timestamps: true });

// module.exports = mongoose.model('Alert', alertSchema);


const mongoose = require('mongoose');

const alertSchema = new mongoose.Schema({
  userId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User' 
    },
    timestamp: String,
    clip_path: String,
    type: String,
    location: String,
    iv: String,
    ciphertext: String
});

module.exports = mongoose.model('Alert', alertSchema);
