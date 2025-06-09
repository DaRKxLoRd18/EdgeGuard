const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const alertRoutes = require('./routes/alerts');
app.use('/api/alerts', alertRoutes);

mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => {
  console.log('✅ MongoDB connected');
  app.listen(process.env.PORT || 5000, () => {
    console.log('🚀 Server running on port', process.env.PORT || 5000);
  });
}).catch(err => console.error('MongoDB error:', err));
