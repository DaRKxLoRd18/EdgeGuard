const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
dotenv.config();

const connectDB = require('./config/db');

// Import routes
const userRoutes = require('./routes/userRoutes');
const alertRoutes = require('./routes/alertRoutes');
const anomalyRoutes = require('./routes/anomalyRoutes');   // ✅ restored
const uploadRoute = require('./routes/upload');            // ✅ added
const streamRoute = require('./routes/stream');            // ✅ added

connectDB();

const app = express();
app.use(cors());
app.use(express.json());

// Mount routes
app.use('/api/users', userRoutes);
app.use('/api/alerts', alertRoutes);
app.use('/api/anomalies', anomalyRoutes);   // ✅ correct
app.use('/api/upload', uploadRoute);        // ✅ new
app.use('/api/stream', streamRoute);        // ✅ new

module.exports = app;
