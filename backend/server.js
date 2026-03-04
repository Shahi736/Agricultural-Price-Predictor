require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

// Flask ML API endpoint
const ML_API_URL = process.env.ML_API_URL || 'http://127.0.0.1:5001/predict';
const ML_DATA_URL = 'http://127.0.0.1:5001/data';

// ✅ Test route
app.get('/', (req, res) => {
  res.send('Backend running successfully and connected to ML API!');
});

// ✅ Predict route (Node → Flask)
app.post('/predict', async (req, res) => {
  const { crop, region } = req.body;

  if (!crop || !region) {
    return res.status(400).json({ error: 'Please send crop and region in body.' });
  }

  try {
    const response = await axios.post(ML_API_URL, { crop, region });
    res.json(response.data);
  } catch (err) {
    console.error('Error connecting to Flask model:', err.message);
    res.status(502).json({ error: 'ML model not reachable', details: err.message });
  }
});

// ✅ Dataset route (Node → Flask)
app.get('/data', async (req, res) => {
  try {
    const response = await axios.get(ML_DATA_URL);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching dataset:', error.message);
    res.status(500).json({ error: 'Error fetching dataset' });
  }
});

// ✅ Start the backend server (ALWAYS LAST)
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`✅ Server started on port ${PORT}`));