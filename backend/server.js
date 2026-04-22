require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

// Flask URLs
const ML_URL = "http://127.0.0.1:5001";

// Home route
app.get('/', (req, res) => {
  res.send(" AgriSense backend running");
});

// Predict route
app.post('/predict', async (req, res) => {
  try {
    console.log("[PREDICT REQUEST]", req.body);

    const response = await axios.post(`${ML_URL}/predict`, req.body);

    console.log("[PREDICT SUCCESS]", response.data);

    res.json(response.data);

  } catch (err) {
    console.error("[PREDICT ERROR]", err.message);
    res.status(500).json({ error: "Flask connection failed" });
  }
});

// Data route
app.get('/data', async (req, res) => {
  try {
    const response = await axios.get(`${ML_URL}/data`);
    res.json(response.data);
  } catch (err) {
    console.error("[DATA ERROR]", err.message);
    res.status(500).json({ error: "Data fetch failed" });
  }
});

// Start server
const PORT = 5000;
app.listen(PORT, () => {
  console.log(` AgriSense backend on http://localhost:${PORT}`);
});