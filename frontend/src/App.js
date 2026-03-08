import React, { useState } from 'react';
import axios from 'axios';
import ImageUpload from './components/ImageUpload';
import './App.css';

const PLANT_ICONS = {
  'Potato'     : '🥔',
  'Tomato'     : '🍅',
  'Bell Pepper': '🫑',
};

function App() {
  const [preview, setPreview]           = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading]       = useState(false);
  const [error, setError]               = useState(null);

  const handleFileDrop = async (files) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setError(null);
    setPredictionData(null);

    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);

    await makePrediction(file);
  };

  const makePrediction = async (file) => {
    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(
        process.env.REACT_APP_API_URL,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setPredictionData(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Prediction failed. Make sure the backend is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPreview(null);
    setPredictionData(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🌿 Plant Disease Detection</h1>
        <p>Upload a leaf image — supports Potato, Tomato &amp; Bell Pepper</p>
      </header>

      <main className="App-main">
        <ImageUpload onDrop={handleFileDrop} />

        {error && (
          <div className="error-message">{error}</div>
        )}

        {isLoading && (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Analyzing leaf...</p>
          </div>
        )}

        {preview && !isLoading && (
          <div className="preview-section">
            <img src={preview} alt="Uploaded leaf" className="preview-image" />
            <button className="reset-btn" onClick={handleReset}>
              Upload Another
            </button>
          </div>
        )}

        {predictionData && !isLoading && (
          <div className={`results-card ${predictionData.is_healthy ? 'result-healthy' : 'result-diseased'}`}>
            <h2>Prediction Result</h2>

            <div className="result-row">
              <span className="label">Plant</span>
              <span className="value">
                {PLANT_ICONS[predictionData.plant] || '🌿'} {predictionData.plant}
              </span>
            </div>

            <div className="result-row">
              <span className="label">Condition</span>
              <span className={`value ${predictionData.is_healthy ? 'healthy-text' : 'disease-text'}`}>
                {predictionData.is_healthy ? '✅ ' : '⚠️ '}
                {predictionData.disease}
              </span>
            </div>

            <div className="result-row">
              <span className="label">Confidence</span>
              <span className="value">
                {(predictionData.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="confidence-bar-wrap">
              <div
                className="confidence-bar"
                style={{ width: `${(predictionData.confidence * 100).toFixed(1)}%` }}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
