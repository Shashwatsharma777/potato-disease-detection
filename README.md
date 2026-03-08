# 🥔 Potato Disease Detection using CNN

An end-to-end web application that detects potato leaf diseases using a Convolutional Neural Network (CNN). Upload a potato leaf photo and get instant disease prediction.

## Disease Classes
- **Early Blight** (`Potato___Early_blight`) — 1,000 images
- **Late Blight** (`Potato___Late_blight`) — 1,000 images
- **Healthy** (`Potato___healthy`) — 152 images

## Project Structure

```
potato-disease-detection/
├── training/
│   ├── potato_disease_training.ipynb    # Model training notebook
│   ├── jupyter_utils.py                 # Inference utility
│   └── saved_models/                    # Model saved here after training
│
├── backend/
│   ├── main.py                          # FastAPI server
│   ├── requirements.txt
│   └── models/                          # Copy model here before starting backend
│
├── frontend/
│   ├── public/index.html
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── components/ImageUpload.js
│   │   └── styles/ImageUpload.css
│   ├── package.json
│   └── .env                             # API URL config
│
├── Dataset/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
│
├── .gitignore
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | TensorFlow / Keras 2.15 |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + Axios |
| File Upload | React Dropzone |

## Quick Start

### Step 1 — Train the Model

```bash
# Activate virtual environment
source venv/bin/activate

# Install training dependencies
pip install -r backend/requirements.txt
pip install matplotlib seaborn scikit-learn jupyter

# Launch notebook
cd training
jupyter notebook potato_disease_training.ipynb
```

Run all cells. The model will be saved to:
- `training/saved_models/potato_model.h5`
- `backend/models/potato_model.h5` (auto-copied)

### Step 2 — Start Backend

```bash
source venv/bin/activate
cd backend
python main.py
```

API runs at `http://localhost:8000`

### Step 3 — Start Frontend

In a new terminal:

```bash
cd frontend
npm install
npm start
```

App opens at `http://localhost:3000`

## API Endpoints

### Health Check
```
GET /ping
→ {"message": "pong", "model_loaded": true}
```

### Predict Disease
```
POST /predict
Body: multipart/form-data  { file: <image> }
→ {"class": "Early Blight", "confidence": 0.9452}
```

## Model Architecture

```
Input (256x256 RGB)
  → Rescaling (1/255)
  → Conv2D(32) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool
  → Conv2D(128) + BatchNorm + MaxPool
  → Conv2D(128) + BatchNorm + MaxPool
  → Flatten
  → Dense(256) + Dropout(0.4)
  → Dense(64)  + Dropout(0.2)
  → Dense(3, softmax)
```

**Training features:**
- Class weights for imbalanced data (Healthy: 152 vs others: 1000)
- Data augmentation: Flip, Rotation, Zoom, Contrast
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not loaded` error | Run the notebook first to train and save the model |
| CORS error | Make sure backend is running on port 8000 |
| Port already in use | `lsof -i :8000` then `kill <PID>` |
| `npm start` fails | Run `npm install` first |

## Author

**Shashwat Sharma**

---
> **Note**: The `.h5` model file is not committed (too large). Train using the notebook to generate it.
