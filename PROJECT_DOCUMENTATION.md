# 🥔 Potato Disease Detection - Project Documentation

## Project Overview

**Potato Disease Detection using CNN** is an end-to-end machine learning project that uses a Convolutional Neural Network (CNN) to classify potato leaf diseases. The system includes:
- A trained deep learning model
- A FastAPI backend server
- A React frontend with drag-and-drop interface
- Complete training pipeline with data preprocessing

---

## What We've Created (Current Status)

### ✅ Completed Files & Structure

```
potato-disease-detection/
├── training/
│   ├── potato_disease_training.ipynb    # Jupyter notebook for model training
│   ├── jupyter_utils.py                 # Utility functions
│   └── saved_models/
│       └── potato_model.h5              # (Will be created after training)
│
├── backend/
│   ├── main.py                          # FastAPI server
│   ├── requirements.txt                 # Python dependencies
│   └── models/
│       └── potato_model.h5              # (Model copy for backend)
│
├── frontend/
│   ├── public/
│   │   └── index.html                   # HTML template
│   ├── src/
│   │   ├── App.js                       # Main React component
│   │   ├── App.css                      # Styling
│   │   ├── components/
│   │   │   └── ImageUpload.js           # Drag-drop upload component
│   │   ├── styles/
│   │   │   └── ImageUpload.css          # Upload component styles
│   │   └── index.js
│   ├── package.json
│   └── .env
│
├── .gitignore                           # Git ignore rules
├── README.md                            # Project README
└── PROJECT_DOCUMENTATION.md             # This file

Dataset/
├── Potato___Early_blight/               # 1000 early blight images
├── Potato___Late_blight/                # 1000 late blight images
└── Potato___Healthy/                    # 152 healthy images
```

---

## What Each Component Does

### 1. **Training (Jupyter Notebook)**
**File**: `training/potato_disease_training.ipynb`

**Purpose**: Train the CNN model using the dataset

**Steps in notebook**:
1. Load dataset from `Dataset/` folder
2. Explore and visualize sample images
3. Preprocess images (256x256 pixels, normalization)
4. Apply data augmentation (flips, rotations)
5. Build CNN architecture with 6 convolutional layers
6. Train model for 50 epochs with class weights
7. Evaluate on validation set
8. Save model to `saved_models/potato_model.h5`

**What's happening**:
- Handles class imbalance (Healthy: 152 vs others: 1000 each) with weighted loss
- Uses 80% train, 10% validation, 10% test split
- Trains on images and learns to distinguish between diseases

### 2. **Backend (FastAPI Server)**
**File**: `backend/main.py`

**Purpose**: REST API server that serves predictions

**Endpoints**:
- `GET /ping` → Health check (returns "pong")
- `POST /predict` → Accept image file and return disease prediction

**How it works**:
1. Loads the trained model from `models/potato_model.h5`
2. Receives image upload from frontend
3. Preprocesses image
4. Runs inference through model
5. Returns JSON with:
   - `class`: Disease name ("Early Blight", "Late Blight", or "Healthy")
   - `confidence`: Prediction confidence (0-1)

**CORS Configuration**: Allows requests from `http://localhost:3000` (React frontend)

### 3. **Frontend (React App)**
**Files**: `frontend/src/App.js`, `frontend/src/components/ImageUpload.js`

**Purpose**: User interface for disease detection

**Features**:
- Drag-and-drop image upload
- Real-time image preview
- Sends image to backend API
- Displays prediction result with confidence
- Color-coded results (Green: Healthy, Red: Disease)

**How it works**:
1. User drags/drops image
2. App sends to backend `/predict` endpoint
3. Shows loading spinner
4. Displays result with disease type and confidence percentage

---

## Git Commands Used

### Initialize Repository (Not Done Yet - Next Step)
```bash
cd /Users/shashwat/Desktop/potato-disease-detection
git init
```

### Stage All Files
```bash
git add .
```

### Create First Commit
```bash
git commit -m "Initial commit: Add CNN model training and web application"
```

### Add Remote Repository (When Ready)
```bash
git remote add origin https://github.com/Shashwatsharma777/potato-disease-detection.git
git branch -M main
git push -u origin main
```

---

## How to Use This Project

### Step 1: Activate Virtual Environment
```bash
cd /Users/shashwat/Desktop/potato-disease-detection
source venv/bin/activate
```

### Step 2: Train the Model
```bash
jupyter notebook potato_disease_training.ipynb
```
- Run all cells (Shift + Enter on each cell)
- Model will be saved to `saved_models/potato_model.h5`
- Also copies to `backend/models/potato_model.h5`

### Step 3: Start Backend Server
```bash
cd /Users/shashwat/Desktop/potato-disease-detection/backend
pip install -r requirements.txt
python main.py
```
- Server runs on `http://localhost:8000`
- API available at `http://localhost:8000/predict`

### Step 4: Start Frontend
In another terminal:
```bash
cd /Users/shashwat/Desktop/potato-disease-detection/frontend
npm install
npm start
```
- Opens at `http://localhost:3000`
- Upload images to get predictions

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **ML Model** | TensorFlow/Keras | 2.15.0 |
| **Backend** | FastAPI | 0.109.0 |
| **Backend Server** | Uvicorn | 0.27.0 |
| **Frontend** | React | 18.2.0 |
| **Upload Library** | React Dropzone | 14.2.3 |
| **HTTP Client** | Axios | 1.6.5 |
| **Image Processing** | Pillow | 10.2.0 |
| **Data Manipulation** | NumPy | 1.26.3 |

---

## Model Architecture

```
Input Layer (256x256 RGB Images)
    ↓
Conv2D(32) + ReLU + MaxPool(2x2)
    ↓
Conv2D(64) + ReLU + MaxPool(2x2)
    ↓
Conv2D(64) + ReLU + MaxPool(2x2)
    ↓
Conv2D(64) + ReLU + MaxPool(2x2)
    ↓
Conv2D(64) + ReLU + MaxPool(2x2)
    ↓
Conv2D(64) + ReLU + MaxPool(2x2)
    ↓
Flatten()
    ↓
Dense(64) + ReLU
    ↓
Dense(3) + Softmax
    ↓
Output: [Early Blight, Late Blight, Healthy]
```

---

## Dependencies Installed

### Backend (`backend/requirements.txt`)
- tensorflow==2.15.0
- fastapi==0.109.0
- uvicorn==0.27.0
- python-multipart==0.0.9
- pillow==10.2.0
- numpy==1.26.3

### Frontend (`frontend/package.json`)
- react@^18.2.0
- react-dom@^18.2.0
- react-dropzone@^14.2.3
- axios@^1.6.5

### Training
- All backend dependencies plus:
- matplotlib
- seaborn
- scikit-learn
- jupyter

---

## Environment Variables

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:8000/predict
```

This tells the React app where to send image files for prediction.

---

## Expected Results After Training

- **Training Accuracy**: >90%
- **Validation Accuracy**: >85%
- **Test Accuracy**: >85%
- **Model Size**: ~50-100 MB (h5 format)
- **Training Time**: ~10-20 minutes (depends on hardware)

---

## Next Steps

1. ✅ Files created and organized
2. ⏳ Train the model (run Jupyter notebook)
3. ⏳ Copy model to backend folder
4. ⏳ Test backend API
5. ⏳ Test frontend interface
6. ⏳ Push to GitHub
7. ⏳ Deploy to cloud (optional)

---

## File Sizes Reference

```
Dataset/
├── Early Blight images: ~1 GB
├── Late Blight images: ~1 GB
└── Healthy images: ~150 MB

trained potato_model.h5: ~50-100 MB
```

---

## Troubleshooting

### Issue: "Dataset not found"
- **Solution**: Ensure `Dataset/` folder is in `/Users/shashwat/Desktop/potato-disease-detection/`

### Issue: CORS error in frontend
- **Solution**: Check `backend/main.py` has `allow_origins=["http://localhost:3000"]`

### Issue: Port 8000 already in use
- **Solution**: `lsof -i :8000` to find process, then `kill <PID>`

### Issue: Port 3000 already in use
- **Solution**: `lsof -i :3000` to find process, then `kill <PID>`

---

## Project Creator
**Shashwat Sharma**

---

## Last Updated
**February 9, 2026**

---

Generated with ❤️ using Claude Code
