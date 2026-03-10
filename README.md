# 🌿 Plant Disease Detection System

A complete end-to-end machine learning application for detecting diseases in plant leaves using deep learning. Upload a plant leaf image and get instant disease prediction with confidence scores.

**Supported Plants:** 🥔 Potato | 🍅 Tomato | 🫑 Bell Pepper

## ✅ Project Status: COMPLETE & RUNNING

**Test Accuracy:** 92.79% | **Best Val Accuracy:** 92.68% | **Total Classes:** 15 diseases + healthy

## 📊 Disease Classes

### 🥔 Potato (3 classes)
- Early Blight — 1,000 images
- Late Blight — 1,000 images
- Healthy — 152 images

### 🍅 Tomato (10 classes)
- Bacterial Spot — 2,127 images
- Early Blight — 1,000 images
- Late Blight — 1,909 images
- Leaf Mold — 952 images
- Septoria Leaf Spot — 1,771 images
- Spider Mites Two Spotted Spider Mite — 1,676 images
- Target Spot — 1,404 images
- Tomato Yellowleaf Curl Virus — 3,208 images
- Tomato Mosaic Virus — 373 images
- Healthy — 1,591 images

### 🫑 Bell Pepper (2 classes)
- Bacterial Spot — 997 images
- Healthy — 1,478 images

**Total Dataset:** 20,638 images across 15 classes

## 📁 Project Structure

```
potato-disease-detection/
├── training/
│   ├── potato_disease_training.ipynb    # Multi-plant Jupyter notebook
│   ├── train_model.py                   # Python script (non-interactive)
│   ├── saved_models/
│   │   ├── plant_model.h5               # Trained model (5.1 MB)
│   │   ├── class_names.txt              # 15 class names
│   │   └── best_plant_model.h5          # Best checkpoint
│   ├── training_outputs/                # Generated plots (6 images)
│   └── jupyter_utils.py
│
├── backend/
│   ├── main.py                          # FastAPI server
│   ├── requirements.txt                 # Dependencies
│   └── models/
│       ├── plant_model.h5               # Model copy for inference
│       └── class_names.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js                       # Main React component
│   │   ├── App.css                      # Green theme styling
│   │   ├── components/
│   │   │   ├── ImageUpload.js
│   │   │   └── PredictionResult.js
│   │   └── index.js
│   ├── public/index.html
│   ├── package.json
│   └── .env                             # API endpoint config
│
├── PlantVillage/                        # Dataset directory
│   ├── Pepper__bell___*
│   ├── Potato___*
│   └── Tomato_*
│
├── venv/                                # Python virtual environment
├── .gitignore
├── PROJECT_DOCUMENTATION.md
└── README.md                            # This file
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | TensorFlow 2.15.0 + Keras |
| **Training** | Python 3.9, scikit-learn, NumPy |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Axios |
| **File Upload** | React Dropzone |
| **Database** | File-based (class_names.txt) |

## 🚀 Quick Start

### Prerequisites
```bash
# Check versions
python --version          # 3.9+
node --version           # 14+
npm --version            # 6+
```

### Step 1 — Activate Virtual Environment

```bash
cd /Users/shashwat/Desktop/potato-disease-detection
source venv/bin/activate
```

### Step 2 — Start Backend

```bash
cd backend
python main.py
```

✅ Backend runs at: **http://localhost:8000**

Test it:
```bash
curl http://localhost:8000/ping
# Response: {"message":"pong","model_loaded":true,"num_classes":15}
```

### Step 3 — Start Frontend (New Terminal)

```bash
cd frontend
npm start
```

✅ Frontend opens at: **http://localhost:3000**

### Step 4 — Test the Application
1. Open http://localhost:3000 in browser
2. Drag & drop a plant leaf image (Potato, Tomato, or Bell Pepper)
3. Get instant prediction with disease name and confidence

---

## 📡 API Endpoints

### Health Check
```
GET /ping

Response:
{
  "message": "pong",
  "model_loaded": true,
  "num_classes": 15
}
```

### Predict Disease
```
POST /predict

Headers: Content-Type: multipart/form-data
Body: file: <image.jpg>

Response:
{
  "plant": "Tomato",
  "disease": "Bacterial Spot",
  "is_healthy": false,
  "confidence": 0.97,
  "raw_class": "Tomato_Bacterial_spot"
}
```

## 🧠 Model Architecture

```
Input (256×256 RGB Image)
  ↓
Resizing + Rescaling (1/255)
  ↓
Conv2D(32, 3×3) + BatchNorm + MaxPool(2×2)
Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2)
Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2)
Conv2D(128, 3×3) + BatchNorm + MaxPool(2×2)
Conv2D(128, 3×3) + BatchNorm + MaxPool(2×2)
Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2)
  ↓
Flatten → 256 units
  ↓
Dense(256) + Dropout(0.4) + ReLU
Dense(64) + Dropout(0.2) + ReLU
Dense(15) + Softmax
  ↓
Output: 15-class probability distribution
```

### Training Configuration

| Setting | Value |
|---------|-------|
| **Total Parameters** | 436,687 |
| **Image Size** | 256×256 pixels |
| **Batch Size** | 32 |
| **Epochs** | 50 (stopped at 33) |
| **Learning Rate** | 0.001 (reduced to 0.0001) |
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |

### Data Augmentation
- Random horizontal & vertical flips
- Random rotations (0-20°)
- Random zoom (±10%)
- Random contrast (±10%)

### Regularization Techniques
- **BatchNormalization** after each Conv layer
- **Dropout(0.4)** after first Dense layer
- **Dropout(0.2)** after second Dense layer
- **Class weights** for imbalanced data (Healthy classes underrepresented)
- **EarlyStopping** with patience=8
- **ReduceLROnPlateau** to reduce learning rate when stuck

### Training Results

| Metric | Value |
|--------|-------|
| **Training Time** | ~9 hours (CPU) |
| **Final Epoch** | 33 (EarlyStopping) |
| **Train Accuracy** | 99.78% |
| **Validation Accuracy** | 92.68% |
| **Test Accuracy** | **92.79%** ✅ |
| **Model Size** | 5.1 MB |

### Performance by Plant

| Plant | Classes | Test Accuracy | Notes |
|-------|---------|---------------|-------|
| 🥔 Potato | 3 | 91-99% | Excellent precision |
| 🍅 Tomato | 10 | 82-99% | High recall on diseases |
| 🫑 Bell Pepper | 2 | 94-100% | Perfect on healthy class |

---

## 📈 Training History

- **Epoch 1:** Val Acc = 48.4%
- **Epoch 6:** Val Acc = 78.3% (improved)
- **Epoch 13:** Val Acc = 85.5% (improved)
- **Epoch 22:** Val Acc = 91.3% (improved)
- **Epoch 25:** Val Acc = **92.7%** (BEST) ✅
- **Epoch 33:** Early stopping triggered (no improvement for 8 epochs)



## 📚 Training on GPU (Faster)

For faster training (~2-3 hours instead of 9 hours):

1. **Use Google Colab:**
   - Upload `Plant_Disease_Detection_COLAB.ipynb` to Colab
   - Upload `PlantVillage/` dataset to Google Drive
   - Enable GPU: Runtime → Change runtime type → GPU
   - Run all cells

2. **Download trained model:**
   - Copy `plant_model.h5` and `class_names.txt` from Colab
   - Place in `backend/models/`
   - Restart backend

---

## 📊 Classification Report (Test Set)

```
                         precision  recall  f1-score  support
Bell Pepper - Bact Spot      0.90    0.97      0.94      115
    Bell Pepper - Healthy    0.94    1.00      0.97      144
   Potato - Early Blight     0.91    0.99      0.95      102
    Potato - Late Blight     0.92    0.90      0.91       79
        Potato - Healthy     0.96    0.88      0.92       25
 Tomato - Bacterial Spot     0.97    0.96      0.96      206
   Tomato - Early Blight     0.91    0.82      0.86       94
    Tomato - Late Blight     0.97    0.87      0.92      205
      Tomato - Leaf Mold     1.00    0.86      0.93      110
Tomato - Septoria Leaf Spot  0.86    0.95      0.90      173
Tomato - Spider Mites        0.90    0.85      0.87      151
    Tomato - Target Spot     0.91    0.83      0.87      132
Tomato - Yellowleaf Virus    0.99    0.96      0.97      340
 Tomato - Mosaic Virus       0.93    0.98      0.96       44
        Tomato - Healthy     0.83    1.00      0.91      160

              Accuracy                         0.93     2080
           Macro Avg          0.93    0.92      0.92     2080
        Weighted Avg          0.93    0.93      0.93     2080
```

---

## 🎯 Next Steps (Optional Improvements)

- [ ] Deploy to AWS/GCP/Heroku
- [ ] Add more plant species
- [ ] Implement batch prediction
- [ ] Add prediction history logging
- [ ] Create mobile app (React Native)
- [ ] Use transfer learning (ResNet50, EfficientNet)
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions

---

## 📞 Support

**GitHub Repository:** https://github.com/Shashwatsharma777/potato-disease-detection

For issues or questions, please create a GitHub issue.

---

## 📜 License

This project uses the PlantVillage dataset which is publicly available for research purposes.

PlantVillage Dataset: https://plantvillage.psu.edu/

---

## ✨ Project Completion Summary

| Task | Status | Details |
|------|--------|---------|
| **Data Collection** | ✅ Complete | 20,638 images, 15 classes |
| **Model Training** | ✅ Complete | 92.79% test accuracy |
| **Backend API** | ✅ Complete | FastAPI, fully functional |
| **Frontend UI** | ✅ Complete | React, drag-and-drop upload |
| **GitHub Push** | ✅ Complete | All code uploaded |
| **Documentation** | ✅ Complete | README + PROJECT_DOCUMENTATION.md |
| **Testing** | ✅ Complete | Model tested successfully |

---

**Created:** March 2026
**Status:** 🎉 **COMPLETE AND RUNNING**
**Test Accuracy:** 92.79% ✅
