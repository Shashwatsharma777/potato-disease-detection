# 🌿 Plant Disease Detection System

A machine learning application for detecting diseases in plant leaves using deep learning. Just upload a leaf image to get an instant disease prediction.

**Supported Plants:** Potato, Tomato, and Bell Pepper

## 🚀 Quick Start
To run this project locally, follow these steps:

### Prerequisites
Make sure you have Python and Node.js installed.

### Step 1 — Activate the Environment
```bash
cd potato-disease-detection
source venv/bin/activate
```

### Step 2 — Start Backend
```bash
cd backend
python main.py
```
Backend runs at `http://localhost:8000`.

### Step 3 — Start Frontend
In a new terminal:
```bash
cd frontend
npm start
```
Frontend opens at `http://localhost:3000`. 

Now you can upload an image from the frontend and test the model!

---

## 📁 Project Structure

* `training/` - Contains the Jupyter notebook used for model training and data preprocessing
* `backend/` - FastAPI server handling the inference (loads the trained model to serve predictions)
* `frontend/` - React frontend handling file upload and displaying results
* `PlantVillage/` - Dataset directory (images for potato, tomato, and bell pepper leaves)
* `saved_models/` - Important saved weights and class names generated during training

---

## 🧠 Model Info
* **Architecture**: Custom CNN
* **Framework**: TensorFlow / Keras
* **Total Classes**: 15 (Various diseases + Healthy leaves across 3 plant types)
* **Test Accuracy**: ~92.7%

If you want to re-train the model quickly, you can use the `training/potato_disease_training.ipynb` notebook (Google Colab with a GPU is recommended for speed).

