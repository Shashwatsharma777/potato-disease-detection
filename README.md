# Plant Disease Detection System

Hey there! 👋 Welcome to my Plant Disease Detection project. 

This application is built to help farmers, gardeners, or anyone who loves plants figure out if their crops are sick. By just uploading a clear picture of a leaf, the system uses deep learning to identify the specific disease affecting the plant. 

**How does it work?**
I trained a Convolutional Neural Network (CNN) from scratch using TensorFlow and Keras. The model looks at the patterns, spots, and colors on the leaf image and compares them against thousands of images it has seen before. It can currently recognize 15 different classes (including both healthy and diseased states) with an accuracy of around 92.7%. It's designed to be fast and user-friendly, with a clean React frontend and a FastAPI backend doing the heavy lifting.

**Supported Plants:** Potato, Tomato, and Bell Pepper


##  Quick Start
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

## Project Structure

* `training/` - Contains the Jupyter notebook used for model training and data preprocessing
* `backend/` - FastAPI server handling the inference (loads the trained model to serve predictions)
* `frontend/` - React frontend handling file upload and displaying results
* `PlantVillage/` - Dataset directory (images for potato, tomato, and bell pepper leaves)
* `saved_models/` - Important saved weights and class names generated during training

---

## Model Info
* **Architecture**: Custom CNN
* **Framework**: TensorFlow / Keras
* **Total Classes**: 15 (Various diseases + Healthy leaves across 3 plant types)
* **Test Accuracy**: ~92.7%

If you want to re-train the model quickly, you can use the `training/potato_disease_training.ipynb` notebook (Google Colab with a GPU is recommended for speed).

