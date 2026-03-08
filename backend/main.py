from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from io import BytesIO
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Plant Disease Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Plant name mapping ────────────────────────────────────────────────────────
# Handles inconsistent separators: _, __, ___ 
PLANT_KEYS = [
    ('Pepper__bell', 'Bell Pepper'),
    ('Potato',       'Potato'),
    ('Tomato',       'Tomato'),
]

def parse_class_name(folder_name: str):
    """Parse PlantVillage folder name into (plant_display, disease_display)."""
    for plant_key, plant_disp in PLANT_KEYS:
        if folder_name.startswith(plant_key):
            rest    = folder_name[len(plant_key):].lstrip('_')
            disease = rest.replace('_', ' ').replace('  ', ' ').strip().title()
            return plant_disp, disease or 'Unknown'
    return folder_name.replace('_', ' ').title(), 'Unknown'


# ── Model & class names loading ───────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
MODEL_PATH     = os.path.join(BASE_DIR, 'models', 'plant_model.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'class_names.txt')

model       = None
CLASS_NAMES = []

def load_model_on_startup():
    global model, CLASS_NAMES

    # Load class names from file (saved by notebook after training)
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(CLASS_NAMES)} class names")
    else:
        print("[WARNING] class_names.txt not found. Train the model first.")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at: {MODEL_PATH}")
        print("[WARNING] Run the Jupyter notebook to train and save the model.")
        return
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print(f"[INFO] Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

load_model_on_startup()


# ── Helper ────────────────────────────────────────────────────────────────────
def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/ping")
async def ping():
    return {
        "message"     : "pong",
        "model_loaded": model is not None,
        "num_classes" : len(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the Jupyter notebook to train the model first.",
        )

    image       = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)     # (1, 256, 256, 3)

    predictions     = model.predict(image_batch)
    predicted_index = int(np.argmax(predictions[0]))
    confidence      = float(predictions[0][predicted_index])

    raw_class          = CLASS_NAMES[predicted_index] if CLASS_NAMES else str(predicted_index)
    plant_name, disease = parse_class_name(raw_class)
    is_healthy         = 'healthy' in raw_class.lower()

    return {
        "plant"     : plant_name,
        "disease"   : disease,
        "is_healthy": is_healthy,
        "confidence": round(confidence, 4),
        "raw_class" : raw_class,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
