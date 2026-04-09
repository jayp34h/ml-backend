from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
import pickle
import joblib
import warnings
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models as keras_models
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------
# Fix for pickles saved with NumPy 2.x being loaded in NumPy 1.x
# NumPy 2.0 renamed numpy.core → numpy._core. This custom unpickler
# transparently redirects those references so the file loads cleanly.
# ---------------------------------------------------------------
class RenameUnpickler(pickle.Unpickler):
    _MAP = {
        'numpy._core.numeric':              'numpy.core.numeric',
        'numpy._core.multiarray':           'numpy.core.multiarray',
        'numpy._core.umath':                'numpy.core.umath',
        'numpy._core._multiarray_umath':    'numpy.core._multiarray_umath',
        'numpy._core':                      'numpy.core',
    }
    def find_class(self, module, name):
        module = self._MAP.get(module, module)
        return super().find_class(module, name)


# ---------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------
app = FastAPI(title="AgriNova ML API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# MODEL 1 — Crop Disease Detection  (plant_disease_model.pkl)
#   Type  : TensorFlow/Keras weights wrapped in pickle
#   Arch  : MobileNetV2
#   Input : Image (224×224 RGB by default)
# ---------------------------------------------------------------
disease_model = None
disease_class_names: list = []
disease_img_size: int = 224
disease_model_info: str = "not loaded"

try:
    with open('plant_disease_model.pkl', 'rb') as f:
        data = RenameUnpickler(f).load()

    disease_class_names = data.get('class_names', [])
    raw_size = data.get('img_size', 224)
    disease_img_size = int(raw_size[0]) if isinstance(raw_size, (tuple, list)) else int(raw_size)
    model_weights = data.get('model_weights', None)
    num_classes = data.get('num_classes', len(disease_class_names))

    print(f"[Disease Model] classes={num_classes}, img_size={disease_img_size}")

    if model_weights is not None:
        base = MobileNetV2(weights=None, include_top=False, input_shape=(disease_img_size, disease_img_size, 3))
        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        preds = layers.Dense(num_classes, activation='softmax')(x)
        disease_model = keras_models.Model(inputs=base.input, outputs=preds)
        disease_model.set_weights(model_weights)
        disease_model_info = f"MobileNetV2 loaded OK — {num_classes} classes"
    else:
        disease_model_info = "model_weights key is None in pkl"

    print(f"[Disease Model] {disease_model_info}")

except Exception as e:
    disease_model_info = f"Error during loading: {e}"
    print(f"[Disease Model] {disease_model_info}")


# ---------------------------------------------------------------
# MODEL 2 — Crop Recommendation  (best_model.pkl)
#   Type  : scikit-learn classifier
#   Input : [N, P, K, temperature, humidity, ph, rainfall]
#   Output: crop label string (e.g. "rice", "wheat")
# ---------------------------------------------------------------
rec_model = None
rec_model_info: str = "not loaded"

try:
    rec_model = joblib.load('best_model.pkl')
    rec_model_info = f"Loaded OK — type: {type(rec_model).__name__}"
    print(f"[Recommendation Model] {rec_model_info}")
except Exception as e:
    rec_model_info = f"Error during loading: {e}"
    print(f"[Recommendation Model] {rec_model_info}")


# ---------------------------------------------------------------
# Pydantic schema for Crop Recommendation input
# ---------------------------------------------------------------
class CropData(BaseModel):
    n: float
    p: float
    k: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def preprocess_image(image_bytes: bytes, size: int) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((size, size))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "AgriNova ML API is running!",
        "disease_model_ready": disease_model is not None,
        "recommendation_model_ready": rec_model is not None,
    }


@app.get("/debug")
def debug():
    return {
        "disease_model_info": disease_model_info,
        "disease_model_loaded": disease_model is not None,
        "disease_class_count": len(disease_class_names),
        "disease_img_size": disease_img_size,
        "rec_model_info": rec_model_info,
        "rec_model_loaded": rec_model is not None,
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if disease_model is None:
            return {"status": "error", "message": f"Model not ready: {disease_model_info}"}

        image_bytes = await file.read()
        processed = preprocess_image(image_bytes, disease_img_size)
        predictions = disease_model.predict(processed)
        idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        # --- SIMPLE VALIDATION (OOD Detection) ---
        avg_r = np.mean(processed[0, :, :, 0])
        avg_g = np.mean(processed[0, :, :, 1])
        avg_b = np.mean(processed[0, :, :, 2])
        
        # 1. Reject skin/flesh colored and red objects (e.g., legs, hands, brick walls)
        is_skin_colored = (avg_r > avg_g + 0.15) and (avg_r > avg_b + 0.20)
        if is_skin_colored:
            return {
                "status": "error", 
                "message": "Incorrect photo detected. The image appears to be a body part or unrelated object. Please upload a clear photo of a crop leaf."
            }

        # 2. Reject grayscale/neutral/dark objects (laptops, keyboards, plain walls)
        # Realistic plant images have noticeable color variance across RGB channels (green/yellow/brown).
        # Grayscale objects have R ≈ G ≈ B, leading to very low standard deviation across the color channels.
        channel_std = np.std(processed[0], axis=2)
        mean_color_variance = np.mean(channel_std)
        is_neutral = mean_color_variance < 0.05
        
        # Also reject extremely dark photos where features can't be distinguished
        is_very_dark = np.max([avg_r, avg_g, avg_b]) < 0.15

        if is_neutral or is_very_dark:
            return {
                "status": "error",
                "message": "Incorrect photo detected. This appears to be a non-plant object (e.g., device, table, plain background) or the photo is too dark. Please upload a clear crop leaf."
            }

        # 3. Reject predominantly blue objects (e.g., clothing, screens, sky)
        is_blueish = (avg_b > avg_g + 0.05) and (avg_b > avg_r + 0.05)
        if is_blueish:
            return {
                "status": "error",
                "message": "Incorrect photo detected. This appears to be an unrelated object (too much blue). Please upload a valid crop leaf."
            }

        # 4. Reject low confidence predictions (often indicates unrelated objects)
        if confidence < 0.75:
            return {
                "status": "error", 
                "message": "Incorrect photo. The AI could not confidently detect a crop disease. Please upload a clear photo of a crop leaf."
            }

        if disease_class_names and idx < len(disease_class_names):
            raw = disease_class_names[idx]
            display = raw.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
        else:
            display = f"Class {idx}"

        return {"disease_name": display, "confidence": round(confidence, 4), "status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/recommend-crop")
def recommend_crop(data: CropData):
    if rec_model is None:
        return {"status": "error", "message": f"Model not ready: {rec_model_info}"}

    try:
        features = np.array([[
            data.n, data.p, data.k,
            data.temperature, data.humidity,
            data.ph, data.rainfall,
        ]])

        prediction = rec_model.predict(features)
        
        crop_dict = {
            0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea",
            4: "coconut", 5: "coffee", 6: "cotton", 7: "grapes",
            8: "jute", 9: "kidneybeans", 10: "lentil", 11: "maize",
            12: "mango", 13: "mothbeans", 14: "mungbean", 15: "muskmelon",
            16: "orange", 17: "papaya", 18: "pigeonpeas", 19: "pomegranate",
            20: "rice", 21: "watermelon"
        }

        if len(prediction) > 0:
            pred_val = int(prediction[0])
            crop = crop_dict.get(pred_val, "Unknown")
        else:
            crop = "Unknown"

        # Nicely capitalise (e.g. "kidneybeans" → "Kidney Beans")
        if crop == "kidneybeans":
            crop = "Kidney Beans"
        elif crop == "mothbeans":
            crop = "Moth Beans"
        elif crop == "mungbean":
            crop = "Mung Bean"
        elif crop == "pigeonpeas":
            crop = "Pigeon Peas"
        elif crop == "blackgram":
            crop = "Black Gram"
        else:
            crop = crop.title()

        return {"recommended_crop": crop, "status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
