from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
import joblib
import warnings
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf
warnings.filterwarnings("ignore", category=UserWarning)

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
# MODEL 1 — Crop Disease Detection  (plant_disease.tflite)
#   Type  : TensorFlow Lite model
#   Arch  : CNN optimised for plant disease classification
#   Input : Image (224×224 RGB, normalised to [0, 1])
#   Output: 38-class softmax probabilities (PlantVillage dataset)
# ---------------------------------------------------------------

# Standard PlantVillage 38-class labels (alphabetical order used during training)
DISEASE_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

disease_interpreter = None
disease_input_details = None
disease_output_details = None
disease_class_names: list = DISEASE_CLASS_NAMES
disease_img_size: int = 224
disease_model_info: str = "not loaded"

try:
    disease_interpreter = tf.lite.Interpreter(model_path="plant_disease.tflite")
    disease_interpreter.allocate_tensors()
    disease_input_details  = disease_interpreter.get_input_details()
    disease_output_details = disease_interpreter.get_output_details()

    # Auto-detect image size from the model's input tensor shape [1, H, W, C]
    input_shape = disease_input_details[0]['shape']
    disease_img_size = int(input_shape[1])   # height dimension

    disease_model_info = (
        f"TFLite loaded OK — {len(disease_class_names)} classes, "
        f"input {disease_img_size}×{disease_img_size}"
    )
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
@app.head("/")
def root():
    return {
        "message": "AgriNova ML API is running!",
        "disease_model_ready": disease_interpreter is not None,
        "recommendation_model_ready": rec_model is not None,
    }


@app.get("/debug")
def debug():
    return {
        "disease_model_info": disease_model_info,
        "disease_model_loaded": disease_interpreter is not None,
        "disease_class_count": len(disease_class_names),
        "disease_img_size": disease_img_size,
        "rec_model_info": rec_model_info,
        "rec_model_loaded": rec_model is not None,
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if disease_interpreter is None:
            return {"status": "error", "message": f"Model not ready: {disease_model_info}"}

        image_bytes = await file.read()
        processed = preprocess_image(image_bytes, disease_img_size)   # shape: (1, H, W, 3) float32

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

        # --- TFLite Inference ---
        # Ensure input dtype matches what the model expects (usually float32)
        input_dtype = disease_input_details[0]['dtype']
        input_tensor = processed.astype(input_dtype)

        disease_interpreter.set_tensor(disease_input_details[0]['index'], input_tensor)
        disease_interpreter.invoke()
        output_data = disease_interpreter.get_tensor(disease_output_details[0]['index'])  # shape: (1, num_classes)

        idx = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))

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
