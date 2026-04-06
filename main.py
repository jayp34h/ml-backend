from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import pickle
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI(title="AgriNova ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- LOAD AND INSPECT MODEL ----
raw_pkl = None
model = None
labels = None
model_type_info = "unknown"

try:
    with open('plant_disease_model.pkl', 'rb') as f:
        raw_pkl = pickle.load(f)
    print(f"Pickle loaded. Type: {type(raw_pkl)}")

    if isinstance(raw_pkl, dict):
        print(f"Dict keys: {list(raw_pkl.keys())}")
        for key in ['model', 'classifier', 'estimator', 'net', 'clf']:
            if key in raw_pkl:
                model = raw_pkl[key]
                break
        for key in ['labels', 'classes', 'class_names', 'label_names']:
            if key in raw_pkl:
                labels = raw_pkl[key]
                break
        if model is None:
            for v in raw_pkl.values():
                if hasattr(v, 'predict'):
                    model = v
                    break
        model_type_info = f"dict-wrapped, model type: {type(model)}"

    elif isinstance(raw_pkl, list):
        for item in raw_pkl:
            if hasattr(item, 'predict') and model is None:
                model = item
            elif isinstance(item, (list, dict)) and labels is None:
                labels = item
        model_type_info = f"list-wrapped, model type: {type(model)}"

    elif hasattr(raw_pkl, 'predict'):
        model = raw_pkl
        model_type_info = f"direct model: {type(model)}"

    else:
        model_type_info = f"UNRECOGNIZED: {type(raw_pkl)}"

    print(f"Model info: {model_type_info}")

except Exception as e:
    print(f"Error loading model: {e}")


def process_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def read_root():
    return {"message": "AgriNova Crop Detection Server is Running!"}


@app.get("/debug")
def debug_model():
    return {
        "model_type_info": model_type_info,
        "model_loaded": model is not None,
        "labels": list(labels)[:10] if labels else "none",
        "raw_pkl_type": str(type(raw_pkl)),
        "raw_pkl_keys": list(raw_pkl.keys()) if isinstance(raw_pkl, dict) else "N/A",
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if model is None:
            return {
                "status": "error",
                "message": f"Model not ready. Visit /debug. Info: {model_type_info}"
            }

        image_bytes = await file.read()
        processed_image = process_image(image_bytes)
        prediction = model.predict(processed_image)

        confidence = 0.99
        try:
            raw_pred = prediction[0]
            if labels is not None:
                if isinstance(raw_pred, (int, np.integer)):
                    detected_disease = str(labels[int(raw_pred)])
                elif isinstance(raw_pred, np.ndarray):
                    idx = int(np.argmax(raw_pred))
                    detected_disease = str(labels[idx])
                    confidence = float(np.max(raw_pred))
                else:
                    detected_disease = str(raw_pred)
            else:
                if isinstance(raw_pred, np.ndarray):
                    detected_disease = str(int(np.argmax(raw_pred)))
                    confidence = float(np.max(raw_pred))
                else:
                    detected_disease = str(raw_pred)
        except Exception:
            detected_disease = str(prediction)
            confidence = 0.0

        return {
            "disease_name": detected_disease,
            "confidence": round(float(confidence), 4),
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
