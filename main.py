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

# ---- LOAD MODEL ----
model = None
class_names = []
img_size = 224
model_info = "not loaded"

try:
    with open('plant_disease_model.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"PKL keys: {list(data.keys())}")

    class_names = data.get('class_names', [])
    img_size = data.get('img_size', 224)
    model_weights = data.get('model_weights', None)
    num_classes = data.get('num_classes', len(class_names))

    print(f"Classes: {num_classes}, img_size: {img_size}")
    print(f"class_names: {class_names}")

    if model_weights is not None:
        # Reconstruct a MobileNetV2 model (most common lightweight plant disease architecture)
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, models

        base = MobileNetV2(weights=None, include_top=False, input_shape=(img_size, img_size, 3))
        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        predictions = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base.input, outputs=predictions)
        model.set_weights(model_weights)
        model_info = f"MobileNetV2 reconstructed. Classes: {num_classes}"
        print(model_info)
    else:
        model_info = "model_weights key is None in pkl"
        print(model_info)

except Exception as e:
    model_info = f"Error during loading: {str(e)}"
    print(model_info)


def process_image(image_bytes, size):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((size, size))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


@app.get("/")
def read_root():
    return {"message": "AgriNova Crop Detection Server is Running!", "model_ready": model is not None}


@app.get("/debug")
def debug_model():
    return {
        "model_info": model_info,
        "model_loaded": model is not None,
        "class_names": class_names,
        "img_size": img_size,
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if model is None:
            return {
                "status": "error",
                "message": f"Model not ready: {model_info}"
            }

        image_bytes = await file.read()
        processed = process_image(image_bytes, img_size)
        predictions = model.predict(processed)
        idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        if class_names and idx < len(class_names):
            # Clean up underscore-formatted names for display
            raw_name = class_names[idx]
            display_name = raw_name.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
        else:
            display_name = f"Class {idx}"

        return {
            "disease_name": display_name,
            "confidence": round(confidence, 4),
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
