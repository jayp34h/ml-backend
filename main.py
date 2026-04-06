from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import pickle
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI(title="AgriNova ML API")

# Add CORS so the Flutter web/app can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LOAD YOUR MODEL HERE
# We load the crop detection model
try:
    with open('plant_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def process_image(image_bytes):
    """
    Transforms the incoming image bytes into a format your model understands.
    You MUST adjust the image size (e.g., 224x224) based on what your model expects!
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224)) # Change to your model's input size
    
    # Convert image to numpy array and normalize if needed
    img_array = np.array(image) / 255.0
    
    # Add batch dimension (1, 224, 224, 3) generally expected by models
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.get("/")
def read_root():
    return {"message": "AgriNova ML Server is Running!"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # Read the image uploaded from Flutter
        image_bytes = await file.read()
        
        # Process image
        processed_image = process_image(image_bytes)
        
        # 2. RUN INFERENCE 
        if model is None:
            return {"status": "error", "message": "Model not loaded on server."}
        
        # Predict the image! Assuming the model returns a direct prediction format
        prediction = model.predict(processed_image)
        
        # Depending on how the `.pkl` was trained, the prediction format might be an array or string.
        # usually pred[0] is the result for standard scikit-learn/tensorflow.
        # We try to cleanly serialize it to string so JSON doesn't crash
        try:
            detected_disease = str(prediction[0])
            confidence = 0.99 # if model outputs probs, you extract them, otherwise putting placeholder
        except:
            detected_disease = str(prediction)
            confidence = 0.0
        
        result = {
            "disease_name": detected_disease,
            "confidence": confidence,
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
